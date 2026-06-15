"""Sandbox execution backend for running benchmark agents in-sandbox."""

import io
import json
import shlex
import zipfile
from pathlib import Path, PurePosixPath

from benchmark_runner.sandbox.bundle import BUNDLE_DIR, AgentBundle
from benchmark_runner.sandbox.contract import AgentContract
from benchmark_runner.sandbox.protocols import SandboxLike
from benchmark_runner.schemas import GenerationResult, GenerationStatus

INSTALL_TIMEOUT_SEC = 600  # ceiling for the install step so a hung install can't block a run


def _format_exc(exc: BaseException) -> str:
    """Render as 'TypeName: message'; some exceptions (httpx.ReadTimeout) stringify
    to '', and the type name alone is what keeps those errors legible."""
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _error_result(*, task_id: str, model: str, error: str) -> GenerationResult:
    return GenerationResult(
        task_id=task_id,
        status=GenerationStatus.ERROR,
        data="",
        model=model,
        error=error,
    )


def _max_time_result(*, task_id: str, model: str, error: str) -> GenerationResult:
    return GenerationResult(
        task_id=task_id,
        status=GenerationStatus.MAX_TIME,
        data="",
        model=model,
        error=error,
    )


def _bundle_uploads(bundle: AgentBundle) -> list[tuple[str, bytes]]:
    uploads: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(bundle.zip_bytes)) as zf:
        for entry in zf.infolist():
            if entry.is_dir():
                continue
            path = PurePosixPath(entry.filename)
            if (
                len(path.parts) < 2
                or path.is_absolute()
                or "\\" in entry.filename
                or ".." in path.parts
                or path.parts[0] != bundle.root
            ):
                raise ValueError(f"agent bundle has an unsafe entry path: {entry.filename!r}")
            uploads.append((f"{BUNDLE_DIR}/{path.as_posix()}", zf.read(entry)))
    if not uploads:
        raise ValueError("agent bundle contains no files")
    return uploads


class SandboxGenerationBackend:
    async def generate(
        self,
        *,
        sandbox: SandboxLike,
        contract: AgentContract,
        task_id: str,
        model: str,
        problem_path: str,
        cwd: str,
        agent_timeout: float | None,
        log_dir: Path,
        bundle: AgentBundle | None = None,
    ) -> GenerationResult:
        try:
            # Without final_output there is no result file to read; fail before any exec.
            final_output = contract.final_output
            if final_output is None:
                return _error_result(
                    task_id=task_id,
                    model=model,
                    error="contract.final_output is not set; no generation file to read",
                )

            # Step 0: services may return a cwd the image does not pre-create.
            await sandbox.exec(f"mkdir -p {shlex.quote(cwd)}")

            install_cwd = bundle.install_path if bundle else cwd

            # A pinned bundle must be present before the install command runs.
            if bundle is not None:
                install_path_arg = shlex.quote(bundle.install_path)
                bundle_uploads = _bundle_uploads(bundle)
                parent_dirs = sorted({str(PurePosixPath(path).parent) for path, _ in bundle_uploads})
                prepare_result = await sandbox.exec(
                    f"rm -rf {install_path_arg}"
                    f" && mkdir -p {' '.join(shlex.quote(path) for path in parent_dirs)}"
                )
                if prepare_result.exit_code != 0:
                    return _error_result(
                        task_id=task_id,
                        model=model,
                        error=(
                            f"bundle prepare failed (exit {prepare_result.exit_code}): "
                            f"{prepare_result.output[:1024]}"
                        ),
                    )
                for remote_path, content in bundle_uploads:
                    await sandbox.upload_file(remote_path, content)

            # Timeouts are shell `timeout` prefixes inside the command string: passing
            # timeout= to exec would prefix OUTSIDE the `cd && ...` chain and break it.
            if contract.install_cmd:
                install_result = await sandbox.exec(
                    f"cd {shlex.quote(install_cwd)}"
                    f" && timeout -k 10 {INSTALL_TIMEOUT_SEC} {contract.install_cmd}"
                )
                if install_result.exit_code != 0:
                    return _error_result(
                        task_id=task_id,
                        model=model,
                        error=(
                            f"install failed (exit {install_result.exit_code}): "
                            f"{install_result.output[:1024]}"
                        ),
                    )

            # Step 2: run the agent. `timeout -k 10` SIGKILLs a process that ignores SIGTERM.
            run_cmd = (
                contract.run_cmd
                .replace("{problem_statement_path}", shlex.quote(problem_path))
                .replace("{task_id}", shlex.quote(task_id))
            )
            if agent_timeout:
                run_cmd = f"timeout -k 10 {int(agent_timeout)} {run_cmd}"
            result = await sandbox.exec(
                f"cd {shlex.quote(cwd)} && PYTHONSAFEPATH=1 {run_cmd}",
            )

            # Step 3: exit 124 = the shell timeout fired → MAX_TIME, not ERROR; nothing to download.
            if result.exit_code == 124:
                timeout_note = f" after {int(agent_timeout)}s" if agent_timeout else ""
                return _max_time_result(
                    task_id=task_id,
                    model=model,
                    error=f"agent timed out{timeout_note} (exit 124)",
                )

            # Step 4: download the agent's generation.json — even on a nonzero exit, since
            # the runner writes a structured result on its own failures; fall back to stdout
            # only when the file is genuinely absent.
            output_path = f"{final_output.rstrip('/')}/{task_id}/generation.json"
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            try:
                content = await sandbox.download_file(output_path)
            except Exception as download_exc:
                if result.exit_code != 0:
                    return _error_result(
                        task_id=task_id,
                        model=model,
                        error=(
                            f"agent exited {result.exit_code}, no generation file: "
                            f"{result.output[:4096]}"
                        ),
                    )
                return _error_result(
                    task_id=task_id,
                    model=model,
                    error=f"could not read generation file {output_path}: {download_exc}",
                )

            (log_dir / "generation_raw.json").write_bytes(content)
            parsed = GenerationResult.model_validate(json.loads(content))

            parsed.task_id = task_id
            if parsed.model is None:
                parsed.model = model

            return parsed
        except TimeoutError as exc:
            # SDK-level timeout (asyncio.TimeoutError aliases TimeoutError on 3.11+)
            return _max_time_result(task_id=task_id, model=model, error=_format_exc(exc))
        except Exception as exc:
            return _error_result(task_id=task_id, model=model, error=_format_exc(exc))
