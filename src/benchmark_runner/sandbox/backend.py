"""Sandbox execution backend for running benchmark agents in-sandbox."""

import json
import shlex
from pathlib import Path
from typing import Protocol

from benchmark_runner.sandbox.contract import AgentContract
from benchmark_runner.schemas import GenerationResult, GenerationStatus

# cbs DaytonaSandbox.exec prefixes `timeout {t} {command}` as a shell command, so
# passing timeout= would produce `timeout 120 cd /app && ...` which breaks the cd chain.
# We rely solely on shell-level `timeout` prefixed into the command string instead.
#
# INSTALL_TIMEOUT_SEC: install step ceiling so a hung install can't block indefinitely.
INSTALL_TIMEOUT_SEC = 600  # 10 min; generous for slow image setups


class ExecResultLike(Protocol):
    exit_code: int
    output: str


class SandboxLike(Protocol):
    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResultLike: ...
    async def download_file(self, remote_path: str) -> bytes: ...


def _format_exc(exc: BaseException) -> str:
    """Render an exception as 'TypeName: message', keeping the type when the
    message is empty. Some exceptions (notably httpx.ReadTimeout) stringify to
    '', so a bare str(exc) saves a useless blank error; the type name is the
    signal that makes a timeout/connection failure legible in the artifact."""
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
    ) -> GenerationResult:
        try:
            # Step 0: ensure cwd exists before any `cd` into it.
            # The benchmark service may return a cwd that the image does not
            # pre-create (e.g. legal-research returns /workspace on an image with
            # WORKDIR /app); Valkyrie's tracker mkdir -p's it first, so mirror that.
            await sandbox.exec(f"mkdir -p {shlex.quote(cwd)}")

            # Step 1: optional install
            # Shell-prefix the install with a timeout so a hung install doesn't block forever.
            # Do NOT pass timeout= to sandbox.exec — cbs prefixes it as a shell command which
            # would break the `cd && ...` chain.
            if contract.install_cmd:
                install_result = await sandbox.exec(
                    f"cd {shlex.quote(cwd)} && timeout {INSTALL_TIMEOUT_SEC} {contract.install_cmd}"
                )
                if install_result.exit_code != 0:
                    # A broken install means the agent would fail with a confusing
                    # downstream error (ModuleNotFoundError etc.); fail fast instead.
                    return _error_result(
                        task_id=task_id,
                        model=model,
                        error=(
                            f"install failed (exit {install_result.exit_code}): "
                            f"{install_result.output[:4096]}"
                        ),
                    )

            # Step 2: build and run the agent command
            # Use `timeout -k 10` so a process ignoring SIGTERM is killed 10s later by SIGKILL.
            # Exit code 124 means the timeout fired (not a task failure → MAX_TIME, not ERROR).
            run_cmd = (
                contract.run_cmd
                .replace("{problem_statement_path}", problem_path)
                .replace("{task_id}", task_id)
            )
            if agent_timeout:
                run_cmd = f"timeout -k 10 {int(agent_timeout)} {run_cmd}"
            result = await sandbox.exec(
                f"cd {shlex.quote(cwd)} && PYTHONSAFEPATH=1 {run_cmd}",
            )

            # Step 3: no final_output configured → cannot read result
            if contract.final_output is None:
                return _error_result(
                    task_id=task_id,
                    model=model,
                    error="contract.final_output is not set; no generation file to read",
                )

            # Step 4: shell-level timeout → the process was killed before it could
            # write a result file, so don't attempt a download.
            # Exit code 124 = `timeout` fired → MAX_TIME, not ERROR.
            if result.exit_code == 124:
                timeout_note = f" after {int(agent_timeout)}s" if agent_timeout else ""
                return _max_time_result(
                    task_id=task_id,
                    model=model,
                    error=f"agent timed out{timeout_note} (exit 124)",
                )

            # Step 5: download and parse the agent's generation.json.
            # Attempt this even on a nonzero exit: the runner writes a structured
            # GenerationResult (status + a real `error`) on its own failures, which
            # is far more useful than raw stdout. Only fall back to stdout when the
            # file is genuinely absent (the agent never got far enough to write it).
            output_path = f"{contract.final_output.rstrip('/')}/{task_id}/generation.json"
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            try:
                content = await sandbox.download_file(output_path)
            except Exception as download_exc:
                # No result file → surface the run's exit code + captured stdout.
                if result.exit_code != 0:
                    return _error_result(
                        task_id=task_id,
                        model=model,
                        error=f"agent exited {result.exit_code}, no generation file: {result.output[:4096]}",
                    )
                return _error_result(
                    task_id=task_id,
                    model=model,
                    error=f"could not read generation file {output_path}: {download_exc}",
                )

            (log_dir / "generation_raw.json").write_bytes(content)
            parsed = GenerationResult.model_validate(json.loads(content))

            # Ensure task_id and model are set correctly
            parsed.task_id = task_id
            if parsed.model is None:
                parsed.model = model

            return parsed
        except TimeoutError as exc:
            # SDK-level timeout (asyncio.TimeoutError is an alias for TimeoutError on 3.11+)
            return _max_time_result(task_id=task_id, model=model, error=_format_exc(exc))
        except Exception as exc:
            return _error_result(task_id=task_id, model=model, error=_format_exc(exc))
