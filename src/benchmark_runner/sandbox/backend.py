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
            # Step 1: optional install
            # Shell-prefix the install with a timeout so a hung install doesn't block forever.
            # Do NOT pass timeout= to sandbox.exec — cbs prefixes it as a shell command which
            # would break the `cd && ...` chain.
            if contract.install_cmd:
                await sandbox.exec(
                    f"cd {shlex.quote(cwd)} && timeout {INSTALL_TIMEOUT_SEC} {contract.install_cmd}"
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

            # Step 4: check exit code before attempting download
            # Exit code 124 = shell timeout fired → classify as MAX_TIME, not ERROR.
            if result.exit_code == 124:
                return _max_time_result(
                    task_id=task_id,
                    model=model,
                    error=f"agent timed out after {agent_timeout}s (exit 124)",
                )
            if result.exit_code != 0:
                return _error_result(
                    task_id=task_id,
                    model=model,
                    error=result.output[:4096],
                )

            # Step 5: download and parse
            output_path = f"{contract.final_output.rstrip('/')}/{task_id}/generation.json"
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            content = await sandbox.download_file(output_path)
            (log_dir / "generation_raw.json").write_bytes(content)
            parsed = GenerationResult.model_validate(json.loads(content))

            # Ensure task_id and model are set correctly
            parsed.task_id = task_id
            if parsed.model is None:
                parsed.model = model

            return parsed
        except TimeoutError as exc:
            # SDK-level timeout (asyncio.TimeoutError is an alias for TimeoutError on 3.11+)
            return _max_time_result(task_id=task_id, model=model, error=str(exc))
        except Exception as exc:
            return _error_result(task_id=task_id, model=model, error=str(exc))
