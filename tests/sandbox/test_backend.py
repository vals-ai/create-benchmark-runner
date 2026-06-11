"""Tests for SandboxGenerationBackend (TDD)."""

import json
import shlex
from pathlib import Path

from benchmark_runner.sandbox.backend import SandboxGenerationBackend, _format_exc
from benchmark_runner.sandbox.contract import AgentContract
from benchmark_runner.schemas import GenerationStatus


def test_format_exc_keeps_type_when_message_blank() -> None:
    """An empty-message exception (e.g. httpx.ReadTimeout('')) must still record its
    type — a bare str() would save a useless empty error string."""

    class ReadTimeout(Exception):
        pass

    assert _format_exc(ReadTimeout("")) == "ReadTimeout"
    assert _format_exc(ReadTimeout("boom")) == "ReadTimeout: boom"


class FakeExecResult:
    def __init__(self, exit_code: int = 0, output: str = "") -> None:
        self.exit_code = exit_code
        self.output = output


class FakeSandbox:
    def __init__(
        self,
        *,
        exec_exit_code: int = 0,
        exec_output: str = "",
        exec_raises: Exception | None = None,
        download_bytes: bytes | None = None,
        download_raises: Exception | None = None,
    ) -> None:
        self._exec_exit_code = exec_exit_code
        self._exec_output = exec_output
        self._exec_raises = exec_raises
        self._download_bytes = download_bytes
        self._download_raises = download_raises
        self.commands: list[str] = []
        self.timeouts: list[float | None] = []
        self.download_path: str | None = None

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> FakeExecResult:
        self.commands.append(command)
        self.timeouts.append(timeout)
        if self._exec_raises is not None:
            raise self._exec_raises
        return FakeExecResult(exit_code=self._exec_exit_code, output=self._exec_output)

    async def download_file(self, remote_path: str) -> bytes:
        self.download_path = remote_path
        if self._download_raises is not None:
            raise self._download_raises
        if self._download_bytes is not None:
            return self._download_bytes
        raise FileNotFoundError(f"no file at {remote_path}")


def _make_generation_json(task_id: str = "task-1") -> bytes:
    return json.dumps({
        "task_id": task_id,
        "status": "success",
        "data": "my answer",
        "model": "openai/gpt-5",
    }).encode()


def _make_contract(*, with_install: bool = True) -> AgentContract:
    return AgentContract(
        name="test-agent",
        run_cmd="agent run --problem {problem_statement_path} --task {task_id}",
        install_cmd="bash setup.sh" if with_install else None,
        final_output="/app/results",
    )


async def test_success_path(tmp_path: Path) -> None:
    """Install and run are executed; run command has substitutions; returns parsed result."""
    raw = _make_generation_json("task-1")
    sandbox = FakeSandbox(download_bytes=raw)
    backend = SandboxGenerationBackend()
    contract = _make_contract(with_install=True)

    result = await backend.generate(
        sandbox=sandbox,
        contract=contract,
        task_id="task-1",
        model="openai/gpt-5",
        problem_path="/problems/task-1.json",
        cwd="/app",
        agent_timeout=60.0,
        log_dir=tmp_path,
    )

    # Three commands: mkdir cwd + install + run
    assert len(sandbox.commands) == 3
    mkdir_cmd, install_cmd, run_cmd = sandbox.commands

    # cwd is created before anything cd's into it
    assert mkdir_cmd.startswith("mkdir -p ")

    # Install command contains the install_cmd
    assert "bash setup.sh" in install_cmd

    # Run command: substitutions applied
    assert "/problems/task-1.json" in run_cmd
    assert "--task task-1" in run_cmd
    assert "{problem_statement_path}" not in run_cmd
    assert "{task_id}" not in run_cmd
    # timeout -k prefix present (SIGKILL 10s after SIGTERM)
    assert "timeout -k 10 60" in run_cmd
    # PYTHONSAFEPATH set
    assert "PYTHONSAFEPATH=1" in run_cmd

    # SDK timeout= is NOT passed (cbs shell-prefixes it, which would break the cd chain)
    assert sandbox.timeouts[-1] is None

    # Download used the correct remote path
    assert sandbox.download_path == "/app/results/task-1/generation.json"

    # Raw bytes written to log_dir
    assert (tmp_path / "generation_raw.json").read_bytes() == raw

    # Result is the parsed generation
    assert result.status == GenerationStatus.SUCCESS
    assert result.data == "my answer"
    assert result.task_id == "task-1"


class _InstallFailsSandbox(FakeSandbox):
    """Install command exits 127; everything else succeeds."""

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> FakeExecResult:
        self.commands.append(command)
        self.timeouts.append(timeout)
        if "bash setup.sh" in command:
            return FakeExecResult(exit_code=127, output="bash: setup.sh: No such file or directory")
        return FakeExecResult(exit_code=0, output="")


async def test_install_failure_fails_fast_without_running_agent(tmp_path: Path) -> None:
    """Nonzero install exit → ERROR immediately; the agent never runs, so a broken
    install surfaces as 'install failed' instead of a confusing downstream error."""
    raw = _make_generation_json("task-1")
    sandbox = _InstallFailsSandbox(download_bytes=raw)
    backend = SandboxGenerationBackend()
    contract = _make_contract(with_install=True)

    result = await backend.generate(
        sandbox=sandbox,
        contract=contract,
        task_id="task-1",
        model="openai/gpt-5",
        problem_path="/problems/task-1.json",
        cwd="/app",
        agent_timeout=60.0,
        log_dir=tmp_path,
    )

    assert result.status == GenerationStatus.ERROR
    assert "install failed (exit 127)" in (result.error or "")
    # mkdir + install only — the agent did NOT run.
    assert len(sandbox.commands) == 2


async def test_missing_final_output_errors_before_any_exec(tmp_path: Path) -> None:
    """A contract without final_output is rejected before the sandbox runs anything."""
    sandbox = FakeSandbox()
    backend = SandboxGenerationBackend()
    contract = _make_contract(with_install=False).model_copy(update={"final_output": None})

    result = await backend.generate(
        sandbox=sandbox,
        contract=contract,
        task_id="task-1",
        model="openai/gpt-5",
        problem_path="/problems/task-1.json",
        cwd="/app",
        agent_timeout=60.0,
        log_dir=tmp_path,
    )

    assert result.status == GenerationStatus.ERROR
    assert "final_output is not set" in (result.error or "")
    assert sandbox.commands == []


async def test_nonzero_exit_returns_error(tmp_path: Path) -> None:
    """Non-zero exec exit code → ERROR status with exec output in error field."""
    exec_output = "agent crashed: out of memory"
    sandbox = FakeSandbox(exec_exit_code=1, exec_output=exec_output)
    backend = SandboxGenerationBackend()
    contract = _make_contract(with_install=False)

    result = await backend.generate(
        sandbox=sandbox,
        contract=contract,
        task_id="task-2",
        model="openai/gpt-5",
        problem_path="/problems/task-2.json",
        cwd="/app",
        agent_timeout=None,
        log_dir=tmp_path,
    )

    assert result.status == GenerationStatus.ERROR
    assert result.task_id == "task-2"
    assert exec_output in (result.error or "")


async def test_nonzero_exit_with_generation_file_parses_it(tmp_path: Path) -> None:
    """Nonzero exit BUT the agent wrote a structured generation.json → parse it
    (the agent's own status/error is richer than raw stdout)."""
    raw = json.dumps({
        "task_id": "task-2",
        "status": "error",
        "data": "",
        "model": "openai/gpt-5",
        "error": "agent-reported: tool call failed",
    }).encode()
    sandbox = FakeSandbox(exec_exit_code=1, exec_output="noisy stdout", download_bytes=raw)
    backend = SandboxGenerationBackend()
    contract = _make_contract(with_install=False)

    result = await backend.generate(
        sandbox=sandbox,
        contract=contract,
        task_id="task-2",
        model="openai/gpt-5",
        problem_path="/problems/task-2.json",
        cwd="/app",
        agent_timeout=None,
        log_dir=tmp_path,
    )

    assert result.status == GenerationStatus.ERROR
    assert result.error == "agent-reported: tool call failed"  # the agent's error, not stdout


async def test_missing_output_file_returns_error(tmp_path: Path) -> None:
    """download_file raises → ERROR status, no exception escapes."""
    sandbox = FakeSandbox(
        exec_exit_code=0,
        download_raises=FileNotFoundError("generation.json not found"),
    )
    backend = SandboxGenerationBackend()
    contract = _make_contract(with_install=False)

    result = await backend.generate(
        sandbox=sandbox,
        contract=contract,
        task_id="task-3",
        model="openai/gpt-5",
        problem_path="/problems/task-3.json",
        cwd="/app",
        agent_timeout=None,
        log_dir=tmp_path,
    )

    assert result.status == GenerationStatus.ERROR
    assert result.task_id == "task-3"
    assert result.error is not None


async def test_run_exec_raises_timeout_error_returns_max_time(tmp_path: Path) -> None:
    """SDK TimeoutError from run exec → MAX_TIME, not ERROR; no exception escapes generate()."""
    sandbox = FakeSandbox(exec_raises=TimeoutError("agent hung"))
    backend = SandboxGenerationBackend()
    contract = _make_contract(with_install=False)

    result = await backend.generate(
        sandbox=sandbox,
        contract=contract,
        task_id="task-4",
        model="openai/gpt-5",
        problem_path="/problems/task-4.json",
        cwd="/app",
        agent_timeout=60.0,
        log_dir=tmp_path,
    )

    assert result.status == GenerationStatus.MAX_TIME
    assert result.task_id == "task-4"
    assert result.error is not None


async def test_exit_code_124_returns_max_time(tmp_path: Path) -> None:
    """Exit code 124 (shell timeout) → MAX_TIME, not ERROR."""
    sandbox = FakeSandbox(exec_exit_code=124, exec_output="Killed")
    backend = SandboxGenerationBackend()
    contract = _make_contract(with_install=False)

    result = await backend.generate(
        sandbox=sandbox,
        contract=contract,
        task_id="task-5",
        model="openai/gpt-5",
        problem_path="/problems/task-5.json",
        cwd="/app",
        agent_timeout=60.0,
        log_dir=tmp_path,
    )

    assert result.status == GenerationStatus.MAX_TIME
    assert result.task_id == "task-5"
    assert result.error is not None


async def test_runtime_placeholders_are_shell_quoted(tmp_path: Path) -> None:
    """Service-provided paths and task IDs must not become shell syntax in the agent command."""
    raw = _make_generation_json("task weird; echo nope")
    sandbox = FakeSandbox(download_bytes=raw)
    backend = SandboxGenerationBackend()
    contract = AgentContract(
        name="test-agent",
        run_cmd="agent run --problem {problem_statement_path} --task {task_id}",
        final_output="/app/results",
    )

    await backend.generate(
        sandbox=sandbox,
        contract=contract,
        task_id="task weird; echo nope",
        model="openai/gpt-5",
        problem_path="/app/problems/task 1.txt; touch /tmp/pwned",
        cwd="/app",
        agent_timeout=None,
        log_dir=tmp_path,
    )

    run_cmd = sandbox.commands[-1]
    assert shlex.quote("/app/problems/task 1.txt; touch /tmp/pwned") in run_cmd
    assert shlex.quote("task weird; echo nope") in run_cmd
