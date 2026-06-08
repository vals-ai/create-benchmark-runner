"""Behavioral tests for the sandbox CLI entry point."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from benchmark_runner.sandbox.cli import cli


@pytest.fixture
def contract_file(tmp_path: Path) -> str:
    p = tmp_path / "contract.yaml"
    p.write_text(
        "name: test-agent\n"
        "run_cmd: agent run --model {model} --problem {problem_statement_path} --task {task_id}\n"
        "final_output: /app/results\n"
    )
    return str(p)


def test_run_maps_args_to_run_sandbox(contract_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI args are forwarded correctly to run_sandbox."""
    calls: list[dict] = []

    async def fake_run_sandbox(**kwargs) -> None:  # type: ignore[return]
        calls.append(kwargs)

    fake_client = MagicMock()
    monkeypatch.setattr("benchmark_runner.sandbox.cli.run_sandbox", fake_run_sandbox)
    monkeypatch.setattr("benchmark_runner.sandbox.cli.BenchmarkServiceClient", lambda *a, **kw: fake_client)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "--model", "m", "--run-id", "r", "--contract", contract_file, "t1", "t2"],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["task_ids"] == ["t1", "t2"]
    assert calls[0]["model"] == "m"
    assert calls[0]["run_id"] == "r"
    assert calls[0]["parallelism"] == 10


def test_run_no_task_ids_exits_nonzero(contract_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invoking run with no TASK_IDs exits non-zero with a UsageError."""
    monkeypatch.setattr("benchmark_runner.sandbox.cli.BenchmarkServiceClient", lambda *a, **kw: MagicMock())

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "--model", "m", "--run-id", "r", "--contract", contract_file],
    )

    assert result.exit_code != 0
