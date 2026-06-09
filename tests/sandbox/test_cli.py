"""Behavioral tests for the sandbox CLI entry point."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner

from tests.sandbox.conftest import make_manifest
from benchmark_runner.sandbox.cli import cli
from benchmark_runner.sandbox.store import install_manifest


@pytest.fixture
def contract_file(tmp_path: Path) -> str:
    p = tmp_path / "contract.yaml"
    p.write_text(
        "name: test-agent\n"
        "run_cmd: agent run --model {model} --problem {problem_statement_path} --task {task_id}\n"
        "final_output: /app/results\n"
    )
    return str(p)


def test_run_maps_args_to_run_benchmark(contract_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI args are forwarded correctly to run_benchmark."""
    calls: list[dict] = []

    async def fake_run_benchmark(**kwargs) -> None:  # type: ignore[return]
        calls.append(kwargs)

    fake_client = MagicMock()
    monkeypatch.setattr("benchmark_runner.sandbox.cli.run_benchmark", fake_run_benchmark)
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
    # Direct mode: contract comes from the file, never from the manifest store
    assert calls[0]["contract_path"] == Path(contract_file)
    assert calls[0]["contract"] is None
    assert calls[0]["results_dir"] == "results"


def test_run_no_task_ids_exits_nonzero(contract_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invoking run with no TASK_IDs exits non-zero with a UsageError."""
    monkeypatch.setattr("benchmark_runner.sandbox.cli.BenchmarkServiceClient", lambda *a, **kw: MagicMock())

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "--model", "m", "--run-id", "r", "--contract", contract_file],
    )

    assert result.exit_code != 0


def test_run_manifest_mode_uses_installed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --contract, the first positional is an installed benchmark name:
    empty task ids expand to all manifest tasks, the contract (incl. secrets) is
    built in memory, dataset/service URL come from the manifest, and results
    nest under <results-dir>/<benchmark>."""
    calls: list[dict] = []
    client_urls: list[str] = []

    async def fake_run_sandbox(**kwargs) -> None:  # type: ignore[return]
        calls.append(kwargs)

    def fake_client(url: str, **kwargs: object) -> MagicMock:
        client_urls.append(url)
        return MagicMock()

    monkeypatch.setattr("benchmark_runner.sandbox.cli.run_sandbox", fake_run_sandbox)
    monkeypatch.setattr("benchmark_runner.sandbox.cli.BenchmarkServiceClient", fake_client)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        install_manifest(make_manifest("mybench"))
        result = runner.invoke(cli, ["run", "--model", "m", "--run-id", "r", "mybench"])

    assert result.exit_code == 0, result.output
    (call,) = calls
    assert call["task_ids"] == ["task-1", "task-2"]  # empty task ids = all tasks
    assert call["contract_path"] is None
    assert call["contract"].secrets == {"GOOGLE_API_KEY": "projects/x/google"}
    assert "{problem_statement_path}" in call["contract"].run_cmd
    assert call["dataset"] == "mybench-dataset"
    assert call["results_dir"] == str(Path("results") / "mybench")
    assert client_urls == ["http://svc"]  # manifest's service.url


def test_run_manifest_mode_unknown_name_lists_installed(tmp_path: Path) -> None:
    """An uninstalled benchmark name fails fast and names what IS installed."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        install_manifest(make_manifest("mybench"))
        result = runner.invoke(cli, ["run", "--model", "m", "--run-id", "r", "nope"])

    assert result.exit_code != 0
    assert "not installed" in result.output
    assert "mybench" in result.output


def test_add_and_list_flow(tmp_path: Path) -> None:
    """add installs into ./benchmarks with a summary; re-add prints a pin diff
    (or 'no pin changes'); list shows installed manifests with short digests."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No benchmarks installed" in result.output

        manifest_file = Path("mybench.yaml")
        manifest_file.write_text(yaml.safe_dump(make_manifest("mybench").model_dump(), sort_keys=False))

        result = runner.invoke(cli, ["add", str(manifest_file)])
        assert result.exit_code == 0, result.output
        assert Path("benchmarks/mybench.manifest.yaml").exists()
        assert "2 tasks" in result.output
        assert "service version: 0.6.1" in result.output

        # Re-add identical → explicit "no pin changes"
        result = runner.invoke(cli, ["add", str(manifest_file)])
        assert result.exit_code == 0
        assert "no pin changes" in result.output

        # Re-add with a new agent digest → pin diff line before replacing
        changed = make_manifest("mybench", image="ghcr.io/vals-ai/agent@sha256:" + "b" * 64)
        manifest_file.write_text(yaml.safe_dump(changed.model_dump(), sort_keys=False))
        result = runner.invoke(cli, ["add", str(manifest_file)])
        assert result.exit_code == 0
        assert "agent.image" in result.output and "→" in result.output

        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "mybench" in result.output
        assert "b" * 12 in result.output  # short digest shown ...
        assert "b" * 64 not in result.output  # ... not the full one
