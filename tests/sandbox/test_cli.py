"""Behavioral tests for the sandbox CLI entry point."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner

from benchmark_service.sandbox import Resources
from benchmark_service.sandbox.types import ImageSource
from tests.sandbox.conftest import make_manifest
from benchmark_runner.sandbox.bundle import build_bundle_zip
from benchmark_runner.sandbox.cli import cli
from benchmark_runner.sandbox.manifest import AgentSpec, BundleSpec, DatasetSpec, EvalSpec, Manifest, ServiceSpec, TaskEntry
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


def _make_manifest_with_distinct_problem_paths(name: str = "mybench") -> Manifest:
    """Manifest where task-1 and task-2 have DIFFERENT problem_paths, proving
    _task_specs_from_manifest fans out per-task rather than reading agent-level."""
    image = "ghcr.io/vals-ai/agent@sha256:" + "a" * 64
    return Manifest(
        benchmark=name,
        service=ServiceSpec(url="http://svc", framework_version="1.0.0", service_version="0.6.1"),
        dataset=DatasetSpec(name=f"{name}-dataset"),
        agent=AgentSpec(
            install_cmd=None,
            run_cmd="agent run --model {model} --problem {problem_statement_path}",
            final_output="/app/results",
            required_env=["GOOGLE_API_KEY"],
        ),
        eval=EvalSpec(
            evaluate_endpoint="/evaluate-response/",
            score_endpoint="/final-score/",
            payload_schema=f"{name}.text.v1",
        ),
        tasks=[
            TaskEntry(
                id="task-1",
                question="Q1",
                timeout=60.0,
                image=image,
                resources=Resources(vcpu=2, memory=4, disk=10),
                cwd="/app",
                problem_path="/app/problem.txt",
            ),
            TaskEntry(
                id="task-2",
                question="Q2",
                timeout=60.0,
                image=image,
                resources=Resources(vcpu=2, memory=4, disk=10),
                cwd="/app",
                problem_path="/app/task2_problem.txt",
            ),
        ],
    )


def test_run_manifest_mode_uses_installed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --contract, the first positional is an installed benchmark name:
    empty task ids expand to all manifest tasks, the contract (incl. required_env) is
    built in memory, dataset/service URL come from the manifest, and results
    nest under <results-dir>/<benchmark>. problem_path is read per-task (not agent-level)."""
    calls: list[dict] = []
    client_urls: list[str] = []

    async def fake_run_benchmark(**kwargs) -> None:  # type: ignore[return]
        calls.append(kwargs)

    def fake_client(url: str, **kwargs: object) -> MagicMock:
        client_urls.append(url)
        return MagicMock()

    monkeypatch.setattr("benchmark_runner.sandbox.cli.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr("benchmark_runner.sandbox.cli.BenchmarkServiceClient", fake_client)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        install_manifest(_make_manifest_with_distinct_problem_paths("mybench"))
        result = runner.invoke(cli, ["run", "--model", "m", "--run-id", "r", "mybench"])

    assert result.exit_code == 0, result.output
    (call,) = calls
    assert call["task_ids"] == ["task-1", "task-2"]  # empty task ids = all tasks
    assert call["contract_path"] is None
    # manifest carries names only; reconstructed contract maps each name to itself
    assert call["contract"].secrets == {"GOOGLE_API_KEY": "GOOGLE_API_KEY"}
    assert "{problem_statement_path}" in call["contract"].run_cmd
    assert call["dataset"] == "mybench-dataset"
    assert call["results_dir"] == str(Path("results") / "mybench")
    assert client_urls == ["http://svc"]  # manifest's service.url
    assert call["task_specs"]["task-1"].source == ImageSource(
        image="ghcr.io/vals-ai/agent@sha256:" + "a" * 64
    )
    assert call["task_specs"]["task-1"].resources == Resources(vcpu=2, memory=4, disk=10)
    assert call["task_specs"]["task-1"].cwd == "/app"
    assert call["task_specs"]["task-1"].agent_timeout == 60.0
    # problem_path is per-task: each task entry has its own value (not agent-level)
    assert call["task_specs"]["task-1"].question == "Q1"
    assert call["task_specs"]["task-1"].problem_path == "/app/problem.txt"
    assert call["task_specs"]["task-2"].question == "Q2"
    assert call["task_specs"]["task-2"].problem_path == "/app/task2_problem.txt"


def test_run_manifest_mode_unknown_name_lists_installed(tmp_path: Path) -> None:
    """An uninstalled benchmark name fails fast and names what IS installed."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        install_manifest(make_manifest("mybench"))
        result = runner.invoke(cli, ["run", "--model", "m", "--run-id", "r", "nope"])

    assert result.exit_code != 0
    assert "not installed" in result.output
    assert "mybench" in result.output


def test_add_replaces_unreadable_installed_manifest(tmp_path: Path) -> None:
    """An installed manifest from an older schema must not block reinstalling:
    add warns, skips the pin diff, and replaces (a lab upgrading across a
    manifest-schema change hits exactly this)."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        store = Path("benchmarks")
        store.mkdir()
        (store / "mybench.manifest.yaml").write_text("benchmark: mybench\nagent: [not, the, schema]\n")

        manifest_file = Path("mybench.yaml")
        manifest_file.write_text(yaml.safe_dump(make_manifest("mybench").model_dump(), sort_keys=False))

        result = runner.invoke(cli, ["add", str(manifest_file)])
        assert result.exit_code == 0, result.output
        assert "unreadable" in result.output
        assert "Installed mybench" in result.output
        # Replaced copy is now loadable
        assert "mybench" in runner.invoke(cli, ["list"]).output


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

        # Re-add with a new image digest → per-task pin diff lines before replacing
        changed = make_manifest("mybench", image="ghcr.io/vals-ai/agent@sha256:" + "b" * 64)
        manifest_file.write_text(yaml.safe_dump(changed.model_dump(), sort_keys=False))
        result = runner.invoke(cli, ["add", str(manifest_file)])
        assert result.exit_code == 0
        assert "tasks.task-1.image" in result.output and "→" in result.output

        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "mybench" in result.output
        assert "b" * 12 in result.output  # short digest shown ...
        assert "b" * 64 not in result.output  # ... not the full one

def _write_bundled_manifest(name: str = "mybench", sha256: str | None = None) -> None:
    """In the CliRunner cwd: an agent dir, its bundle zip, and a manifest pinning it."""
    agent_dir = Path("my_agent")
    agent_dir.mkdir()
    (agent_dir / "setup.sh").write_text("true")
    built_sha = build_bundle_zip(agent_dir, Path("my_agent.zip"))
    mf = make_manifest(name, bundle=BundleSpec(file="my_agent.zip", sha256=sha256 or built_sha))
    Path(f"{name}.yaml").write_text(yaml.safe_dump(mf.model_dump(), sort_keys=False))


def test_add_and_run_deliver_pinned_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`add` verifies + copies the pinned bundle into the store; `run` loads the
    installed copy (digest-checked) and hands it to run_benchmark; a tampered
    store copy fails the run instead of shipping modified agent code."""
    calls: list[dict] = []

    async def fake_run_benchmark(**kwargs) -> None:  # type: ignore[return]
        calls.append(kwargs)

    monkeypatch.setattr("benchmark_runner.sandbox.cli.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        "benchmark_runner.sandbox.cli.BenchmarkServiceClient", lambda *a, **kw: MagicMock()
    )

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _write_bundled_manifest()

        result = runner.invoke(cli, ["add", "mybench.yaml"])
        assert result.exit_code == 0, result.output
        assert "agent bundle: my_agent.zip" in result.output
        assert Path("benchmarks/my_agent.zip").exists()

        result = runner.invoke(cli, ["run", "--model", "m", "--run-id", "r", "mybench"])
        assert result.exit_code == 0, result.output
        (call,) = calls
        assert call["bundle"].root == "my_agent"
        assert call["bundle"].zip_bytes == Path("benchmarks/my_agent.zip").read_bytes()

        Path("benchmarks/my_agent.zip").write_bytes(b"tampered")
        result = runner.invoke(cli, ["run", "--model", "m", "--run-id", "r2", "mybench"])
        assert result.exit_code != 0
        assert "digest mismatch" in result.output


def test_add_rejects_bundle_digest_mismatch(tmp_path: Path) -> None:
    """A bundle that does not match the manifest's pin fails `add` before
    anything is installed."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _write_bundled_manifest(sha256="0" * 64)

        result = runner.invoke(cli, ["add", "mybench.yaml"])
        assert result.exit_code != 0
        assert "digest mismatch" in result.output
        assert not Path("benchmarks/mybench.manifest.yaml").exists()


def test_run_bundle_flag_zips_directory(
    contract_file: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--bundle with an agent DIRECTORY zips it on the fly and passes it through
    — the dev/custom-agent path needs no manifest or prebuilt zip."""
    calls: list[dict] = []

    async def fake_run_benchmark(**kwargs) -> None:  # type: ignore[return]
        calls.append(kwargs)

    monkeypatch.setattr("benchmark_runner.sandbox.cli.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        "benchmark_runner.sandbox.cli.BenchmarkServiceClient", lambda *a, **kw: MagicMock()
    )

    agent_dir = tmp_path / "custom_agent"
    agent_dir.mkdir()
    (agent_dir / "run.py").write_text("x = 1")

    result = CliRunner().invoke(
        cli,
        ["run", "--model", "m", "--run-id", "r", "--contract", contract_file,
         "--bundle", str(agent_dir), "t1"],
    )

    assert result.exit_code == 0, result.output
    (call,) = calls
    assert call["bundle"].root == "custom_agent"
    assert call["bundle"].zip_bytes  # the on-the-fly zip made it through
