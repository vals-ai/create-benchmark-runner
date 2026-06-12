"""Behavioral tests for the split run phases: `run --skip-eval`, evaluate_run, score_run."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from tests.sandbox.conftest import FakeClient, FakeProvider, make_manifest
from benchmark_runner.artifacts import RunArtifacts
from benchmark_runner.sandbox.cli import cli
from benchmark_runner.sandbox.orchestrator import evaluate_run, run_benchmark, score_run
from benchmark_runner.sandbox.store import install_manifest
from benchmark_runner.schemas import (
    EvalResult,
    EvalStatus,
    GenerationResult,
    GenerationStatus,
    ScoreResult,
)


class CountingClient(FakeClient):
    """FakeClient that counts judge calls, to assert resume skips clean evals."""

    def __init__(self) -> None:
        super().__init__()
        self.evaluate_call_count = 0

    async def evaluate_response(
        self,
        task_id: str,
        response: str,
        dataset: str | None = None,
    ) -> dict[str, object]:
        self.evaluate_call_count += 1
        return await super().evaluate_response(task_id, response, dataset=dataset)


def _seed_generation(
    artifacts: RunArtifacts,
    task_id: str,
    status: GenerationStatus = GenerationStatus.SUCCESS,
    data: str = "ANSWER",
) -> None:
    artifacts.save_generation(
        task_id, GenerationResult(task_id=task_id, status=status, data=data)
    )


@pytest.mark.asyncio
async def test_run_benchmark_skip_eval_generates_only(
    tmp_path: Path, contract_yaml: Path
) -> None:
    """--skip-eval produces generation.json per task but no eval.json and no
    final_score.json: the run is a generation slice, evaluated and scored later."""
    client = FakeClient()
    await run_benchmark(
        run_id="r1",
        model="openai/gpt-5",
        task_ids=["t1"],
        dataset=None,
        results_dir=str(tmp_path),
        contract_path=contract_yaml,
        client=client,
        provider=FakeProvider(),
        skip_eval=True,
    )
    artifacts = RunArtifacts(results_dir=tmp_path, run_id="r1")
    assert artifacts.load_generation("t1") is not None
    assert artifacts.load_eval("t1") is None
    assert not artifacts.final_score_path.exists()
    assert client.last_final_score_args is None  # final score never reached the service


@pytest.mark.asyncio
async def test_direct_skip_eval_slices_merge_into_scored_task_set(
    tmp_path: Path, contract_yaml: Path
) -> None:
    client = FakeClient()
    run_id = "r1"

    await run_benchmark(
        run_id=run_id,
        model="openai/gpt-5",
        task_ids=["t1"],
        dataset="ds",
        results_dir=str(tmp_path),
        contract_path=contract_yaml,
        client=client,
        provider=FakeProvider(),
        skip_eval=True,
    )
    await run_benchmark(
        run_id=run_id,
        model="openai/gpt-5",
        task_ids=["t2"],
        dataset="ds",
        results_dir=str(tmp_path),
        contract_path=contract_yaml,
        client=client,
        provider=FakeProvider(),
        skip_eval=True,
    )

    artifacts = RunArtifacts(results_dir=tmp_path, run_id=run_id)
    config = artifacts.load_run_config()
    assert config is not None
    assert config["tasks"] == ["t1", "t2"]

    await evaluate_run(run_id=run_id, results_dir=str(tmp_path), client=client)
    await score_run(run_id=run_id, results_dir=str(tmp_path), client=client)
    assert client.last_final_score_args is not None
    assert set(client.last_final_score_args) == {"t1", "t2"}


@pytest.mark.asyncio
async def test_evaluate_run_discovers_slices_and_resumes(tmp_path: Path) -> None:
    """With no explicit task ids, every on-disk generation is evaluated — slices
    written by separate `run --skip-eval` invocations included. Failed generations
    record GENERATION_ERROR without a judge call, and re-running evaluates nothing
    that already completed cleanly."""
    artifacts = RunArtifacts(results_dir=tmp_path, run_id="r1")
    _seed_generation(artifacts, "t1")
    _seed_generation(artifacts, "t2", status=GenerationStatus.ERROR, data="")
    client = CountingClient()

    await evaluate_run(run_id="r1", results_dir=str(tmp_path), client=client)
    t1 = artifacts.load_eval("t1")
    t2 = artifacts.load_eval("t2")
    assert t1 is not None and t1.status == EvalStatus.EVALUATED
    assert t2 is not None and t2.status == EvalStatus.GENERATION_ERROR
    assert client.evaluate_call_count == 1  # only the successful generation hit the judge

    await evaluate_run(run_id="r1", results_dir=str(tmp_path), client=client)
    assert client.evaluate_call_count == 1  # resume: the clean eval is not redone


@pytest.mark.asyncio
async def test_evaluate_run_raises_artifact_read_failures(tmp_path: Path) -> None:
    artifacts = RunArtifacts(results_dir=tmp_path, run_id="r1")
    generation_path = artifacts.generation_path("t1")
    generation_path.parent.mkdir(parents=True)
    generation_path.write_text("{not json")

    with pytest.raises(json.JSONDecodeError):
        await evaluate_run(run_id="r1", results_dir=str(tmp_path), client=FakeClient())


@pytest.mark.asyncio
async def test_evaluate_run_no_generations_errors(tmp_path: Path) -> None:
    """An empty run dir fails loudly instead of silently evaluating nothing."""
    with pytest.raises(ValueError, match="nothing to evaluate"):
        await evaluate_run(run_id="empty", results_dir=str(tmp_path), client=FakeClient())


@pytest.mark.asyncio
async def test_score_run_defaults_to_run_config_and_flags_incomplete(
    tmp_path: Path,
) -> None:
    """The scored set defaults to the run_config's frozen task list; a task with
    no eval result submits as None (scores zero server-side, never omitted) and
    marks the run incomplete."""
    artifacts = RunArtifacts(results_dir=tmp_path, run_id="r1")
    artifacts.save_run_config({"run_id": "r1", "tasks": ["t1", "t2"], "dataset_name": "ds"})
    artifacts.save_eval("t1", EvalResult(task_id="t1", status=EvalStatus.EVALUATED))
    client = FakeClient()

    score = await score_run(run_id="r1", results_dir=str(tmp_path), client=client)
    assert client.last_final_score_args is not None
    assert set(client.last_final_score_args) == {"t1", "t2"}
    assert client.last_final_score_args["t2"] is None  # submitted as None, not omitted
    assert score.complete is False
    assert artifacts.final_score_path.exists()


@pytest.mark.asyncio
async def test_score_run_without_config_requires_task_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no run_config"):
        await score_run(run_id="r1", results_dir=str(tmp_path), client=FakeClient())


def test_cli_eval_manifest_mode_resolves_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`benchmark eval <name>` resolves service URL, dataset, and the nested
    results dir from the installed manifest; no task ids means discover-from-disk
    (task_ids=None passed through)."""
    calls: list[dict] = []
    client_urls: list[str] = []

    async def fake_evaluate_run(**kwargs: object) -> None:
        calls.append(dict(kwargs))

    monkeypatch.setattr("benchmark_runner.sandbox.cli.evaluate_run", fake_evaluate_run)
    monkeypatch.setattr(
        "benchmark_runner.sandbox.cli.BenchmarkServiceClient",
        lambda url, **kw: client_urls.append(url) or MagicMock(),
    )

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        install_manifest(make_manifest("mybench"))
        result = runner.invoke(cli, ["eval", "--run-id", "r1", "mybench"])

    assert result.exit_code == 0, result.output
    (call,) = calls
    assert call["task_ids"] is None
    assert call["dataset"] == "mybench-dataset"
    assert call["results_dir"] == str(Path("results") / "mybench")
    assert client_urls == ["http://svc"]


def test_cli_eval_unknown_manifest_name_fails(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["eval", "--run-id", "r1", "missing-benchmark"])

    assert result.exit_code != 0
    assert "benchmark 'missing-benchmark' is not installed" in result.output


def test_cli_eval_direct_mode_requires_explicit_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    async def fake_evaluate_run(**kwargs: object) -> None:
        calls.append(dict(kwargs))

    monkeypatch.setattr("benchmark_runner.sandbox.cli.evaluate_run", fake_evaluate_run)
    monkeypatch.setattr(
        "benchmark_runner.sandbox.cli.BenchmarkServiceClient",
        lambda url, **kw: MagicMock(),
    )

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["eval", "--run-id", "r1", "--direct", "task-1"])

    assert result.exit_code == 0, result.output
    (call,) = calls
    assert call["task_ids"] == ["task-1"]
    assert call["results_dir"] == "results"


def test_cli_score_manifest_mode_defaults_to_full_task_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`benchmark score <name>` scores over the manifest's FULL task list by
    default, so a partial run cannot drop tasks out of the final score."""
    calls: list[dict] = []

    async def fake_score_run(**kwargs: object) -> ScoreResult:
        calls.append(dict(kwargs))
        return ScoreResult(
            tasks_evaluated=["task-1"], final_score=0.5, metadata={}, complete=False
        )

    monkeypatch.setattr("benchmark_runner.sandbox.cli.score_run", fake_score_run)
    monkeypatch.setattr(
        "benchmark_runner.sandbox.cli.BenchmarkServiceClient", lambda url, **kw: MagicMock()
    )

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        install_manifest(make_manifest("mybench"))
        result = runner.invoke(cli, ["score", "--run-id", "r1", "mybench"])

    assert result.exit_code == 0, result.output
    (call,) = calls
    assert call["task_ids"] == ["task-1", "task-2"]
    assert "final_score=0.5" in result.output


@pytest.mark.asyncio
async def test_run_benchmark_raises_artifact_read_failures(
    tmp_path: Path, contract_yaml: Path
) -> None:
    """run_benchmark surfaces unexpected per-task exceptions (e.g. a corrupt
    pre-existing artifact read on resume) the same way evaluate_run does — the
    command fails instead of silently scoring around the bad task."""
    artifacts = RunArtifacts(results_dir=tmp_path, run_id="r1")
    generation_path = artifacts.generation_path("t1")
    generation_path.parent.mkdir(parents=True)
    generation_path.write_text("{not json")

    with pytest.raises(json.JSONDecodeError):
        await run_benchmark(
            run_id="r1",
            model="openai/gpt-5",
            task_ids=["t1"],
            dataset=None,
            results_dir=str(tmp_path),
            contract_path=contract_yaml,
            client=FakeClient(),
            provider=FakeProvider(),
        )
    assert not artifacts.final_score_path.exists()  # failed before scoring
