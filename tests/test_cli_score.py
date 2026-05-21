import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from benchmark_runner import (
    BenchmarkRunner,
    EvalResult,
    EvalStatus,
    GenerationResult,
    GenerationStatus,
    Task,
)
from benchmark_runner.artifacts import RunArtifacts
from benchmark_runner.cli import make_cli
from benchmark_runner.schemas import FinalScoreResponse


def _seed_run(tmp_path, *, all_evaluated: bool):
    art = RunArtifacts(results_dir=tmp_path, run_id="r1")
    art.save_run_config({
        "run_id": "r1", "model": "m", "tasks": ["t1", "t2"],
        "dataset_file": None, "payload_schema": "test-bench.text.v1",
        "payload_type": "text", "runner_version": "x", "generation_version": "x",
    })
    for tid in ("t1", "t2"):
        art.save_generation(tid, GenerationResult(task_id=tid, status=GenerationStatus.SUCCESS, data="x"))
    art.save_eval("t1", EvalResult(task_id="t1", status=EvalStatus.EVALUATED))
    if all_evaluated:
        art.save_eval("t2", EvalResult(task_id="t2", status=EvalStatus.EVALUATED))
    return art


def test_score_writes_final_score_when_complete(make_test_adapter, tmp_path, monkeypatch):
    monkeypatch.delenv("VALS_AUTH_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)

    _seed_run(tmp_path, all_evaluated=True)

    TestRunner = make_test_adapter()
    cli = make_cli(TestRunner)

    with patch("benchmark_runner.base.build_client") as builder:
        builder.return_value = AsyncMock()
        builder.return_value.final_score = AsyncMock(return_value=FinalScoreResponse(
            tasks_evaluated=["t1", "t2"], final_score=0.42, metadata={},
        ))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "score", "--run-id", "r1",
            "--results-dir", str(tmp_path),
            "--service-url", "http://svc",
        ])
        assert result.exit_code == 0, result.output

    fs = json.loads((tmp_path / "r1" / "final_score.json").read_text())
    assert fs["final_score"] == 0.42
    assert fs["complete"] is True


def test_score_incomplete_run_requires_force(make_test_adapter, tmp_path, monkeypatch):
    monkeypatch.delenv("VALS_AUTH_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)

    _seed_run(tmp_path, all_evaluated=False)

    TestRunner = make_test_adapter()
    cli = make_cli(TestRunner)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "score", "--run-id", "r1",
        "--results-dir", str(tmp_path),
        "--service-url", "http://svc",
    ])
    assert result.exit_code == 0
    assert "incomplete" in result.output.lower()
    assert not (tmp_path / "r1" / "final_score.json").exists()

    with patch("benchmark_runner.base.build_client") as builder:
        builder.return_value = AsyncMock()
        builder.return_value.final_score = AsyncMock(return_value=FinalScoreResponse(
            tasks_evaluated=["t1"], final_score=0.5, metadata={},
        ))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "score", "--run-id", "r1",
            "--results-dir", str(tmp_path),
            "--service-url", "http://svc",
            "--force",
        ])
        assert result.exit_code == 0, result.output

    fs = json.loads((tmp_path / "r1" / "final_score.json").read_text())
    assert fs["final_score"] == 0.5
    assert fs["complete"] is False


def test_score_dataset_file_override_sets_dataset_for_final_score(tmp_path, monkeypatch):
    monkeypatch.delenv("VALS_AUTH_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)

    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(json.dumps({
        "dataset_name": "validation",
        "tests": [
            {"id": "t1", "question": "q1"},
            {"id": "t2", "question": "q2"},
        ],
    }))
    _seed_run(tmp_path, all_evaluated=True)

    class DatasetRunner(BenchmarkRunner):
        NAME = "test-bench"
        GENERATION_VERSION_ENV = "TEST_BENCH_GENERATION_VERSION"

        def load_tasks(self, dataset_file: str | None) -> list[Task]:
            assert dataset_file is not None
            payload = json.loads(Path(dataset_file).read_text())
            self._dataset = payload["dataset_name"]
            return [Task(id=item["id"], question=item["question"]) for item in payload["tests"]]

        async def generate(self, task: Task, model: str, llm_config=None, log_dir=None) -> GenerationResult:
            return GenerationResult(task_id=task.id, status=GenerationStatus.SUCCESS, data="unused")

    cli = make_cli(DatasetRunner)

    with patch("benchmark_runner.base.build_client") as builder:
        builder.return_value = AsyncMock()
        builder.return_value.final_score = AsyncMock(return_value=FinalScoreResponse(
            tasks_evaluated=["t1", "t2"], final_score=0.42, metadata={},
        ))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "score", "--run-id", "r1",
            "--results-dir", str(tmp_path),
            "--service-url", "http://svc",
            "--dataset-file", str(dataset_file),
        ])

    assert result.exit_code == 0, result.output
    builder.return_value.final_score.assert_called_once()
    assert builder.return_value.final_score.call_args.kwargs["dataset"] == "validation"
