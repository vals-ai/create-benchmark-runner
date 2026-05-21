"""CLI integration tests: invoke `make_cli(TestRunner)`'s `run` against tmpdir."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from benchmark_runner import BenchmarkRunner, GenerationResult, GenerationStatus, Task
from benchmark_runner.artifacts import RunArtifacts
from benchmark_runner.cli import make_cli
from benchmark_runner.schemas import EvalResult, EvalStatus, FinalScoreResponse


def test_run_writes_generation_and_eval(make_test_adapter, tmp_path, monkeypatch):
    monkeypatch.delenv("VALS_AUTH_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)

    TestRunner = make_test_adapter()
    cli = make_cli(TestRunner)

    with patch("benchmark_runner.base.build_client") as builder:
        builder.return_value = AsyncMock()
        builder.return_value.evaluate_response = AsyncMock(return_value={
            "pass_percentage": 0.7,
            "eval_version": "v1",
        })
        builder.return_value.final_score = AsyncMock(return_value=FinalScoreResponse(
            tasks_evaluated=["t1", "t2"], final_score=0.7, metadata={},
        ))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--model", "m", "--run-id", "r1",
            "--results-dir", str(tmp_path),
            "--service-url", "http://svc",
            "--parallelism", "2",
        ])
        assert result.exit_code == 0, result.output

    for tid in ("t1", "t2"):
        gen_path = tmp_path / "r1" / tid / "generation.json"
        ev_path = tmp_path / "r1" / tid / "eval.json"
        assert gen_path.exists()
        assert ev_path.exists()
        gen = json.loads(gen_path.read_text())
        assert gen["status"] == "success"
        assert "data" in gen
        assert "answer" not in gen

    cfg = json.loads((tmp_path / "r1" / "run_config.json").read_text())
    assert cfg["payload_schema"] == "test-bench.text.v1"
    assert cfg["payload_type"] == "text"
    assert cfg["dataset_name"] == "validation"
    assert "runner_version" in cfg


def test_run_skip_eval_writes_generation_only(make_test_adapter, tmp_path, monkeypatch):
    monkeypatch.delenv("VALS_AUTH_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)

    TestRunner = make_test_adapter()
    cli = make_cli(TestRunner)

    with patch("benchmark_runner.base.build_client") as builder:
        builder.return_value = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--model", "m", "--run-id", "r1",
            "--results-dir", str(tmp_path),
            "--service-url", "http://svc",
            "--skip-eval",
        ])
        assert result.exit_code == 0, result.output

    for tid in ("t1", "t2"):
        assert (tmp_path / "r1" / tid / "generation.json").exists()
        assert not (tmp_path / "r1" / tid / "eval.json").exists()


def test_run_resume_skips_already_done(make_test_adapter, tmp_path, monkeypatch):
    """Pre-existing generation.json/eval.json with non-error status are not redone."""
    monkeypatch.delenv("VALS_AUTH_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)

    TestRunner = make_test_adapter()
    cli = make_cli(TestRunner)

    art = RunArtifacts(results_dir=tmp_path, run_id="r1")
    art.save_generation("t1", GenerationResult(task_id="t1", status=GenerationStatus.SUCCESS, data="seeded"))
    art.save_eval("t1", EvalResult(task_id="t1", status=EvalStatus.EVALUATED))
    art.save_run_config({
        "run_id": "r1", "model": "m", "tasks": ["t1", "t2"],
        "dataset_file": None, "payload_schema": "test-bench.text.v1",
        "payload_type": "text", "runner_version": "x", "generation_version": "x",
    })

    with patch("benchmark_runner.base.build_client") as builder:
        builder.return_value = AsyncMock()
        builder.return_value.evaluate_response = AsyncMock(return_value={
            "pass_percentage": 1.0, "eval_version": "v1",
        })
        builder.return_value.final_score = AsyncMock(return_value=FinalScoreResponse(
            tasks_evaluated=["t1", "t2"], final_score=1.0, metadata={},
        ))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--model", "m", "--run-id", "r1",
            "--results-dir", str(tmp_path),
            "--service-url", "http://svc",
        ])
        assert result.exit_code == 0, result.output

    seeded = json.loads((tmp_path / "r1" / "t1" / "generation.json").read_text())
    assert seeded["data"] == "seeded"


def test_run_explicit_task_ids_filters(make_test_adapter, tmp_path, monkeypatch):
    monkeypatch.delenv("VALS_AUTH_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)

    TestRunner = make_test_adapter()
    cli = make_cli(TestRunner)

    with patch("benchmark_runner.base.build_client") as builder:
        builder.return_value = AsyncMock()
        builder.return_value.evaluate_response = AsyncMock(return_value={
            "pass_percentage": 1.0, "eval_version": "v1",
        })
        builder.return_value.final_score = AsyncMock(return_value=FinalScoreResponse(
            tasks_evaluated=["t1"], final_score=1.0, metadata={},
        ))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--model", "m", "--run-id", "r1",
            "--results-dir", str(tmp_path),
            "--service-url", "http://svc",
            "t1",
        ])
        assert result.exit_code == 0, result.output

    assert (tmp_path / "r1" / "t1" / "generation.json").exists()
    assert not (tmp_path / "r1" / "t2" / "generation.json").exists()


def test_run_problem_mode_creates_single_task(make_test_adapter, tmp_path, monkeypatch):
    """--problem PATH TASK_ID reads the file, registers a Task with that question."""
    monkeypatch.delenv("VALS_AUTH_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)

    problem = tmp_path / "problem.txt"
    problem.write_text("solve this please")

    TestRunner = make_test_adapter()
    cli = make_cli(TestRunner)

    with patch("benchmark_runner.base.build_client") as builder:
        builder.return_value = AsyncMock()
        builder.return_value.evaluate_response = AsyncMock(return_value={
            "pass_percentage": 1.0, "eval_version": "v1",
        })
        builder.return_value.final_score = AsyncMock(return_value=FinalScoreResponse(
            tasks_evaluated=["custom"], final_score=1.0, metadata={},
        ))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--model", "m", "--run-id", "r1",
            "--results-dir", str(tmp_path),
            "--service-url", "http://svc",
            "--problem", str(problem),
            "custom",
        ])
        assert result.exit_code == 0, result.output

    gen = json.loads((tmp_path / "r1" / "custom" / "generation.json").read_text())
    assert gen["question"] == "solve this please"


def test_run_problem_requires_exactly_one_task_id(make_test_adapter, tmp_path):
    TestRunner = make_test_adapter()
    cli = make_cli(TestRunner)
    runner = CliRunner()
    result = runner.invoke(cli, [
        "run", "--model", "m", "--run-id", "r1",
        "--results-dir", str(tmp_path),
        "--service-url", "http://svc",
        "--problem", "/nope.txt",
        "t1", "t2",
    ])
    assert result.exit_code != 0


def test_run_exits_nonzero_when_eval_errors(make_test_adapter, tmp_path, monkeypatch):
    monkeypatch.delenv("VALS_AUTH_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)

    TestRunner = make_test_adapter()
    cli = make_cli(TestRunner)

    with patch("benchmark_runner.base.build_client") as builder:
        builder.return_value = AsyncMock()
        builder.return_value.evaluate_response = AsyncMock(side_effect=RuntimeError("service down"))
        builder.return_value.final_score = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--model", "m", "--run-id", "r1",
            "--results-dir", str(tmp_path),
            "--service-url", "http://svc",
        ])

    assert result.exit_code != 0
    assert "eval errors" in result.output
    builder.return_value.final_score.assert_not_called()

    ev = json.loads((tmp_path / "r1" / "t1" / "eval.json").read_text())
    assert ev["status"] == "error"


def test_run_enforces_task_timeout(tmp_path, monkeypatch):
    monkeypatch.delenv("VALS_AUTH_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)

    class SlowRunner(BenchmarkRunner):
        NAME = "slow"
        GENERATION_VERSION_ENV = "SLOW_GENERATION_VERSION"

        def load_tasks(self, dataset_file: str | None) -> list[Task]:
            return [Task(id="slow-task", question="q", timeout=0.01)]

        async def generate(
            self,
            task: Task,
            model: str,
            llm_config=None,
            log_dir=None,
        ) -> GenerationResult:
            await asyncio.sleep(0.05)
            return GenerationResult(task_id=task.id, status=GenerationStatus.SUCCESS, data="late")

    cli = make_cli(SlowRunner)

    with patch("benchmark_runner.base.build_client") as builder:
        builder.return_value = AsyncMock()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--model", "m", "--run-id", "r1",
            "--results-dir", str(tmp_path),
            "--service-url", "http://svc",
            "--skip-eval",
        ])

    assert result.exit_code == 0, result.output
    gen = json.loads((tmp_path / "r1" / "slow-task" / "generation.json").read_text())
    assert gen["status"] == "max_time"
    assert gen["data"] == ""
    assert "timed out" in gen["error"]
