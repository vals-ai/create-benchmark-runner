import asyncio
import json
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from benchmark_runner import BenchmarkRunner, GenerationResult, GenerationStatus, Task
from benchmark_runner.artifacts import RunArtifacts
from benchmark_runner.cli import make_cli
from benchmark_runner.schemas import EvalResult, EvalStatus, FinalScoreResponse
from benchmark_service.client import BenchmarkServiceError
from benchmark_service.v1_schemas import V1DatasetTasksResponse, V1Task


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
        assert gen["data"] == f"answer-{tid}"
        assert gen["answer"] == f"answer-{tid}"

    cfg = json.loads((tmp_path / "r1" / "run_config.json").read_text())
    assert cfg["payload_schema"] == "test-bench.text.v1"
    assert cfg["payload_type"] == "text"
    assert cfg["dataset_name"] == "validation"
    assert cfg["task_source"] == "file"
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
    cfg = json.loads((tmp_path / "r1" / "run_config.json").read_text())
    assert cfg["task_source"] == "problem"


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


def test_run_with_dataset_name_fetches_from_service_and_skips_file(
    make_test_adapter, tmp_path, monkeypatch,
):
    """--dataset-name triggers service fetch; --dataset-file is ignored."""
    monkeypatch.delenv("VALS_AUTH_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)

    TestRunner = make_test_adapter()

    fetched: list[str] = []

    async def stub_list_tasks(self, dataset: str) -> V1DatasetTasksResponse:
        fetched.append(dataset)
        return V1DatasetTasksResponse(
            dataset=dataset,
            tasks=[V1Task(id="svc-t1", question="from service")],
        )

    monkeypatch.setattr(
        "benchmark_service.client.BenchmarkServiceClient.list_tasks",
        stub_list_tasks,
    )

    async def stub_generate(self, task, model, llm_config=None, log_dir=None) -> GenerationResult:
        return GenerationResult(
            task_id=task.id, status=GenerationStatus.SUCCESS,
            data="ok", question=task.question, model=model,
        )

    monkeypatch.setattr(TestRunner, "generate", stub_generate)

    cli = make_cli(TestRunner, default_dataset_file=None, default_results_dir=str(tmp_path))
    result = CliRunner().invoke(cli, [
        "run", "--model", "m", "--run-id", "r",
        "--service-url", "http://svc",
        "--dataset-name", "validation",
        "--skip-eval",
    ])

    assert result.exit_code == 0, result.output
    assert fetched == ["validation"]
    assert (tmp_path / "r" / "svc-t1" / "generation.json").exists()
    cfg = json.loads((tmp_path / "r" / "run_config.json").read_text())
    assert cfg["task_source"] == "service"


def test_run_rejects_both_dataset_name_and_dataset_file(make_test_adapter, tmp_path):
    TestRunner = make_test_adapter()
    cli = make_cli(TestRunner, default_dataset_file=None, default_results_dir=str(tmp_path))

    result = CliRunner().invoke(cli, [
        "run", "--model", "m", "--run-id", "r",
        "--dataset-name", "validation",
        "--dataset-file", str(tmp_path / "x.json"),
    ])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower() or "cannot pass both" in result.output.lower()


def test_run_rejects_dataset_name_with_explicit_default_dataset_file(
    make_test_adapter, tmp_path, monkeypatch,
):
    TestRunner = make_test_adapter()
    default_dataset_file = str(tmp_path / "dataset.json")

    async def fail_list_tasks(self, dataset: str):
        raise AssertionError(f"unexpected service task fetch for {dataset}")

    monkeypatch.setattr(
        "benchmark_service.client.BenchmarkServiceClient.list_tasks",
        fail_list_tasks,
    )

    cli = make_cli(
        TestRunner,
        default_dataset_file=default_dataset_file,
        default_results_dir=str(tmp_path),
    )
    result = CliRunner().invoke(cli, [
        "run", "--model", "m", "--run-id", "r",
        "--dataset-name", "validation",
        "--dataset-file", default_dataset_file,
    ])

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_run_rejects_problem_and_dataset_name(make_test_adapter, tmp_path):
    """--problem (single-task Valkyrie mode) and --dataset-name (service-loading)
    are conceptually different sources of task content and must not be combined."""
    TestRunner = make_test_adapter()
    cli = make_cli(TestRunner, default_dataset_file=None, default_results_dir=str(tmp_path))
    problem_file = tmp_path / "q.txt"
    problem_file.write_text("what is fair use?")

    result = CliRunner().invoke(cli, [
        "run", "--model", "m", "--run-id", "r",
        "--problem", str(problem_file),
        "--dataset-name", "validation",
        "VL-1",
    ])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_run_with_dataset_name_converts_service_error_to_click_error(
    make_test_adapter, tmp_path, monkeypatch,
):
    """If the benchmark service returns a non-200 (e.g. 501 because the deploy
    hasn't overridden list_tasks), the runner should surface a clear
    ClickException instead of letting BenchmarkServiceError bubble as a
    Python traceback."""
    TestRunner = make_test_adapter()

    async def stub_list_tasks(self, dataset: str):
        raise BenchmarkServiceError("list_tasks not implemented for this benchmark")

    monkeypatch.setattr(
        "benchmark_service.client.BenchmarkServiceClient.list_tasks",
        stub_list_tasks,
    )

    cli = make_cli(TestRunner, default_dataset_file=None, default_results_dir=str(tmp_path))
    result = CliRunner().invoke(cli, [
        "run", "--model", "m", "--run-id", "r",
        "--dataset-name", "validation",
        "--skip-eval",
    ])
    assert result.exit_code != 0
    # The error should mention the dataset and the underlying service-error text,
    # not be a raw traceback.
    assert "validation" in result.output.lower()
    assert "list_tasks" in result.output.lower() or "not implemented" in result.output.lower()


def test_run_resume_honors_service_loaded_run_config(
    make_test_adapter, tmp_path, monkeypatch,
):
    """If a run was service-loaded, a resume without --dataset-name must still
    re-fetch from the service rather than falling back to the bundled file."""
    TestRunner = make_test_adapter()

    # Seed an existing run_config that records service-loading.
    artifacts = RunArtifacts(results_dir=str(tmp_path), run_id="r")
    artifacts.save_run_config({
        "run_id": "r",
        "model": "m",
        "tasks": ["svc-t1"],
        "dataset_file": None,
        "dataset_name": "validation",
        "task_source": "service",
        "payload_schema": "x.text.v1",
        "payload_type": "text",
        "runner_version": "0.0.0",
        "generation_version": "dev",
    })

    fetched: list[str] = []

    async def stub_list_tasks(self, dataset: str):
        fetched.append(dataset)
        return V1DatasetTasksResponse(
            dataset=dataset,
            tasks=[V1Task(id="svc-t1", question="from service")],
        )

    monkeypatch.setattr(
        "benchmark_service.client.BenchmarkServiceClient.list_tasks",
        stub_list_tasks,
    )

    async def stub_generate(self, task, model, llm_config=None, log_dir=None):
        return GenerationResult(
            task_id=task.id, status=GenerationStatus.SUCCESS,
            data="ok", question=task.question, model=model,
        )
    monkeypatch.setattr(TestRunner, "generate", stub_generate)

    cli = make_cli(TestRunner, default_dataset_file=None, default_results_dir=str(tmp_path))
    # Note: no --dataset-name flag passed; the framework should pick it up from
    # the existing run_config.
    result = CliRunner().invoke(cli, [
        "run", "--model", "m", "--run-id", "r", "--skip-eval",
    ])

    assert result.exit_code == 0, result.output
    assert fetched == ["validation"]
    assert (tmp_path / "r" / "svc-t1" / "generation.json").exists()


def test_run_resume_file_task_source_does_not_infer_service_loading(
    make_test_adapter, tmp_path, monkeypatch,
):
    """A file-loaded run can still have dataset_name from the adapter; task_source
    is the disambiguator that prevents accidental service fetches on resume."""
    TestRunner = make_test_adapter()

    artifacts = RunArtifacts(results_dir=str(tmp_path), run_id="r")
    artifacts.save_run_config({
        "run_id": "r",
        "model": "m",
        "tasks": ["t1", "t2"],
        "dataset_file": None,
        "dataset_name": "validation",
        "task_source": "file",
        "payload_schema": "test-bench.text.v1",
        "payload_type": "text",
        "runner_version": "0.0.0",
        "generation_version": "dev",
    })

    async def fail_list_tasks(self, dataset: str):
        raise AssertionError(f"unexpected service task fetch for {dataset}")

    monkeypatch.setattr(
        "benchmark_service.client.BenchmarkServiceClient.list_tasks",
        fail_list_tasks,
    )

    cli = make_cli(TestRunner, default_dataset_file=None, default_results_dir=str(tmp_path))
    result = CliRunner().invoke(cli, [
        "run", "--model", "m", "--run-id", "r", "--skip-eval",
    ])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "r" / "t1" / "generation.json").exists()
    assert (tmp_path / "r" / "t2" / "generation.json").exists()


def test_run_failure_summary_surfaces_and_groups_error_reasons(tmp_path, monkeypatch):
    """When generation fails, the run summary surfaces the error reason grouped by
    identical message, instead of just listing task IDs with no explanation."""
    monkeypatch.delenv("VALS_AUTH_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)

    class FailingRunner(BenchmarkRunner):
        NAME = "failing"
        GENERATION_VERSION_ENV = "FAILING_GENERATION_VERSION"

        def load_tasks(self, dataset_file: str | None) -> list[Task]:
            return [Task(id="t1", question="q1"), Task(id="t2", question="q2")]

        async def generate(self, task, model, llm_config=None, log_dir=None) -> GenerationResult:
            return GenerationResult(
                task_id=task.id,
                status=GenerationStatus.ERROR,
                data="",
                question=task.question,
                model=model,
                error="Model m not found in registry",
            )

    cli = make_cli(FailingRunner)

    with patch("benchmark_runner.base.build_client") as builder:
        builder.return_value = AsyncMock()

        result = CliRunner().invoke(cli, [
            "run", "--model", "m", "--run-id", "r1",
            "--results-dir", str(tmp_path),
            "--service-url", "http://svc",
            "--skip-eval",
        ])

    assert result.exit_code != 0
    # The actual reason is surfaced, not just the task IDs.
    assert "Model m not found in registry" in result.output
    # Both failing tasks are still named.
    assert "t1" in result.output and "t2" in result.output
    # Identical errors are grouped: the shared message is printed once, not per-task.
    assert result.output.count("Model m not found in registry") == 1


def test_run_resume_legacy_config_does_not_infer_service_loading(
    make_test_adapter, tmp_path, monkeypatch,
):
    """Configs without task_source predate --dataset-name, so they are file-loaded."""
    TestRunner = make_test_adapter()

    artifacts = RunArtifacts(results_dir=str(tmp_path), run_id="r")
    artifacts.save_run_config({
        "run_id": "r",
        "model": "m",
        "tasks": ["t1", "t2"],
        "dataset_file": None,
        "dataset_name": "validation",
        "payload_schema": "test-bench.text.v1",
        "payload_type": "text",
        "runner_version": "0.0.0",
        "generation_version": "dev",
    })

    async def fail_list_tasks(self, dataset: str):
        raise AssertionError(f"unexpected service task fetch for {dataset}")

    monkeypatch.setattr(
        "benchmark_service.client.BenchmarkServiceClient.list_tasks",
        fail_list_tasks,
    )

    cli = make_cli(TestRunner, default_dataset_file=None, default_results_dir=str(tmp_path))
    result = CliRunner().invoke(cli, [
        "run", "--model", "m", "--run-id", "r", "--skip-eval",
    ])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "r" / "t1" / "generation.json").exists()
    assert (tmp_path / "r" / "t2" / "generation.json").exists()
