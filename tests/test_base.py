import pytest

from benchmark_runner import (
    EvalResult,
    EvalResultData,
    EvalStatus,
    GenerationResult,
    GenerationStatus,
    Task,
    FinalScoreResponse,
)


def test_runner_identity_and_generation_version(make_test_adapter, monkeypatch):
    monkeypatch.setenv("TEST_BENCH_GENERATION_VERSION", "abc123")
    TestRunner = make_test_adapter()
    runner = TestRunner(service_url="http://x")
    assert runner.payload_schema == "test-bench.text.v1"
    assert runner.generation_version == "abc123"

    monkeypatch.delenv("TEST_BENCH_GENERATION_VERSION", raising=False)
    assert runner.generation_version == "dev"


def test_task_registration_helpers(make_test_adapter):
    TestRunner = make_test_adapter()
    runner = TestRunner(service_url="http://x")
    runner._register_tasks(runner.load_tasks(dataset_file=None))
    assert [t.id for t in runner.get_tasks()] == ["t1", "t2"]
    runner.add_task(Task(id="single", question="only-task"))
    assert [t.id for t in runner.get_tasks()] == ["single", "t1", "t2"]


@pytest.mark.asyncio
async def test_default_evaluate_short_circuits_non_success_generations(make_test_adapter, mock_client):
    TestRunner = make_test_adapter()
    runner = TestRunner(service_url="http://x")
    runner._client = mock_client

    timed_out = await runner.evaluate("t1", GenerationResult(task_id="t1", status=GenerationStatus.MAX_TIME, data=""))
    errored = await runner.evaluate(
        "t2",
        GenerationResult(task_id="t2", status=GenerationStatus.ERROR, data="", error="boom"),
    )

    assert timed_out.status == EvalStatus.DID_NOT_COMPLETE
    assert errored.status == EvalStatus.GENERATION_ERROR
    assert errored.error == "boom"
    mock_client.evaluate_response.assert_not_called()


@pytest.mark.asyncio
async def test_default_evaluate_posts_to_service_on_success(make_test_adapter, mock_client):
    TestRunner = make_test_adapter()
    runner = TestRunner(service_url="http://x")
    runner._client = mock_client
    runner._dataset = "validation"
    gen = GenerationResult(task_id="t1", status=GenerationStatus.SUCCESS, data="42")
    ev = await runner.evaluate("t1", gen)
    assert ev.status == EvalStatus.EVALUATED
    assert ev.result is not None
    assert ev.result.pass_percentage == 0.8
    mock_client.evaluate_response.assert_called_once_with(
        task_id="t1", response="42", dataset="validation"
    )


@pytest.mark.asyncio
async def test_default_score_pads_missing_tasks_with_null(make_test_adapter, mock_client):
    TestRunner = make_test_adapter()
    runner = TestRunner(service_url="http://x")
    runner._client = mock_client
    runner._register_tasks(runner.load_tasks(dataset_file=None))

    mock_client.final_score.return_value = FinalScoreResponse(
        tasks_evaluated=["t1", "t2"], final_score=0.5, metadata={},
    )

    eval_results = [EvalResult(task_id="t1", status=EvalStatus.EVALUATED,
                               result=EvalResultData(pass_percentage=1.0, eval_version="v1"))]
    sr = await runner.score(eval_results)
    assert sr.final_score == 0.5

    call_args = mock_client.final_score.call_args
    submitted = call_args.kwargs.get("evaluation_results") or call_args.args[0]
    assert "t1" in submitted
    assert "t2" in submitted
    assert submitted["t2"] is None
