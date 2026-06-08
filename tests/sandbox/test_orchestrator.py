"""Behavioral tests for run_sandbox orchestrator loop (TDD)."""

import json
import stat
from pathlib import Path

import pytest

from benchmark_service.sandbox import SandboxCreateRequest
from tests.sandbox.conftest import FakeClient, FakeProvider
from benchmark_runner.artifacts import RunArtifacts
from benchmark_runner.sandbox import run_sandbox
from benchmark_runner.schemas import EvalStatus, GenerationResult, GenerationStatus, ScoreResult


@pytest.mark.asyncio
async def test_run_sandbox_produces_artifacts_and_score(
    tmp_path: Path,
    contract_yaml: Path,
) -> None:
    """Full happy path: generation.json + eval.json per task + final_score.json."""
    run_id = "run-001"
    task_ids = ["task-a", "task-b"]
    client = FakeClient()
    provider = FakeProvider()

    await run_sandbox(
        run_id=run_id,
        model="openai/gpt-5",
        task_ids=task_ids,
        dataset=None,
        results_dir=str(tmp_path),
        contract_path=contract_yaml,
        client=client,
        provider=provider,
    )

    artifacts = RunArtifacts(results_dir=str(tmp_path), run_id=run_id)
    run_dir = tmp_path / run_id
    for tid in task_ids:
        task_dir = run_dir / tid
        assert (task_dir / "generation.json").exists(), f"missing generation.json for {tid}"
        assert (task_dir / "eval.json").exists(), f"missing eval.json for {tid}"
        gen = artifacts.load_generation(tid)
        assert gen is not None
        assert gen.status == GenerationStatus.SUCCESS, f"expected SUCCESS for {tid}, got {gen.status}"

    assert (run_dir / "final_score.json").exists()

    # Two sandboxes created, one per task
    assert len(provider.created) == 2

    # The sandbox request source came from retrieve_task
    retrieved_source = (await client.retrieve_task("any")).source
    assert provider.last_request is not None
    assert provider.last_request.source == retrieved_source


@pytest.mark.asyncio
async def test_run_sandbox_resume_skips_sandboxes(
    tmp_path: Path,
    contract_yaml: Path,
) -> None:
    """Second run with completed artifacts creates no new sandboxes."""
    run_id = "run-002"
    task_ids = ["task-x", "task-y"]
    client = FakeClient()

    first_provider = FakeProvider()
    await run_sandbox(
        run_id=run_id,
        model="openai/gpt-5",
        task_ids=task_ids,
        dataset=None,
        results_dir=str(tmp_path),
        contract_path=contract_yaml,
        client=client,
        provider=first_provider,
    )
    assert len(first_provider.created) == 2

    # Second pass — inject a fresh provider to detect any new sandbox creation
    second_provider = FakeProvider()
    await run_sandbox(
        run_id=run_id,
        model="openai/gpt-5",
        task_ids=task_ids,
        dataset=None,
        results_dir=str(tmp_path),
        contract_path=contract_yaml,
        client=client,
        provider=second_provider,
    )

    assert second_provider.created == [], "resume should not create any new sandboxes"


@pytest.mark.asyncio
async def test_eval_status_mapping(
    tmp_path: Path,
    contract_yaml: Path,
) -> None:
    """Non-success generation statuses map to the correct eval status.

    MAX_TIME / MAX_TURNS → DID_NOT_COMPLETE.
    A task whose sandbox creation failed (ERROR generation) → GENERATION_ERROR.
    """

    class ExplodingProvider(FakeProvider):
        async def create_sandbox(self, request: SandboxCreateRequest) -> object:  # type: ignore[override]
            raise RuntimeError("create failed")

    run_id = "run-eval-map"
    client = FakeClient()
    artifacts = RunArtifacts(results_dir=str(tmp_path), run_id=run_id)

    # Pre-seed MAX_TIME and MAX_TURNS generations (both are non-redoable,
    # so the orchestrator skips generation and goes straight to eval)
    artifacts.save_generation(
        "task-max-time",
        GenerationResult(task_id="task-max-time", status=GenerationStatus.MAX_TIME, data=""),
    )
    artifacts.save_generation(
        "task-max-turns",
        GenerationResult(task_id="task-max-turns", status=GenerationStatus.MAX_TURNS, data=""),
    )

    await run_sandbox(
        run_id=run_id,
        model="openai/gpt-5",
        task_ids=["task-max-time", "task-max-turns", "task-infra-fail"],
        dataset=None,
        results_dir=str(tmp_path),
        contract_path=contract_yaml,
        client=client,
        provider=ExplodingProvider(),
    )

    ev_time = artifacts.load_eval("task-max-time")
    ev_turns = artifacts.load_eval("task-max-turns")
    ev_infra = artifacts.load_eval("task-infra-fail")
    assert ev_time is not None and ev_time.status == EvalStatus.DID_NOT_COMPLETE
    assert ev_turns is not None and ev_turns.status == EvalStatus.DID_NOT_COMPLETE
    assert ev_infra is not None and ev_infra.status == EvalStatus.GENERATION_ERROR


@pytest.mark.asyncio
async def test_partial_resume_runs_eval_only(
    tmp_path: Path,
    contract_yaml: Path,
) -> None:
    """When generation.json already exists but eval.json is absent, eval runs and zero sandboxes are created."""
    run_id = "run-partial"
    task_ids = ["task-p"]
    client = FakeClient()
    artifacts = RunArtifacts(results_dir=str(tmp_path), run_id=run_id)

    # Pre-seed a successful generation; no eval yet
    artifacts.save_generation(
        "task-p",
        GenerationResult(task_id="task-p", status=GenerationStatus.SUCCESS, data="MY_ANSWER"),
    )

    provider = FakeProvider()
    await run_sandbox(
        run_id=run_id,
        model="openai/gpt-5",
        task_ids=task_ids,
        dataset=None,
        results_dir=str(tmp_path),
        contract_path=contract_yaml,
        client=client,
        provider=provider,
    )

    assert provider.created == [], "partial resume must not create new sandboxes"
    ev = artifacts.load_eval("task-p")
    assert ev is not None and ev.status == EvalStatus.EVALUATED


@pytest.mark.asyncio
async def test_setup_task_failure_deletes_sandbox_and_records_error(
    tmp_path: Path,
    contract_yaml: Path,
) -> None:
    """When setup_task raises for one task, that sandbox is deleted, an ERROR generation is saved, and final_score.json is still written."""

    class FailingSetupClient(FakeClient):
        async def setup_task(  # type: ignore[override]
            self,
            task_id: str,
            instance_id: str,
            on_message: object = None,
            dataset: str | None = None,
            sandbox_provider: object = None,
        ) -> object:
            if task_id == "task-bad":
                raise RuntimeError("infra exploded")
            return await super().setup_task(
                task_id=task_id,
                instance_id=instance_id,
                on_message=on_message,
                dataset=dataset,
                sandbox_provider=sandbox_provider,
            )

    run_id = "run-fault"
    task_ids = ["task-ok", "task-bad"]
    client = FailingSetupClient()
    provider = FakeProvider()

    await run_sandbox(
        run_id=run_id,
        model="openai/gpt-5",
        task_ids=task_ids,
        dataset=None,
        results_dir=str(tmp_path),
        contract_path=contract_yaml,
        client=client,
        provider=provider,
    )

    artifacts = RunArtifacts(results_dir=str(tmp_path), run_id=run_id)

    # Both tasks must have generation artifacts
    gen_ok = artifacts.load_generation("task-ok")
    gen_bad = artifacts.load_generation("task-bad")
    assert gen_ok is not None and gen_ok.status == GenerationStatus.SUCCESS
    assert gen_bad is not None and gen_bad.status == GenerationStatus.ERROR

    # Every created sandbox must be deleted (no leaks)
    created_ids = {f"sandbox-{run_id}-{tid}" for tid in task_ids}
    assert set(provider.deleted) == created_ids

    # final_score.json must always be written
    assert (tmp_path / run_id / "final_score.json").exists()


@pytest.mark.asyncio
async def test_none_fill_in_final_score_when_eval_missing(
    tmp_path: Path,
    contract_yaml: Path,
) -> None:
    """If a task's eval cannot be saved (e.g. filesystem error), final_score receives None for that task."""
    run_id = "run-none"
    task_ids = ["task-good", "task-nofs"]
    client = FakeClient()
    provider = FakeProvider()

    # First run to seed task-good's generation so it's not redoable
    artifacts = RunArtifacts(results_dir=str(tmp_path), run_id=run_id)
    artifacts.save_generation(
        "task-nofs",
        GenerationResult(task_id="task-nofs", status=GenerationStatus.SUCCESS, data="X"),
    )

    # Make the task-nofs directory read-only so save_eval fails
    task_nofs_dir = tmp_path / run_id / "task-nofs"
    task_nofs_dir.mkdir(parents=True, exist_ok=True)
    task_nofs_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # read + execute, no write

    try:
        await run_sandbox(
            run_id=run_id,
            model="openai/gpt-5",
            task_ids=task_ids,
            dataset=None,
            results_dir=str(tmp_path),
            contract_path=contract_yaml,
            client=client,
            provider=provider,
        )
    finally:
        # Restore permissions so tmp_path cleanup works
        task_nofs_dir.chmod(stat.S_IRWXU)

    # final_score.json written despite the error
    assert (tmp_path / run_id / "final_score.json").exists()

    # task-nofs has no eval on disk → submitted as None
    assert client.last_final_score_args is not None
    assert client.last_final_score_args["task-nofs"] is None
    assert client.last_final_score_args["task-good"] is not None


@pytest.mark.asyncio
async def test_delete_sandbox_failure_preserves_generation_result(
    tmp_path: Path,
    contract_yaml: Path,
) -> None:
    """delete_sandbox raising must not overwrite a successful generation or prevent final_score.json."""

    class FailingDeleteProvider(FakeProvider):
        async def delete_sandbox(self, instance_id: str) -> None:
            raise RuntimeError("network blip during delete")

    run_id = "run-del-fail"
    task_ids = ["task-del"]
    client = FakeClient()
    provider = FailingDeleteProvider()

    await run_sandbox(
        run_id=run_id,
        model="openai/gpt-5",
        task_ids=task_ids,
        dataset=None,
        results_dir=str(tmp_path),
        contract_path=contract_yaml,
        client=client,
        provider=provider,
    )

    artifacts = RunArtifacts(results_dir=str(tmp_path), run_id=run_id)
    gen = artifacts.load_generation("task-del")
    assert gen is not None, "generation must be saved even when delete_sandbox raises"
    assert gen.status == GenerationStatus.SUCCESS, f"expected SUCCESS, got {gen.status}"
    assert (tmp_path / run_id / "final_score.json").exists(), "final_score.json must be written"


@pytest.mark.asyncio
async def test_final_score_complete_flag_happy_path(
    tmp_path: Path,
    contract_yaml: Path,
) -> None:
    """A fully-evaluated run writes final_score.json with complete=True."""
    run_id = "run-complete"
    task_ids = ["task-c1", "task-c2"]
    client = FakeClient()
    provider = FakeProvider()

    await run_sandbox(
        run_id=run_id,
        model="openai/gpt-5",
        task_ids=task_ids,
        dataset=None,
        results_dir=str(tmp_path),
        contract_path=contract_yaml,
        client=client,
        provider=provider,
    )

    score_path = tmp_path / run_id / "final_score.json"
    score = ScoreResult.model_validate(json.loads(score_path.read_text()))
    assert score.complete is True


@pytest.mark.asyncio
async def test_final_score_complete_flag_with_generation_error(
    tmp_path: Path,
    contract_yaml: Path,
) -> None:
    """A run with a generation error writes final_score.json with complete=False."""

    class ExplodingProvider(FakeProvider):
        async def create_sandbox(self, request: SandboxCreateRequest) -> object:  # type: ignore[override]
            raise RuntimeError("infra exploded")

    run_id = "run-incomplete"
    task_ids = ["task-fail"]
    client = FakeClient()

    await run_sandbox(
        run_id=run_id,
        model="openai/gpt-5",
        task_ids=task_ids,
        dataset=None,
        results_dir=str(tmp_path),
        contract_path=contract_yaml,
        client=client,
        provider=ExplodingProvider(),
    )

    score_path = tmp_path / run_id / "final_score.json"
    score = ScoreResult.model_validate(json.loads(score_path.read_text()))
    assert score.complete is False


@pytest.mark.asyncio
async def test_parallelism_zero_raises(
    tmp_path: Path,
    contract_yaml: Path,
) -> None:
    """parallelism < 1 must raise ValueError before any work begins."""
    with pytest.raises(ValueError, match="parallelism"):
        await run_sandbox(
            run_id="run-zero",
            model="openai/gpt-5",
            task_ids=["task-x"],
            dataset=None,
            results_dir=str(tmp_path),
            contract_path=contract_yaml,
            client=FakeClient(),
            provider=FakeProvider(),
            parallelism=0,
        )
