from benchmark_runner.artifacts import RunArtifacts
from benchmark_runner.checkpoint import (
    is_eval_redoable,
    is_generation_redoable,
    load_or_create_run_config,
)
from benchmark_runner.schemas import (
    EvalResult,
    EvalStatus,
    GenerationResult,
    GenerationStatus,
)


def test_load_or_create_stamps_new_run_and_resumes_existing(tmp_path):
    art = RunArtifacts(results_dir=tmp_path, run_id="r1")
    config, was_resumed = load_or_create_run_config(
        artifacts=art,
        model="m",
        task_ids=["t1", "t2"],
        dataset_file="/data/x.json",
        dataset_name="validation",
        task_source="file",
        payload_schema="test.text.v1",
        payload_type="text",
        runner_version="0.1.0",
        generation_version="abc",
    )
    assert was_resumed is False
    assert config["run_id"] == "r1"
    assert config["tasks"] == ["t1", "t2"]
    assert config["dataset_name"] == "validation"
    assert config["task_source"] == "file"
    assert config["payload_schema"] == "test.text.v1"
    assert config["payload_type"] == "text"
    assert config["runner_version"] == "0.1.0"
    assert config["generation_version"] == "abc"

    saved = art.load_run_config()
    assert saved == config

    resumed, was_resumed = load_or_create_run_config(
        artifacts=RunArtifacts(results_dir=tmp_path, run_id="r1"),
        model="new-model",
        task_ids=["different"],
        dataset_file=None,
        dataset_name=None,
        task_source="service",
        payload_schema="x",
        payload_type="x",
        runner_version="x",
        generation_version="x",
    )
    assert was_resumed is True
    assert resumed == config


def test_generation_redoable_for_missing_or_error_only(tmp_path):
    art = RunArtifacts(results_dir=tmp_path, run_id="r1")
    assert is_generation_redoable(art, "missing") is True

    art.save_generation("t1", GenerationResult(task_id="t1", status=GenerationStatus.ERROR, data=""))
    assert is_generation_redoable(art, "t1") is True

    art.save_generation("t1", GenerationResult(task_id="t1", status=GenerationStatus.SUCCESS, data="x"))
    assert is_generation_redoable(art, "t1") is False


def test_eval_redoable_for_missing_error_or_generation_error_only(tmp_path):
    art = RunArtifacts(results_dir=tmp_path, run_id="r1")
    assert is_eval_redoable(art, "missing") is True

    art.save_eval("t1", EvalResult(task_id="t1", status=EvalStatus.ERROR, error="boom"))
    assert is_eval_redoable(art, "t1") is True

    art.save_eval("t1", EvalResult(task_id="t1", status=EvalStatus.GENERATION_ERROR, error="agent"))
    assert is_eval_redoable(art, "t1") is True

    art.save_eval("t1", EvalResult(task_id="t1", status=EvalStatus.EVALUATED))
    assert is_eval_redoable(art, "t1") is False

    art.save_eval("t1", EvalResult(task_id="t1", status=EvalStatus.DID_NOT_COMPLETE))
    assert is_eval_redoable(art, "t1") is False
