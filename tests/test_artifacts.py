import json

import pytest

from benchmark_runner.artifacts import RunArtifacts
from benchmark_runner.schemas import (
    EvalResult,
    EvalResultData,
    EvalStatus,
    GenerationResult,
    GenerationStatus,
    ScoreResult,
)


def test_paths_and_missing_loads(tmp_path):
    art = RunArtifacts(results_dir=tmp_path, run_id="r1")
    assert art.run_dir == tmp_path / "r1"
    assert art.run_config_path == tmp_path / "r1" / "run_config.json"
    assert art.final_score_path == tmp_path / "r1" / "final_score.json"
    assert art.task_dir("t1") == tmp_path / "r1" / "t1"
    assert art.generation_path("t1") == tmp_path / "r1" / "t1" / "generation.json"
    assert art.eval_path("t1") == tmp_path / "r1" / "t1" / "eval.json"
    assert art.agent_logs_dir("t1") == tmp_path / "r1" / "t1" / "agent_logs"

    assert art.load_run_config() is None
    assert art.load_generation("t1") is None


def test_run_config_round_trip(tmp_path):
    art = RunArtifacts(results_dir=tmp_path, run_id="r1")
    art.save_run_config({
        "run_id": "r1",
        "model": "m",
        "tasks": ["t1", "t2"],
        "dataset_file": "/data/x.json",
        "payload_schema": "fabv2.text.v1",
        "payload_type": "text",
        "runner_version": "0.1.0",
        "generation_version": "abc123",
    })
    loaded = art.load_run_config()
    assert loaded is not None
    assert loaded["run_id"] == "r1"
    assert loaded["tasks"] == ["t1", "t2"]


def test_generation_eval_and_final_score_round_trip(tmp_path):
    art = RunArtifacts(results_dir=tmp_path, run_id="r1")
    gen = GenerationResult(
        task_id="t1",
        status=GenerationStatus.SUCCESS,
        data="42",
        generation_version="abc",
    )
    art.save_generation("t1", gen)
    loaded = art.load_generation("t1")
    assert loaded == gen

    ev = EvalResult(
        task_id="t1",
        status=EvalStatus.EVALUATED,
        result=EvalResultData(pass_percentage=0.8, eval_version="v1"),
    )
    art.save_eval("t1", ev)
    assert art.load_eval("t1") == ev

    sr = ScoreResult(tasks_evaluated=["t1"], final_score=0.8, metadata={}, complete=True)
    art.save_final_score(sr)
    assert art.load_final_score() == sr


def test_saved_generation_includes_legacy_answer_field(tmp_path):
    art = RunArtifacts(results_dir=tmp_path, run_id="r1")
    gen = GenerationResult(
        task_id="t1",
        status=GenerationStatus.SUCCESS,
        data="42",
        generation_version="abc",
    )

    art.save_generation("t1", gen)

    raw = json.loads(art.generation_path("t1").read_text())
    assert raw["data"] == "42"
    assert raw["answer"] == "42"
    assert art.load_generation("t1") == gen


def test_creates_parent_dirs_on_save(tmp_path):
    art = RunArtifacts(results_dir=tmp_path / "deep" / "nested", run_id="r1")
    gen = GenerationResult(task_id="t1", status=GenerationStatus.SUCCESS, data="x")
    art.save_generation("t1", gen)
    assert (tmp_path / "deep" / "nested" / "r1" / "t1" / "generation.json").exists()


@pytest.mark.parametrize("task_id", ["../escape", "nested/task", r"nested\task", "", "."])
def test_rejects_unsafe_task_ids(tmp_path, task_id):
    art = RunArtifacts(results_dir=tmp_path, run_id="r1")
    with pytest.raises(ValueError):
        art.generation_path(task_id)
