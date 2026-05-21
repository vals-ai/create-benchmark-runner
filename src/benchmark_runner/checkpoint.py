"""Run-config bootstrap and resume detection.

Centralizes the logic that both existing runners had inline in their
`_load_run` and per-task `process()` functions.
"""

from typing import Any

from benchmark_runner.artifacts import RunArtifacts
from benchmark_runner.schemas import EvalStatus, GenerationStatus


def load_or_create_run_config(
    *,
    artifacts: RunArtifacts,
    model: str,
    task_ids: list[str],
    dataset_file: str | None,
    dataset_name: str | None,
    payload_schema: str,
    payload_type: str,
    runner_version: str,
    generation_version: str,
) -> tuple[dict[str, Any], bool]:
    """Load an existing run_config.json or stamp a new one.

    Returns (config, was_resumed). On resume, the on-disk config wins;
    the caller's arguments are ignored.
    """
    existing = artifacts.load_run_config()
    if existing is not None:
        return existing, True

    config: dict[str, Any] = {
        "run_id": artifacts._run_id,
        "model": model,
        "tasks": task_ids,
        "dataset_file": dataset_file,
        "dataset_name": dataset_name,
        "payload_schema": payload_schema,
        "payload_type": payload_type,
        "runner_version": runner_version,
        "generation_version": generation_version,
    }
    artifacts.save_run_config(config)
    return config, False


def is_generation_redoable(artifacts: RunArtifacts, task_id: str) -> bool:
    """True if generation.json is missing or in an error state."""
    gen = artifacts.load_generation(task_id)
    if gen is None:
        return True
    return gen.status == GenerationStatus.ERROR


def is_eval_redoable(artifacts: RunArtifacts, task_id: str) -> bool:
    """True if eval.json is missing or in an error / generation_error state."""
    ev = artifacts.load_eval(task_id)
    if ev is None:
        return True
    return ev.status in (EvalStatus.ERROR, EvalStatus.GENERATION_ERROR)


__all__ = ["is_eval_redoable", "is_generation_redoable", "load_or_create_run_config"]
