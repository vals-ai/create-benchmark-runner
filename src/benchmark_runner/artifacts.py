"""On-disk artifact layout for a benchmark run.

results/<run_id>/
    run_config.json
    final_score.json
    <task_id>/
        generation.json
        eval.json
        agent_logs/
"""

import json
from pathlib import Path
from typing import Any

from benchmark_runner.schemas import EvalResult, GenerationResult, ScoreResult


class RunArtifacts:
    def __init__(self, results_dir: Path | str, run_id: str):
        self._results_dir = Path(results_dir)
        self._run_id = run_id

    @property
    def run_dir(self) -> Path:
        return self._results_dir / self._run_id

    @property
    def run_config_path(self) -> Path:
        return self.run_dir / "run_config.json"

    @property
    def final_score_path(self) -> Path:
        return self.run_dir / "final_score.json"

    def task_dir(self, task_id: str) -> Path:
        _validate_task_id(task_id)
        return self.run_dir / task_id

    def generation_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "generation.json"

    def eval_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "eval.json"

    def agent_logs_dir(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "agent_logs"

    def load_run_config(self) -> dict[str, Any] | None:
        return _load_json(self.run_config_path)

    def save_run_config(self, config: dict[str, Any]) -> None:
        _save_json(self.run_config_path, config)

    def load_generation(self, task_id: str) -> GenerationResult | None:
        raw = _load_json(self.generation_path(task_id))
        if raw is None:
            return None
        return GenerationResult.model_validate(raw)

    def save_generation(self, task_id: str, gen: GenerationResult) -> None:
        _save_json(self.generation_path(task_id), gen.model_dump(mode="json"))

    def load_eval(self, task_id: str) -> EvalResult | None:
        raw = _load_json(self.eval_path(task_id))
        if raw is None:
            return None
        return EvalResult.model_validate(raw)

    def save_eval(self, task_id: str, ev: EvalResult) -> None:
        _save_json(self.eval_path(task_id), ev.model_dump(mode="json"))

    def load_final_score(self) -> ScoreResult | None:
        raw = _load_json(self.final_score_path)
        if raw is None:
            return None
        return ScoreResult.model_validate(raw)

    def save_final_score(self, sr: ScoreResult) -> None:
        _save_json(self.final_score_path, sr.model_dump(mode="json"))


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _validate_task_id(task_id: str) -> None:
    if not task_id or task_id in {".", ".."} or "/" in task_id or "\\" in task_id:
        raise ValueError(f"Unsafe task ID for artifact path: {task_id!r}")


__all__ = ["RunArtifacts"]
