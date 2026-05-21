"""BenchmarkRunner ABC. Adapters subclass this and implement `load_tasks`
and `generate`; `evaluate` and `score` have default implementations that
POST to the existing internal benchmark service endpoints."""

import asyncio
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from model_library.base import LLMConfig

from benchmark_runner.client import build_client
from benchmark_runner.schemas import (
    EvalResult,
    EvalResultData,
    EvalStatus,
    GenerationResult,
    GenerationStatus,
    ScoreResult,
    Task,
)


class BenchmarkRunner(ABC):
    """Adapter base class for a benchmark runner.

    Subclasses set four class constants identifying the runner, and
    implement `load_tasks` and `generate`. Defaults for `evaluate` and
    `score` POST to the benchmark service through `self._client`.
    """

    NAME: str
    PAYLOAD_TYPE: str = "text"
    PAYLOAD_SCHEMA_VERSION: int = 1
    GENERATION_VERSION_ENV: str

    def __init__(
        self,
        service_url: str,
        dataset_name: str | None = None,
        eval_concurrency: int = 10,
    ):
        self._client = build_client(service_url)
        self._eval_sem = asyncio.Semaphore(eval_concurrency)
        self._dataset = dataset_name
        self._tasks: dict[str, Task] = {}

    @abstractmethod
    def load_tasks(self, dataset_file: str | None) -> list[Task]:
        """Load tasks from a bundled file (or wherever the adapter sources them).

        The framework registers each returned Task before invoking generate/score.
        Adapters may also set `self._dataset` (the benchmark service's dataset
        name, often from the dataset file's `dataset_name` field) so the default
        evaluate/score calls forward it to the service.
        """

    @abstractmethod
    async def generate(
        self,
        task: Task,
        model: str,
        llm_config: LLMConfig | None = None,
        log_dir: Path | None = None,
    ) -> GenerationResult:
        """Run the benchmark's agent on one task and return a GenerationResult."""

    async def evaluate(self, task_id: str, generation: GenerationResult) -> EvalResult:
        """Default per-task evaluation. Short-circuits on DID_NOT_COMPLETE or
        GENERATION_ERROR; otherwise POSTs the generated data to /evaluate-response/."""
        if generation.status in (GenerationStatus.MAX_TIME, GenerationStatus.MAX_TURNS):
            return EvalResult(task_id=task_id, status=EvalStatus.DID_NOT_COMPLETE)
        if generation.status != GenerationStatus.SUCCESS:
            return EvalResult(
                task_id=task_id,
                status=EvalStatus.GENERATION_ERROR,
                error=generation.error,
            )
        try:
            async with self._eval_sem:
                raw = await self._client.evaluate_response(
                    task_id=task_id,
                    response=generation.data,
                    dataset=self._dataset,
                )
            data = EvalResultData.model_validate(raw) if raw is not None else None
            return EvalResult(task_id=task_id, status=EvalStatus.EVALUATED, result=data)
        except Exception as e:
            return EvalResult(task_id=task_id, status=EvalStatus.ERROR, error=str(e))

    async def score(self, eval_results: list[EvalResult]) -> ScoreResult:
        """Default final scoring. Pads missing tasks with null per the contract
        and posts to /final-score/. Returns a ScoreResult."""
        submitted: dict[str, Any] = {ev.task_id: ev.model_dump(mode="json") for ev in eval_results}
        for tid in self._tasks:
            if tid not in submitted:
                submitted[tid] = None
        resp = await self._client.final_score(submitted, dataset=self._dataset)
        payload = resp.model_dump()
        return ScoreResult(
            tasks_evaluated=payload.get("tasks_evaluated", []),
            final_score=float(payload.get("final_score", 0.0)),
            metadata=payload.get("metadata", {}),
        )

    async def _fetch_task(self, task_id: str) -> Task:
        """Forward-compat hook for `GET /v1/tasks/{task_id}`. Adapters override
        to enable service-side dataset loading; default raises."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support service-side task fetch"
        )

    def _register_task(self, task: Task) -> None:
        self._tasks[task.id] = task

    def _register_tasks(self, tasks: list[Task]) -> None:
        self._tasks = {task.id: task for task in tasks}

    def add_task(self, task: Task) -> None:
        """Public alias for `_register_task`. Used by the `--problem` single-task path."""
        self._register_task(task)

    def get_tasks(self) -> list[Task]:
        """Deterministic sorted-by-id task list."""
        return [t for _, t in sorted(self._tasks.items())]

    @property
    def payload_schema(self) -> str:
        return f"{self.NAME}.{self.PAYLOAD_TYPE}.v{self.PAYLOAD_SCHEMA_VERSION}"

    @property
    def generation_version(self) -> str:
        return os.environ.get(self.GENERATION_VERSION_ENV, "dev")


__all__ = ["BenchmarkRunner"]
