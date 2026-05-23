"""Benchmark runner base class."""

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
        """Load task definitions for a run."""

    @abstractmethod
    async def generate(
        self,
        task: Task,
        model: str,
        llm_config: LLMConfig | None = None,
        log_dir: Path | None = None,
    ) -> GenerationResult:
        """Generate a task result."""

    async def evaluate(self, task_id: str, generation: GenerationResult) -> EvalResult:
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

    async def load_tasks_from_service(self, dataset_name: str) -> list[Task]:
        """Fetch tasks for `dataset_name` via the service's /v1/ surface.

        Concrete framework method — adapters do not override. Stamps
        `self._dataset = dataset_name` so default `evaluate` / `score`
        forward it to the service alongside per-task calls.
        """
        response = await self._client.list_tasks(dataset=dataset_name)
        self._dataset = dataset_name
        # V1Task and Task have the same shape (id, question, timeout) and
        # both accept extras; convert by model_dump round-trip.
        return [Task.model_validate(t.model_dump()) for t in response.tasks]

    async def _fetch_task(self, task_id: str) -> Task:
        raise NotImplementedError(
            f"{type(self).__name__} does not support service-side task fetch"
        )

    def _register_task(self, task: Task) -> None:
        self._tasks[task.id] = task

    def _register_tasks(self, tasks: list[Task]) -> None:
        self._tasks = {task.id: task for task in tasks}

    def add_task(self, task: Task) -> None:
        self._register_task(task)

    def get_tasks(self) -> list[Task]:
        return [t for _, t in sorted(self._tasks.items())]

    @property
    def payload_schema(self) -> str:
        return f"{self.NAME}.{self.PAYLOAD_TYPE}.v{self.PAYLOAD_SCHEMA_VERSION}"

    @property
    def generation_version(self) -> str:
        return os.environ.get(self.GENERATION_VERSION_ENV, "dev")


__all__ = ["BenchmarkRunner"]
