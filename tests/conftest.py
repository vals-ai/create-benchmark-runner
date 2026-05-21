"""Shared test fixtures: a mock BenchmarkServiceClient and a minimal test adapter."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from benchmark_runner import BenchmarkRunner, GenerationResult, GenerationStatus, Task


@pytest.fixture
def mock_client():
    """Returns a Mock with the BenchmarkServiceClient async methods stubbed."""
    client = AsyncMock()
    client.evaluate_response = AsyncMock(return_value={
        "pass_percentage": 0.8,
        "eval_version": "v1",
    })
    client.final_score = AsyncMock()
    return client


@pytest.fixture
def make_test_adapter():
    """Returns a factory that builds a configurable test adapter class."""

    def _make(*, generate_status: GenerationStatus = GenerationStatus.SUCCESS):
        class TestRunner(BenchmarkRunner):
            NAME = "test-bench"
            PAYLOAD_TYPE = "text"
            PAYLOAD_SCHEMA_VERSION = 1
            GENERATION_VERSION_ENV = "TEST_BENCH_GENERATION_VERSION"

            def load_tasks(self, dataset_file: str | None) -> list[Task]:
                self._dataset = "validation"
                return [Task(id="t1", question="q1"), Task(id="t2", question="q2")]

            async def generate(
                self,
                task: Task,
                model: str,
                llm_config: Any = None,
                log_dir: Any = None,
            ) -> GenerationResult:
                return GenerationResult(
                    task_id=task.id,
                    status=generate_status,
                    data=f"answer-{task.id}",
                    question=task.question,
                    model=model,
                    generation_version=self.generation_version,
                )

        return TestRunner

    return _make
