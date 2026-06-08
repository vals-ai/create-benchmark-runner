"""Shared fakes and fixtures for sandbox orchestrator tests."""

import json
from pathlib import Path

import pytest

from benchmark_service.sandbox import ImageSource, Resources, SandboxCreateRequest
from benchmark_service.schemas import FinalScoreResponse, RetrieveTaskResponse, SetupTaskResponse
from benchmark_runner.schemas import GenerationStatus


class FakeExecResult:
    def __init__(self, exit_code: int = 0, output: str = "") -> None:
        self.exit_code = exit_code
        self.output = output


class FakeSandbox:
    """Sandbox double that returns a canned GenerationResult."""

    def __init__(
        self,
        sandbox_id: str,
        task_answer: str = "ANSWER",
        generation_status: GenerationStatus = GenerationStatus.SUCCESS,
    ) -> None:
        self._id = sandbox_id
        self._task_answer = task_answer
        self._generation_status = generation_status

    @property
    def id(self) -> str:
        return self._id

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> FakeExecResult:
        return FakeExecResult(exit_code=0, output="")

    async def download_file(self, remote_path: str) -> bytes:
        # Parse task_id from path: "<final_output>/<task_id>/generation.json"
        task_id = remote_path.rstrip("/").split("/")[-2]
        payload = {
            "task_id": task_id,
            "status": self._generation_status,
            "data": self._task_answer,
        }
        return json.dumps(payload).encode()


class FakeProvider:
    """SandboxProvider double. Records created sandbox names and deleted ids for assertion."""

    def __init__(self, generation_status: GenerationStatus = GenerationStatus.SUCCESS) -> None:
        self.created: list[str] = []
        self.deleted: list[str] = []
        self.last_request: SandboxCreateRequest | None = None
        self._generation_status = generation_status

    async def create_sandbox(self, request: SandboxCreateRequest) -> FakeSandbox:
        self.created.append(request.name)
        self.last_request = request
        sandbox_id = f"sandbox-{request.name}"
        return FakeSandbox(sandbox_id=sandbox_id, generation_status=self._generation_status)

    async def delete_sandbox(self, instance_id: str) -> None:
        self.deleted.append(instance_id)


class FakeClient:
    """BenchmarkServiceClient double."""

    def __init__(self) -> None:
        self._source = ImageSource(image="img:latest")
        self._resources = Resources(vcpu=1, memory=2, disk=5)
        self.last_final_score_args: dict[str, object] | None = None

    async def retrieve_task(
        self,
        task_id: str,
        skip_validation: bool = False,
        dataset: str | None = None,
    ) -> RetrieveTaskResponse:
        return RetrieveTaskResponse(
            source=self._source,
            cwd="/app",
            problem_path="/app/problem.txt",
            agent_timeout=60.0,
            resources=self._resources,
        )

    async def setup_task(
        self,
        task_id: str,
        instance_id: str,
        on_message: object = None,
        dataset: str | None = None,
        sandbox_provider: object = None,
    ) -> SetupTaskResponse:
        return SetupTaskResponse(status="ok")

    async def evaluate_response(
        self,
        task_id: str,
        response: str,
        dataset: str | None = None,
    ) -> dict[str, object]:
        return {"pass_percentage": 1.0, "eval_version": "v1"}

    async def final_score(
        self,
        evaluation_results: dict[str, object],
        dataset: str | None = None,
    ) -> FinalScoreResponse:
        self.last_final_score_args = evaluation_results
        evaluated = [k for k, v in evaluation_results.items() if v is not None]
        return FinalScoreResponse(
            tasks_evaluated=evaluated,
            final_score=float(len(evaluated)) / max(len(evaluation_results), 1),
            metadata={},
        )

    def get_sandbox_provider(self) -> FakeProvider:
        return FakeProvider()


@pytest.fixture
def contract_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "contract.yaml"
    p.write_text(
        "name: test-agent\n"
        "run_cmd: agent run --model {model} --problem {problem_statement_path} --task {task_id}\n"
        "final_output: /app/results\n"
    )
    return p
