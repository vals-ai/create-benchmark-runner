"""Behavioral tests for the manifest generator."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from benchmark_service.sandbox import ImageSource, SnapshotSource, Resources
from benchmark_service.schemas import RetrieveTaskResponse

from benchmark_runner.sandbox.cli import cli
from benchmark_runner.sandbox.manifest import generate_manifest


def _make_contract(tmp_path: Path) -> Path:
    p = tmp_path / "contract.yaml"
    p.write_text(
        "name: test-agent\n"
        "install_cmd: pip install -e .\n"
        "run_cmd: agent run --problem {problem_statement_path}\n"
        "final_output: /app/results\n"
        "secrets:\n"
        "  GOOGLE_API_KEY: projects/x/google\n"
        "required_env:\n"
        "  - GOOGLE_API_KEY\n"
    )
    return p


def _make_retrieve_response(
    source: ImageSource | SnapshotSource,
    resources: Resources | None = None,
    cwd: str = "/app",
) -> RetrieveTaskResponse:
    return RetrieveTaskResponse(
        source=source,
        cwd=cwd,
        problem_path="/app/problem.txt",
        agent_timeout=120.0,
        resources=resources or Resources(vcpu=2, memory=4, disk=10),
    )


class FakeV1Task:
    """Minimal V1Task-like object with an extra field that must NOT be leaked."""

    def __init__(self, task_id: str, extra_field: str = "secret") -> None:
        self.id = task_id
        self.question = f"Question for {task_id}"
        self.timeout = 60.0
        self.extra_field = extra_field  # must NOT appear in manifest


class FakeListTasksResponse:
    def __init__(self, tasks: list[FakeV1Task]) -> None:
        self.tasks = tasks
        self.dataset = "test-dataset"


def _make_version_http_response(
    framework_version: str, service_version: str | None = None, service_name: str | None = None
) -> MagicMock:
    """Build a fake httpx.Response-like object for a /version endpoint."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()  # no-op (success)
    resp.json = MagicMock(
        return_value={
            "framework_version": framework_version,
            "service_version": service_version,
            "service_name": service_name,
        }
    )
    return resp


class FakeClient:
    """Fake BenchmarkServiceClient with controllable retrieve_task responses."""

    def __init__(
        self,
        retrieve_responses: dict[str, RetrieveTaskResponse],
        version_response: MagicMock | None = None,
    ) -> None:
        self._retrieve_responses = retrieve_responses
        self._list_response = FakeListTasksResponse(
            [FakeV1Task(task_id) for task_id in retrieve_responses]
        )
        # Expose a fake _http_client so the real _fetch_version can call it.
        # Default: raise to simulate no network (version fields stay None).
        self._http_client = MagicMock()
        if version_response is not None:
            self._http_client.get = AsyncMock(return_value=version_response)
        else:
            self._http_client.get = AsyncMock(side_effect=Exception("no network in tests"))

    async def list_tasks(self, dataset: str) -> FakeListTasksResponse:
        return self._list_response

    async def retrieve_task(
        self, task_id: str, skip_validation: bool = False, dataset: str | None = None
    ) -> RetrieveTaskResponse:
        return self._retrieve_responses[task_id]


@pytest.mark.asyncio
async def test_generator_fully_populates_every_task(tmp_path: Path) -> None:
    """Every task entry carries its own image/resources/cwd/question/timeout;
    problem_path is emitted agent-level. Uses the real _fetch_version body with
    a stubbed HTTP client so version mapping is covered end-to-end."""
    version_resp = _make_version_http_response(
        framework_version="1.0.0", service_version="0.6.1", service_name="mybench"
    )
    client = FakeClient(
        {
            "task-1": _make_retrieve_response(
                ImageSource(image="registry.example.com/agent-a:1.0"),
                resources=Resources(vcpu=2, memory=4, disk=10),
            ),
            "task-2": _make_retrieve_response(
                ImageSource(image="registry.example.com/agent-b:2.0"),
                resources=Resources(vcpu=4, memory=8, disk=20),
            ),
        },
        version_response=version_resp,
    )
    contract_path = _make_contract(tmp_path)

    manifest = await generate_manifest(
        client=client,
        service_url="http://svc",
        dataset="my-dataset",
        contract_path=contract_path,
        benchmark="mybench",
    )

    assert manifest.agent.problem_path == "/app/problem.txt"

    # Contract fields — the manifest publishes only the explicit lab-facing
    # required_env declaration. The contract's internal `secrets` map (Vals
    # provider keys + secret-manager references like "projects/x/google") is an
    # internal implementation detail and must never appear in a manifest.
    assert manifest.agent.contract.install_cmd == "pip install -e ."
    assert manifest.agent.contract.final_output == "/app/results"
    assert "{problem_statement_path}" in manifest.agent.contract.run_cmd
    assert manifest.agent.contract.required_env == ["GOOGLE_API_KEY"]
    dumped = manifest.model_dump()
    assert "secrets" not in dumped["agent"]["contract"]
    assert "projects/x/google" not in str(dumped)

    # Eval block
    assert manifest.eval.evaluate_endpoint == "/evaluate-response/"
    assert manifest.eval.score_endpoint == "/final-score/"
    assert manifest.eval.payload_schema == "mybench.text.v1"

    # Every task entry is fully populated — no inherit-from-agent fallback exists.
    task_map = {t.id: t for t in manifest.tasks}
    assert set(task_map) == {"task-1", "task-2"}
    assert task_map["task-1"].image == "registry.example.com/agent-a:1.0"
    assert task_map["task-2"].image == "registry.example.com/agent-b:2.0"
    assert task_map["task-1"].resources == Resources(vcpu=2, memory=4, disk=10)
    assert task_map["task-2"].resources == Resources(vcpu=4, memory=8, disk=20)
    for entry in manifest.tasks:
        assert entry.cwd == "/app"
        assert entry.question == f"Question for {entry.id}"
        assert entry.timeout == 60.0
    # FakeV1Task.extra_field must not leak into the serialized manifest
    assert "extra_field" not in yaml.safe_dump(manifest.model_dump())

    # /version fields are direct copies (no cross-field fallback)
    assert manifest.service.framework_version == "1.0.0"
    assert manifest.service.service_version == "0.6.1"


@pytest.mark.asyncio
async def test_snapshot_source_raises(tmp_path: Path) -> None:
    """A SnapshotSource is not lab-pullable → generate_manifest raises."""
    source = SnapshotSource(snapshot="snap-abc123")
    client = FakeClient({"task-1": _make_retrieve_response(source)})
    contract_path = _make_contract(tmp_path)

    with patch("benchmark_runner.sandbox.manifest._fetch_version", new=AsyncMock(return_value=None)):
        with pytest.raises(ValueError, match="is not supported"):
            await generate_manifest(
                client=client,
                service_url="http://svc",
                dataset="my-dataset",
                contract_path=contract_path,
                benchmark="mybench",
            )


@pytest.mark.asyncio
async def test_legacy_snapshot_prefixed_image_source_raises(tmp_path: Path) -> None:
    """Legacy services return docker_image="snapshot:<name>", which cbs auto-wraps
    into an ImageSource — the generator must refuse it like a real SnapshotSource
    instead of emitting "snapshot:..." as a pullable ref (found live against
    production legal-research)."""
    legacy = RetrieveTaskResponse.model_validate({
        "docker_image": "snapshot:legal-research-runner-pkg-962a5e8-run-14",
        "cwd": "/workspace",
        "problem_path": "/workspace/problem.txt",
        "agent_timeout": 60,
        "resources": {"vcpu": 2, "memory": 4, "disk": 10},
    })
    assert isinstance(legacy.source, ImageSource)  # the cbs auto-wrap this guards against
    client = FakeClient({"task-1": legacy})
    contract_path = _make_contract(tmp_path)

    with patch("benchmark_runner.sandbox.manifest._fetch_version", new=AsyncMock(return_value=None)):
        with pytest.raises(ValueError, match="is not supported"):
            await generate_manifest(
                client=client,
                service_url="http://svc",
                dataset="my-dataset",
                contract_path=contract_path,
                benchmark="mybench",
            )


def test_cli_manifest_fails_on_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI exits non-zero with a clear error when the source is a Daytona snapshot, and writes no file."""
    source = SnapshotSource(snapshot="snap-abc123")
    retrieve_resp = _make_retrieve_response(source)

    class SnapshotClient(FakeClient):
        def __init__(self) -> None:
            super().__init__({"task-1": retrieve_resp})

    monkeypatch.setenv("SERVICE_URL", "http://svc")
    monkeypatch.setattr("benchmark_runner.sandbox.cli.BenchmarkServiceClient", lambda *a, **kw: SnapshotClient())
    monkeypatch.setattr(
        "benchmark_runner.sandbox.manifest._fetch_version",
        AsyncMock(return_value=None),
    )

    contract_path = _make_contract(tmp_path)
    output_path = tmp_path / "manifest.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "manifest",
            "--dataset", "my-dataset",
            "--contract", str(contract_path),
            "--benchmark", "mybench",
            "--output", str(output_path),
        ],
    )

    assert result.exit_code != 0
    assert "is not supported. Please contact Vals." in result.output
    assert not output_path.exists()


@pytest.mark.asyncio
async def test_divergent_problem_path_raises(tmp_path: Path) -> None:
    """Tasks with different problem_path values must raise ValueError mentioning
    problem_path: it is emitted agent-level, so divergent per-task values can't
    be represented in a manifest."""
    source = ImageSource(image="registry.example.com/agent:1.0")
    resp_a = RetrieveTaskResponse(
        source=source,
        cwd="/app",
        problem_path="/app/problem.txt",
        agent_timeout=60.0,
        resources=Resources(vcpu=2, memory=4, disk=10),
    )
    resp_b = RetrieveTaskResponse(
        source=source,
        cwd="/app",
        problem_path="/app/other_problem.txt",
        agent_timeout=60.0,
        resources=Resources(vcpu=2, memory=4, disk=10),
    )
    client = FakeClient({"task-1": resp_a, "task-2": resp_b})
    contract_path = _make_contract(tmp_path)

    with patch("benchmark_runner.sandbox.manifest._fetch_version", new=AsyncMock(return_value=None)):
        with pytest.raises(ValueError, match="problem_path"):
            await generate_manifest(
                client=client,
                service_url="http://svc",
                dataset="my-dataset",
                contract_path=contract_path,
                benchmark="mybench",
            )


@pytest.mark.asyncio
async def test_empty_dataset_raises(tmp_path: Path) -> None:
    """list_tasks returning zero tasks should raise ValueError with a clear message."""

    class EmptyClient(FakeClient):
        def __init__(self) -> None:
            super().__init__({})  # no tasks

    contract_path = _make_contract(tmp_path)
    with patch("benchmark_runner.sandbox.manifest._fetch_version", new=AsyncMock(return_value=None)):
        with pytest.raises(ValueError, match="has no tasks"):
            await generate_manifest(
                client=EmptyClient(),
                service_url="http://svc",
                dataset="empty-dataset",
                contract_path=contract_path,
                benchmark="mybench",
            )


@pytest.mark.asyncio
async def test_no_http_client_produces_null_versions(tmp_path: Path) -> None:
    """A client without _http_client still produces a manifest; version fields are None."""
    source = ImageSource(image="registry.example.com/agent:1.0")

    class ClientWithoutHttpClient:
        """Minimal fake client with no _http_client attribute."""

        def __init__(self) -> None:
            resp = _make_retrieve_response(source)
            self._list_response = FakeListTasksResponse([FakeV1Task("task-1")])
            self._retrieve_responses = {"task-1": resp}

        async def list_tasks(self, dataset: str) -> FakeListTasksResponse:
            return self._list_response

        async def retrieve_task(
            self, task_id: str, skip_validation: bool = False, dataset: str | None = None
        ) -> RetrieveTaskResponse:
            return self._retrieve_responses[task_id]

    contract_path = _make_contract(tmp_path)
    manifest = await generate_manifest(
        client=ClientWithoutHttpClient(),
        service_url="http://svc",
        dataset="my-dataset",
        contract_path=contract_path,
        benchmark="mybench",
    )

    assert manifest.service.framework_version is None
    assert manifest.service.service_version is None
