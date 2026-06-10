"""Behavioral tests for the manifest generator."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from benchmark_service.sandbox import ImageSource, SnapshotSource, Resources
from benchmark_service.schemas import RetrieveTaskResponse

from benchmark_runner.sandbox.cli import cli
from benchmark_runner.sandbox.manifest import Manifest, generate_manifest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_contract(tmp_path: Path) -> Path:
    p = tmp_path / "contract.yaml"
    p.write_text(
        "name: test-agent\n"
        "install_cmd: pip install -e .\n"
        "run_cmd: agent run --problem {problem_statement_path}\n"
        "final_output: /app/results\n"
        "secrets:\n"
        "  GOOGLE_API_KEY: projects/x/google\n"
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


# ---------------------------------------------------------------------------
# Test 1: shared-image benchmark
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_image_manifest(tmp_path: Path) -> None:
    """All tasks share one ImageSource → single agent.image, no per-task image.

    Uses the real _fetch_version body with a stubbed HTTP client so version
    mapping is covered end-to-end.
    """
    source = ImageSource(image="registry.example.com/agent:1.0")
    version_resp = _make_version_http_response(
        framework_version="1.0.0", service_version="0.6.1", service_name="mybench"
    )
    client = FakeClient(
        {
            "task-1": _make_retrieve_response(source),
            "task-2": _make_retrieve_response(source),
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

    assert isinstance(manifest, Manifest)

    # Agent block has shared image
    assert manifest.agent.image == "registry.example.com/agent:1.0"
    assert manifest.agent.resources is not None
    assert manifest.agent.cwd == "/app"

    # Contract fields — secrets are env-var NAMES only; the contract's Vals-internal
    # reference value ("projects/x/google") must be dropped, never shipped in a manifest.
    assert manifest.agent.contract.install_cmd == "pip install -e ."
    assert manifest.agent.contract.final_output == "/app/results"
    assert "{problem_statement_path}" in manifest.agent.contract.run_cmd
    assert manifest.agent.contract.secrets == ["GOOGLE_API_KEY"]

    # Eval block
    assert manifest.eval.evaluate_endpoint == "/evaluate-response/"
    assert manifest.eval.score_endpoint == "/final-score/"
    assert manifest.eval.payload_schema == "mybench.text.v1"

    # Tasks are baked (id/question/timeout only — extra_field NOT leaked into serialized manifest)
    assert len(manifest.tasks) == 2
    task_ids = {t.id for t in manifest.tasks}
    assert task_ids == {"task-1", "task-2"}
    manifest_yaml = yaml.safe_dump(manifest.model_dump())
    assert "extra_field" not in manifest_yaml
    for t in manifest.tasks:
        # Per-task image should be absent in shared mode
        assert t.image is None

    # Version fields mapped correctly:
    # service.framework_version and service.service_version are direct copies (no cross-field fallback)
    assert manifest.service.framework_version == "1.0.0"
    assert manifest.service.service_version == "0.6.1"
    # versions.benchmark_service prefers service_version
    assert manifest.versions.benchmark_service == "0.6.1"


# ---------------------------------------------------------------------------
# Test 2: snapshot source → generation fails (a lab-hosted manifest must
# reference a pullable registry image, never a Daytona snapshot)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_source_raises(tmp_path: Path) -> None:
    """A SnapshotSource is not lab-pullable → generate_manifest raises."""
    source = SnapshotSource(snapshot="snap-abc123")
    client = FakeClient({"task-1": _make_retrieve_response(source)})
    contract_path = _make_contract(tmp_path)

    with patch("benchmark_runner.sandbox.manifest._fetch_version", new=AsyncMock(return_value=None)):
        with pytest.raises(ValueError, match="Daytona snapshot"):
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
    instead of emitting agent.image "snapshot:..." (found live against production
    legal-research, which emitted snapshot:...-run-14 as a pullable ref)."""
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
        with pytest.raises(ValueError, match="Daytona snapshot"):
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
    assert "Daytona snapshot" in result.output
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Test 3: per-task images (different ImageSource per task)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_task_images(tmp_path: Path) -> None:
    """Two tasks with different ImageSource → no shared agent.image; each task has its own image."""
    source_a = ImageSource(image="registry.example.com/agent-a:1.0")
    source_b = ImageSource(image="registry.example.com/agent-b:2.0")
    client = FakeClient(
        {
            "task-1": _make_retrieve_response(source_a),
            "task-2": _make_retrieve_response(source_b),
        }
    )
    contract_path = _make_contract(tmp_path)

    with patch("benchmark_runner.sandbox.manifest._fetch_version", new=AsyncMock(return_value=None)):
        manifest = await generate_manifest(
            client=client,
            service_url="http://svc",
            dataset="my-dataset",
            contract_path=contract_path,
            benchmark="mybench",
        )

    # No shared agent image in per-task mode
    assert manifest.agent.image is None

    # Each task carries its own image ref
    task_map = {t.id: t for t in manifest.tasks}
    assert task_map["task-1"].image == "registry.example.com/agent-a:1.0"
    assert task_map["task-2"].image == "registry.example.com/agent-b:2.0"


# ---------------------------------------------------------------------------
# Test 4: shared source but differing resources → per-task branch (regression)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_source_different_resources_is_per_task(tmp_path: Path) -> None:
    """Same ImageSource but different Resources → per-task branch.

    This is the regression case for the shared-detection bug where only `source`
    was compared: tasks would silently inherit the first task's resources.
    """
    source = ImageSource(image="registry.example.com/agent:1.0")
    client = FakeClient(
        {
            "task-1": _make_retrieve_response(source, resources=Resources(vcpu=2, memory=4, disk=10)),
            "task-2": _make_retrieve_response(source, resources=Resources(vcpu=4, memory=8, disk=20)),
        }
    )
    contract_path = _make_contract(tmp_path)

    with patch("benchmark_runner.sandbox.manifest._fetch_version", new=AsyncMock(return_value=None)):
        manifest = await generate_manifest(
            client=client,
            service_url="http://svc",
            dataset="my-dataset",
            contract_path=contract_path,
            benchmark="mybench",
        )

    # Must take per-task branch
    assert manifest.agent.image is None
    assert manifest.agent.resources is None

    task_map = {t.id: t for t in manifest.tasks}
    assert task_map["task-1"].resources == {"vcpu": 2, "memory": 4, "disk": 10}
    assert task_map["task-2"].resources == {"vcpu": 4, "memory": 8, "disk": 20}
    assert task_map["task-1"].image == "registry.example.com/agent:1.0"
    assert task_map["task-2"].image == "registry.example.com/agent:1.0"


# ---------------------------------------------------------------------------
# Test 5: empty dataset → ValueError
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 6: client without _http_client → manifest with null versions
# ---------------------------------------------------------------------------


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

    assert isinstance(manifest, Manifest)
    assert manifest.service.framework_version is None
    assert manifest.service.service_version is None
    assert manifest.versions.benchmark_service is None
