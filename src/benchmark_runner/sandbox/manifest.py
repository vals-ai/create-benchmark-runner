"""Benchmark manifest generator.

Produces a versioned, self-contained manifest describing a benchmark for
lab-hosted consumers (labs that run sandboxes on their own infra and only
call Vals for scoring).
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from benchmark_service.sandbox import ImageSource, SnapshotSource
from benchmark_service.schemas import RetrieveTaskResponse, VersionResponse

from benchmark_runner.sandbox.contract import AgentContract
from benchmark_runner.sandbox.orchestrator import _normalize_source

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typed manifest models
# ---------------------------------------------------------------------------


class ContractSpec(BaseModel):
    install_cmd: str | None
    run_cmd: str
    final_output: str | None
    # Declared secret env-var names the agent needs at runtime (same shape as
    # AgentContract.secrets). The orchestrator's _resolve_secret_env injects only
    # these into the sandbox, so a manifest without them would run agents with no
    # model/tool credentials. Defaulted for manifests generated before this field.
    secrets: dict[str, str] = {}


class AgentSpec(BaseModel):
    image: str | None
    resources: dict[str, Any] | None
    cwd: str | None
    contract: ContractSpec


class EvalSpec(BaseModel):
    evaluate_endpoint: str
    score_endpoint: str
    payload_schema: str


class TaskEntry(BaseModel):
    id: str
    question: str
    timeout: float | None
    image: str | None = None
    resources: dict[str, Any] | None = None
    cwd: str | None = None


class ServiceSpec(BaseModel):
    url: str
    framework_version: str | None
    service_version: str | None


class DatasetSpec(BaseModel):
    name: str
    version: str | None


class VersionsSpec(BaseModel):
    benchmark_service: str | None
    eval: str | None
    dataset: str | None
    image: str | None


class Manifest(BaseModel):
    benchmark: str
    service: ServiceSpec
    dataset: DatasetSpec
    agent: AgentSpec
    eval: EvalSpec
    tasks: list[TaskEntry]
    versions: VersionsSpec


# ---------------------------------------------------------------------------
# Version fetch — injectable for tests, graceful on failure
# ---------------------------------------------------------------------------


async def _fetch_version(http_client: Any, service_url: str) -> VersionResponse | None:
    """GET {service_url}/version using the provided async HTTP client.

    Returns None on any error so a failed version fetch never aborts the manifest.
    """
    try:
        resp = await http_client.get(f"{service_url}/version")
        resp.raise_for_status()
        return VersionResponse.model_validate(resp.json())
    except Exception as exc:
        logger.warning("Could not fetch /version from %s: %s", service_url, exc)
        return None


# ---------------------------------------------------------------------------
# Image source helpers
# ---------------------------------------------------------------------------


def _registry_ref(source: ImageSource | SnapshotSource) -> str:
    """Return the pullable registry image reference for a sandbox source.

    A lab-hosted manifest must reference registry images a lab can pull, so a
    Daytona snapshot (Vals-internal, not pullable) is rejected here rather than
    emitted into an unusable manifest. Once the image-publishing workstream
    lands and retrieve_task returns a registry digest, this passes through.
    """
    # Legacy services return docker_image="snapshot:<name>", which cbs auto-wraps
    # into an ImageSource — normalize first so a snapshot masquerading as an image
    # ref is refused instead of emitted into an unrunnable manifest (found live:
    # production legal-research emitted agent.image "snapshot:...-run-14").
    source = _normalize_source(source)
    if isinstance(source, ImageSource):
        return source.image
    raise ValueError(
        f"sandbox source is a Daytona snapshot ('{source.snapshot}'), which a lab cannot "
        "pull; publish a registry image (digest-pinned) and have retrieve_task return it "
        "before generating a lab-hosted manifest"
    )


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------


async def generate_manifest(
    *,
    client: Any,
    service_url: str,
    dataset: str,
    contract_path: Path,
    benchmark: str,
    payload_schema: str | None = None,
) -> Manifest:
    """Generate a self-contained benchmark manifest.

    Args:
        client: BenchmarkServiceClient (or compatible fake in tests).
        service_url: Base URL of the deployed benchmark service.
        dataset: Dataset name to enumerate.
        contract_path: Path to the agent contract.yaml.
        benchmark: Benchmark identifier (used in payload_schema default).
        payload_schema: Override for the eval payload schema.
            Defaults to ``{benchmark}.text.v1``.

    Returns:
        A populated Manifest ready to serialize.
    """
    payload_schema = payload_schema or f"{benchmark}.text.v1"

    # --- Version (graceful) -------------------------------------------------
    # Use getattr so a client without _http_client (e.g. in tests or alternate
    # implementations) simply skips the version fetch rather than crashing.
    http_client = getattr(client, "_http_client", None)
    if http_client is None:
        logger.warning("Client has no _http_client; version fields will be None")
        version_resp = None
    else:
        version_resp = await _fetch_version(http_client, service_url)

    # --- Task list ----------------------------------------------------------
    list_resp = await client.list_tasks(dataset)
    tasks_raw = list_resp.tasks

    if not tasks_raw:
        raise ValueError(f"dataset '{dataset}' has no tasks; cannot generate a manifest")

    # --- Retrieve details for each task (bounded concurrency) ---------------
    semaphore = asyncio.Semaphore(10)

    async def _retrieve(task: Any) -> RetrieveTaskResponse:
        async with semaphore:
            try:
                return await client.retrieve_task(task.id, dataset=dataset)
            except Exception as exc:
                raise RuntimeError(f"retrieve_task failed for task {task.id}: {exc}") from exc

    details: list[Any] = await asyncio.gather(*[_retrieve(t) for t in tasks_raw])

    # --- Shared vs per-task detection ---------------------------------------
    # All three of source, resources, and cwd must be identical across tasks
    # for the shared branch.  Checking only source was a bug: if tasks shared
    # an image but differed in resources or cwd, later tasks' requirements
    # would be silently dropped (first task's values promoted to agent block).
    first = details[0]
    shared = all(
        d.source == first.source and d.resources == first.resources and d.cwd == first.cwd
        for d in details
    )

    if shared:
        agent_spec = AgentSpec(
            image=_registry_ref(first.source),
            resources=first.resources.model_dump(),
            cwd=first.cwd,
            contract=_contract_spec(contract_path),
        )
        task_entries = [
            TaskEntry(id=t.id, question=t.question, timeout=t.timeout)
            for t in tasks_raw
        ]
    else:
        # Per-task images; agent block has no shared image but still carries contract.
        # Each TaskEntry carries its own image, resources, and cwd.
        agent_spec = AgentSpec(
            image=None,
            resources=None,
            cwd=None,
            contract=_contract_spec(contract_path),
        )

        def _per_task_entry(t: Any, d: Any) -> TaskEntry:
            try:
                ref = _registry_ref(d.source)
            except ValueError as exc:
                raise ValueError(f"task {t.id}: {exc}") from exc
            return TaskEntry(
                id=t.id,
                question=t.question,
                timeout=t.timeout,
                image=ref,
                resources=d.resources.model_dump(),
                cwd=d.cwd,
            )

        task_entries = [_per_task_entry(t, d) for t, d in zip(tasks_raw, details)]

    # --- Versions block -----------------------------------------------------
    # /version gives framework_version + service_version.  We copy each field
    # directly into ServiceSpec (no cross-field fallback) so consumers can tell
    # them apart.  For the single-value versions.benchmark_service summary we
    # prefer service_version and fall back to framework_version, since the
    # framework version is always populated but service_version is more specific.
    # eval/dataset/image versions are not available from this endpoint.
    fw_version: str | None = None
    svc_version: str | None = None
    if version_resp is not None:
        fw_version = version_resp.framework_version
        svc_version = version_resp.service_version

    return Manifest(
        benchmark=benchmark,
        service=ServiceSpec(
            url=service_url,
            framework_version=fw_version,
            service_version=svc_version,
        ),
        dataset=DatasetSpec(name=dataset, version=None),
        agent=agent_spec,
        # Near-term internal Vals eval endpoints by deliberate design decision.
        # Vals owns eval for initial lab/partner integrations; the /v1 eval surface
        # (/v1/evaluate + /v1/score) is deferred and not yet implemented.
        # Regenerate with /v1/evaluate + /v1/score once that surface ships.
        eval=EvalSpec(
            evaluate_endpoint="/evaluate-response/",
            score_endpoint="/final-score/",
            payload_schema=payload_schema,
        ),
        tasks=task_entries,
        versions=VersionsSpec(
            benchmark_service=svc_version or fw_version,
            eval=None,
            dataset=None,
            image=None,
        ),
    )


def _contract_spec(contract_path: Path) -> ContractSpec:
    c = AgentContract.from_yaml(contract_path)
    return ContractSpec(
        install_cmd=c.install_cmd,
        run_cmd=c.run_cmd,
        final_output=c.final_output,
        secrets=c.secrets,
    )


__all__ = ["Manifest", "generate_manifest", "_fetch_version"]
