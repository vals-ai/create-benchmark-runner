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

from benchmark_service.sandbox import ImageSource, Resources, SnapshotSource
from benchmark_service.schemas import RetrieveTaskResponse, VersionResponse

from benchmark_runner.sandbox.contract import AgentContract

logger = logging.getLogger(__name__)


class ContractSpec(BaseModel):
    install_cmd: str | None
    run_cmd: str
    final_output: str | None
    # Env var NAMES the lab must set for the agent to function. Sourced from the
    # contract's explicit `required_env` declaration; internal secrets never ship.
    required_env: list[str] = []


class AgentSpec(BaseModel):
    # In-sandbox path where the agent expects the problem statement.
    problem_path: str
    contract: ContractSpec


class EvalSpec(BaseModel):
    evaluate_endpoint: str
    score_endpoint: str
    payload_schema: str


class TaskEntry(BaseModel):
    id: str
    question: str
    timeout: float | None
    image: str
    resources: Resources
    cwd: str


class ServiceSpec(BaseModel):
    url: str
    framework_version: str | None
    service_version: str | None


class DatasetSpec(BaseModel):
    name: str
    version: str | None


class Manifest(BaseModel):
    benchmark: str
    service: ServiceSpec
    dataset: DatasetSpec
    agent: AgentSpec
    eval: EvalSpec
    tasks: list[TaskEntry]


async def _fetch_version(http_client: Any, service_url: str) -> VersionResponse | None:
    """GET {service_url}/version; None on any error so a failed fetch never aborts the manifest."""
    try:
        resp = await http_client.get(f"{service_url}/version")
        resp.raise_for_status()
        return VersionResponse.model_validate(resp.json())
    except Exception as exc:
        logger.warning("Could not fetch /version from %s: %s", service_url, exc)
        return None


def _registry_ref(source: ImageSource | SnapshotSource) -> str:
    """Pullable registry image ref for a sandbox source; snapshots are refused."""
    # Legacy services return docker_image="snapshot:<name>", which cbs auto-wraps
    # into an ImageSource — refuse that prefix like a real SnapshotSource.
    if isinstance(source, ImageSource) and not source.image.startswith("snapshot:"):
        return source.image
    raise ValueError(f"Snapshot source={source} is not supported. Please contact Vals.")


async def generate_manifest(
    *,
    client: Any,
    service_url: str,
    dataset: str,
    contract_path: Path,
    benchmark: str,
    payload_schema: str | None = None,
) -> Manifest:
    """Generate a self-contained benchmark manifest for lab-hosted consumers.

    payload_schema defaults to ``{benchmark}.text.v1``.
    """
    payload_schema = payload_schema or f"{benchmark}.text.v1"

    # getattr so a client without _http_client (tests, alternate implementations)
    # skips the version fetch rather than crashing.
    http_client = getattr(client, "_http_client", None)
    if http_client is None:
        logger.warning("Client has no _http_client; version fields will be None")
        version_resp = None
    else:
        version_resp = await _fetch_version(http_client, service_url)

    list_resp = await client.list_tasks(dataset)
    tasks_raw = list_resp.tasks

    if not tasks_raw:
        raise ValueError(f"dataset '{dataset}' has no tasks; cannot generate a manifest")

    semaphore = asyncio.Semaphore(10)

    async def _retrieve(task: Any) -> RetrieveTaskResponse:
        async with semaphore:
            try:
                return await client.retrieve_task(task.id, dataset=dataset)
            except Exception as exc:
                raise RuntimeError(f"retrieve_task failed for task {task.id}: {exc}") from exc

    details: list[Any] = await asyncio.gather(*[_retrieve(t) for t in tasks_raw])

    # problem_path is emitted agent-level, so divergent per-task values can't be
    # represented in a manifest; reject early rather than silently dropping all
    # but the first task's value.
    first = details[0]
    if any(d.problem_path != first.problem_path for d in details):
        raise ValueError(
            "tasks have divergent problem_path values; manifests carry problem_path at the "
            "agent level (not per-task), so all tasks must share the same problem_path"
        )

    def _task_entry(t: Any, d: Any) -> TaskEntry:
        try:
            ref = _registry_ref(d.source)
        except ValueError as exc:
            raise ValueError(f"task {t.id}: {exc}") from exc
        return TaskEntry(
            id=t.id,
            question=t.question,
            timeout=t.timeout,
            image=ref,
            resources=d.resources,
            cwd=d.cwd,
        )

    task_entries = [_task_entry(t, d) for t, d in zip(tasks_raw, details)]

    return Manifest(
        benchmark=benchmark,
        service=ServiceSpec(
            url=service_url,
            framework_version=version_resp.framework_version if version_resp else None,
            service_version=version_resp.service_version if version_resp else None,
        ),
        dataset=DatasetSpec(name=dataset, version=None),
        agent=AgentSpec(
            problem_path=first.problem_path,
            contract=_contract_spec(contract_path),
        ),
        # Vals-internal eval endpoints by design: the /v1 eval surface is deferred.
        # Regenerate with /v1/evaluate + /v1/score once it ships.
        eval=EvalSpec(
            evaluate_endpoint="/evaluate-response/",
            score_endpoint="/final-score/",
            payload_schema=payload_schema,
        ),
        tasks=task_entries,
    )


def _contract_spec(contract_path: Path) -> ContractSpec:
    c = AgentContract.from_yaml(contract_path)
    return ContractSpec(
        install_cmd=c.install_cmd,
        run_cmd=c.run_cmd,
        final_output=c.final_output,
        # Only the explicit lab-facing declaration — the internal `secrets` map never ships.
        required_env=sorted(c.required_env),
    )


__all__ = ["Manifest", "generate_manifest", "_fetch_version"]
