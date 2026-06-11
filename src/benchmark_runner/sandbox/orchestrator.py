"""Sandbox orchestrator: drives one cloud sandbox per task through the full run loop."""

import asyncio
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_service.sandbox import SandboxCreateRequest
from benchmark_service.sandbox.types import ImageSource, Resources, SandboxSource

from benchmark_runner.artifacts import RunArtifacts
from benchmark_runner.checkpoint import is_eval_redoable, is_generation_redoable
from benchmark_runner.sandbox.backend import SandboxGenerationBackend, _format_exc
from benchmark_runner.sandbox.contract import AgentContract, format_run_cmd
from benchmark_runner.sandbox.protocols import BenchmarkServiceClientLike, SandboxProviderLike
from benchmark_runner.schemas import (
    EvalResult,
    EvalResultData,
    EvalStatus,
    GenerationResult,
    GenerationStatus,
    ScoreResult,
)

logger = logging.getLogger(__name__)

# Bring-your-own-model-endpoint env vars, forwarded into every sandbox when set,
# independent of the contract. A lab evaluating its own (often unreleased) model
# points generation at its own endpoint; the baked runner reads these from env
# (create-benchmark-runner cli.py CUSTOM_ENDPOINT/CUSTOM_API_KEY fallbacks), so no
# run_cmd or per-agent contract change is needed. These are universal generation
# parameters, not agent-specific secrets, so they live here rather than in any
# contract's `secrets:` — keeping the lab's model and endpoint off Vals infra.
_BYO_ENDPOINT_ENV_VARS = (
    "CUSTOM_ENDPOINT",
    "CUSTOM_API_KEY",
)


def _require_image_source(source: SandboxSource) -> ImageSource:
    """Only pullable registry images are supported as sandbox sources.

    Daytona snapshots are Vals-internal and unusable for lab-hosted sandboxes, so
    a SnapshotSource — or a legacy service returning the string
    ``docker_image="snapshot:<name>"``, which cbs wraps into an ImageSource — is
    rejected with a clear error instead of surfacing as a failed image pull.
    """
    if isinstance(source, ImageSource) and not source.image.startswith("snapshot:"):
        return source
    raise ValueError(
        f"unsupported sandbox source {source!r}: the service must return a pullable "
        "registry image (snapshots are not supported)"
    )


def _resolve_secret_env(contract: AgentContract) -> dict[str, str]:
    """Resolve the contract's declared secret env-var names from the orchestrator's
    own environment. Keying on the contract keeps unrelated orchestrator env
    (SERVICE_URL, Daytona creds) out of the sandbox; a declared name missing from
    the env is an error, since the agent would only fail later without it.

    The BYO-endpoint vars (_BYO_ENDPOINT_ENV_VARS) are forwarded on top whenever
    set, independent of the contract, so a lab can point generation at its own
    model endpoint without editing any agent's contract.
    """
    env: dict[str, str] = {}
    missing: list[str] = []
    for var_name in contract.secrets:
        value = os.environ.get(var_name)
        if value:
            env[var_name] = value
        else:
            missing.append(var_name)
    if missing:
        raise ValueError(
            "contract declares secret(s) not set in the orchestrator env: "
            + ", ".join(sorted(missing))
        )
    for var_name in _BYO_ENDPOINT_ENV_VARS:
        value = os.environ.get(var_name)
        if value:
            env[var_name] = value
    return env

# Sandbox lifecycle constants
# auto_stop_interval is in minutes (Daytona SDK); 30 min ensures an undeleted sandbox
# eventually stops even if orchestrator cleanup fails.
SANDBOX_AUTO_STOP_INTERVAL = 30
SANDBOX_CREATE_TIMEOUT = 600  # seconds to wait for sandbox readiness


@dataclass(frozen=True)
class SandboxTaskSpec:
    source: SandboxSource
    resources: Resources
    cwd: str
    agent_timeout: float | None
    question: str
    problem_path: str


async def run_benchmark(
    *,
    run_id: str,
    model: str,
    task_ids: list[str],
    dataset: str | None,
    results_dir: str,
    contract_path: Path | str | None = None,
    contract: AgentContract | None = None,
    client: BenchmarkServiceClientLike,
    provider: SandboxProviderLike | None = None,
    parallelism: int = 10,
    source_override: ImageSource | None = None,
    task_specs: Mapping[str, SandboxTaskSpec] | None = None,
) -> None:
    """Run the full benchmark loop against cloud sandboxes, one per task.

    Handles resume: skips generation/eval if valid artifacts already exist.

    Exactly one of `contract_path` (load contract.yaml from disk) or `contract`
    (an in-memory AgentContract, e.g. built from an installed manifest's
    ContractSpec) must be provided.

    source_override: when set, boot every sandbox from this image instead of the
    one retrieve_task returns. Eval is text-based (the generation string is sent to
    the service judge, not the sandbox), so this lets you point generation at a
    different/known-good agent image while still using the service's real dataset
    and rubric eval — e.g. to validate a candidate agent build before repointing
    the deployed service.
    """
    if parallelism < 1:
        raise ValueError(f"parallelism must be >= 1, got {parallelism}")

    if contract_path is not None:
        if contract is not None:
            raise ValueError("provide exactly one of contract or contract_path, not both")
        contract = AgentContract.from_yaml(Path(contract_path))
    elif contract is None:
        raise ValueError("provide exactly one of contract or contract_path")

    if provider is None:
        provider = client.get_sandbox_provider()

    # Without final_output there is no result file to read; fail before booting sandboxes.
    if contract.final_output is None:
        raise ValueError("contract.final_output is not set; no generation file to read")
    contract = contract.model_copy(update={"run_cmd": format_run_cmd(contract.run_cmd, {"model": model})})
    secret_env = _resolve_secret_env(contract)  # constant across tasks

    artifacts = RunArtifacts(results_dir=results_dir, run_id=run_id)
    config = artifacts.load_run_config()
    if config is None:
        config = {
            "run_id": run_id,
            "model": model,
            "tasks": task_ids,
            "dataset_name": dataset,
            "task_source": "sandbox",
            "payload_schema": f"{contract.name}.text.v1",
            "payload_type": "text",
        }
        artifacts.save_run_config(config)
    elif dataset is None:
        dataset = config.get("dataset_name")
    config_task_ids: list[str] = config["tasks"]

    backend = SandboxGenerationBackend()
    sem = asyncio.Semaphore(parallelism)

    async def _run_task(tid: str) -> None:
        # The semaphore gates generation only: eval is an HTTP call to the service
        # judge (can take minutes) and holding a sandbox slot through it would
        # throttle throughput.
        async with sem:
            await _process_generation(tid)
        await _process_eval(tid)

    async def _process_generation(tid: str) -> None:
        if not is_generation_redoable(artifacts, tid):
            return

        sandbox = None
        try:
            task_spec = task_specs.get(tid) if task_specs is not None else None
            # Manifest-native tasks (spec present) skip the service callbacks: the
            # service cannot reach lab/local sandboxes, so the orchestrator uploads
            # the question itself and lab credentials stay lab-side.
            if task_spec is not None:
                source = task_spec.source
                resources, cwd = task_spec.resources, task_spec.cwd
                agent_timeout, problem_path = task_spec.agent_timeout, task_spec.problem_path
            else:
                td = await client.retrieve_task(task_id=tid, dataset=dataset)
                source, resources, cwd = td.source, td.resources, td.cwd
                agent_timeout, problem_path = td.agent_timeout, td.problem_path
            if source_override is not None:
                source = source_override
            else:
                source = _require_image_source(source)
            req = SandboxCreateRequest(
                source=source,
                resources=resources,
                name=f"{run_id}-{tid}",
                labels={},
                env_vars=secret_env,
                auto_stop_interval=SANDBOX_AUTO_STOP_INTERVAL,
                create_timeout=SANDBOX_CREATE_TIMEOUT,
            )
            sandbox = await provider.create_sandbox(req)
            try:
                if task_spec is not None:
                    await sandbox.upload_file(problem_path, task_spec.question.encode())
                else:
                    await client.setup_task(task_id=tid, instance_id=sandbox.id, dataset=dataset)
                gen = await backend.generate(
                    sandbox=sandbox,
                    contract=contract,
                    task_id=tid,
                    model=model,
                    problem_path=problem_path,
                    cwd=cwd,
                    agent_timeout=agent_timeout,
                    log_dir=artifacts.agent_logs_dir(tid),
                )
            finally:
                try:
                    await provider.delete_sandbox(sandbox.id)
                except Exception as del_exc:
                    logger.warning("failed to delete sandbox %s: %s", sandbox.id, del_exc)
        except Exception as exc:
            logger.warning("task %s generation failed: %s", tid, _format_exc(exc))
            gen = GenerationResult(
                task_id=tid,
                status=GenerationStatus.ERROR,
                data="",
                model=model,
                error=_format_exc(exc),
            )

        artifacts.save_generation(tid, gen)

    async def _process_eval(tid: str) -> None:
        if not is_eval_redoable(artifacts, tid):
            return

        gen = artifacts.load_generation(tid)
        if gen is None:
            ev = EvalResult(
                task_id=tid,
                status=EvalStatus.GENERATION_ERROR,
                error="generation result missing",
            )
        elif gen.status in (GenerationStatus.MAX_TIME, GenerationStatus.MAX_TURNS):
            ev = EvalResult(task_id=tid, status=EvalStatus.DID_NOT_COMPLETE)
        elif gen.status != GenerationStatus.SUCCESS:
            ev = EvalResult(
                task_id=tid,
                status=EvalStatus.GENERATION_ERROR,
                error=gen.error,
            )
        else:
            try:
                raw = await client.evaluate_response(
                    task_id=tid,
                    response=gen.data,
                    dataset=dataset,
                )
                data = EvalResultData.model_validate(raw) if raw is not None else None
                ev = EvalResult(task_id=tid, status=EvalStatus.EVALUATED, result=data)
            except Exception as exc:
                ev = EvalResult(task_id=tid, status=EvalStatus.ERROR, error=_format_exc(exc))

        artifacts.save_eval(tid, ev)

    results = await asyncio.gather(*(_run_task(tid) for tid in task_ids), return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            logger.error("unexpected task exception (should have been caught): %s", r)

    # Final score: build submitted dict, fill missing as None
    submitted: dict[str, Any] = {}
    missing = 0
    gen_errors = 0
    eval_errors = 0
    for tid in config_task_ids:
        ev = artifacts.load_eval(tid)
        submitted[tid] = ev.model_dump(mode="json") if ev is not None else None
        if ev is None:
            missing += 1
        elif ev.status == EvalStatus.GENERATION_ERROR:
            gen_errors += 1
        elif ev.status == EvalStatus.ERROR:
            eval_errors += 1
    # Complete iff no missing evals and no generation/eval errors; DID_NOT_COMPLETE
    # (timed-out tasks) does not break completeness.
    complete = missing == 0 and gen_errors == 0 and eval_errors == 0

    resp = await client.final_score(submitted, dataset=dataset)
    payload = resp.model_dump()
    score = ScoreResult(
        tasks_evaluated=payload.get("tasks_evaluated", []),
        final_score=float(payload.get("final_score", 0.0)),
        metadata=payload.get("metadata", {}),
        complete=complete,
    )
    artifacts.save_final_score(score)


__all__ = ["SandboxTaskSpec", "run_benchmark"]
