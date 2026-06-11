"""Click CLI for the sandbox orchestrator."""

import asyncio
import os
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

from benchmark_service.client import BenchmarkServiceClient
from benchmark_service.sandbox.types import ImageSource

from benchmark_runner.client import auth_headers
from benchmark_runner.sandbox.contract import AgentContract
from benchmark_runner.sandbox.manifest import Manifest, generate_manifest
from benchmark_runner.sandbox.orchestrator import SandboxTaskSpec, run_benchmark
from benchmark_runner.sandbox.store import (
    install_manifest,
    list_installed,
    load_installed,
    load_manifest_file,
    pin_diff,
)


@click.group()
def cli() -> None:
    load_dotenv(Path(".env"), override=True)


def _contract_from_manifest(mf: Manifest) -> AgentContract:
    return AgentContract(
        name=mf.benchmark,
        run_cmd=mf.agent.run_cmd,
        install_cmd=mf.agent.install_cmd,
        final_output=mf.agent.final_output,
        # The manifest publishes lab-facing required_env NAMES only.
        # AgentContract.secrets is a name→reference map but the orchestrator reads
        # only the keys, so map each name to itself; the lab supplies the values
        # via its own environment. BYO model-endpoint vars are forwarded
        # unconditionally by the orchestrator and are deliberately not listed.
        secrets={name: name for name in mf.agent.required_env},
    )


def _task_specs_from_manifest(mf: Manifest) -> dict[str, SandboxTaskSpec]:
    return {
        task.id: SandboxTaskSpec(
            source=ImageSource(image=task.image),
            resources=task.resources,
            cwd=task.cwd,
            agent_timeout=task.timeout,
            question=task.question,
            problem_path=task.problem_path,
        )
        for task in mf.tasks
    }


@cli.command()
@click.option("--model", required=True, help="Model identifier")
@click.option("--run-id", required=True, help="Unique run identifier")
@click.option("--contract", default=None, help="Path to the agent contract.yaml (direct mode). Omit to run an installed benchmark by name (manifest mode).")
@click.option("--dataset", default=None, help="Dataset name (manifest mode: overrides the manifest's dataset)")
@click.option("--results-dir", default="results", help="Results output directory")
@click.option("--service-url", default=None, help="Benchmark service URL (direct mode: falls back to $SERVICE_URL; manifest mode: overrides the manifest's service.url)")
@click.option("--parallelism", default=10, type=int, help="Number of concurrent tasks")
@click.option("--image", default=None, help="Override: boot every sandbox from this registry image (e.g. to validate a candidate agent build) instead of the source retrieve_task returns. Eval still uses the service.")
@click.option("--eval-timeout", default=1800, type=int, help="HTTP timeout (s) for service calls incl. the eval judge. Default 1800; the rubric judge can take minutes, and the 60s client default times out (httpx.ReadTimeout) on slow tasks.")
@click.argument("args", nargs=-1)
def run(
    model: str,
    run_id: str,
    contract: str | None,
    dataset: str | None,
    results_dir: str,
    service_url: str | None,
    parallelism: int,
    image: str | None,
    eval_timeout: int,
    args: tuple[str, ...],
) -> None:
    """Run the sandbox orchestrator.

    Direct mode (--contract given): ARGS are task ids; at least one is required.

    Manifest mode (no --contract): the first ARG is an installed benchmark name
    (see `benchmark add` / `benchmark list`); remaining ARGS are task ids, and
    none means every task in the manifest. Service URL, dataset, and contract
    come from the manifest; results nest under <results-dir>/<benchmark>.
    """
    source_override = ImageSource(image=image) if image else None

    agent_contract: AgentContract | None = None
    contract_path: Path | None = None
    task_specs: dict[str, SandboxTaskSpec] | None = None
    if contract is not None:
        # Direct mode: every positional is a task id (unchanged behavior).
        if not args:
            raise click.UsageError("At least one TASK_ID is required.")
        task_ids = list(args)
        contract_path = Path(contract)
        # Resolve after the group callback's load_dotenv so a .env-only SERVICE_URL is honored.
        service_url = service_url or os.environ.get("SERVICE_URL", "")
    else:
        # Manifest mode: first positional is the installed benchmark name.
        if not args:
            raise click.UsageError("BENCHMARK name is required (or pass --contract for direct mode).")
        benchmark_name, *manifest_task_ids = args
        mf = load_installed(benchmark_name)
        if mf is None:
            installed = ", ".join(m.benchmark for m in list_installed()) or "none"
            raise click.ClickException(
                f"benchmark '{benchmark_name}' is not installed (installed: {installed}); "
                "install it with `benchmark add <manifest.yaml>`"
            )
        task_ids = manifest_task_ids or [t.id for t in mf.tasks]
        agent_contract = _contract_from_manifest(mf)
        task_specs = _task_specs_from_manifest(mf)
        service_url = service_url or mf.service.url
        dataset = dataset or mf.dataset.name
        results_dir = str(Path(results_dir) / mf.benchmark)

    headers = auth_headers()

    daytona_api_key = os.environ.get("DAYTONA_API_KEY")
    daytona_api_url = os.environ.get("DAYTONA_API_URL")
    daytona_target = os.environ.get("DAYTONA_TARGET")
    if daytona_api_key:
        headers["x-api-key"] = daytona_api_key
    if daytona_api_url:
        headers["x-api-url"] = daytona_api_url
    if daytona_target:
        headers["x-target"] = daytona_target
    headers["x-sandbox-provider"] = "daytona"

    client = BenchmarkServiceClient(service_url, headers=headers, timeout=eval_timeout)

    asyncio.run(
        run_benchmark(
            run_id=run_id,
            model=model,
            task_ids=task_ids,
            dataset=dataset,
            results_dir=results_dir,
            contract_path=contract_path,
            contract=agent_contract,
            client=client,
            parallelism=parallelism,
            source_override=source_override,
            task_specs=task_specs,
        )
    )


@cli.command()
@click.option("--service-url", default=None, help="Benchmark service URL (falls back to $SERVICE_URL)")
@click.option("--dataset", required=True, help="Dataset name")
@click.option("--contract", required=True, help="Path to the agent contract.yaml")
@click.option("--benchmark", required=True, help="Benchmark identifier")
@click.option(
    "--required-env",
    multiple=True,
    help="Lab-facing env var required by the agent. Repeat for multiple values.",
)
@click.option("--output", required=True, help="Output path for the manifest YAML")
def manifest(
    service_url: str | None,
    dataset: str,
    contract: str,
    benchmark: str,
    required_env: tuple[str, ...],
    output: str,
) -> None:
    """Generate a self-contained benchmark manifest for lab-hosted consumers."""
    service_url = service_url or os.environ.get("SERVICE_URL", "")
    headers = auth_headers()
    client = BenchmarkServiceClient(service_url, headers=headers)

    try:
        mf = asyncio.run(
            generate_manifest(
                client=client,
                service_url=service_url,
                dataset=dataset,
                contract_path=Path(contract),
                benchmark=benchmark,
                required_env=list(required_env),
            )
        )
    except ValueError as exc:
        # A non-pullable source (Daytona snapshot) or an empty dataset cannot
        # produce a valid lab-hosted manifest; surface it cleanly (exit non-zero).
        raise click.ClickException(str(exc)) from exc

    output_path = Path(output)
    output_path.write_text(yaml.safe_dump(mf.model_dump(), default_flow_style=False, sort_keys=False))

    click.echo(f"Manifest written to {output_path} ({len(mf.tasks)} tasks, benchmark={benchmark})")


def _short_image_ref(image: str) -> str:
    """Compact display ref: digest-pinned images show only the first 12 digest chars."""
    repo, sep, digest = image.partition("@sha256:")
    if sep:
        return f"{repo}@sha256:{digest[:12]}"
    return image


def _image_summary(mf: Manifest) -> str:
    """One short ref when every task shares an image, else the distinct-image count."""
    images = {task.image for task in mf.tasks}
    if len(images) == 1:
        return _short_image_ref(images.pop())
    return f"per-task ({len(images)} images)"


@cli.command()
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def add(manifest_path: Path) -> None:
    """Install a benchmark manifest into the project-local ./benchmarks store."""
    try:
        mf = load_manifest_file(manifest_path)
    except Exception as exc:
        raise click.ClickException(f"invalid manifest {manifest_path}: {exc}") from exc

    # The pin diff needs the previously-installed copy, but an unreadable one
    # (e.g. written by an older manifest schema) must not block reinstalling —
    # a lab upgrading across a schema change hits exactly this.
    try:
        existing = load_installed(mf.benchmark)
    except Exception:
        click.echo(
            f"Warning: installed manifest for '{mf.benchmark}' is unreadable "
            "(older format?); replacing without pin diff",
            err=True,
        )
        existing = None
    if existing is not None:
        changes = pin_diff(existing, mf)
        if changes:
            click.echo(f"Replacing installed '{mf.benchmark}' with pin changes:")
            for line in changes:
                click.echo(f"  {line}")
        else:
            click.echo(f"Replacing installed '{mf.benchmark}': no pin changes")

    installed_path = install_manifest(mf)
    click.echo(f"Installed {mf.benchmark} -> {installed_path}")
    click.echo(f"  dataset: {mf.dataset.name} ({len(mf.tasks)} tasks)")
    click.echo(f"  image: {_image_summary(mf)}")
    click.echo(f"  service version: {mf.service.service_version}, framework version: {mf.service.framework_version}")


@cli.command(name="list")
def list_cmd() -> None:
    """List installed benchmark manifests."""
    manifests = list_installed()
    if not manifests:
        click.echo("No benchmarks installed; install one with `benchmark add <manifest.yaml>`.")
        return
    for mf in manifests:
        click.echo(
            f"{mf.benchmark}  dataset={mf.dataset.name} ({len(mf.tasks)} tasks)  "
            f"image={_image_summary(mf)}  service={mf.service.service_version}"
        )


__all__ = ["cli"]
