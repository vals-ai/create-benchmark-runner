"""Click CLI for the sandbox orchestrator."""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

from benchmark_service.client import BenchmarkServiceClient
from benchmark_service.sandbox.types import ImageSource

from benchmark_runner.client import auth_headers
from benchmark_runner.sandbox.bundle import (
    AgentBundle,
    build_bundle_zip,
    file_sha256,
    load_bundle,
    zip_root,
)
from benchmark_runner.sandbox.contract import AgentContract
from benchmark_runner.sandbox.manifest import BundleSpec, Manifest, generate_manifest
from benchmark_runner.sandbox.orchestrator import (
    SandboxTaskSpec,
    evaluate_run,
    run_benchmark,
    score_run,
)
from benchmark_runner.sandbox.store import (
    install_manifest,
    installed_bundle_path,
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


def _load_bundle_arg(path: Path) -> AgentBundle:
    """--bundle accepts a prebuilt zip or an agent directory (zipped on the fly)."""
    if path.is_dir():
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / f"{path.name}.zip"
            build_bundle_zip(path, zip_path)
            return load_bundle(zip_path)
    return load_bundle(path)


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
@click.option("--skip-eval", is_flag=True, help="Generation only: skip per-task evaluation and the final score. Evaluate later with `benchmark eval`, then `benchmark score` — lets generation be sliced across invocations into one shared results/<run_id>/.")
@click.option("--bundle", "bundle_arg", default=None, type=click.Path(exists=True, path_type=Path), help="Agent bundle (zip, or a directory zipped on the fly) installed into each sandbox at /bundle/<name>. Overrides the manifest's pinned bundle — e.g. to run a custom agent against pinned tasks.")
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
    skip_eval: bool,
    bundle_arg: Path | None,
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
    mf: Manifest | None = None
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

    # Resolve the agent bundle before booting anything: an explicit --bundle wins,
    # else the manifest's pin (digest-verified against the installed copy).
    bundle: AgentBundle | None = None
    try:
        if bundle_arg is not None:
            bundle = _load_bundle_arg(bundle_arg)
        elif mf is not None and mf.agent.bundle is not None:
            bundle = load_bundle(
                installed_bundle_path(mf.benchmark, mf.agent.bundle.sha256),
                expected_sha256=mf.agent.bundle.sha256,
            )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

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
            skip_eval=skip_eval,
            bundle=bundle,
        )
    )


def _resolve_results_target(
    args: tuple[str, ...],
    service_url: str | None,
    dataset: str | None,
    results_dir: str,
    *,
    direct: bool = False,
) -> tuple[str, str | None, str, list[str], Manifest | None]:
    """Resolve eval/score target from either an installed benchmark or direct task ids."""
    if direct or not args:
        return (
            service_url or os.environ.get("SERVICE_URL", ""),
            dataset,
            results_dir,
            list(args),
            None,
        )
    mf = load_installed(args[0])
    if mf is None:
        installed = ", ".join(m.benchmark for m in list_installed()) or "none"
        raise click.ClickException(
            f"benchmark '{args[0]}' is not installed (installed: {installed}); "
            "install it with `benchmark add <manifest.yaml>` or pass --direct for task ids"
        )
    return (
        service_url or mf.service.url,
        dataset or mf.dataset.name,
        str(Path(results_dir) / mf.benchmark),
        list(args[1:]),
        mf,
    )


@cli.command(name="eval")
@click.option("--run-id", required=True, help="Run identifier to evaluate")
@click.option("--dataset", default=None, help="Dataset name (defaults to the manifest's, then the run config's)")
@click.option("--results-dir", default="results", help="Results root directory")
@click.option("--service-url", default=None, help="Benchmark service URL (direct mode: falls back to $SERVICE_URL; manifest mode: overrides the manifest's service.url)")
@click.option("--parallelism", default=10, type=int, help="Number of concurrent evaluation calls")
@click.option("--eval-timeout", default=1800, type=int, help="HTTP timeout (s) for the eval judge; see `run --eval-timeout`.")
@click.option("--direct", is_flag=True, help="Treat ARGS as direct task ids instead of an installed benchmark name")
@click.argument("args", nargs=-1)
def eval_cmd(
    run_id: str,
    dataset: str | None,
    results_dir: str,
    service_url: str | None,
    parallelism: int,
    eval_timeout: int,
    direct: bool,
    args: tuple[str, ...],
) -> None:
    """Evaluate a run's existing generations, without re-running generation.

    Manifest mode: the first ARG is an installed benchmark name; remaining ARGS
    are task ids. Direct mode: pass --direct to treat ARGS as task ids. With no
    task ids, every task with a generation.json under results/<run_id>/ is
    evaluated. Tasks that already evaluated cleanly are skipped (resume).
    """
    service_url, dataset, results_dir, task_ids, _ = _resolve_results_target(
        args, service_url, dataset, results_dir, direct=direct
    )
    client = BenchmarkServiceClient(service_url, headers=auth_headers(), timeout=eval_timeout)
    try:
        asyncio.run(
            evaluate_run(
                run_id=run_id,
                results_dir=results_dir,
                client=client,
                dataset=dataset,
                task_ids=task_ids or None,
                parallelism=parallelism,
            )
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option("--run-id", required=True, help="Run identifier to score")
@click.option("--dataset", default=None, help="Dataset name (defaults to the manifest's, then the run config's)")
@click.option("--results-dir", default="results", help="Results root directory")
@click.option("--service-url", default=None, help="Benchmark service URL (direct mode: falls back to $SERVICE_URL; manifest mode: overrides the manifest's service.url)")
@click.option("--eval-timeout", default=1800, type=int, help="HTTP timeout (s) for the final-score call.")
@click.option("--direct", is_flag=True, help="Treat ARGS as direct task ids instead of an installed benchmark name")
@click.argument("args", nargs=-1)
def score(
    run_id: str,
    dataset: str | None,
    results_dir: str,
    service_url: str | None,
    eval_timeout: int,
    direct: bool,
    args: tuple[str, ...],
) -> None:
    """Final-score a run from its on-disk eval results.

    Manifest mode (first ARG = installed benchmark name) scores over the
    manifest's FULL task list by default — tasks without an eval submit as None
    and score zero, so a partial run cannot inflate its score by omission; pass
    task ids after the name to override. Direct mode defaults to the run
    config's frozen task list; pass --direct to score specific direct task ids.
    """
    service_url, dataset, results_dir, task_ids, mf = _resolve_results_target(
        args, service_url, dataset, results_dir, direct=direct
    )
    if not task_ids and mf is not None:
        task_ids = [t.id for t in mf.tasks]
    client = BenchmarkServiceClient(service_url, headers=auth_headers(), timeout=eval_timeout)
    try:
        result = asyncio.run(
            score_run(
                run_id=run_id,
                results_dir=results_dir,
                client=client,
                dataset=dataset,
                task_ids=task_ids or None,
            )
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"final_score={result.final_score} "
        f"tasks_evaluated={len(result.tasks_evaluated)} complete={result.complete}"
    )


def _package_bundle(source: Path, out_dir: Path) -> BundleSpec:
    """Build (directory) or copy (prebuilt zip) the agent bundle next to the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        zip_path = out_dir / f"{source.name}.zip"
        sha256 = build_bundle_zip(source, zip_path)
    else:
        zip_root(source)  # layout check before accepting a prebuilt zip
        zip_path = out_dir / source.name
        if zip_path.resolve() != source.resolve():
            shutil.copyfile(source, zip_path)
        sha256 = file_sha256(zip_path)
    return BundleSpec(file=zip_path.name, sha256=sha256)


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
@click.option(
    "--agent-bundle",
    "agent_bundle",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Agent directory (zipped with the standard exclusions) or prebuilt bundle zip; "
    "written next to the manifest and pinned by sha256. Omit only for benchmarks whose "
    "task images prebake the agent — without a bundle the contract's install_cmd is dropped.",
)
@click.option("--output", required=True, help="Output path for the manifest YAML")
def manifest(
    service_url: str | None,
    dataset: str,
    contract: str,
    benchmark: str,
    required_env: tuple[str, ...],
    agent_bundle: Path | None,
    output: str,
) -> None:
    """Generate a self-contained benchmark manifest for lab-hosted consumers."""
    service_url = service_url or os.environ.get("SERVICE_URL", "")
    output_path = Path(output)

    # Package the bundle before any network work: a bad agent dir/zip fails here.
    bundle_spec: BundleSpec | None = None
    if agent_bundle is not None:
        try:
            bundle_spec = _package_bundle(agent_bundle, output_path.parent)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

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
                bundle=bundle_spec,
            )
        )
    except ValueError as exc:
        # A non-pullable source (Daytona snapshot) or an empty dataset cannot
        # produce a valid lab-hosted manifest; surface it cleanly (exit non-zero).
        raise click.ClickException(str(exc)) from exc

    output_path.write_text(yaml.safe_dump(mf.model_dump(), default_flow_style=False, sort_keys=False))

    click.echo(f"Manifest written to {output_path} ({len(mf.tasks)} tasks, benchmark={benchmark})")
    if bundle_spec is not None:
        click.echo(f"  agent bundle: {_bundle_summary(bundle_spec.file, bundle_spec.sha256)}")


def _short_image_ref(image: str) -> str:
    """Compact display ref: digest-pinned images show only the first 12 digest chars."""
    repo, sep, digest = image.partition("@sha256:")
    if sep:
        return f"{repo}@sha256:{digest[:12]}"
    return image


def _bundle_summary(file: str, sha256: str) -> str:
    return f"{file} (sha256:{sha256[:12]}…)"


def _image_summary(mf: Manifest) -> str:
    """One short ref when every task shares an image, else the distinct-image count."""
    images = {task.image for task in mf.tasks}
    if len(images) == 1:
        return _short_image_ref(images.pop())
    return f"per-task ({len(images)} images)"


@cli.command()
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def add(manifest_path: Path) -> None:
    """Install a benchmark manifest (and its agent bundle) into the project-local ./benchmarks store."""
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

    bundle_src = (
        manifest_path.parent / mf.agent.bundle.file if mf.agent.bundle is not None else None
    )
    try:
        installed_path = install_manifest(mf, bundle_src=bundle_src)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Installed {mf.benchmark} -> {installed_path}")
    click.echo(f"  dataset: {mf.dataset.name} ({len(mf.tasks)} tasks)")
    click.echo(f"  image: {_image_summary(mf)}")
    if mf.agent.bundle is not None:
        bundle_name = installed_bundle_path(mf.benchmark, mf.agent.bundle.sha256).name
        click.echo(f"  agent bundle: {_bundle_summary(bundle_name, mf.agent.bundle.sha256)}")
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
