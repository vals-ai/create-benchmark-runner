"""Click CLI for the sandbox orchestrator."""

import asyncio
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from benchmark_service.client import BenchmarkServiceClient

from benchmark_runner.sandbox.orchestrator import run_sandbox


@click.group()
def cli() -> None:
    load_dotenv(Path(".env"), override=True)


@cli.command()
@click.option("--model", required=True, help="Model identifier")
@click.option("--run-id", required=True, help="Unique run identifier")
@click.option("--contract", required=True, help="Path to the agent contract.yaml")
@click.option("--dataset", default=None, help="Dataset name")
@click.option("--results-dir", default="results", help="Results output directory")
@click.option("--service-url", default=None, help="Benchmark service URL (falls back to $SERVICE_URL)")
@click.option("--parallelism", default=10, type=int, help="Number of concurrent tasks")
@click.argument("task_ids", nargs=-1)
def run(
    model: str,
    run_id: str,
    contract: str,
    dataset: str | None,
    results_dir: str,
    service_url: str | None,
    parallelism: int,
    task_ids: tuple[str, ...],
) -> None:
    """Run the sandbox orchestrator against one or more tasks."""
    if not task_ids:
        raise click.UsageError("At least one TASK_ID is required.")

    # Resolve after the group callback's load_dotenv so a .env-only SERVICE_URL is honored.
    service_url = service_url or os.environ.get("SERVICE_URL", "")

    headers: dict[str, str] = {}
    descope_key = os.environ.get("VALS_AUTH_KEY")
    bearer_key = os.environ.get("BENCHMARK_API_KEY")
    if descope_key:
        headers["x-descope-api-key"] = descope_key
    elif bearer_key:
        headers["Authorization"] = f"Bearer {bearer_key}"

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

    client = BenchmarkServiceClient(service_url, headers=headers)

    asyncio.run(
        run_sandbox(
            run_id=run_id,
            model=model,
            task_ids=list(task_ids),
            dataset=dataset,
            results_dir=results_dir,
            contract_path=Path(contract),
            client=client,
            parallelism=parallelism,
        )
    )


__all__ = ["cli"]
