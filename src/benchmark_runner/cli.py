"""Click CLI factory for benchmark runners."""

import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from benchmark_service.client import BenchmarkServiceError

from benchmark_runner.artifacts import RunArtifacts
from benchmark_runner.base import BenchmarkRunner
from benchmark_runner.checkpoint import (
    is_eval_redoable,
    is_generation_redoable,
    load_or_create_run_config,
)
from benchmark_runner.llm import build_llm_config
from benchmark_runner.schemas import EvalStatus, GenerationResult, GenerationStatus, Task


def _runner_framework_version() -> str:
    try:
        return importlib.metadata.version("create-benchmark-runner")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def make_cli(
    runner_cls: type[BenchmarkRunner],
    *,
    default_dataset_file: str | None = None,
    default_results_dir: str = "results",
    default_parallelism: int = 10,
    default_timeout: int = 1800,
    extra_run_options: list[Any] | None = None,
) -> click.Group:
    @click.group()
    def cli():
        load_dotenv(Path(".env"), override=True)

    @cli.command()
    @click.option("--model", required=True, help="Model identifier")
    @click.option("--run-id", required=True, help="Unique run identifier")
    @click.argument("task_ids", nargs=-1)
    @click.option("--skip-eval", is_flag=True, help="Generate only, skip evaluation")
    @click.option("--problem", "problem_path", default=None,
                  help="Path to a problem-statement file")
    @click.option("--dataset-file", default=None)
    @click.option("--dataset-name", default=None,
                  help="Fetch task list from the service via GET /v1/datasets/{name}/tasks. "
                       "Mutually exclusive with --dataset-file.")
    @click.option("--results-dir", default=default_results_dir)
    @click.option("--service-url", default=None,
                  help="Override SERVICE_URL env for this invocation")
    @click.option("--parallelism", default=default_parallelism, type=int)
    @click.option("--timeout", "task_timeout", default=default_timeout, type=float, show_default=True,
                  help="Default per-task generation timeout in seconds when a task has no timeout")
    @click.option("--max-tokens", type=int, default=None)
    @click.option("--temperature", type=float, default=None)
    @click.option("--top-p", type=float, default=None)
    @click.option("--top-k", type=int, default=None)
    @click.option("--reasoning-effort", type=str, default=None)
    @click.option("--custom-endpoint", type=str, default=None)
    @click.option("--custom-api-key", type=str, default=None)
    @click.option("--chat-completions", is_flag=True, default=False,
                  help="Use OpenAI-compatible chat completions API instead of native provider API")
    @click.option("--disable-streaming", is_flag=True, default=False,
                  help="Disable streaming for chat completions. Requires --chat-completions.")
    def run(
        model: str, run_id: str, task_ids: tuple[str, ...],
        skip_eval: bool, problem_path: str | None,
        dataset_file: str | None, dataset_name: str | None,
        results_dir: str, service_url: str | None,
        parallelism: int, task_timeout: float | None,
        max_tokens: int | None, temperature: float | None,
        top_p: float | None, top_k: int | None, reasoning_effort: str | None,
        custom_endpoint: str | None, custom_api_key: str | None,
        chat_completions: bool, disable_streaming: bool,
    ):
        """Generate and evaluate tasks."""
        if disable_streaming and not chat_completions:
            raise click.UsageError("--disable-streaming requires --chat-completions")
        if problem_path and len(task_ids) != 1:
            raise click.UsageError("--problem requires exactly one TASK_ID")
        if dataset_name and dataset_file is not None:
            raise click.UsageError("--dataset-name and --dataset-file are mutually exclusive")
        if problem_path and dataset_name:
            raise click.UsageError("--problem and --dataset-name are mutually exclusive")

        custom_endpoint = custom_endpoint or os.environ.get("CUSTOM_ENDPOINT")
        custom_api_key = custom_api_key or os.environ.get("CUSTOM_API_KEY")
        service_url_resolved = service_url or os.environ.get("SERVICE_URL", "")

        llm_config = build_llm_config(
            max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, reasoning_effort=reasoning_effort,
            custom_endpoint=custom_endpoint, custom_api_key=custom_api_key,
            chat_completions=chat_completions, disable_streaming=disable_streaming,
        )

        asyncio.run(_run_impl(
            runner_cls=runner_cls,
            model=model, run_id=run_id, task_ids=list(task_ids),
            skip_eval=skip_eval, problem_path=problem_path,
            dataset_file=dataset_file, dataset_name=dataset_name,
            default_dataset_file=default_dataset_file,
            results_dir=results_dir,
            service_url=service_url_resolved, parallelism=parallelism,
            default_timeout=task_timeout,
            llm_config=llm_config,
        ))

    if extra_run_options:
        for opt in extra_run_options:
            run = opt(run)  # type: ignore[assignment]

    _add_score_command(cli, runner_cls, default_results_dir)

    return cli


async def _run_impl(
    *,
    runner_cls: type[BenchmarkRunner],
    model: str,
    run_id: str,
    task_ids: list[str],
    skip_eval: bool,
    problem_path: str | None,
    dataset_file: str | None,
    dataset_name: str | None,
    default_dataset_file: str | None,
    results_dir: str,
    service_url: str,
    parallelism: int,
    default_timeout: float | None,
    llm_config: Any,
) -> None:
    artifacts = RunArtifacts(results_dir=results_dir, run_id=run_id)
    runner = runner_cls(service_url=service_url)

    # If resuming, restore the task source frozen in run_config (service dataset
    # name or file path) so it isn't lost when the flag isn't re-passed.
    existing_config = artifacts.load_run_config()
    if existing_config and not problem_path and not dataset_name:
        saved_task_source = existing_config.get("task_source")
        if saved_task_source == "service":
            resumed_dataset_name = existing_config.get("dataset_name")
            if not resumed_dataset_name:
                raise click.ClickException("run_config.json has task_source=service but no dataset_name")
            if dataset_file is not None:
                click.echo(
                    f"Warning: --dataset-file ignored; resuming service-backed run "
                    f"with dataset '{resumed_dataset_name}' from run_config.json",
                    err=True,
                )
            dataset_name = resumed_dataset_name
        elif saved_task_source == "file" and dataset_file is None:
            # Restore the dataset file frozen in run_config so a resumed
            # file-backed run reloads the same tasks instead of silently
            # falling back to default_dataset_file.
            dataset_file = existing_config.get("dataset_file")

    # Resolve the bundled-file default only after any resume restoration, so an
    # un-passed --dataset-file on a fresh run still uses the adapter default.
    if dataset_file is None and not problem_path and not dataset_name:
        dataset_file = default_dataset_file

    if problem_path:
        assert len(task_ids) == 1
        question = Path(problem_path).read_text(encoding="utf-8").strip()
        runner.add_task(Task(id=task_ids[0], question=question))
        task_source = "problem"
    elif dataset_name:
        try:
            tasks = await runner.load_tasks_from_service(dataset_name)
        except BenchmarkServiceError as exc:
            raise click.ClickException(
                f"Failed to load dataset '{dataset_name}' from {service_url}: {exc}"
            ) from exc
        runner._register_tasks(tasks)
        task_source = "service"
    else:
        runner._register_tasks(runner.load_tasks(dataset_file))
        task_source = "file"

    all_task_ids = [t.id for t in runner.get_tasks()]
    config, was_resumed = load_or_create_run_config(
        artifacts=artifacts,
        model=model,
        task_ids=all_task_ids if not problem_path else task_ids,
        dataset_file=dataset_file if not dataset_name else None,
        dataset_name=dataset_name or runner._dataset,
        task_source=task_source,
        payload_schema=runner.payload_schema,
        payload_type=runner.PAYLOAD_TYPE,
        runner_version=_runner_framework_version(),
        generation_version=runner.generation_version,
    )
    click.echo(f"{'Resuming' if was_resumed else 'Starting'} {run_id}: {len(config['tasks'])} tasks")

    config_task_ids: list[str] = config["tasks"]
    tasks_by_id = {t.id: t for t in runner.get_tasks()}

    process_ids = config_task_ids
    if task_ids and not problem_path:
        invalid = sorted(set(task_ids) - set(config_task_ids))
        if invalid:
            raise click.ClickException(f"Unknown task IDs: {invalid}")
        process_ids = list(task_ids)
        click.echo(f"Processing {len(process_ids)}/{len(config_task_ids)} tasks")

    gen_sem = asyncio.Semaphore(parallelism)
    pbar = tqdm(total=len(process_ids), desc="Running")

    async def process(tid: str) -> None:
        task = tasks_by_id[tid]
        if is_generation_redoable(artifacts, tid):
            async with gen_sem:
                timeout = task.timeout if task.timeout is not None else default_timeout
                generate_coro = runner.generate(
                    task=task,
                    model=model,
                    llm_config=llm_config,
                    log_dir=artifacts.agent_logs_dir(tid),
                )
                try:
                    if timeout is not None and timeout > 0:
                        gen = await asyncio.wait_for(generate_coro, timeout=timeout)
                    else:
                        gen = await generate_coro
                except TimeoutError:
                    gen = GenerationResult(
                        task_id=tid,
                        status=GenerationStatus.MAX_TIME,
                        data="",
                        question=task.question,
                        model=model,
                        error=f"generation timed out after {timeout:g} seconds",
                        generation_version=runner.generation_version,
                    )
                if gen.generation_version is None:
                    gen.generation_version = runner.generation_version
                artifacts.save_generation(tid, gen)
        else:
            gen = artifacts.load_generation(tid)
            assert gen is not None

        if not skip_eval and is_eval_redoable(artifacts, tid):
            ev = await runner.evaluate(tid, gen)
            artifacts.save_eval(tid, ev)

        pbar.set_postfix_str(tid)
        pbar.update(1)

    await asyncio.gather(*[process(t) for t in process_ids])
    pbar.close()

    failed = [
        tid for tid in process_ids
        if (g := artifacts.load_generation(tid)) and g.status == GenerationStatus.ERROR
    ]

    if skip_eval:
        click.echo("Eval skipped (--skip-eval)")
    else:
        evals = {tid: artifacts.load_eval(tid) for tid in process_ids}
        evaluated = sum(1 for ev in evals.values() if ev is not None and ev.status == EvalStatus.EVALUATED)
        eval_errors = [tid for tid, ev in evals.items() if ev is not None and ev.status == EvalStatus.ERROR]
        missing_eval = [tid for tid, ev in evals.items() if ev is None]

        parts = [f"{evaluated}/{len(process_ids)} evaluated"]
        if eval_errors:
            parts.append(f"{len(eval_errors)} eval errors")
        if missing_eval:
            parts.append(f"{len(missing_eval)} missing evals")
        click.echo(f"Done: {', '.join(parts)}.")

        if eval_errors:
            click.echo(f"Evaluation failed for: {', '.join(eval_errors)}", err=True)
        if missing_eval:
            click.echo(f"Evaluation missing for: {', '.join(missing_eval)}", err=True)

        if eval_errors or missing_eval:
            raise SystemExit(1)

        if not failed:
            try:
                await _score_impl(
                    runner_cls=runner_cls,
                    run_id=run_id,
                    results_dir=results_dir,
                    service_url=service_url,
                    force=False,
                    dataset_file=dataset_file if not dataset_name else None,
                )
            except click.ClickException:
                raise
            except Exception as e:
                raise click.ClickException(f"Auto-score failed: {e}") from e

    if failed:
        click.echo(f"Generation failed for: {', '.join(failed)}", err=True)
        raise SystemExit(1)


def _add_score_command(cli_group: click.Group, runner_cls: type[BenchmarkRunner],
                       default_results_dir: str) -> None:
    @cli_group.command()
    @click.option("--run-id", required=True)
    @click.option("--results-dir", default=default_results_dir)
    @click.option("--service-url", default=None,
                  help="Override SERVICE_URL env for this invocation")
    @click.option("--dataset-file", default=None,
                  help="Optional override for dataset file used during scoring")
    @click.option("--force", is_flag=True,
                  help="Re-score past the cache and proceed on incomplete runs (missing tasks = 0).")
    def score(run_id: str, results_dir: str, service_url: str | None,
              dataset_file: str | None, force: bool):
        """Aggregate evaluated tasks into a final score."""
        service_url_resolved = service_url or os.environ.get("SERVICE_URL", "")
        asyncio.run(_score_impl(
            runner_cls=runner_cls,
            run_id=run_id,
            results_dir=results_dir,
            service_url=service_url_resolved,
            force=force,
            dataset_file=dataset_file,
        ))


async def _score_impl(
    *,
    runner_cls: type[BenchmarkRunner],
    run_id: str,
    results_dir: str,
    service_url: str,
    force: bool,
    dataset_file: str | None,
) -> None:
    artifacts = RunArtifacts(results_dir=results_dir, run_id=run_id)
    config = artifacts.load_run_config()
    if config is None:
        raise click.ClickException(f"No run_config.json for run {run_id} in {results_dir}")
    task_ids: list[str] = config["tasks"]
    dataset_name: str | None = config.get("dataset_name")

    runner = runner_cls(service_url=service_url, dataset_name=dataset_name)
    if dataset_file is not None:
        runner.load_tasks(dataset_file)
    runner._register_tasks([Task(id=tid, question="") for tid in task_ids])

    eval_results = []
    missing = 0
    gen_errors = 0
    eval_errors = 0
    timed_out = 0
    evaluated = 0
    for tid in task_ids:
        ev = artifacts.load_eval(tid)
        if ev is None:
            missing += 1
            continue
        eval_results.append(ev)
        if ev.status == EvalStatus.EVALUATED:
            evaluated += 1
        elif ev.status == EvalStatus.DID_NOT_COMPLETE:
            timed_out += 1
        elif ev.status == EvalStatus.GENERATION_ERROR:
            gen_errors += 1
        elif ev.status == EvalStatus.ERROR:
            eval_errors += 1

    parts = [f"{evaluated}/{len(task_ids)} evaluated"]
    if timed_out:
        parts.append(f"{timed_out} timed out")
    if gen_errors:
        parts.append(f"{gen_errors} generation errors")
    if eval_errors:
        parts.append(f"{eval_errors} eval errors")
    if missing:
        parts.append(f"{missing} missing")
    click.echo(f"Scoring: {', '.join(parts)}")

    complete = missing == 0 and gen_errors == 0 and eval_errors == 0

    if not force:
        cached = artifacts.load_final_score()
        if cached is not None and cached.complete and complete:
            click.echo(f"Final score (cached): {cached.final_score}")
            return
        if not complete:
            click.echo("Run is incomplete. Use --force to score anyway (missing tasks scored as 0).")
            return

    sr = await runner.score(eval_results)
    sr.complete = complete
    artifacts.save_final_score(sr)
    click.echo(f"Final score: {sr.final_score}")
    if sr.metadata:
        click.echo(f"Metadata: {json.dumps(sr.metadata, indent=2, default=str)}")


__all__ = ["make_cli"]
