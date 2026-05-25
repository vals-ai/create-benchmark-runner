# create-benchmark-runner

Shared runner framework and scaffolder for Vals benchmarks. Mirrors `create-benchmark-service` in shape: one repo containing a library package (`benchmark_runner`), a scaffolder CLI (`create-benchmark-runner`), and Jinja templates for a generated `<name>-runner` repo.

## Quick start

```bash
uv tool install git+ssh://git@github.com/vals-ai/create-benchmark-runner.git@main
create-benchmark-runner my-bench
cd my-bench-runner
# Edit runner/benchmark.py: implement load_tasks() and generate()
# Drop your dataset into data/
make install
make docker-build
```

## Repo structure

- `src/benchmark_runner/` — the runtime library
  - `schemas.py` — `Task`, `GenerationResult`, `EvalResult`, `EvalResultData`, `ScoreResult`, status enums
  - `base.py` — `BenchmarkRunner` ABC with default `evaluate()` and `score()`
  - `cli.py` — `make_cli(adapter_cls)` factory returning a Click group
  - `artifacts.py` — on-disk results layout helper (`RunArtifacts`)
  - `checkpoint.py` — run config + resume detection
  - `client.py` — env-driven service client builder
  - `llm.py` — `LLMConfig` assembly from CLI kwargs
  - `scaffolder/` — `create-benchmark-runner` scaffolder CLI (`main.py`) and template renderer (`generator.py`)
  - `templates/` — Jinja templates for generated runner repos
- `tests/` — framework tests

## Implementing a benchmark runner

Authors typically only write `runner/benchmark.py`:

```python
from benchmark_runner import BenchmarkRunner, GenerationResult, Task

class MyBenchRunner(BenchmarkRunner):
    NAME = "my-bench"
    PAYLOAD_TYPE = "text"
    PAYLOAD_SCHEMA_VERSION = 1
    GENERATION_VERSION_ENV = "MY_BENCH_GENERATION_VERSION"

    def load_tasks(self, dataset_file):
        # Read dataset and return Task objects. The framework registers them.
        ...

    async def generate(self, task, model, llm_config=None, log_dir=None):
        # Run your agent. Return a GenerationResult.
        ...
```

The framework's defaults handle `evaluate()` and `score()` for text-response benchmarks against the legacy `/evaluate-response/` and `/final-score/` endpoints. Override them only if your benchmark needs special pre/post-processing.

For per-task fields beyond `(id, question, timeout)` (system prompt override, docker image, problem path in a sandbox), subclass `Task`:

```python
from benchmark_runner import Task

class MyTask(Task):
    docker_image: str
    cwd: str
```

The framework only ever touches the base `Task` fields, so subclass-specific data flows freely through `load_tasks` → `generate`.

## Service-loaded datasets

By default a runner reads tasks from its bundled JSON file (the `default_dataset_file` argument to `make_cli`). When the benchmark service supports the `/v1/datasets/{name}/tasks` endpoint and the deploy has overridden `BenchmarkService.list_tasks`, runners can fetch the task list at runtime instead. The same tenant/dataset allowlist that gates `/v1/evaluate` and `/v1/score` gates the dataset list, so granting a customer access to a sample is a one-line YAML change in `benchmark-services-registry/allowlist.yaml` rather than a custom image build.

```bash
<benchmark>-runner run \
  --model M --run-id R \
  --service-url https://<svc>.benchmarks.vals.ai \
  --dataset-name validation
```

Auth: Descope only — the runner forwards `VALS_AUTH_KEY` as `x-descope-api-key`. Legacy bearer auth (`BENCHMARK_API_KEY`) is rejected by `/v1/*` with 403, so service-loading requires the deploy to have Descope configured.

`--dataset-name` and `--dataset-file` are mutually exclusive. The existing `--problem <file>` Valkyrie path is unaffected (it never touches the dataset API). If the benchmark service hasn't implemented `list_tasks`, the runner gets a 501 from the endpoint and the run fails with a clear error.

If service-loaded tasks expose benchmark-specific fields, set `TASK_MODEL` on the runner so the framework validates those fields after fetching them:

```python
class SWEBenchTask(Task):
    repo: str
    base_commit: str

class SWEBenchRunner(BenchmarkRunner):
    TASK_MODEL = SWEBenchTask
```

## Development

```bash
make install
make test
make lint
make typecheck
```
