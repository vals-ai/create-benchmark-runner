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

## Design

See `docs/superpowers/specs/2026-05-20-runner-framework-design.md` (in the parent workspace) for the full design rationale.

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

## Development

```bash
make install
make test
make lint
make typecheck
```
