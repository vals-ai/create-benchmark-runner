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
  - `sandbox/` — sandbox orchestrator (`benchmark` CLI): runs each task in its own sandbox via a pluggable provider, eval/score through the service
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

## Sandbox orchestrator

`benchmark run` drives the full benchmark loop with one sandbox per task: create a sandbox from a registry image → write the problem statement into it → run the agent per the contract (`install_cmd`, then `run_cmd`) → download `<final_output>/<task_id>/generation.json` → delete the sandbox → eval/score through the service. Only pullable registry images are supported as sandbox sources (no Daytona snapshots).

The sandbox backend is pluggable. The default is Daytona, built by the cbs client from `DAYTONA_API_KEY`/`DAYTONA_API_URL`/`DAYTONA_TARGET`; any other backend plugs in as a provider object.

### Implementing a sandbox provider

A provider implements the `benchmark_service.sandbox` interface: a `SandboxProvider` and the `Sandbox` it returns. This is the whole abstraction (copied from create-benchmark-service `sandbox/types.py`):

```python
class ImageSource(BaseModel):
    type: Literal["image"] = "image"
    image: str            # pullable registry ref, digest-pinned in practice


class Resources(BaseModel):
    vcpu: int             # logical sandbox CPU count
    memory: int           # sandbox memory
    disk: int             # sandbox ephemeral disk


class SandboxCreateRequest(BaseModel):
    source: SandboxSource         # the orchestrator always sends an ImageSource
    resources: Resources
    name: str
    labels: dict[str, str]
    env_vars: dict[str, str]      # must reach the agent process environment
    auto_stop_interval: int       # minutes; backstop if cleanup never runs
    create_timeout: int           # seconds to wait for sandbox readiness


class ExecResult(BaseModel):
    exit_code: int
    output: str = ""


class Sandbox(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def state(self) -> str: ...

    @abstractmethod
    async def exec(
        self, command: str, *, cwd: str | None = None, timeout: float | None = None
    ) -> ExecResult: ...

    @abstractmethod
    def command(
        self, command: str, *, cwd: str | None = None, timeout: float | None = None
    ) -> AsyncGenerator[str, None]: ...

    @abstractmethod
    async def upload_file(self, remote_path: str, content: bytes) -> None: ...

    @abstractmethod
    async def download_file(self, remote_path: str) -> bytes: ...


class SandboxProvider(ABC):
    @abstractmethod
    async def create_sandbox(self, request: SandboxCreateRequest) -> Sandbox: ...

    @abstractmethod
    async def get_sandbox(self, instance_id: str) -> Sandbox: ...

    @abstractmethod
    async def delete_sandbox(self, instance_id: str) -> None: ...

    @abstractmethod
    def list_sandboxes(self, query: SandboxQuery) -> AsyncGenerator[Sandbox, None]: ...

    # plus an optional `async def close()` hook (default no-op) + async-context support
```

Failures are reported through `SandboxError` and its subtypes (`SandboxNotFoundError`, `SandboxCommandError(exit_code)`), also in `benchmark_service.sandbox.types`.

Reference implementation: `src/benchmark_runner/sandbox/local_docker.py` — a Daytona-free provider that runs sandboxes as local Docker containers, in ~300 lines.

The orchestrator loop itself uses a narrow core of that interface. Implement these exactly; the rest of the ABC matters for service-side setup and tooling, not the loop:

| Method | Called | Contract the orchestrator relies on |
|---|---|---|
| `provider.create_sandbox(request)` | once per task | `request.source` is always an `ImageSource` (snapshots are rejected before the provider sees them). `request.env_vars` must reach the agent process env — it carries the contract-declared secrets. `request.name` is `{run_id}-{task_id}`. Raise on failure: the task is recorded as a generation ERROR. |
| `sandbox.exec(command)` | cwd mkdir, install, agent run | `command` is a shell string (`cd X && …`, `timeout N …`) — run it through a shell. Return `ExecResult(exit_code, output)` with stderr merged into `output` (on a nonzero exit, `output` becomes the recorded error text). Exit code `124` must propagate untouched — it classifies the task as MAX_TIME rather than ERROR. The orchestrator never passes `cwd=`/`timeout=` kwargs; both are baked into the command string. |
| `sandbox.upload_file(path, content)` | task setup | Writes the problem statement into the sandbox; `path` is absolute. Manifest-native runs upload it directly; callback runs do the same write service-side through this interface. |
| `sandbox.download_file(path)` | after the agent run | MUST raise when the file is missing — an absent `generation.json` is how a failed run is classified. Returning empty bytes would silently corrupt results. |
| `provider.delete_sandbox(id)` | always, in a `finally` | Best-effort cleanup; failures are logged, not fatal. |

Wire it in programmatically (the CLI builds the default Daytona provider):

```python
from benchmark_runner.sandbox.orchestrator import run_benchmark

await run_benchmark(..., provider=MyProvider())
```

## Development

```bash
make install
make test
make lint
make typecheck
```
