"""Local Docker implementation of the create-benchmark-service sandbox ABCs.

Lets the sandbox orchestrator run against containers on the local Docker daemon
instead of Daytona — the "test without Daytona" path. It implements the same
``benchmark_service.sandbox`` ``SandboxProvider`` / ``Sandbox`` interface that
``DaytonaSandbox`` does, so ``run_benchmark(provider=LocalDockerSandboxProvider())``
is a drop-in. Built on docker-py (no Daytona SDK); requires a reachable local
Docker daemon.

Image requirements: containers are kept alive with ``tail -f /dev/null`` and
commands run under ``sh -c``, so the image must contain ``tail`` and a POSIX
``sh`` — distroless/scratch images are unsupported. Container paths must be
absolute. CPU and memory limits are mapped to Docker limits; disk limits and
auto-stop have no faithful local equivalent and are intentionally ignored.
Snapshots are rejected outright. This is a local test/dev harness, not a
production sandbox.
"""

from __future__ import annotations

import asyncio
import io
import posixpath
import shlex
import tarfile
from collections.abc import AsyncGenerator, Mapping
from contextlib import suppress
from typing import cast

import docker
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.models.containers import Container

from benchmark_service.sandbox.types import (
    ExecResult,
    ImageSource,
    Sandbox,
    SandboxCommandError,
    SandboxCreateRequest,
    SandboxError,
    SandboxNotFoundError,
    SandboxProvider,
    SandboxQuery,
)

# Marks containers this provider created, so pre-create cleanup and
# list_sandboxes only ever touch containers this provider owns.
_PROVIDER_LABEL = "benchmark-runner-local-sandbox"
# Keep-alive so the container stays up for exec. `tail -f /dev/null` works on
# virtually every base image; `sleep infinity` does not (busybox sleep rejects
# "infinity"). The entrypoint/command split matters: command replaces the image
# CMD, whereas putting the whole keep-alive in entrypoint with command=None
# would leave the image CMD to be appended as bogus extra args to tail.
_KEEPALIVE_ENTRYPOINT = ["tail"]
_KEEPALIVE_COMMAND = ["-f", "/dev/null"]
_NANO_CPUS_PER_VCPU = 1_000_000_000


def _command(command: str, cwd: str | None, timeout: float | None) -> str:
    """Mirror ``benchmark_service.sandbox.daytona._command`` exactly.

    timeout wraps the command first (inner), then cwd prepends a ``cd`` (outer).
    The orchestrator backend passes neither (it bakes both into the command
    string itself), but matching Daytona keeps this a faithful drop-in.
    """
    if timeout is not None:
        command = f"timeout {timeout:g} {command}"
    if cwd:
        command = f"cd {shlex.quote(cwd)} && {command}"
    return command


def _require_absolute(remote_path: str) -> None:
    """Docker's archive API resolves container-side relative paths against ``/``
    while exec'd commands resolve against the image WORKDIR, so a relative path
    would read and write in different places. Require absolute paths instead of
    picking a side; all real callers (problem statements, final_output) pass them.
    """
    if not posixpath.isabs(remote_path):
        raise SandboxError(f"container path must be absolute, got {remote_path!r}")


def _tar_archive(remote_path: str, content: bytes) -> bytes:
    """An in-memory tar that lands ``content`` at ``remote_path`` when extracted at /.

    Carries explicit entries for every ancestor directory so parents are created
    (put_archive does not mkdir -p), and mode 0o644 on the file so non-root agent
    processes can read it (TarInfo's default would land it root-owned 0600).
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        ancestors: list[str] = []
        directory = posixpath.dirname(remote_path)
        while directory != "/":
            ancestors.append(directory)
            directory = posixpath.dirname(directory)
        for ancestor in reversed(ancestors):
            dir_info = tarfile.TarInfo(ancestor.lstrip("/"))
            dir_info.type = tarfile.DIRTYPE
            dir_info.mode = 0o755
            tar.addfile(dir_info)
        file_info = tarfile.TarInfo(remote_path.lstrip("/"))
        file_info.mode = 0o644
        file_info.size = len(content)
        tar.addfile(file_info, io.BytesIO(content))
    return buf.getvalue()


def _is_owned(container: Container) -> bool:
    return _PROVIDER_LABEL in (container.labels or {})


class LocalDockerSandbox(Sandbox):
    """A single local Docker container exposed through the cbs Sandbox interface."""

    def __init__(self, container: Container) -> None:
        self._container = container

    @property
    def id(self) -> str:
        return self._container.id or ""

    @property
    def name(self) -> str:
        return self._container.name or ""

    @property
    def state(self) -> str:
        # The ABC's state is a sync property and the orchestrator never reads it
        # (only Daytona internals do), so report the container object's cached
        # status rather than block the event loop on a reload here.
        return self._container.status

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        full = _command(command, cwd, timeout)
        try:
            raw = await asyncio.to_thread(self._container.exec_run, ["sh", "-c", full], demux=True)
        except (APIError, DockerException) as e:
            raise SandboxError(f"exec failed in {self.id}: {e}") from e
        # exec_run is untyped; with demux=True and no streaming it returns
        # (exit_code, (stdout | None, stderr | None)).
        exit_code, (stdout, stderr) = cast(tuple[int, tuple[bytes | None, bytes | None]], raw)
        # Combine stdout then stderr (not interleaved like Daytona's stream):
        # `backend.generate` surfaces `result.output` as the error text on a
        # nonzero exit, so the agent's stderr must be present in it.
        output = "\n".join(stream.decode("utf-8", "replace") for stream in (stdout, stderr) if stream)
        # Exit code passes through untouched: 124 classifies the task as MAX_TIME upstream.
        return ExecResult(exit_code=exit_code, output=output)

    async def command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[str, None]:
        # Minimal streaming impl (the orchestrator does not use it): run once,
        # emit the output, then raise on failure to match Daytona's contract.
        result = await self.exec(command, cwd=cwd, timeout=timeout)
        if result.output:
            yield result.output
        if result.exit_code != 0:
            raise SandboxCommandError(result.exit_code)

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        _require_absolute(remote_path)
        archive = _tar_archive(remote_path, content)
        try:
            ok = await asyncio.to_thread(self._container.put_archive, "/", archive)
        except (APIError, DockerException) as e:
            raise SandboxError(f"upload_file failed for {remote_path} in {self.id}: {e}") from e
        if not ok:
            raise SandboxError(f"upload_file failed for {remote_path} in {self.id}")

    async def download_file(self, remote_path: str) -> bytes:
        _require_absolute(remote_path)
        try:
            # get_archive streams a tar of the requested path; drain it on the
            # worker thread since reading the chunks is also blocking I/O.
            archive = await asyncio.to_thread(
                lambda: b"".join(self._container.get_archive(remote_path)[0])
            )
        except NotFound as e:
            # Must RAISE on a missing file (not return empty): `backend.generate`
            # relies on this to classify a missing generation.json as ERROR.
            raise SandboxError(f"download_file failed for {remote_path} in {self.id}: {e}") from e
        except (APIError, DockerException) as e:
            raise SandboxError(f"download_file failed for {remote_path} in {self.id}: {e}") from e
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar.getmembers():
                if member.isfile():
                    extracted = tar.extractfile(member)
                    assert extracted is not None  # isfile() members are always extractable
                    return extracted.read()
        raise SandboxError(f"download_file: {remote_path} in {self.id} is not a regular file")


class LocalDockerSandboxProvider(SandboxProvider):
    """Creates and manages sandboxes as containers on the local Docker daemon."""

    def __init__(
        self,
        *,
        extra_env: Mapping[str, str] | None = None,
        client: docker.DockerClient | None = None,
    ) -> None:
        # docker.from_env() is the daemon preflight: it raises DockerException
        # with a clear message when no daemon is reachable, so constructing the
        # provider fails fast. `client=` is the injection seam for tests.
        self._client = client or docker.from_env()
        # Local-testing convenience: env merged into every sandbox, for anything
        # the agent needs beyond the contract-declared secrets the orchestrator
        # passes via request.env_vars (which win on conflict).
        self._extra_env = dict(extra_env or {})

    async def create_sandbox(self, request: SandboxCreateRequest) -> LocalDockerSandbox:
        if not isinstance(request.source, ImageSource) or request.source.image.startswith("snapshot:"):
            raise SandboxError(
                "LocalDockerSandboxProvider supports only ImageSource (a pullable registry "
                f"image); got {request.source!r}. Daytona snapshots cannot run locally."
            )
        image = request.source.image
        create_task = asyncio.create_task(asyncio.to_thread(self._create_container, request, image))
        try:
            container = await asyncio.wait_for(
                asyncio.shield(create_task),
                timeout=request.create_timeout,
            )
        except TimeoutError as e:
            self._cleanup_late_create(create_task)
            raise SandboxError(f"sandbox create timed out after {request.create_timeout}s (image pull included)") from e
        except (APIError, DockerException) as e:
            raise SandboxError(f"sandbox create failed for image {image!r}: {e}") from e
        return LocalDockerSandbox(container)

    def _cleanup_late_create(self, create_task: asyncio.Task[Container]) -> None:
        def cleanup(done: asyncio.Task[Container]) -> None:
            try:
                container = done.result()
            except asyncio.CancelledError:
                return
            except Exception:
                return
            if _is_owned(container):
                with suppress(APIError, DockerException):
                    container.remove(force=True)

        create_task.add_done_callback(cleanup)

    def _create_container(self, request: SandboxCreateRequest, image: str) -> Container:
        """Blocking create path, run on a worker thread under request.create_timeout."""
        # Remove any stale same-named container so re-runs are idempotent — but
        # only if this provider made it; never force-remove someone else's.
        try:
            stale = self._client.containers.get(request.name)
        except NotFound:
            pass
        else:
            if not _is_owned(stale):
                raise SandboxError(
                    f"container name {request.name!r} is taken by a container this provider does not own"
                )
            stale.remove(force=True)
        try:
            self._client.images.get(image)
        except ImageNotFound:
            self._client.images.pull(image)
        container = self._client.containers.run(
            image,
            detach=True,
            name=request.name,
            entrypoint=_KEEPALIVE_ENTRYPOINT,
            command=_KEEPALIVE_COMMAND,
            labels={_PROVIDER_LABEL: "", **request.labels},
            environment={**self._extra_env, **request.env_vars},
            nano_cpus=request.resources.vcpu * _NANO_CPUS_PER_VCPU,
            mem_limit=f"{request.resources.memory}g",
        )
        try:
            container.reload()
            if container.status != "running":
                logs = container.logs(tail=20).decode("utf-8", "replace").strip()
                raise SandboxError(
                    f"container for image {image!r} exited immediately instead of staying up; the image "
                    f"must contain `tail` (used as the keep-alive entrypoint). Last logs:\n{logs}"
                )
            try:
                sh_exit_code, _output = container.exec_run(["sh", "-c", "true"])
            except APIError:
                sh_exit_code = -1
            if sh_exit_code != 0:
                raise SandboxError(
                    f"image {image!r} cannot run `sh -c`; sandbox commands run under a POSIX `sh`, "
                    "which must be present in the image"
                )
        except Exception:
            with suppress(APIError, DockerException):
                container.remove(force=True)
            raise
        return container

    async def get_sandbox(self, instance_id: str) -> LocalDockerSandbox:
        try:
            container = await asyncio.to_thread(self._client.containers.get, instance_id)
        except NotFound as e:
            raise SandboxNotFoundError(f"Sandbox not found: {instance_id}") from e
        return LocalDockerSandbox(container)

    async def delete_sandbox(self, instance_id: str) -> None:
        try:
            container = await asyncio.to_thread(self._client.containers.get, instance_id)
        except NotFound:
            return  # idempotent: already gone
        if not _is_owned(container):
            raise SandboxError(f"container {instance_id!r} exists but this provider does not own it")
        try:
            await asyncio.to_thread(container.remove, force=True)
        except APIError as e:
            raise SandboxError(f"delete_sandbox failed for {instance_id}: {e}") from e

    async def list_sandboxes(self, query: SandboxQuery) -> AsyncGenerator[LocalDockerSandbox, None]:
        label_filters = [_PROVIDER_LABEL, *(f"{key}={value}" for key, value in query.labels.items())]
        containers = await asyncio.to_thread(
            self._client.containers.list, all=True, filters={"label": label_filters}
        )
        for container in containers:
            yield LocalDockerSandbox(container)


__all__ = ["LocalDockerSandbox", "LocalDockerSandboxProvider"]
