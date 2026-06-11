"""Local Docker implementation of the create-benchmark-service sandbox ABCs.

Resource limits, auto-stop, and snapshots are Daytona concepts with no faithful
local equivalent and are intentionally ignored (snapshots are rejected outright).
"""

from __future__ import annotations

import asyncio
import os
import posixpath
import shlex
import tempfile
from collections.abc import AsyncGenerator
from contextlib import suppress

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

# Marks containers this provider created, so list_sandboxes can find them.
_PROVIDER_LABEL = "benchmark-runner-local-sandbox"
# Keep-alive entrypoint so the container stays up for `docker exec`. `tail -f
# /dev/null` works on virtually every base image; `sleep infinity` does not
# (busybox sleep rejects "infinity").
_KEEPALIVE_ENTRYPOINT = "tail"
_KEEPALIVE_ARGS = ("-f", "/dev/null")


# ---------------------------------------------------------------------------
# Pure helpers (no I/O) and the single subprocess seam.
# ---------------------------------------------------------------------------


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


def _create_argv(
    *,
    name: str,
    image: str,
    labels: dict[str, str],
    env_vars: dict[str, str],
    env_file: str | None = None,
) -> list[str]:
    """Build the ``docker run`` argv that boots a keep-alive sandbox container."""
    argv = ["docker", "run", "-d", "--name", name, "--label", _PROVIDER_LABEL]
    for key, value in labels.items():
        argv += ["--label", f"{key}={value}"]
    if env_file:
        argv += ["--env-file", env_file]
    for key, value in env_vars.items():
        argv += ["-e", f"{key}={value}"]
    argv += ["--entrypoint", _KEEPALIVE_ENTRYPOINT, image, *_KEEPALIVE_ARGS]
    return argv


def _exec_argv(container_id: str, full_command: str) -> list[str]:
    """Build the ``docker exec`` argv. The command runs under ``sh -c`` because
    the orchestrator passes shell strings (``cd x && ...``, ``timeout ...``)."""
    return ["docker", "exec", container_id, "sh", "-c", full_command]


async def _run(*argv: str) -> tuple[int, bytes, bytes]:
    """The single subprocess seam: run a docker CLI command.

    Returns ``(exit_code, stdout, stderr)``. Everything funnels through here so
    tests can fake Docker by monkeypatching this one function.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    assert proc.returncode is not None  # communicate() waits for process exit
    return proc.returncode, stdout, stderr


def _require_absolute(remote_path: str) -> None:
    """``docker cp`` resolves container-side relative paths against ``/`` while
    exec'd commands resolve against the image WORKDIR, so a relative path would
    mkdir in one place and copy to another. Require absolute paths instead of
    picking a side; all real callers (problem statements, final_output) pass them.
    """
    if not posixpath.isabs(remote_path):
        raise SandboxError(f"container path must be absolute, got {remote_path!r}")


async def _inspect_name(container_id: str) -> str | None:
    """Return a container's name (sans leading '/'), or None if it does not exist."""
    code, out, _err = await _run("docker", "inspect", "-f", "{{.Name}}", container_id)
    return out.decode().strip().lstrip("/") if code == 0 else None


# ---------------------------------------------------------------------------
# Sandbox + Provider
# ---------------------------------------------------------------------------


class LocalDockerSandbox(Sandbox):
    """A single local Docker container exposed through the cbs Sandbox interface."""

    def __init__(self, container_id: str, name: str) -> None:
        self._id = container_id
        self._name = name

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> str:
        # The ABC's state is a sync property and the orchestrator never reads it
        # (only Daytona internals do), so we report the post-create state rather
        # than block the event loop on a `docker inspect` here.
        return "running"

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        full = _command(command, cwd, timeout)
        code, out, err = await _run(*_exec_argv(self._id, full))
        # Combine stdout then stderr (not interleaved like Daytona's stream):
        # `backend.generate` surfaces `result.output` as the error text on a
        # nonzero exit, so the agent's stderr must be present in it.
        output = out.decode("utf-8", "replace")
        if err:
            output += err.decode("utf-8", "replace")
        return ExecResult(exit_code=code, output=output)

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
        # docker cp will not create parent directories, so ensure they exist.
        parent = posixpath.dirname(remote_path)
        await _run(*_exec_argv(self._id, f"mkdir -p {shlex.quote(parent)}"))
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(content)
            tmp.flush()
            code, _out, err = await _run("docker", "cp", tmp.name, f"{self._id}:{remote_path}")
        if code != 0:
            raise SandboxError(
                f"upload_file failed for {remote_path} in {self._id}: {err.decode('utf-8', 'replace').strip()}"
            )

    async def download_file(self, remote_path: str) -> bytes:
        _require_absolute(remote_path)
        # Must RAISE on a missing file (not return empty): `backend.generate`
        # relies on this to classify a missing generation.json as ERROR.
        fd, tmp_path = tempfile.mkstemp()
        os.close(fd)
        try:
            code, _out, err = await _run("docker", "cp", f"{self._id}:{remote_path}", tmp_path)
            if code != 0:
                raise SandboxError(
                    f"download_file failed for {remote_path} in {self._id}: {err.decode('utf-8', 'replace').strip()}"
                )
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            with suppress(FileNotFoundError):
                os.unlink(tmp_path)


class LocalDockerSandboxProvider(SandboxProvider):
    """Creates and manages sandboxes as containers on the local Docker daemon."""

    def __init__(self, *, extra_env_file: str | None = None) -> None:
        # Local-testing convenience: a `docker --env-file` injected into every
        # sandbox, for env the agent needs beyond the contract-declared secrets
        # the orchestrator passes via env_vars. Not a Daytona concept; local only.
        self._extra_env_file = extra_env_file

    async def create_sandbox(self, request: SandboxCreateRequest) -> LocalDockerSandbox:
        if not isinstance(request.source, ImageSource):
            raise SandboxError(
                "LocalDockerSandboxProvider supports only ImageSource (a pullable registry "
                f"image); got {type(request.source).__name__}. Daytona snapshots cannot run locally."
            )
        image = request.source.image
        name = request.name
        # Remove any stale container with this name so re-runs are idempotent.
        # (Daytona reuses a same-named running sandbox; a local harness just rebuilds.)
        await _run("docker", "rm", "-f", name)
        code, out, err = await _run(
            *_create_argv(
                name=name,
                image=image,
                labels=request.labels,
                env_vars=request.env_vars,
                env_file=self._extra_env_file,
            )
        )
        if code != 0:
            raise SandboxError(f"docker run failed for image {image!r}: {err.decode('utf-8', 'replace').strip()}")
        return LocalDockerSandbox(container_id=out.decode().strip(), name=name)

    async def get_sandbox(self, instance_id: str) -> LocalDockerSandbox:
        name = await _inspect_name(instance_id)
        if name is None:
            raise SandboxNotFoundError(f"Sandbox not found: {instance_id}")
        return LocalDockerSandbox(container_id=instance_id, name=name)

    async def delete_sandbox(self, instance_id: str) -> None:
        # Best-effort and idempotent: a missing container is not an error.
        await _run("docker", "rm", "-f", instance_id)

    async def list_sandboxes(self, query: SandboxQuery) -> AsyncGenerator[LocalDockerSandbox, None]:
        argv = ["docker", "ps", "-aq", "--filter", f"label={_PROVIDER_LABEL}"]
        for key, value in query.labels.items():
            argv += ["--filter", f"label={key}={value}"]
        code, out, _err = await _run(*argv)
        if code != 0:
            return
        for container_id in out.decode().split():
            yield LocalDockerSandbox(container_id=container_id, name=await _inspect_name(container_id) or container_id)


__all__ = ["LocalDockerSandbox", "LocalDockerSandboxProvider"]
