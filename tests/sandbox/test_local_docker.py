"""Docker-free unit tests for the local Docker sandbox provider.

These exercise the provider through its public surface by faking the single
subprocess seam (`local_docker._run`). They never touch the Docker daemon, so
they are safe to run in CI. The real-container behaviour is covered by one-off
smoke scaffolding, not committed tests.
"""

import pytest

from benchmark_service.sandbox.types import (
    ExecResult,
    ImageSource,
    Resources,
    SandboxCommandError,
    SandboxCreateRequest,
    SandboxError,
    SandboxNotFoundError,
    SnapshotSource,
)

from benchmark_runner.sandbox import local_docker
from benchmark_runner.sandbox.local_docker import (
    LocalDockerSandbox,
    LocalDockerSandboxProvider,
    _command,
)


def _fake_run(result: tuple[int, bytes, bytes]):
    async def run(*_argv: str) -> tuple[int, bytes, bytes]:
        return result

    return run


def _recording_run(calls: list[tuple[str, ...]], result: tuple[int, bytes, bytes] = (0, b"", b"")):
    async def run(*argv: str) -> tuple[int, bytes, bytes]:
        calls.append(argv)
        return result

    return run


def _request(source) -> SandboxCreateRequest:
    return SandboxCreateRequest(
        source=source,
        resources=Resources(vcpu=1, memory=1, disk=1),
        name="r1-t1",
        labels={"k": "v"},
        env_vars={"FOO": "bar"},
        auto_stop_interval=0,
        create_timeout=60,
    )


def test_command_matches_daytona_order():
    # timeout wraps first (inner), cwd prepends second (outer) — same as cbs _command.
    assert _command("run x", "/app", 5) == "cd /app && timeout 5 run x"
    assert _command("run x", None, None) == "run x"
    assert _command("run x", "/w", None) == "cd /w && run x"


async def test_create_sandbox_boots_keepalive_container(monkeypatch):
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(local_docker, "_run", _recording_run(calls, (0, b"abc123\n", b"")))

    provider = LocalDockerSandboxProvider(extra_env_file="/tmp/keys.env")
    sandbox = await provider.create_sandbox(_request(ImageSource(image="img:tag")))

    assert sandbox.id == "abc123"
    assert sandbox.name == "r1-t1"
    # stale same-named container removed first, so re-runs are idempotent
    assert calls[0] == ("docker", "rm", "-f", "r1-t1")
    run_argv = calls[1]
    assert run_argv[:3] == ("docker", "run", "-d")
    assert "img:tag" in run_argv
    # keepalive entrypoint so the container stays up for `docker exec`
    assert run_argv[run_argv.index("--entrypoint") + 1] == "tail"
    # provider label (so list_sandboxes finds it) plus the request's labels
    assert local_docker._PROVIDER_LABEL in run_argv
    assert "k=v" in run_argv
    # env: request env_vars via -e, provider extra env file via --env-file
    assert "FOO=bar" in run_argv
    assert "/tmp/keys.env" in run_argv


async def test_create_sandbox_raises_on_docker_run_failure(monkeypatch):
    monkeypatch.setattr(local_docker, "_run", _fake_run((125, b"", b"pull access denied")))
    with pytest.raises(SandboxError, match="pull access denied"):
        await LocalDockerSandboxProvider().create_sandbox(_request(ImageSource(image="img:tag")))


async def test_create_sandbox_rejects_snapshot():
    with pytest.raises(SandboxError, match="ImageSource"):
        await LocalDockerSandboxProvider().create_sandbox(_request(SnapshotSource(snapshot="snap-1")))


async def test_get_sandbox_raises_not_found(monkeypatch):
    monkeypatch.setattr(local_docker, "_run", _fake_run((1, b"", b"Error: No such object: gone")))
    with pytest.raises(SandboxNotFoundError):
        await LocalDockerSandboxProvider().get_sandbox("gone")


async def test_exec_merges_stderr_and_returns_exit_code(monkeypatch):
    monkeypatch.setattr(local_docker, "_run", _fake_run((1, b"out-", b"err")))
    res = await LocalDockerSandbox("cid", "n").exec("boom")
    assert isinstance(res, ExecResult)
    assert res.exit_code == 1
    assert res.output == "out-err"


async def test_command_yields_output_then_raises(monkeypatch):
    monkeypatch.setattr(local_docker, "_run", _fake_run((2, b"partial", b"")))
    chunks: list[str] = []
    with pytest.raises(SandboxCommandError) as excinfo:
        async for chunk in LocalDockerSandbox("cid", "n").command("boom"):
            chunks.append(chunk)
    assert chunks == ["partial"]
    assert excinfo.value.exit_code == 2


async def test_download_file_raises_when_missing(monkeypatch):
    monkeypatch.setattr(local_docker, "_run", _fake_run((1, b"", b"No such file or directory")))
    with pytest.raises(SandboxError):
        await LocalDockerSandbox("cid", "n").download_file("/app/missing.json")


async def test_upload_file_mkdirs_parent_and_rejects_relative_paths(monkeypatch):
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(local_docker, "_run", _recording_run(calls))

    await LocalDockerSandbox("cid", "n").upload_file("/app/problem.txt", b"x")
    assert calls[0] == ("docker", "exec", "cid", "sh", "-c", "mkdir -p /app")
    assert calls[1][:2] == ("docker", "cp")

    # docker cp and exec'd commands resolve relative paths against different
    # bases (/ vs WORKDIR), so relative container paths are rejected outright.
    with pytest.raises(SandboxError, match="absolute"):
        await LocalDockerSandbox("cid", "n").upload_file("results/x.json", b"x")
    with pytest.raises(SandboxError, match="absolute"):
        await LocalDockerSandbox("cid", "n").download_file("results/x.json")
