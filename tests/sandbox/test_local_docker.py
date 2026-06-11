"""Daemon-free unit tests for the local Docker sandbox provider.

These exercise the provider through its public surface by injecting a fake
docker client (the `client=` seam), so they never touch the Docker daemon. The
real-daemon path is covered by the `docker`-marked smoke test at the bottom,
which is deselected by default (pyproject addopts) and needs a local daemon.
"""

import io
import tarfile

import pytest
from docker.errors import APIError, ImageNotFound, NotFound

from benchmark_service.sandbox.types import (
    ExecResult,
    ImageSource,
    Resources,
    SandboxCommandError,
    SandboxCreateRequest,
    SandboxError,
    SandboxNotFoundError,
    SandboxQuery,
    SnapshotSource,
)

from benchmark_runner.sandbox.local_docker import (
    _PROVIDER_LABEL,
    LocalDockerSandbox,
    LocalDockerSandboxProvider,
    _command,
)

ExecOutcome = tuple[int, tuple[bytes | None, bytes | None]] | Exception


class FakeContainer:
    def __init__(
        self,
        *,
        container_id: str = "cid",
        name: str = "r1-t1",
        labels: dict[str, str] | None = None,
        status: str = "running",
        exec_outcomes: list[ExecOutcome] | None = None,
        archive: bytes | None = None,
        put_archive_ok: bool = True,
    ) -> None:
        self.id = container_id
        self.name = name
        self.labels = labels if labels is not None else {_PROVIDER_LABEL: ""}
        self.status = status
        self.archive = archive
        self.put_archive_ok = put_archive_ok
        self._exec_outcomes = list(exec_outcomes or [])
        self.exec_calls: list[tuple[list[str], bool]] = []
        self.put_archive_calls: list[tuple[str, bytes]] = []
        self.removed = False

    def reload(self) -> None:
        pass

    def logs(self, **_kwargs: object) -> bytes:
        return b"tail: not found"

    def remove(self, *, force: bool = False) -> None:
        assert force
        self.removed = True

    def exec_run(self, cmd: list[str], demux: bool = False) -> tuple[int, object]:
        self.exec_calls.append((cmd, demux))
        outcome = self._exec_outcomes.pop(0) if self._exec_outcomes else (0, (b"", b""))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def put_archive(self, path: str, data: bytes) -> bool:
        self.put_archive_calls.append((path, data))
        return self.put_archive_ok

    def get_archive(self, path: str) -> tuple[object, dict[str, str]]:
        if self.archive is None:
            raise NotFound(f"no such path: {path}")
        return iter([self.archive[:7], self.archive[7:]]), {"name": path}


class FakeContainers:
    def __init__(self, *, existing: dict[str, FakeContainer] | None = None, run_result: FakeContainer | None = None):
        self.existing = existing or {}
        self.run_result = run_result or FakeContainer()
        self.run_calls: list[dict[str, object]] = []
        self.list_calls: list[dict[str, object]] = []

    def get(self, name: str) -> FakeContainer:
        try:
            return self.existing[name]
        except KeyError:
            raise NotFound(f"no such container: {name}") from None

    def run(self, image: str, **kwargs: object) -> FakeContainer:
        self.run_calls.append({"image": image, **kwargs})
        return self.run_result

    def list(self, *, all: bool = False, filters: dict[str, object] | None = None) -> list[FakeContainer]:
        self.list_calls.append({"all": all, "filters": filters})
        return list(self.existing.values())


class FakeImages:
    def __init__(self, *, present: bool = True):
        self.present = present
        self.pulled: list[str] = []

    def get(self, image: str) -> object:
        if not self.present:
            raise ImageNotFound(f"no such image: {image}")
        return object()

    def pull(self, image: str) -> None:
        self.pulled.append(image)


class FakeDockerClient:
    def __init__(self, *, containers: FakeContainers | None = None, images: FakeImages | None = None):
        self.containers = containers or FakeContainers()
        self.images = images or FakeImages()


def _provider(client: FakeDockerClient, **kwargs) -> LocalDockerSandboxProvider:
    return LocalDockerSandboxProvider(client=client, **kwargs)  # pyright: ignore[reportArgumentType]


def _request(source, *, name: str = "r1-t1") -> SandboxCreateRequest:
    return SandboxCreateRequest(
        source=source,
        resources=Resources(vcpu=1, memory=1, disk=1),
        name=name,
        labels={"k": "v"},
        env_vars={"FOO": "bar"},
        auto_stop_interval=0,
        create_timeout=60,
    )


def _sandbox(container: FakeContainer) -> LocalDockerSandbox:
    return LocalDockerSandbox(container)  # pyright: ignore[reportArgumentType]


def test_command_matches_daytona_order():
    # timeout wraps first (inner), cwd prepends second (outer) — same as cbs _command.
    assert _command("run x", "/app", 5) == "cd /app && timeout 5 run x"
    assert _command("run x", None, None) == "run x"
    assert _command("run x", "/w", None) == "cd /w && run x"


async def test_create_sandbox_rejects_snapshots():
    client = FakeDockerClient()
    with pytest.raises(SandboxError, match="ImageSource"):
        await _provider(client).create_sandbox(_request(SnapshotSource(snapshot="snap-1")))
    # A legacy service can hand back docker_image="snapshot:<name>" wrapped in an
    # ImageSource; that is still a snapshot and must be rejected, not pulled.
    with pytest.raises(SandboxError, match="ImageSource"):
        await _provider(client).create_sandbox(_request(ImageSource(image="snapshot:legacy")))
    assert client.containers.run_calls == []


async def test_create_sandbox_boots_keepalive_container():
    container = FakeContainer(container_id="abc123")
    client = FakeDockerClient(containers=FakeContainers(run_result=container))
    provider = _provider(client, extra_env={"FOO": "loses", "EXTRA": "1"})

    sandbox = await provider.create_sandbox(_request(ImageSource(image="img:tag")))

    assert sandbox.id == "abc123"
    assert sandbox.name == "r1-t1"
    (run_call,) = client.containers.run_calls
    assert run_call["image"] == "img:tag"
    assert run_call["detach"] is True
    assert run_call["name"] == "r1-t1"
    # entrypoint/command split: command replaces the image CMD; a combined
    # entrypoint would get the image CMD appended as bogus extra args to tail.
    assert run_call["entrypoint"] == ["tail"]
    assert run_call["command"] == ["-f", "/dev/null"]
    # provider label (so cleanup/list find it) plus the request's labels
    assert run_call["labels"] == {_PROVIDER_LABEL: "", "k": "v"}
    # extra_env merged in, request env_vars winning on conflict
    assert run_call["environment"] == {"FOO": "bar", "EXTRA": "1"}
    # the sh preflight ran against the new container
    assert container.exec_calls == [(["sh", "-c", "true"], False)]


async def test_create_sandbox_removes_stale_owned_container():
    stale = FakeContainer(labels={_PROVIDER_LABEL: ""})
    client = FakeDockerClient(containers=FakeContainers(existing={"r1-t1": stale}))
    await _provider(client).create_sandbox(_request(ImageSource(image="img:tag")))
    assert stale.removed


async def test_create_sandbox_refuses_unowned_name():
    unowned = FakeContainer(labels={"someone": "else"})
    client = FakeDockerClient(containers=FakeContainers(existing={"r1-t1": unowned}))
    with pytest.raises(SandboxError, match="does not own"):
        await _provider(client).create_sandbox(_request(ImageSource(image="img:tag")))
    assert not unowned.removed
    assert client.containers.run_calls == []


async def test_create_sandbox_pulls_image_only_when_missing():
    missing = FakeDockerClient(images=FakeImages(present=False))
    await _provider(missing).create_sandbox(_request(ImageSource(image="img:tag")))
    assert missing.images.pulled == ["img:tag"]

    present = FakeDockerClient(images=FakeImages(present=True))
    await _provider(present).create_sandbox(_request(ImageSource(image="img:tag")))
    assert present.images.pulled == []


async def test_create_sandbox_diagnoses_exited_container():
    # A container that is not running right after create means the image lacks
    # the keep-alive binary; the error must say so rather than fail later.
    container = FakeContainer(status="exited")
    client = FakeDockerClient(containers=FakeContainers(run_result=container))
    with pytest.raises(SandboxError, match="tail"):
        await _provider(client).create_sandbox(_request(ImageSource(image="img:tag")))


@pytest.mark.parametrize("outcome", [(127, (b"", b"")), APIError("no sh")])
async def test_create_sandbox_diagnoses_missing_sh(outcome: ExecOutcome):
    container = FakeContainer(exec_outcomes=[outcome])
    client = FakeDockerClient(containers=FakeContainers(run_result=container))
    with pytest.raises(SandboxError, match="sh"):
        await _provider(client).create_sandbox(_request(ImageSource(image="img:tag")))


async def test_exec_joins_demuxed_streams_with_newline():
    # `backend.generate` surfaces output as the error text on nonzero exit, so
    # stderr must be present in it; the newline keeps the streams readable.
    container = FakeContainer(exec_outcomes=[(1, (b"out", b"err"))])
    res = await _sandbox(container).exec("boom")
    assert isinstance(res, ExecResult)
    assert res.exit_code == 1
    assert res.output == "out\nerr"
    assert container.exec_calls == [(["sh", "-c", "boom"], True)]


async def test_exec_handles_stderr_only_and_propagates_exit_code():
    container = FakeContainer(exec_outcomes=[(124, (None, b"timed out"))])
    res = await _sandbox(container).exec("slow")
    # 124 must pass through untouched — it classifies the task as MAX_TIME upstream.
    assert res.exit_code == 124
    assert res.output == "timed out"


async def test_command_yields_output_then_raises():
    container = FakeContainer(exec_outcomes=[(2, (b"partial", None))])
    chunks: list[str] = []
    with pytest.raises(SandboxCommandError) as excinfo:
        async for chunk in _sandbox(container).command("boom"):
            chunks.append(chunk)
    assert chunks == ["partial"]
    assert excinfo.value.exit_code == 2


async def test_upload_file_archives_file_with_ancestor_dirs():
    container = FakeContainer()
    await _sandbox(container).upload_file("/app/sub/problem.txt", b"hello")

    ((path, data),) = container.put_archive_calls
    assert path == "/"
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        members = {member.name: member for member in tar.getmembers()}
        # ancestor dirs are explicit entries (put_archive does not mkdir -p)
        assert members["app"].isdir() and members["app"].mode == 0o755
        assert members["app/sub"].isdir() and members["app/sub"].mode == 0o755
        file_member = members["app/sub/problem.txt"]
        # 0o644, not the root-owned 0600 default: non-root agents must read it
        assert file_member.isfile() and file_member.mode == 0o644
        extracted = tar.extractfile(file_member)
        assert extracted is not None and extracted.read() == b"hello"


async def test_upload_file_rejects_relative_path_and_failed_put():
    container = FakeContainer(put_archive_ok=False)
    # Docker's archive API and exec'd commands resolve relative paths against
    # different bases (/ vs WORKDIR), so relative paths are rejected outright.
    with pytest.raises(SandboxError, match="absolute"):
        await _sandbox(container).upload_file("results/x.json", b"x")
    assert container.put_archive_calls == []
    with pytest.raises(SandboxError, match="upload_file failed"):
        await _sandbox(container).upload_file("/app/x.json", b"x")


def _tar_bytes(name: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


async def test_download_file_extracts_bytes():
    container = FakeContainer(archive=_tar_bytes("generation.json", b'{"answer": 42}'))
    assert await _sandbox(container).download_file("/app/generation.json") == b'{"answer": 42}'


async def test_download_file_raises_when_missing():
    # Must raise (not return empty): `backend.generate` relies on this to
    # classify a missing generation.json as ERROR.
    container = FakeContainer(archive=None)
    with pytest.raises(SandboxError, match="download_file failed"):
        await _sandbox(container).download_file("/app/missing.json")
    with pytest.raises(SandboxError, match="absolute"):
        await _sandbox(container).download_file("results/x.json")


async def test_get_sandbox_raises_not_found():
    client = FakeDockerClient()
    with pytest.raises(SandboxNotFoundError, match="gone"):
        await _provider(client).get_sandbox("gone")


async def test_delete_sandbox_is_idempotent():
    client = FakeDockerClient(containers=FakeContainers(existing={"cid": FakeContainer()}))
    provider = _provider(client)
    await provider.delete_sandbox("cid")
    assert client.containers.existing["cid"].removed
    await provider.delete_sandbox("never-existed")  # NotFound is not an error


async def test_list_sandboxes_filters_on_provider_label():
    client = FakeDockerClient(containers=FakeContainers(existing={"cid": FakeContainer()}))
    sandboxes = [s async for s in _provider(client).list_sandboxes(SandboxQuery(labels={"k": "v"}))]
    assert [s.id for s in sandboxes] == ["cid"]
    (list_call,) = client.containers.list_calls
    assert list_call["all"] is True
    assert list_call["filters"] == {"label": [_PROVIDER_LABEL, "k=v"]}


@pytest.mark.docker
async def test_real_daemon_smoke():
    """End-to-end against the local Docker daemon (busybox has both sh and tail)."""
    provider = LocalDockerSandboxProvider()
    request = SandboxCreateRequest(
        source=ImageSource(image="busybox"),
        resources=Resources(vcpu=1, memory=1, disk=1),
        name="cbr-local-docker-smoke",
        labels={"cbr-test": "smoke"},
        env_vars={"SMOKE_VAR": "hello"},
        auto_stop_interval=0,
        create_timeout=120,
    )
    sandbox = await provider.create_sandbox(request)
    try:
        echo = await sandbox.exec('echo "$SMOKE_VAR"')
        assert echo.exit_code == 0
        assert "hello" in echo.output

        await sandbox.upload_file("/tmp/smoke/data.txt", b"roundtrip")
        assert await sandbox.download_file("/tmp/smoke/data.txt") == b"roundtrip"
        cat = await sandbox.exec("cat /tmp/smoke/data.txt")
        assert cat.exit_code == 0
        assert "roundtrip" in cat.output

        with pytest.raises(SandboxError):
            await sandbox.download_file("/tmp/smoke/never-written.json")
    finally:
        await provider.delete_sandbox(sandbox.id)
