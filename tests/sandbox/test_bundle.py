"""Behavioral tests for agent bundle packaging and in-sandbox delivery."""

import zipfile
from pathlib import Path

import pytest

from tests.sandbox.conftest import FakeSandbox
from benchmark_runner.sandbox.backend import SandboxGenerationBackend
from benchmark_runner.sandbox.bundle import (
    AgentBundle,
    build_bundle_zip,
    file_sha256,
    load_bundle,
    zip_root,
)
from benchmark_runner.sandbox.contract import AgentContract
from benchmark_runner.schemas import GenerationStatus


def _make_agent_dir(tmp_path: Path) -> Path:
    agent = tmp_path / "my_agent"
    (agent / "sub").mkdir(parents=True)
    (agent / "run.py").write_text("print('hi')")
    (agent / "setup.sh").write_text("true")
    (agent / "sub" / "helper.py").write_text("x = 1")
    # Must never ship: dev junk and the contract file itself.
    (agent / "contract.yaml").write_text("internal: true")
    (agent / ".env").write_text("SECRET=1")
    pycache = agent / "__pycache__"
    pycache.mkdir()
    (pycache / "run.cpython-312.pyc").write_bytes(b"\x00")
    return agent


def test_build_bundle_zip_layout_exclusions_and_roundtrip(tmp_path: Path) -> None:
    """The zip nests everything under the agent dir name, drops junk and the
    contract file, and round-trips through load_bundle with a matching digest."""
    agent = _make_agent_dir(tmp_path)
    out = tmp_path / "my_agent.zip"
    sha256 = build_bundle_zip(agent, out)

    assert sha256 == file_sha256(out)
    names = zipfile.ZipFile(out).namelist()
    assert "my_agent/run.py" in names
    assert "my_agent/sub/helper.py" in names
    assert not [n for n in names if "contract.yaml" in n or ".env" in n or ".pyc" in n]

    bundle = load_bundle(out, expected_sha256=sha256)
    assert bundle.root == "my_agent"
    assert bundle.install_path == "/bundle/my_agent"
    assert bundle.zip_bytes == out.read_bytes()


def test_load_bundle_rejects_digest_mismatch(tmp_path: Path) -> None:
    out = tmp_path / "my_agent.zip"
    build_bundle_zip(_make_agent_dir(tmp_path), out)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_bundle(out, expected_sha256="0" * 64)


def test_zip_root_rejects_loose_top_level_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("my_agent/run.py", "x")
        zf.writestr("loose.txt", "x")
    with pytest.raises(ValueError, match="exactly one top-level directory"):
        zip_root(bad)


@pytest.mark.asyncio
async def test_backend_installs_bundle_at_bundle_root(tmp_path: Path) -> None:
    """With a bundle, the backend uploads the zip, extracts it under /bundle, and
    runs install_cmd from /bundle/<root> instead of the task cwd — the layout
    internal sandboxes use, which run_cmds reference by absolute path."""
    sandbox = FakeSandbox("s1")
    contract = AgentContract(
        name="my-agent",
        install_cmd="bash setup.sh",
        run_cmd="run --problem {problem_statement_path}",
        final_output="/app/results",
    )
    result = await SandboxGenerationBackend().generate(
        sandbox=sandbox,
        contract=contract,
        task_id="t1",
        model="m",
        problem_path="/app/problem.txt",
        cwd="/app",
        agent_timeout=None,
        log_dir=tmp_path / "logs",
        bundle=AgentBundle(root="my_agent", zip_bytes=b"zipbytes"),
    )

    assert result.status == GenerationStatus.SUCCESS
    assert ("/tmp/my_agent.zip", b"zipbytes") in sandbox.uploads
    extract_cmd = next(c for c in sandbox.commands if "unzip" in c)
    assert "-d /bundle" in extract_cmd
    install_cmd = next(c for c in sandbox.commands if "setup.sh" in c)
    assert install_cmd.startswith("cd /bundle/my_agent && ")
