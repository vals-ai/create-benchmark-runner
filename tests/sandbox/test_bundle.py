"""Behavioral tests for agent bundle packaging."""

import zipfile
from pathlib import Path

import pytest

from benchmark_runner.sandbox.bundle import (
    build_bundle_zip,
    file_sha256,
    load_bundle,
    zip_root,
)


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


def test_zip_root_rejects_unsafe_entry_paths(tmp_path: Path) -> None:
    """Entries escaping the bundle root (zip-slip) are rejected up front, never
    left to unzip's sanitization behavior."""
    for evil in ("../evil.py", "/abs/evil.py", "my_agent/../../evil.py"):
        bad = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("my_agent/run.py", "x")
            zf.writestr(evil, "x")
        with pytest.raises(ValueError, match="unsafe entry"):
            zip_root(bad)


def test_build_bundle_zip_rejects_output_inside_agent_dir(tmp_path: Path) -> None:
    """Writing the zip into the directory being bundled would absorb the
    previous zip (and manifest) on every regeneration."""
    agent = _make_agent_dir(tmp_path)
    with pytest.raises(ValueError, match="inside the agent directory"):
        build_bundle_zip(agent, agent / "my_agent.zip")
