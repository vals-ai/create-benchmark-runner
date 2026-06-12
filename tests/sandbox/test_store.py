"""Behavioral tests for the project-local manifest store."""

from pathlib import Path

import pytest

from benchmark_service.sandbox import Resources
from tests.sandbox.conftest import make_manifest
from benchmark_runner.sandbox.bundle import build_bundle_zip, file_sha256
from benchmark_runner.sandbox.manifest import BundleSpec
from benchmark_runner.sandbox.store import (
    install_manifest,
    list_installed,
    load_installed,
    pin_diff,
)


def test_install_load_list_round_trip(tmp_path: Path) -> None:
    """install writes <name>.manifest.yaml (creating the dir); load/list round-trip
    the full model, secrets included; missing store/name are graceful."""
    store = tmp_path / "benchmarks"
    assert list_installed(store) == []
    assert load_installed("legal-research", store) is None

    mf = make_manifest("legal-research")
    path = install_manifest(mf, store)
    assert path == store / "legal-research.manifest.yaml"
    assert load_installed("legal-research", store) == mf

    install_manifest(make_manifest("cyber-bench"), store)
    assert [m.benchmark for m in list_installed(store)] == ["cyber-bench", "legal-research"]


def test_install_rejects_unsafe_benchmark_names(tmp_path: Path) -> None:
    """Manifest names are file names in the local store, so path separators must be rejected."""
    with pytest.raises(ValueError, match="unsafe benchmark name"):
        install_manifest(make_manifest("../escape"), tmp_path / "benchmarks")


def test_pin_diff_reports_changed_pins_only() -> None:
    """Identical manifests diff empty; changed service version surfaces as
    `field: old → new`."""
    old = make_manifest()
    assert pin_diff(old, make_manifest()) == []

    new = make_manifest(service_version="0.7.0")
    assert pin_diff(old, new) == [
        "service.service_version: 0.6.1 → 0.7.0",
    ]


def test_pin_diff_reports_per_task_execution_pin_changes() -> None:
    """Executable env (image/resources/cwd/timeout/problem_path) is pinned per task entry."""
    old = make_manifest()
    new = make_manifest()
    new.tasks[0].image = "ghcr.io/vals-ai/agent@sha256:" + "b" * 64
    new.tasks[0].resources = Resources(vcpu=4, memory=8, disk=20)
    new.tasks[0].problem_path = "/app/other_problem.txt"

    assert pin_diff(old, new) == [
        f"tasks.task-1.image: ghcr.io/vals-ai/agent@sha256:{'a' * 64} → ghcr.io/vals-ai/agent@sha256:{'b' * 64}",
        f"tasks.task-1.resources: {old.tasks[0].resources} → {new.tasks[0].resources}",
        "tasks.task-1.problem_path: /app/problem.txt → /app/other_problem.txt",
    ]


def _bundle_source(tmp_path: Path, name: str, payload: str) -> tuple[Path, str]:
    agent = tmp_path / name / "agent"
    agent.mkdir(parents=True)
    (agent / "run.py").write_text(payload)
    zip_path = tmp_path / name / "agent.zip"
    sha256 = build_bundle_zip(agent, zip_path)
    return zip_path, sha256


def test_failed_reinstall_keeps_previous_bundle_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed manifest write during reinstall must not replace the old bundle
    bytes that the currently installed manifest still pins."""
    store = tmp_path / "benchmarks"
    old_zip, old_sha = _bundle_source(tmp_path, "old", "old")
    new_zip, new_sha = _bundle_source(tmp_path, "new", "new")

    old_manifest = make_manifest("mybench", bundle=BundleSpec(file="agent.zip", sha256=old_sha))
    install_manifest(old_manifest, store, bundle_src=old_zip)
    old_installed = load_installed("mybench", store)
    assert old_installed is not None and old_installed.agent.bundle is not None
    old_bundle_path = store / old_installed.agent.bundle.file

    original_write_text = Path.write_text

    def fail_manifest_write(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if path == store / "mybench.manifest.yaml":
            raise OSError("disk full")
        return original_write_text(path, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", fail_manifest_write)

    new_manifest = make_manifest("mybench", bundle=BundleSpec(file="agent.zip", sha256=new_sha))
    with pytest.raises(OSError, match="disk full"):
        install_manifest(new_manifest, store, bundle_src=new_zip)

    assert file_sha256(old_bundle_path) == old_sha
    assert load_installed("mybench", store) == old_installed
