"""Behavioral tests for the project-local manifest store."""

from pathlib import Path

from tests.sandbox.conftest import make_manifest
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


def test_pin_diff_reports_changed_pins_only() -> None:
    """Identical manifests diff empty; changed agent image / dataset version /
    versions fields each surface as `field: old → new`."""
    old = make_manifest()
    assert pin_diff(old, make_manifest()) == []

    new = make_manifest(
        image="ghcr.io/vals-ai/agent@sha256:" + "b" * 64,
        dataset_version="v2",
        benchmark_service_version="0.7.0",
    )
    assert pin_diff(old, new) == [
        f"agent.image: ghcr.io/vals-ai/agent@sha256:{'a' * 64} → ghcr.io/vals-ai/agent@sha256:{'b' * 64}",
        "dataset.version: None → v2",
        "versions.benchmark_service: 0.6.1 → 0.7.0",
    ]
