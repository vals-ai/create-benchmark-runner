"""Project-local store of installed benchmark manifests.

A lab licenses multiple benchmarks; one orchestrator serves all of them.
`benchmark add` installs a manifest at ``./benchmarks/<name>.manifest.yaml``
(name = the manifest's ``benchmark`` field) and `benchmark run <name>` loads
its pins from there. Pure functions over the Manifest model — no network.
"""

from pathlib import Path

import yaml

from benchmark_runner.sandbox.manifest import Manifest, VersionsSpec

DEFAULT_STORE_DIR = Path("benchmarks")
MANIFEST_SUFFIX = ".manifest.yaml"

# The fields a lab pins a benchmark on: changing any of these on re-add means
# the lab is now running a different artifact/dataset/version combination.
_PIN_FIELDS = ("agent.image", "dataset.name", "dataset.version")


def manifest_path(name: str, store_dir: Path = DEFAULT_STORE_DIR) -> Path:
    return store_dir / f"{name}{MANIFEST_SUFFIX}"


def load_manifest_file(path: Path) -> Manifest:
    """Load and validate a manifest YAML file."""
    return Manifest.model_validate(yaml.safe_load(path.read_text()))


def load_installed(name: str, store_dir: Path = DEFAULT_STORE_DIR) -> Manifest | None:
    """Return the installed manifest for `name`, or None if not installed."""
    path = manifest_path(name, store_dir)
    if not path.exists():
        return None
    return load_manifest_file(path)


def install_manifest(manifest: Manifest, store_dir: Path = DEFAULT_STORE_DIR) -> Path:
    """Write `manifest` into the store (creating it), replacing any prior install."""
    store_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(manifest.benchmark, store_dir)
    path.write_text(yaml.safe_dump(manifest.model_dump(), default_flow_style=False, sort_keys=False))
    return path


def list_installed(store_dir: Path = DEFAULT_STORE_DIR) -> list[Manifest]:
    """All installed manifests, sorted by filename. Empty/missing store → []."""
    if not store_dir.is_dir():
        return []
    return [load_manifest_file(p) for p in sorted(store_dir.glob(f"*{MANIFEST_SUFFIX}"))]


def pin_diff(old: Manifest, new: Manifest) -> list[str]:
    """Changed pins between an installed manifest and its replacement.

    Compares the agent image ref, dataset name/version, and every field of the
    versions block. Returns ``field: old → new`` lines; empty means no pin changes.
    """

    def _get(manifest: Manifest, dotted: str) -> object:
        obj: object = manifest
        for part in dotted.split("."):
            obj = getattr(obj, part)
        return obj

    fields = list(_PIN_FIELDS) + [f"versions.{name}" for name in VersionsSpec.model_fields]
    return [
        f"{field}: {_get(old, field)} → {_get(new, field)}"
        for field in fields
        if _get(old, field) != _get(new, field)
    ]


__all__ = [
    "DEFAULT_STORE_DIR",
    "install_manifest",
    "list_installed",
    "load_installed",
    "load_manifest_file",
    "manifest_path",
    "pin_diff",
]
