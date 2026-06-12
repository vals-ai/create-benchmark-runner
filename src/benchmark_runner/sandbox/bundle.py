"""Agent bundle packaging and loading.

A bundle is a zip of the agent directory with exactly one top-level directory
(the agent name). The orchestrator uploads it into each sandbox, extracts it
to ``/bundle/<name>``, and runs the contract's install_cmd there — the same
layout internal sandboxes use, so contracts whose run_cmd references
``/bundle/<name>`` paths work unchanged on lab infra.
"""

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

# In-sandbox directory bundles extract into; the agent lands at BUNDLE_DIR/<root>.
BUNDLE_DIR = "/bundle"

# Never packaged: dev/VCS junk, plus the contract file itself — its lab-facing
# fields already ship in the manifest, and it may reference internal-only tooling.
_EXCLUDED_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    ".env",
    ".DS_Store",
    "contract.py",
    "contract.yaml",
}
_EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".egg-info")


@dataclass(frozen=True)
class AgentBundle:
    """A loaded bundle ready for in-sandbox install."""

    root: str
    zip_bytes: bytes

    @property
    def install_path(self) -> str:
        return f"{BUNDLE_DIR}/{self.root}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(rel: Path) -> bool:
    return any(
        part in _EXCLUDED_NAMES or part.endswith(_EXCLUDED_SUFFIXES) for part in rel.parts
    )


def build_bundle_zip(agent_dir: Path, out_path: Path) -> str:
    """Zip ``agent_dir`` as ``<dirname>/...`` entries and return the zip's sha256."""
    if not agent_dir.is_dir():
        raise ValueError(f"agent bundle source {agent_dir} is not a directory")
    root = agent_dir.name
    files = sorted(
        p
        for p in agent_dir.rglob("*")
        if p.is_file() and not _excluded(p.relative_to(agent_dir))
    )
    if not files:
        raise ValueError(f"agent dir {agent_dir} has no files to bundle")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=f"{root}/{p.relative_to(agent_dir)}")
    return file_sha256(out_path)


def zip_root(zip_path: Path) -> str:
    """The single top-level directory inside a bundle zip (the agent name)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    tops = {name.split("/", 1)[0] for name in names}
    loose_files = [name for name in names if "/" not in name]
    if loose_files or len(tops) != 1:
        raise ValueError(
            f"bundle zip {zip_path.name} must contain exactly one top-level directory "
            f"(found: {sorted(tops)}); re-create it with `benchmark manifest --agent-bundle <dir>`"
        )
    return next(iter(tops))


def load_bundle(zip_path: Path, expected_sha256: str | None = None) -> AgentBundle:
    """Load a bundle zip for delivery into sandboxes, verifying its digest when given."""
    if not zip_path.is_file():
        raise ValueError(f"agent bundle {zip_path} does not exist")
    if expected_sha256 is not None:
        actual = file_sha256(zip_path)
        if actual != expected_sha256:
            raise ValueError(
                f"agent bundle {zip_path.name} digest mismatch: manifest pins "
                f"sha256:{expected_sha256[:12]}… but the file is sha256:{actual[:12]}…; "
                "re-install the benchmark or re-generate the manifest"
            )
    return AgentBundle(root=zip_root(zip_path), zip_bytes=zip_path.read_bytes())


__all__ = [
    "BUNDLE_DIR",
    "AgentBundle",
    "build_bundle_zip",
    "file_sha256",
    "load_bundle",
    "zip_root",
]
