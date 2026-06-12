"""Agent bundle packaging and loading."""

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# In-sandbox directory where bundles extract as BUNDLE_DIR/<root>.
BUNDLE_DIR = "/bundle"

# Never packaged: dev/VCS junk, plus the contract file itself.
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
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _digest_mismatch(zip_path: Path, expected: str, actual: str) -> ValueError:
    return ValueError(
        f"agent bundle {zip_path.name} digest mismatch: manifest pins "
        f"sha256:{expected[:12]}… but the file is sha256:{actual[:12]}…; "
        "re-install the benchmark or re-generate the manifest"
    )


def _excluded(rel: Path) -> bool:
    return any(
        part in _EXCLUDED_NAMES or part.endswith(_EXCLUDED_SUFFIXES) for part in rel.parts
    )


def build_bundle_zip(agent_dir: Path, out_path: Path) -> str:
    """Zip ``agent_dir`` as ``<dirname>/...`` entries and return the zip's sha256."""
    if not agent_dir.is_dir():
        raise ValueError(f"agent bundle source {agent_dir} is not a directory")
    if out_path.resolve().is_relative_to(agent_dir.resolve()):
        raise ValueError(
            f"bundle output {out_path} is inside the agent directory; a rerun would "
            "absorb the previous zip into the bundle — write the manifest and zip elsewhere"
        )
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
    """The single top-level directory inside a bundle zip (the agent name).

    Every entry must be a relative POSIX path under that directory.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    tops: set[str] = set()
    has_top_level_file = False
    for name in names:
        parts = PurePosixPath(name).parts
        if not parts or name.startswith("/") or "\\" in name or ".." in parts:
            raise ValueError(f"bundle zip {zip_path.name} has an unsafe entry path: {name!r}")
        tops.add(parts[0])
        has_top_level_file = has_top_level_file or (len(parts) == 1 and not name.endswith("/"))
    if has_top_level_file or len(tops) != 1:
        raise ValueError(
            f"bundle zip {zip_path.name} must contain exactly one top-level directory "
            f"(found: {sorted(tops)}); re-create it with `benchmark manifest --agent-bundle <dir>`"
        )
    return next(iter(tops))


def verify_bundle(zip_path: Path, expected_sha256: str) -> None:
    """Check a bundle zip against its manifest pin: exists, digest matches, safe layout."""
    if not zip_path.is_file():
        raise ValueError(f"agent bundle {zip_path} does not exist")
    actual = file_sha256(zip_path)
    if actual != expected_sha256:
        raise _digest_mismatch(zip_path, expected_sha256, actual)
    zip_root(zip_path)


def load_bundle(zip_path: Path, expected_sha256: str | None = None) -> AgentBundle:
    """Load a bundle zip for delivery into sandboxes, verifying its digest when given."""
    if not zip_path.is_file():
        raise ValueError(f"agent bundle {zip_path} does not exist")
    zip_bytes = zip_path.read_bytes()
    if expected_sha256 is not None:
        actual = hashlib.sha256(zip_bytes).hexdigest()
        if actual != expected_sha256:
            raise _digest_mismatch(zip_path, expected_sha256, actual)
    return AgentBundle(root=zip_root(zip_path), zip_bytes=zip_bytes)


__all__ = [
    "BUNDLE_DIR",
    "AgentBundle",
    "build_bundle_zip",
    "file_sha256",
    "load_bundle",
    "verify_bundle",
    "zip_root",
]
