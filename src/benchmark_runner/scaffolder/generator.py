"""Scaffolder generator: name transforms, validation, Jinja project rendering."""

import re
import shutil
import sys
from pathlib import Path
from typing import TypedDict

from jinja2 import Environment, FileSystemLoader


class BenchmarkNames(TypedDict):
    benchmark_name: str       # "cyber-bench"
    benchmark_camel: str      # "CyberBench"
    benchmark_upper: str      # "CYBER_BENCH"
    benchmark_package: str    # always "runner"


def transform_name(name: str) -> BenchmarkNames:
    """Build the various forms of a benchmark name used in templates."""
    benchmark_name = name.lower().replace("_", "-")
    parts = benchmark_name.split("-")
    benchmark_camel = "".join(p.capitalize() for p in parts)
    benchmark_upper = benchmark_name.replace("-", "_").upper()
    return {
        "benchmark_name": benchmark_name,
        "benchmark_camel": benchmark_camel,
        "benchmark_upper": benchmark_upper,
        "benchmark_package": "runner",
    }


def validate_name(name: str) -> None:
    """Validate a benchmark name. Mirrors create-benchmark-service's rules."""
    if not name:
        raise ValueError("Benchmark name cannot be empty")
    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError(
            "Benchmark name can only contain alphanumeric characters, hyphens, and underscores"
        )
    if not name[0].isalpha():
        raise ValueError("Benchmark name must start with a letter")
    normalized = name.lower().replace("-", "_")
    if normalized in sys.stdlib_module_names:
        raise ValueError(
            f"'{name}' conflicts with Python standard library module '{normalized}'. "
            f"Please choose a different name."
        )


def generate_project(
    *,
    benchmark_name: str,
    output_dir: Path,
    templates_dir: Path,
) -> None:
    """Render every Jinja template in `templates_dir` into `output_dir`.

    File-name semantics:
        foo.jinja           -> foo
        runner/foo.py.jinja -> runner/foo.py
        path/.gitkeep       -> skipped (placeholder)
    """
    validate_name(benchmark_name)
    if output_dir.exists():
        raise FileExistsError(f"Directory {output_dir} already exists.")
    if not templates_dir.is_dir():
        raise FileNotFoundError(f"Templates directory {templates_dir} not found")

    names = transform_name(benchmark_name)
    output_dir.mkdir(parents=True)

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        keep_trailing_newline=True,
    )

    for src in templates_dir.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(templates_dir)
        if src.name == ".gitkeep":
            continue
        if src.suffix == ".jinja":
            dest_rel = rel.with_name(rel.name[: -len(".jinja")])
            dest = output_dir / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            template = env.get_template(str(rel).replace("\\", "/"))
            dest.write_text(template.render(names))
        else:
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    # Always create data/ and tests/ directories for bundled datasets and authored tests.
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "tests").mkdir(exist_ok=True)
    (output_dir / "tests" / "__init__.py").write_text("")


__all__ = ["BenchmarkNames", "generate_project", "transform_name", "validate_name"]
