import pytest

from benchmark_runner.scaffolder.generator import generate_project, transform_name, validate_name


def test_transform_name_normalizes_template_forms():
    assert transform_name("cyber-bench") == {
        "benchmark_name": "cyber-bench",
        "benchmark_camel": "CyberBench",
        "benchmark_upper": "CYBER_BENCH",
        "benchmark_package": "runner",
    }
    assert transform_name("cyber_bench")["benchmark_name"] == "cyber-bench"
    assert transform_name("swe-bench-lite")["benchmark_camel"] == "SweBenchLite"


def test_validate_name_accepts_project_names_and_rejects_invalid_ones():
    for name in ("cyber-bench", "cyber_bench", "CyberBench42"):
        validate_name(name)

    for name in ("", "bad name!", "3-things", "json"):
        with pytest.raises(ValueError):
            validate_name(name)


def test_generate_project_creates_expected_files(tmp_path, monkeypatch):
    templates_dir = tmp_path / "fake_templates"
    templates_dir.mkdir()
    (templates_dir / "pyproject.toml.jinja").write_text('name = "{{ benchmark_name }}-runner"\n')
    (templates_dir / "Dockerfile.jinja").write_text("FROM python:3.12-slim\nENTRYPOINT [\"{{ benchmark_name }}-runner\"]\n")
    (templates_dir / "push_snapshot.py.jinja").write_text('# {{ benchmark_name }} snapshot\n')
    (templates_dir / "Makefile.jinja").write_text('docker-build:\n\tdocker build -t {{ benchmark_name }}-runner .\n')
    (templates_dir / "README.md.jinja").write_text("# {{ benchmark_name }}-runner\n")
    (templates_dir / ".gitignore.jinja").write_text(".venv/\n")
    runner_subdir = templates_dir / "runner"
    runner_subdir.mkdir()
    (runner_subdir / "__init__.py.jinja").write_text('"""{{ benchmark_name }}."""\n')
    (runner_subdir / "cli.py.jinja").write_text(
        'from benchmark_runner.cli import make_cli\nfrom .benchmark import {{ benchmark_camel }}Runner\n'
        'cli = make_cli({{ benchmark_camel }}Runner)\n'
    )
    (runner_subdir / "benchmark.py.jinja").write_text(
        'class {{ benchmark_camel }}Runner:\n    NAME = "{{ benchmark_name }}"\n'
        '    GENERATION_VERSION_ENV = "{{ benchmark_upper }}_GENERATION_VERSION"\n'
    )

    out_dir = tmp_path / "cyber-bench-runner"
    generate_project(
        benchmark_name="cyber-bench",
        output_dir=out_dir,
        templates_dir=templates_dir,
    )

    for f in (
        "pyproject.toml", "Dockerfile", "push_snapshot.py", "Makefile",
        "README.md", ".gitignore",
        "runner/__init__.py", "runner/cli.py", "runner/benchmark.py",
    ):
        assert (out_dir / f).exists(), f"missing {f}"

    assert "CyberBench" in (out_dir / "runner/benchmark.py").read_text()
    assert "CYBER_BENCH_GENERATION_VERSION" in (out_dir / "runner/benchmark.py").read_text()


def test_generate_project_refuses_existing_dir(tmp_path):
    out_dir = tmp_path / "already-here"
    out_dir.mkdir()
    with pytest.raises(FileExistsError):
        generate_project(
            benchmark_name="x",
            output_dir=out_dir,
            templates_dir=tmp_path / "templates",
        )
