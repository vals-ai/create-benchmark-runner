import subprocess
import tomllib
from pathlib import Path

from click.testing import CliRunner

from benchmark_runner.scaffolder.main import main


PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "src" / "benchmark_runner" / "templates"


def test_e2e_generated_runner_is_well_formed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli_runner = CliRunner()
    result = cli_runner.invoke(main, ["cyber-bench", "--templates-dir", str(TEMPLATES_DIR)])
    assert result.exit_code == 0, result.output

    out = tmp_path / "cyber-bench-runner"

    pyproject = tomllib.loads((out / "pyproject.toml").read_text())
    assert pyproject["project"]["name"] == "cyber-bench-runner"
    assert pyproject["project"]["scripts"]["cyber-bench-runner"] == "runner.cli:cli"

    dockerfile = (out / "Dockerfile").read_text()
    assert "cyber-bench-runner" in dockerfile
    assert "CYBER_BENCH_GENERATION_VERSION" in dockerfile

    bench = (out / "runner/benchmark.py").read_text()
    assert "class CyberBenchRunner(BenchmarkRunner):" in bench
    assert 'NAME = "cyber-bench"' in bench
    assert 'GENERATION_VERSION_ENV = "CYBER_BENCH_GENERATION_VERSION"' in bench
    assert 'self._dataset = data.get("dataset_name")' in bench
    assert "_register_task" not in bench

    cli_py = (out / "runner/cli.py").read_text()
    assert "from .benchmark import CyberBenchRunner" in cli_py
    assert "make_cli(CyberBenchRunner)" in cli_py

    readme = (out / "README.md").read_text()
    assert "--dataset-name NAME" in readme
    assert "VALS_API_KEY" in readme
    assert "/v1/datasets/{name}/tasks" in readme

    assert (out / "data").is_dir()

    snap = (out / "push_snapshot.py").read_text()
    assert "cyber-bench-runner-pkg" in snap


def test_e2e_python_compile_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli_runner = CliRunner()
    result = cli_runner.invoke(main, ["my-bench", "--templates-dir", str(TEMPLATES_DIR)])
    assert result.exit_code == 0, result.output

    out = tmp_path / "my-bench-runner"
    py_files = list(out.rglob("*.py"))
    assert py_files, "no .py files generated"
    for f in py_files:
        proc = subprocess.run(
            ["python", "-m", "py_compile", str(f)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"py_compile failed for {f}:\n{proc.stderr}"
