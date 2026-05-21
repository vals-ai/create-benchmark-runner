"""Scaffolder CLI: validation path. The happy path is covered by tests/test_scaffolder_e2e.py
once templates land in Tasks 13-15."""

from pathlib import Path

from click.testing import CliRunner

from benchmark_runner.scaffolder.main import main


def test_scaffolder_invalid_name_aborts(tmp_path, monkeypatch):
    """Invalid name fails before any template rendering."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    templates_dir = Path(__file__).parent.parent / "src" / "benchmark_runner" / "templates"
    result = runner.invoke(main, ["3-bad", "--templates-dir", str(templates_dir)])
    assert result.exit_code != 0
    assert "must start with a letter" in result.output.lower() or "error" in result.output.lower()
