from pathlib import Path
import pytest
from benchmark_runner.sandbox.contract import AgentContract, format_run_cmd


def test_loads_fields_and_validates_problem_placeholder(tmp_path: Path):
    p = tmp_path / "contract.yaml"
    p.write_text(
        "name: legal_research_agent\n"
        "install_cmd: bash setup.sh\n"
        "run_cmd: >-\n"
        "  legal-research-runner run --model {model} --run-id valkyrie --skip-eval\n"
        "  --results-dir /app/results --problem {problem_statement_path} {task_id}\n"
        "final_output: /app/results/valkyrie\n"
    )
    c = AgentContract.from_yaml(p)
    assert c.name == "legal_research_agent"
    assert c.install_cmd == "bash setup.sh"
    assert c.final_output == "/app/results/valkyrie"
    assert "{problem_statement_path}" in c.run_cmd


def test_run_cmd_must_contain_problem_placeholder(tmp_path: Path):
    p = tmp_path / "contract.yaml"
    p.write_text("name: x\nrun_cmd: my-agent --model {model}\n")
    with pytest.raises(ValueError, match="problem_statement_path"):
        AgentContract.from_yaml(p)


def test_format_run_cmd_fills_model_leaves_runtime_placeholders():
    out = format_run_cmd(
        "a --model {model} --problem {problem_statement_path} {task_id}",
        {"model": "openai/gpt-5"},
    )
    assert out == "a --model openai/gpt-5 --problem {problem_statement_path} {task_id}"
