from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator


class AgentContract(BaseModel):
    name: str
    run_cmd: str
    install_cmd: str | None = None
    final_output: str | None = None
    secrets: dict[str, str] = {}
    defaults: dict[str, Any] = {}

    @field_validator("run_cmd")
    @classmethod
    def _require_problem_placeholder(cls, v: str) -> str:
        if "{problem_statement_path}" not in v:
            raise ValueError("run_cmd must contain {problem_statement_path}")
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> "AgentContract":
        data = yaml.safe_load(Path(path).read_text())
        if data.get("final_output") is not None:
            data["final_output"] = str(data["final_output"])
        return cls.model_validate(data)


def format_run_cmd(run_cmd: str, kwargs: dict[str, Any]) -> str:
    """Fill {model} and other declared kwargs. Leaves {problem_statement_path}
    and {task_id} as literal placeholders for runtime substitution in-sandbox."""
    out = run_cmd
    for key, value in kwargs.items():
        out = out.replace("{" + key + "}", str(value))
    return out
