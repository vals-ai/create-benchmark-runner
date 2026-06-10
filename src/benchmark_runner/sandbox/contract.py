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
    # Lab-facing env var NAMES a lab must supply for the agent to function
    # (e.g. tool API keys). This is the ONLY env list a manifest publishes —
    # `secrets` above maps to Vals-internal secret references and never leaves
    # Vals infra. Declared deliberately by the agent author; default empty so
    # nothing is published by accident. Model access is NOT declared here: labs
    # use CUSTOM_ENDPOINT/CUSTOM_API_KEY, forwarded unconditionally.
    required_env: list[str] = []
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
