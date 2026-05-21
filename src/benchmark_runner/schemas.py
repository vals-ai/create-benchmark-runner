"""Runner-side schemas."""

from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from benchmark_service.schemas import (
    EvaluateResponseRequest,
    FinalScoreResponse,
)


class Task(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    question: str
    timeout: float | None = None


class GenerationStatus(StrEnum):
    SUCCESS = "success"
    MAX_TIME = "max_time"
    MAX_TURNS = "max_turns"
    ERROR = "error"


class GenerationResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str
    status: GenerationStatus
    data: str = Field(validation_alias=AliasChoices("data", "answer"))
    question: str | None = None
    model: str | None = None
    total_turns: int | None = None
    error: str | None = None
    log_dir: str | None = None
    generation_version: str | None = None

    @property
    def answer(self) -> str:
        return self.data


class EvalStatus(StrEnum):
    EVALUATED = "evaluated"
    DID_NOT_COMPLETE = "did_not_complete"
    GENERATION_ERROR = "generation_error"
    ERROR = "error"


class EvalResultData(BaseModel):
    model_config = ConfigDict(extra="allow")
    pass_percentage: float | None = None
    eval_version: str | None = None


class EvalResult(BaseModel):
    task_id: str
    status: EvalStatus
    result: EvalResultData | None = None
    error: str | None = None


class ScoreResult(BaseModel):
    tasks_evaluated: list[str]
    final_score: float
    metadata: dict[str, Any]
    complete: bool = False


__all__ = [
    "EvalResult",
    "EvalResultData",
    "EvalStatus",
    "EvaluateResponseRequest",
    "FinalScoreResponse",
    "GenerationResult",
    "GenerationStatus",
    "ScoreResult",
    "Task",
]
