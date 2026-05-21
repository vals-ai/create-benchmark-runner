"""benchmark_runner: shared runner library for Vals benchmarks.

Public API:
    BenchmarkRunner, make_cli, build_client, build_llm_config,
    RunArtifacts, Task, GenerationResult, GenerationStatus,
    EvalResult, EvalResultData, EvalStatus, ScoreResult,
    EvaluateResponseRequest, FinalScoreResponse  (re-exported from create-benchmark-service)
"""

from benchmark_runner.artifacts import RunArtifacts
from benchmark_runner.base import BenchmarkRunner
from benchmark_runner.cli import make_cli
from benchmark_runner.client import build_client
from benchmark_runner.llm import build_llm_config
from benchmark_runner.schemas import (
    EvalResult,
    EvalResultData,
    EvalStatus,
    EvaluateResponseRequest,
    FinalScoreResponse,
    GenerationResult,
    GenerationStatus,
    ScoreResult,
    Task,
)

__all__ = [
    "BenchmarkRunner",
    "EvalResult",
    "EvalResultData",
    "EvalStatus",
    "EvaluateResponseRequest",
    "FinalScoreResponse",
    "GenerationResult",
    "GenerationStatus",
    "RunArtifacts",
    "ScoreResult",
    "Task",
    "build_client",
    "build_llm_config",
    "make_cli",
]
