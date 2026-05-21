"""Assemble an LLMConfig from the CLI's sampling/provider flags.

Replaces the ~25 lines of LLMConfig assembly duplicated between fabv2-runner
and legal-research-runner. CLI options surfaced by `make_cli()` flow into
this function via keyword.
"""

from typing import Any

from model_library.base import LLMConfig
from model_library.providers.openai import OpenAIConfig
from pydantic import SecretStr


def build_llm_config(
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    reasoning_effort: str | None = None,
    custom_endpoint: str | None = None,
    custom_api_key: str | None = None,
    chat_completions: bool = False,
    disable_streaming: bool = False,
) -> LLMConfig:
    """Build an LLMConfig from CLI-level kwargs."""
    kwargs: dict[str, Any] = {}
    for field, val in [
        ("max_tokens", max_tokens),
        ("temperature", temperature),
        ("top_p", top_p),
        ("top_k", top_k),
        ("reasoning_effort", reasoning_effort),
    ]:
        if val is not None:
            kwargs[field] = val
    if custom_endpoint:
        kwargs["custom_endpoint"] = custom_endpoint
    if custom_api_key:
        kwargs["custom_api_key"] = SecretStr(custom_api_key)
    if chat_completions:
        kwargs["native"] = False
    if disable_streaming:
        kwargs["provider_config"] = OpenAIConfig(stream_completions=False)
    return LLMConfig(**kwargs)


__all__ = ["build_llm_config"]
