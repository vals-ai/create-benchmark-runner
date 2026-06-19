"""Service client construction."""

import os

from benchmark_service.client import BenchmarkServiceClient


def auth_headers() -> dict[str, str]:
    """Build auth headers from environment variables.

    Prefers Descope (VALS_API_KEY → x-descope-api-key) over legacy bearer
    (BENCHMARK_API_KEY → Authorization: Bearer).
    """
    headers: dict[str, str] = {}
    descope_key = os.environ.get("VALS_API_KEY")
    bearer_key = os.environ.get("BENCHMARK_API_KEY")
    if descope_key:
        headers["x-descope-api-key"] = descope_key
    elif bearer_key:
        headers["Authorization"] = f"Bearer {bearer_key}"
    return headers


def build_client(service_url: str, timeout: int = 300) -> BenchmarkServiceClient:
    return BenchmarkServiceClient(service_url, headers=auth_headers(), timeout=timeout)


__all__ = ["auth_headers", "build_client"]
