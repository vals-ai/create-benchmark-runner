"""Service client construction with env-driven auth resolution.

Precedence: VALS_AUTH_KEY (Descope) wins over BENCHMARK_API_KEY (Bearer).
Adapters never construct the client themselves; the framework's
BenchmarkRunner.__init__ calls build_client.
"""

import os

from benchmark_service.client import BenchmarkServiceClient


def build_client(service_url: str, timeout: int = 300) -> BenchmarkServiceClient:
    """Construct a BenchmarkServiceClient with auth headers resolved from env."""
    headers: dict[str, str] = {}
    descope_key = os.environ.get("VALS_AUTH_KEY")
    bearer_key = os.environ.get("BENCHMARK_API_KEY")
    if descope_key:
        headers["x-descope-api-key"] = descope_key
    elif bearer_key:
        headers["Authorization"] = f"Bearer {bearer_key}"
    return BenchmarkServiceClient(service_url, headers=headers, timeout=timeout)


__all__ = ["build_client"]
