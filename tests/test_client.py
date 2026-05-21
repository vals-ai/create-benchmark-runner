"""Auth resolution and client builder."""

from benchmark_runner.client import build_client


def test_build_client_resolves_auth_precedence_and_timeout(monkeypatch):
    monkeypatch.delenv("VALS_AUTH_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)

    client = build_client("http://svc", timeout=42)
    assert client._headers == {}
    assert client._timeout == 42

    monkeypatch.setenv("BENCHMARK_API_KEY", "bearer-key")
    client = build_client("http://svc")
    assert client._headers == {"Authorization": "Bearer bearer-key"}

    monkeypatch.setenv("VALS_AUTH_KEY", "descope-key")
    client = build_client("http://svc")
    assert client._headers == {"x-descope-api-key": "descope-key"}
