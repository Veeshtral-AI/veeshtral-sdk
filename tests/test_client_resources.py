"""Client auth + resource upsert tests (httpx mocked)."""

from __future__ import annotations

import httpx
import pytest

from veeshtral.auth import validate_base_url
from veeshtral.client import Client
from veeshtral.errors import AuthError, CompileError
from veeshtral.resources import Agent


def test_validate_base_url_https_required():
    with pytest.raises(AuthError):
        validate_base_url("http://example.com")
    assert validate_base_url("http://localhost:8000").endswith(":8000")
    assert validate_base_url("https://api.veeshtral.com")


def test_api_key_can_author():
    """API key alone is enough to create workflows (platform accepts X-API-Key)."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["api_key"] = request.headers.get("x-api-key") or ""
        seen["idem"] = request.headers.get("idempotency-key") or ""
        return httpx.Response(201, json={"id": 1, "draft_revision": 1})

    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://127.0.0.1:8000")
    c.credentials.api_key = "secret-key"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    out = c.request("POST", "/api/workflows", json={"name": "x"})
    assert out["id"] == 1
    assert seen["api_key"] == "secret-key"
    assert seen["idem"].startswith("wf-create-")


def test_agent_create_if_missing(httpx_mock=None):
    # Manual transport mock without pytest-httpx
    requests_log: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_log.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/api/business/agents":
            if len([1 for m, p in requests_log if m == "POST"]) == 0:
                return httpx.Response(200, json={"items": [], "total": 0})
            return httpx.Response(
                200,
                json={"items": [{"id": 42, "name": "OCR", "agent_key": "invoice-ocr"}], "total": 1},
            )
        if request.method == "POST" and request.url.path == "/api/business/agents":
            return httpx.Response(201, json={"id": 42, "name": "OCR", "agent_key": "invoice-ocr"})
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        return httpx.Response(404, json={"detail": "missing"})

    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False)

    ag = Agent(
        key="invoice-ocr",
        name="OCR",
        api_endpoint="http://127.0.0.1:9/v1",
        api_key="k",
    )
    ag.ensure(c)
    assert ag.id == 42
    assert ag.api_key is None  # write-once: plaintext cleared after create
    # Second ensure hits GET with existing
    ag2 = Agent(key="invoice-ocr", name="OCR", api_endpoint="http://127.0.0.1:9/v1", api_key="k")
    ag2.ensure(c)
    assert ag2.id == 42
    assert ag2.api_key is None
    assert not any(m == "POST" for m, _ in requests_log[2:])  # no second create after first id set path


def test_agent_missing_endpoint_on_create():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.method == "GET" and request.url.path == "/api/business/agents":
            return httpx.Response(200, json={"items": [], "total": 0})
        return httpx.Response(500, json={"detail": "nope"})

    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    with pytest.raises(CompileError, match="api_endpoint"):
        Agent(key="x").ensure(c)


def test_credentials_repr_redacts():
    from veeshtral.auth import Credentials

    c = Credentials(access_token="secret", api_key="k", email="a@b.com")
    text = repr(c)
    assert "secret" not in text
    assert "***" in text


def test_client_rejects_oversized_response_body():
    from veeshtral.client import _MAX_RESPONSE_BYTES
    from veeshtral.errors import ApiError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * (_MAX_RESPONSE_BYTES + 1))

    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://127.0.0.1:8000")
    c.credentials.api_key = "k"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    with pytest.raises(ApiError, match="response body exceeds"):
        c.request("GET", "/api/workflows")
