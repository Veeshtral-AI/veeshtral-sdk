"""Robustness extras: retries, RunResult helpers, ensure races, marketplace ambiguity."""

from __future__ import annotations

import httpx
import pytest

import veeshtral
from veeshtral.errors import AuthError, CompileError, VeeshtralError
from veeshtral.resources import Agent
from veeshtral.results import RunResult


def test_run_result_helpers():
    assert RunResult(status="completed").ok is True
    assert RunResult(status="pending_review").is_pending_hitl is True
    assert (
        RunResult(
            status="in_progress",
            raw={"has_pending_human_approval": True},
        ).is_pending_hitl
        is True
    )
    assert RunResult(status="paused").is_paused is True
    assert RunResult(status="paused").ok is False
    assert RunResult(status="failed").failed is True
    RunResult(status="completed").raise_for_status()
    with pytest.raises(VeeshtralError, match="failed"):
        RunResult(status="failed", job_id=1).raise_for_status()


def test_client_retries_idempotency_in_progress():
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if hits["n"] < 3:
            return httpx.Response(
                409,
                json={
                    "detail": {
                        "error": "idempotency_request_in_progress",
                        "message": "busy",
                    }
                },
            )
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    assert c.request("GET", "/api/ping") == {"ok": True}
    assert hits["n"] >= 3


def test_client_honors_retry_after():
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if hits["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"detail": "slow"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    assert c.request("GET", "/api/ping") == {"ok": True}


def test_client_auth_error_403():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        return httpx.Response(403, json={"detail": "forbidden"})

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    with pytest.raises(AuthError):
        c.request("GET", "/api/secret")


def test_agent_ensure_409_then_lookup():
    state = {"post": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if path == "/api/business/agents" and request.method == "GET":
            # After conflict, return the winner.
            if state["post"]:
                return httpx.Response(
                    200,
                    json={"items": [{"id": 77, "name": "OCR", "agent_key": "ocr", "status": "active"}]},
                )
            return httpx.Response(200, json={"items": []})
        if path == "/api/business/agents" and request.method == "POST":
            state["post"] += 1
            return httpx.Response(
                409,
                json={"detail": {"type": "byo_agent_key_conflict", "message": "exists"}},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    a = Agent(key="ocr", api_endpoint="http://127.0.0.1:9/v1")
    a.ensure(c)
    assert a.id == 77


def test_agent_ensure_rejects_inactive():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/business/agents":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"id": 3, "name": "OCR", "agent_key": "ocr", "status": "inactive"}
                    ]
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    with pytest.raises(CompileError, match="inactive"):
        Agent(key="ocr", api_endpoint="http://127.0.0.1:9/v1").ensure(c)


def test_marketplace_ambiguous():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/agents":
            return httpx.Response(
                200,
                json=[
                    {"id": 1, "agent_key": "acme", "name": "A", "status": "active"},
                    {"id": 2, "agent_key": "acme", "name": "B", "status": "active"},
                ],
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    with pytest.raises(CompileError, match="multiple marketplace"):
        veeshtral.marketplace.install("acme", client=c)


def test_skill_ensure_matches_key_not_first_item():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/business/skills" and request.method == "GET":
            assert request.url.params.get("source") == "tenant"
            assert request.url.params.get("skill_key") == "target-skill"
            assert request.url.params.get("status") == "active"
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": 2,
                            "skill_key": "target-skill",
                            "name": "Target",
                            "status": "active",
                            "is_platform_catalog": False,
                        },
                    ]
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    sk = veeshtral.Skill(key="target-skill")
    sk.ensure(c)
    assert sk.id == 2


def test_skill_ensure_recovers_422_key_conflict():
    calls = {"get": 0, "post": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/business/skills" and request.method == "GET":
            calls["get"] += 1
            if calls["post"] == 0:
                return httpx.Response(200, json={"items": []})
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": 55,
                            "skill_key": "race-skill",
                            "name": "Race",
                            "status": "active",
                            "is_platform_catalog": False,
                        }
                    ]
                },
            )
        if request.url.path == "/api/business/skills" and request.method == "POST":
            calls["post"] += 1
            return httpx.Response(
                422,
                json={
                    "detail": {
                        "type": "skill_key_conflict",
                        "message": "Invalid or unavailable skill_key",
                    }
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    sk = veeshtral.Skill(key="race-skill")
    sk.ensure(c)
    assert sk.id == 55
    assert calls["post"] == 1


def test_skill_ensure_rejects_draft_twin():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/business/skills" and request.method == "GET":
            # Active lookup misses; conflict recovery scans without status.
            if request.url.params.get("status") == "active":
                return httpx.Response(200, json={"items": []})
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": 9,
                            "skill_key": "draft-skill",
                            "name": "Draft",
                            "status": "draft",
                            "is_platform_catalog": False,
                        }
                    ]
                },
            )
        if request.url.path == "/api/business/skills" and request.method == "POST":
            return httpx.Response(
                422,
                json={
                    "detail": {
                        "type": "skill_key_conflict",
                        "message": "Invalid or unavailable skill_key",
                    }
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    with pytest.raises(CompileError, match="draft"):
        veeshtral.Skill(key="draft-skill").ensure(c)


def test_agent_lookup_rejects_mismatched_key():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/business/agents" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"id": 9, "agent_key": "wrong-key", "name": "Wrong", "status": "active"}
                    ]
                },
            )
        if request.url.path == "/api/business/agents" and request.method == "POST":
            return httpx.Response(
                201, json={"id": 10, "agent_key": "wanted", "name": "wanted"}
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    ag = Agent(key="wanted", api_endpoint="http://127.0.0.1:9/v1")
    ag.ensure(c)
    assert ag.id == 10


def test_rubric_ensure_pages_past_fuzzy_noise():
    pages = {
        0: [{"id": 1, "rubric_key": "other-a", "name": "A", "version": 1}],
        50: [{"id": 2, "rubric_key": "target-rubric", "name": "T", "version": 3}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/business/rubrics" and request.method == "GET":
            offset = int(request.url.params.get("offset") or 0)
            items = pages.get(offset, [])
            # Pad first page to force pagination.
            if offset == 0:
                items = items + [
                    {"id": i, "rubric_key": f"noise-{i}", "name": "n", "version": 1}
                    for i in range(3, 52)
                ]
            return httpx.Response(200, json={"items": items[:50]})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    from veeshtral.resources import QualityRubric

    r = QualityRubric(key="target-rubric")
    r.ensure(c)
    assert r.id == 2
    assert r.version == 3


def test_configure_timeout_accepted():
    c = veeshtral.configure(
        base_url="http://127.0.0.1:8000",
        api_key="vsk_test",
        timeout=12.5,
    )
    assert c._timeout == 12.5
    c.close()
