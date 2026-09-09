"""Regression tests for deep-review HIGH fixes (#552 hardening)."""

from __future__ import annotations

import httpx
import pytest

import veeshtral
from veeshtral.auth import Credentials
from veeshtral.client import Client
from veeshtral.errors import ApiError, AuthError, CompileError
from veeshtral.lambda_compile import compile_condition_string
from veeshtral.resources import Agent, LiveSource, QualityRubric, Skill
from veeshtral.workflow import Workflow


def test_validate_base_url_allows_compose_http():
    from veeshtral.auth import validate_base_url

    assert validate_base_url("http://backend:8000") == "http://backend:8000"
    assert validate_base_url("http://host.docker.internal:8000") == "http://host.docker.internal:8000"


def test_find_by_name_filters_exact_when_server_ignores_name_eq():
    """Older servers ignoring name_eq must not produce false ambiguous upserts."""
    import httpx
    from veeshtral.client import Client
    from veeshtral.workflow import Workflow

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/workflows":
            # Simulate pre-#552 list that ignores name_eq.
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"id": 1, "name": "other"},
                        {"id": 2, "name": "target-wf"},
                        {"id": 3, "name": "also-other"},
                    ]
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    wf = Workflow(name="target-wf", client=c, steps=[])
    matches = wf._find_by_name(c)
    assert len(matches) == 1
    assert matches[0]["id"] == 2

    """502 retries must not double-run — JWT path now auto-mints Idempotency-Key."""
    seen: dict[str, str | None] = {"idem": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path.endswith("/run"):
            seen["idem"] = request.headers.get("Idempotency-Key")
            return httpx.Response(200, json={"job_id": 1, "run_id": "r"})
        if request.url.path == "/api/jobs/1":
            return httpx.Response(200, json={"status": "completed", "workflow_steps": []})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://127.0.0.1:8000", credentials=Credentials(access_token="tok"))
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    a = Agent(key="ocr", name="ocr", id=1, api_endpoint="http://127.0.0.1:9/v1")

    @veeshtral.agent(a, task="do")
    def review(text: str) -> dict: ...

    Workflow(name="jwt-run", workflow_id=3, client=c, steps=[review]).run(poll_timeout_s=5)
    assert seen["idem"] and str(seen["idem"]).startswith("run-")


def test_absolute_request_path_rejected():
    c = Client(base_url="http://127.0.0.1:8000", credentials=Credentials(access_token="t"))
    with pytest.raises(AuthError, match="relative"):
        c.request("GET", "https://evil.example/steal")


def test_missing_job_id_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/run"):
            return httpx.Response(200, json={"run_id": "r", "status": "queued"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://127.0.0.1:8000", credentials=Credentials(access_token="t"))
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    a = Agent(key="ocr", name="ocr", id=1, api_endpoint="http://127.0.0.1:9/v1")

    @veeshtral.agent(a, task="do")
    def review(text: str) -> dict: ...

    with pytest.raises(ApiError) as exc:
        Workflow(name="noj", workflow_id=1, client=c, steps=[review]).run(poll_timeout_s=1)
    assert exc.value.typ == "missing_job_id"


def test_hitl_gate_cannot_stack_on_agent():
    a = Agent(key="ocr", name="ocr", id=1, api_endpoint="http://127.0.0.1:9/v1")

    with pytest.raises(CompileError, match="cannot stack @hitl_gate") as exc:

        @veeshtral.hitl_gate()
        @veeshtral.agent(a, task="do")
        def bad(text: str) -> dict: ...

    assert exc.value.code == "decorator_conflict"


def test_string_condition_rejects_pow_and_calls():
    with pytest.raises(CompileError, match="Pow|not allowed"):
        compile_condition_string("ctx.x ** 2")
    with pytest.raises(CompileError, match="function calls"):
        compile_condition_string("ctx.x.lower()")
    assert compile_condition_string("ctx.amount > 100") == "ctx.amount > 100"


def test_live_source_max_events_bounds():
    with pytest.raises(CompileError, match="max_events"):
        LiveSource(max_events=0)
    with pytest.raises(CompileError, match="max_events"):
        LiveSource(max_events=-1)
    assert LiveSource(max_events=5).max_events == 5


def test_rubric_and_skill_reject_nonpositive_ids():
    with pytest.raises(CompileError, match="rubric id"):
        QualityRubric(key="r", rules_text="x", id=0)
    with pytest.raises(CompileError, match="skill id"):
        Skill(key="s", id=-1)


def test_jwt_refresh_latch_resets_after_success():
    hits = {"login": 0, "n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            hits["login"] += 1
            return httpx.Response(200, json={"access_token": f"tok{hits['login']}"})
        hits["n"] += 1
        auth = request.headers.get("Authorization", "")
        # First wave: stale → 401 → fresh. Second wave later: expire again.
        if auth == "Bearer stale":
            return httpx.Response(401, json={"detail": "expired"})
        if auth == "Bearer tok1" and hits["n"] <= 2:
            return httpx.Response(200, json={"ok": 1})
        if auth == "Bearer tok1":
            return httpx.Response(401, json={"detail": "expired-again"})
        if auth == "Bearer tok2":
            return httpx.Response(200, json={"ok": 2})
        return httpx.Response(401, json={"detail": auth})

    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c.credentials.access_token = "stale"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    assert c.request("GET", "/api/a") == {"ok": 1}
    assert hits["login"] == 1
    # Latch cleared after success — second expiry can re-login.
    assert c.request("GET", "/api/b") == {"ok": 2}
    assert hits["login"] == 2
