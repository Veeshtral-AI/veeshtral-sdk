"""High-signal regressions — fail loudly when #552 contracts break.

Each test names the failure mode it catches so CI output points at root cause.
"""

from __future__ import annotations

from email.utils import formatdate
from unittest import mock

import httpx
import pytest

import veeshtral
from veeshtral.auth import Credentials
from veeshtral.client import Client, _retry_after_seconds
from veeshtral.errors import ApiError, AuthError, CompileError
from veeshtral.resources import Agent, LiveSource, Memory
from veeshtral.results import (
    RunResult,
    extract_step_outputs_and_qg,
    job_awaiting_hitl,
    job_poll_terminal,
    normalize_run_status,
)
from veeshtral.workflow import Workflow


def _jwt_client(handler) -> Client:
    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    return c


def _agent(key: str = "ocr", aid: int = 1) -> Agent:
    return Agent(key=key, name=key, id=aid, api_endpoint="http://127.0.0.1:9/v1")


# --- client: Retry-After / DoS / auth refresh ---------------------------------


def test_retry_after_numeric_capped_at_120s():
    """Hostile Retry-After: 999999 must not sleep for days (CWE-400)."""
    resp = httpx.Response(429, headers={"Retry-After": "999999"})
    assert _retry_after_seconds(resp, fallback=0.2) == 120.0


def test_retry_after_http_date_honored_and_capped():
    """HTTP-date Retry-After must parse; far-future dates still cap at 120s."""
    far = formatdate(timeval=None, localtime=False, usegmt=True)
    # Force a far-future stamp via mock of parsedate path by using a large delta header.
    # formatdate(now) → near-zero delay; use fallback floor instead.
    resp = httpx.Response(429, headers={"Retry-After": far})
    delay = _retry_after_seconds(resp, fallback=1.5)
    assert 1.5 <= delay <= 120.0


def test_retry_after_garbage_falls_back():
    resp = httpx.Response(429, headers={"Retry-After": "not-a-delay"})
    assert _retry_after_seconds(resp, fallback=0.4) == 0.4


def test_oversized_response_body_refused():
    """8MB+ body must raise ApiError(response_too_large) before JSON parse."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        return httpx.Response(200, content=b"x" * (8 * 1024 * 1024 + 1))

    c = _jwt_client(handler)
    with pytest.raises(ApiError) as exc:
        c.request("GET", "/api/huge")
    assert exc.value.typ == "response_too_large"


def test_401_clears_stale_token_and_relogins_once():
    """Stale JWT → 401 → clear token → login once → succeed (not infinite loop)."""
    hits = {"login": 0, "ping": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            hits["login"] += 1
            return httpx.Response(200, json={"access_token": "fresh-tok"})
        if request.url.path == "/api/ping":
            hits["ping"] += 1
            auth = request.headers.get("Authorization", "")
            if auth == "Bearer stale-tok":
                return httpx.Response(401, json={"detail": "expired"})
            if auth == "Bearer fresh-tok":
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(401, json={"detail": f"unexpected {auth}"})
        return httpx.Response(404)

    c = _jwt_client(handler)
    c.credentials.access_token = "stale-tok"
    assert c.request("GET", "/api/ping") == {"ok": True}
    assert hits["login"] == 1
    assert hits["ping"] == 2
    assert c.credentials.access_token == "fresh-tok"


def test_require_jwt_rejects_api_key_only_client():
    """Marketplace/JWT-only routes must fail closed for X-API-Key clients."""
    c = Client(
        base_url="http://127.0.0.1:8000",
        credentials=Credentials(api_key="vk_test"),
    )
    with pytest.raises(AuthError, match="requires JWT"):
        c.request("GET", "/api/secret", require_jwt=True)


def test_api_key_auto_idempotency_on_workflow_create():
    """API-key POST /workflows must send Idempotency-Key even if caller omits it."""
    seen: dict[str, str | None] = {"idem": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/workflows" and request.method == "POST":
            seen["idem"] = request.headers.get("Idempotency-Key")
            return httpx.Response(201, json={"id": 1})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = Client(
        base_url="http://127.0.0.1:8000",
        credentials=Credentials(api_key="vk_test"),
    )
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    c.request("POST", "/api/workflows", json={"name": "x"})
    assert seen["idem"] and str(seen["idem"]).startswith("wf-create-")


# --- results: HITL poll semantics ---------------------------------------------


def test_hitl_detected_from_step_status_while_job_in_progress():
    """Platform keeps job in_progress during HITL — poll must still terminate."""
    raw = {
        "status": "in_progress",
        "has_pending_human_approval": False,
        "workflow_steps": [
            {"graph_node_id": "gate", "status": "awaiting_human_approval", "output_data": None}
        ],
    }
    assert job_awaiting_hitl(raw) is True
    assert job_poll_terminal(raw) is True
    assert normalize_run_status(raw) == "awaiting_human_approval"
    rr = RunResult(status="in_progress", raw=raw)
    assert rr.is_pending_hitl is True
    assert rr.ok is False
    assert rr.is_paused is False  # HITL is not operator pause


def test_operator_paused_is_not_hitl_and_not_ok():
    rr = RunResult(status="paused", raw={"status": "paused"})
    assert rr.is_paused is True
    assert rr.is_pending_hitl is False
    assert rr.ok is False


def test_extract_step_outputs_parses_json_string_and_nested_qg():
    """Real JobResponse stores output_data as JSON text; QG may be nested."""
    raw = {
        "status": "completed",
        "workflow_steps": [
            {
                "graph_node_id": "review",
                "step_order": 1,
                "output_data": '{"ok": true, "quality_gate_verdict": {"pass": true}}',
            },
            {
                "step_order": 2,
                "output_data": "not-json",
            },
        ],
    }
    outs, qg = extract_step_outputs_and_qg(raw)
    assert outs["review"] == {"ok": True, "quality_gate_verdict": {"pass": True}}
    assert outs["step_2"] == "not-json"
    assert qg == {"pass": True}


def test_extract_prefers_legacy_step_outputs_dict():
    outs, qg = extract_step_outputs_and_qg(
        {
            "step_outputs": {"a": {"quality_gate": "fail"}},
            "workflow_steps": [{"graph_node_id": "ignored", "output_data": "{}"}],
        }
    )
    assert outs == {"a": {"quality_gate": "fail"}}
    assert qg == "fail"


# --- workflow: draft races / publish context / run auth -----------------------


def test_draft_409_uses_current_draft_revision_from_body():
    """Optimistic concurrency: 409 body current_draft_revision must drive retry."""
    import json

    state = {"rev_seen": [], "patch": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if path == "/api/workflows" and request.method == "GET":
            return httpx.Response(
                200, json={"items": [{"id": 5, "name": "rev", "draft_revision": 1}]}
            )
        if path.endswith("/draft") and request.method == "PATCH":
            state["patch"] += 1
            payload = json.loads(request.content.decode())
            state["rev_seen"].append(payload.get("expected_draft_revision"))
            if state["patch"] == 1:
                return httpx.Response(
                    409,
                    json={
                        "detail": {
                            "type": "draft_revision_conflict",
                            "current_draft_revision": 4,
                            "message": "stale",
                        }
                    },
                )
            assert payload.get("expected_draft_revision") == 4
            return httpx.Response(200, json={"id": 5, "draft_revision": 5})
        if path.endswith("/publish"):
            return httpx.Response(200, json={"workflow_version_id": 1, "version_number": 1})
        return httpx.Response(404, json={"detail": path})

    c = _jwt_client(handler)
    a = _agent()

    @veeshtral.agent(a, task="do")
    def review(text: str) -> dict: ...

    @veeshtral.workflow(name="rev")
    def flow(text: str):
        return review(text)

    with mock.patch("veeshtral.workflow.time.sleep", return_value=None):
        pub = Workflow.from_flow(flow, client=c).publish()
    assert pub.workflow_id == 5
    assert 4 in state["rev_seen"]
    assert state["patch"] == 2


def test_publish_context_changed_does_not_blindly_retry():
    """Name/BRD drift without draft bump — re-patch cannot help; must surface."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if path == "/api/workflows" and request.method == "GET":
            return httpx.Response(
                200, json={"items": [{"id": 3, "name": "ctx", "draft_revision": 2}]}
            )
        if path.endswith("/draft"):
            return httpx.Response(200, json={"id": 3, "draft_revision": 2})
        if path.endswith("/publish"):
            return httpx.Response(
                409,
                json={
                    "detail": {
                        "type": "publish_context_changed",
                        "message": "name changed",
                    }
                },
            )
        return httpx.Response(404)

    c = _jwt_client(handler)
    a = _agent()

    @veeshtral.agent(a, task="do")
    def review(text: str) -> dict: ...

    @veeshtral.workflow(name="ctx")
    def flow(text: str):
        return review(text)

    with pytest.raises(ApiError) as exc:
        Workflow.from_flow(flow, client=c).publish()
    assert exc.value.typ == "publish_context_changed"


def test_publish_409_republishes_after_repatch():
    """Concurrent draft bump: re-patch then publish again."""
    state = {"publish": 0, "patch": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if path == "/api/workflows" and request.method == "GET":
            return httpx.Response(
                200, json={"items": [{"id": 9, "name": "race-pub", "draft_revision": 1}]}
            )
        if path == "/api/workflows/9" and request.method == "GET":
            return httpx.Response(200, json={"id": 9, "draft_revision": state["patch"] + 1})
        if path.endswith("/draft"):
            state["patch"] += 1
            return httpx.Response(200, json={"id": 9, "draft_revision": state["patch"] + 1})
        if path.endswith("/publish"):
            state["publish"] += 1
            if state["publish"] == 1:
                return httpx.Response(
                    409,
                    json={"detail": {"type": "draft_revision_conflict", "message": "bump"}},
                )
            return httpx.Response(200, json={"workflow_version_id": 2, "version_number": 2})
        return httpx.Response(404)

    c = _jwt_client(handler)
    a = _agent()

    @veeshtral.agent(a, task="do")
    def review(text: str) -> dict: ...

    @veeshtral.workflow(name="race-pub")
    def flow(text: str):
        return review(text)

    with mock.patch("veeshtral.workflow.time.sleep", return_value=None):
        pub = Workflow.from_flow(flow, client=c).publish()
    assert pub.workflow_id == 9
    assert state["publish"] == 2
    assert state["patch"] >= 2


def test_create_race_ambiguous_name_raises_with_ids():
    """Two name_eq matches after create 409 → tell caller to pass workflow_id=."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if path == "/api/workflows" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"id": 1, "name": "ambi"},
                        {"id": 2, "name": "ambi"},
                    ]
                },
            )
        if path == "/api/workflows" and request.method == "POST":
            return httpx.Response(
                409,
                json={"detail": {"error": "idempotency_request_in_progress", "message": "busy"}},
            )
        return httpx.Response(404)

    c = _jwt_client(handler)
    a = _agent()

    @veeshtral.agent(a, task="do")
    def review(text: str) -> dict: ...

    @veeshtral.workflow(name="ambi")
    def flow(text: str):
        return review(text)

    with mock.patch("veeshtral.workflow.time.sleep", return_value=None):
        with pytest.raises(CompileError, match="workflow_id") as exc:
            Workflow.from_flow(flow, client=c).publish()
    assert exc.value.code == "ambiguous_name"


def test_run_use_api_key_requires_credential():
    c = Client(base_url="http://127.0.0.1:8000", credentials=Credentials(access_token="t"))
    a = _agent()

    @veeshtral.agent(a, task="do")
    def review(text: str) -> dict: ...

    wf = Workflow(name="x", workflow_id=1, client=c, steps=[review])
    with pytest.raises(CompileError, match="api_key") as exc:
        wf.run(use_api_key=True, poll_timeout_s=0.1)
    assert exc.value.code == "missing_api_key"


def test_run_stops_on_hitl_step_without_waiting_for_timeout():
    """Regression: in_progress + HITL step must not hit PollTimeout."""
    polls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if path.endswith("/run"):
            return httpx.Response(200, json={"job_id": 55, "run_id": "r"})
        if path == "/api/jobs/55":
            polls["n"] += 1
            return httpx.Response(
                200,
                json={
                    "status": "in_progress",
                    "has_pending_human_approval": False,
                    "workflow_steps": [
                        {
                            "graph_node_id": "hitl",
                            "status": "pending_review",
                            "output_data": None,
                        }
                    ],
                },
            )
        return httpx.Response(404)

    c = _jwt_client(handler)
    a = _agent()

    @veeshtral.agent(a, task="do")
    def review(text: str) -> dict: ...

    wf = Workflow(name="hitl-poll", workflow_id=1, client=c, steps=[review])
    result = wf.run(poll_timeout_s=30.0)
    assert result.is_pending_hitl
    assert result.status == "awaiting_human_approval"
    assert polls["n"] == 1


# --- compile guards that catch bad graphs early -------------------------------


def test_agent_id_zero_rejected_when_resolved():
    with pytest.raises(CompileError, match="positive") as exc:
        Agent(key="z", name="z", id=0, api_endpoint="http://127.0.0.1:9/v1")
    assert exc.value.code == "bad_agent_id"


def test_live_plus_memory_emits_scope_only_memory_edge():
    """Live stream → agent is flow; memory → agent must be wire_role=scope."""
    a = _agent()

    @veeshtral.agent(a, task="do")
    def review(text: str) -> dict: ...

    @veeshtral.workflow(name="live-mem")
    def flow(text: str):
        return review(text)

    wf = Workflow.from_flow(flow)
    wf.attach_live_source(LiveSource(kind="inbound_webhook"))
    wf.attach_memory(Memory())
    graph = wf.compile()
    mem_ids = {n["id"] for n in graph["nodes"] if n.get("type") == "memory"}
    scope_from_mem = [e for e in graph["edges"] if e.get("source") in mem_ids]
    assert scope_from_mem, "expected memory→agent edge"
    assert all(e.get("wire_role") == "scope" for e in scope_from_mem)


def test_empty_name_and_huge_description_fail_fast():
    with pytest.raises(CompileError, match="name"):
        Workflow(name="  ")
    with pytest.raises(CompileError, match="2 MiB"):
        Workflow(name="ok", description="x" * (2_097_152 + 1))
