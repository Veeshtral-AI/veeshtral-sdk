"""Additional coverage for client, run, branch, memory, live, errors."""

from __future__ import annotations

import httpx
import pytest

import veeshtral
from veeshtral.auth import validate_base_url
from veeshtral.client import Client, configure, get_client
from veeshtral.decorators import branch
from veeshtral.errors import ApiError, AuthError, CompileError, GraphRejected
from veeshtral.resources import Agent, LiveSource, Memory, Skill
from veeshtral.workflow import Workflow


def _client(handler) -> Client:
    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    return c


def test_configure_and_get_client():
    c = configure(base_url="http://127.0.0.1:8000", access_token="tok")
    assert get_client() is c
    assert c.credentials.access_token == "tok"
    c.close()


def test_validate_base_url_bad_scheme():
    with pytest.raises(AuthError):
        validate_base_url("ftp://x")


def test_graph_rejected_and_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path.endswith("/draft"):
            return httpx.Response(
                422,
                json={"detail": [{"type": "graph_cycle", "msg": "cycle"}]},
            )
        return httpx.Response(400, json={"detail": {"type": "x", "message": "bad"}})

    c = _client(handler)
    with pytest.raises(GraphRejected) as ge:
        c.request("PATCH", "/api/workflows/1/draft", json={})
    assert ge.value.typ == "graph_cycle"
    with pytest.raises(ApiError) as ae:
        c.request("POST", "/api/workflows", json={})
    assert ae.value.typ == "x"


def test_run_polls_job():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if path == "/api/workflows" and request.method == "GET":
            return httpx.Response(200, json={"items": [], "total": 0})
        if path == "/api/workflows" and request.method == "POST":
            return httpx.Response(201, json={"id": 9, "draft_revision": 1})
        if path.endswith("/publish"):
            return httpx.Response(200, json={"workflow_version_id": 1, "version_number": 1})
        if path.endswith("/run"):
            return httpx.Response(200, json={"job_id": 100, "run_id": "r1"})
        if path == "/api/jobs/100":
            calls["n"] += 1
            # Real JobResponse: status uses JobStatus values; outputs live on workflow_steps.
            if calls["n"] <= 1:
                return httpx.Response(
                    200,
                    json={
                        "status": "in_progress",
                        "has_pending_human_approval": False,
                        "workflow_steps": [],
                    },
                )
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "has_pending_human_approval": False,
                    "workflow_steps": [
                        {
                            "id": 1,
                            "job_id": 100,
                            "agent_id": 1,
                            "step_order": 1,
                            "graph_node_id": "review",
                            "status": "completed",
                            "output_data": (
                                '{"result": 1, "quality_gate_verdict": '
                                '{"passed": false, "on_fail": "hitl"}}'
                            ),
                            "input_data": None,
                            "require_human_approval": False,
                            "cost": 0,
                        }
                    ],
                },
            )
        return httpx.Response(404, json={"detail": path})

    c = _client(handler)
    a = Agent(key="ocr", id=1, api_endpoint="http://127.0.0.1:9/v1")

    @veeshtral.agent(a, task="t")
    def review(invoice_text: str) -> dict: ...

    @veeshtral.workflow_fn(name="run-demo")
    def flow(invoice_text: str):
        return review(invoice_text)

    wf = Workflow.from_flow(flow, client=c)
    result = wf.run(invoice_text="x", poll_timeout_s=5.0)
    assert result.status == "completed"
    assert result.job_id == 100
    assert result.quality_gate_verdict["on_fail"] == "hitl"
    assert result.step_outputs["review"]["result"] == 1


def test_run_stops_on_hitl_while_job_in_progress():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if path == "/api/workflows" and request.method == "GET":
            return httpx.Response(200, json={"items": [{"id": 9, "name": "hitl-demo", "draft_revision": 1}]})
        if path.endswith("/draft") and request.method == "PATCH":
            return httpx.Response(200, json={"id": 9, "draft_revision": 2})
        if path.endswith("/publish"):
            return httpx.Response(200, json={"workflow_version_id": 1, "version_number": 1})
        if path.endswith("/run"):
            return httpx.Response(200, json={"job_id": 101, "run_id": "r-hitl"})
        if path == "/api/jobs/101":
            return httpx.Response(
                200,
                json={
                    "status": "in_progress",
                    "has_pending_human_approval": True,
                    "workflow_steps": [
                        {
                            "id": 2,
                            "job_id": 101,
                            "agent_id": 1,
                            "step_order": 1,
                            "graph_node_id": "escalate",
                            "status": "awaiting_human_approval",
                            "output_data": '{"flag": true}',
                            "input_data": None,
                            "require_human_approval": True,
                            "cost": 0,
                        }
                    ],
                },
            )
        return httpx.Response(404, json={"detail": path})

    c = _client(handler)
    a = Agent(key="ocr", id=1, api_endpoint="http://127.0.0.1:9/v1")

    @veeshtral.agent(a, task="t")
    def review(invoice_text: str) -> dict: ...

    @veeshtral.workflow_fn(name="hitl-demo")
    def flow(invoice_text: str):
        return review(invoice_text)

    wf = Workflow.from_flow(flow, client=c)
    result = wf.run(invoice_text="x", poll_timeout_s=5.0)
    assert result.status == "awaiting_human_approval"
    assert result.is_pending_hitl is True
    assert result.ok is False
    assert result.step_outputs["escalate"]["flag"] is True


def test_memory_and_live_compile():
    a = Agent(key="ocr", id=1, api_endpoint="http://127.0.0.1:9/v1")

    @veeshtral.agent(a, task="t")
    def review(invoice_text: str) -> dict: ...

    @veeshtral.workflow_fn(name="mem-live")
    def flow(invoice_text: str):
        return review(invoice_text)

    wf = Workflow.from_flow(flow)
    wf.attach_memory(Memory(scope="workflow"))
    graph = wf.compile()
    assert any(n["type"] == "memory" for n in graph["nodes"])

    wf2 = Workflow.from_flow(flow)
    wf2.attach_live_source(LiveSource(kind="inbound_webhook"))
    g2 = wf2.compile()
    assert any(n["type"] == "stream_source" for n in g2["nodes"])

    with pytest.raises(CompileError, match="550"):
        Workflow.from_flow(flow).attach_live_source(LiveSource(kind="meeting")).compile()


def test_branch_decorator_marks_edge():
    a = Agent(key="ocr", id=1, api_endpoint="http://127.0.0.1:9/v1")
    b = Agent(key="esc", id=2, api_endpoint="http://127.0.0.1:9/v1")

    @veeshtral.agent(a, task="classify")
    def classify(ticket: str) -> dict: ...

    @branch(condition="ctx.urgent", is_default=False)
    @veeshtral.agent(b, task="escalate")
    def escalate(x: dict) -> dict: ...

    @branch(is_default=True)
    @veeshtral.agent(b, task="auto")
    def auto(x: dict) -> dict: ...

    # linear compile still works; branch meta stamped
    from veeshtral.decorators import get_meta

    assert get_meta(escalate).edge_condition == "ctx.urgent"
    assert get_meta(auto).is_default is True


def test_marketplace_install_success():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/agents":
            return httpx.Response(
                200,
                json=[{"id": 88, "name": "Support", "agent_key": "acme-support"}],
            )
        return httpx.Response(404)

    c = _client(handler)
    ag = veeshtral.marketplace.install("acme-support", client=c)
    assert ag.id == 88
    assert ag._marketplace is True


def test_skill_ensure():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/business/skills" and request.method == "GET":
            return httpx.Response(200, json={"items": []})
        if request.url.path == "/api/business/skills" and request.method == "POST":
            return httpx.Response(201, json={"id": 3, "skill_key": "parse-invoice"})
        return httpx.Response(404)

    c = _client(handler)
    sk = Skill(key="parse-invoice", body_markdown="# hi")
    sk.ensure(c)
    assert sk.id == 3


def test_ambiguous_workflow_name():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/workflows":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"id": 1, "name": "dup"},
                        {"id": 2, "name": "dup"},
                    ]
                },
            )
        return httpx.Response(404)

    c = _client(handler)
    a = Agent(key="ocr", id=1, api_endpoint="http://127.0.0.1:9/v1")

    @veeshtral.agent(a, task="t")
    def review(invoice_text: str) -> dict: ...

    @veeshtral.workflow_fn(name="dup")
    def flow(invoice_text: str):
        return review(invoice_text)

    with pytest.raises(CompileError, match="multiple"):
        Workflow.from_flow(flow, client=c).publish()


def test_context_manager():
    c = Client(base_url="http://127.0.0.1:8000")
    with c:
        assert c.base_url.startswith("http://127.0.0.1")
