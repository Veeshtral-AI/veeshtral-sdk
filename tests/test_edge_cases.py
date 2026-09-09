"""Edge / corner cases: branching, memory+live, poll timeout, disambiguation."""

from __future__ import annotations

import time

import httpx
import pytest

import veeshtral
from veeshtral.errors import CompileError, PollTimeout
from veeshtral.resources import Agent, LiveSource, Memory, QualityRubric
from veeshtral.workflow import Workflow


def _agent(key: str, aid: int) -> Agent:
    return Agent(key=key, name=key, id=aid, api_endpoint="http://127.0.0.1:9/v1")


def test_fan_out_diamond_without_branch_decorator():
    ingest = _agent("ingest", 1)
    sent = _agent("sentiment", 2)
    prio = _agent("priority", 3)
    merge = _agent("merge", 4)

    @veeshtral.agent(ingest, task="Ingest")
    def ingest_step(text: str) -> dict: ...

    @veeshtral.agent(sent, task="Sentiment")
    def sentiment(x: dict) -> dict: ...

    @veeshtral.agent(prio, task="Priority")
    def priority(x: dict) -> dict: ...

    @veeshtral.agent(merge, task="Merge")
    def merge_step(a: dict, b: dict) -> dict: ...

    @veeshtral.workflow(name="fanout-demo")
    def flow(text: str):
        root = ingest_step(text)
        s = sentiment(root)
        p = priority(root)
        return merge_step(s, p)

    graph = Workflow.from_flow(flow).compile()
    ingest_node = next(n for n in graph["nodes"] if n["id"].startswith("ingest"))
    assert ingest_node["config"].get("branch_mode") == "fan_out"
    assert not any(
        e.get("is_default") or e.get("condition")
        for e in graph["edges"]
        if e.get("source") == ingest_node["id"]
    )


def test_first_match_with_lambda_branch_arms():
    classify = _agent("classify", 10)
    esc = _agent("escalate", 11)
    auto = _agent("auto", 12)

    @veeshtral.agent(classify, task="Classify")
    def classify_step(text: str) -> dict: ...

    @veeshtral.branch(condition=lambda ctx: ctx.needs_escalation)
    @veeshtral.agent(esc, task="Escalate")
    def escalate(c: dict) -> dict: ...

    @veeshtral.branch(is_default=True)
    @veeshtral.agent(auto, task="Auto")
    def auto_step(c: dict) -> dict: ...

    @veeshtral.workflow(name="first-match-demo")
    def flow(text: str):
        c = classify_step(text)
        escalate(c)
        auto_step(c)

    graph = Workflow.from_flow(flow).compile()
    parent = next(n for n in graph["nodes"] if n["id"].startswith("classify"))
    assert parent["config"]["branch_mode"] == "first_match"
    outs = [e for e in graph["edges"] if e.get("source") == parent["id"] and not e.get("wire_role")]
    assert len(outs) == 2
    assert sum(1 for e in outs if e.get("is_default") is True) == 1
    assert any("ctx.needs_escalation" in str(e.get("condition") or "") for e in outs)


def test_first_match_if_else_body_traced():
    classify = _agent("classify", 10)
    esc = _agent("escalate", 11)
    auto = _agent("auto", 12)

    @veeshtral.agent(classify, task="Classify")
    def classify_step(text: str) -> dict: ...

    @veeshtral.branch(condition="ctx.needs_escalation")
    @veeshtral.agent(esc, task="Escalate")
    def escalate(c: dict) -> dict: ...

    @veeshtral.branch(is_default=True)
    @veeshtral.agent(auto, task="Auto")
    def auto_step(c: dict) -> dict: ...

    @veeshtral.workflow(name="if-else-demo")
    def flow(text: str):
        c = classify_step(text)
        if True:
            escalate(c)
        else:
            auto_step(c)

    graph = Workflow.from_flow(flow).compile()
    parent = next(n for n in graph["nodes"] if n["id"].startswith("classify"))
    assert parent["config"]["branch_mode"] == "first_match"


def test_memory_and_live_single_spine():
    a = _agent("ocr", 5)

    @veeshtral.agent(a, task="Extract")
    def review(text: str) -> dict: ...

    @veeshtral.workflow(name="mem-live")
    def flow(text: str):
        return review(text)

    wf = Workflow.from_flow(flow)
    wf.attach_memory(Memory())
    wf.attach_live_source(LiveSource(kind="inbound_webhook"))
    graph = wf.compile()
    ids = {n["id"] for n in graph["nodes"]}
    assert "stream-1" in ids and "memory-1" in ids
    # Live path: trigger → stream → agent (feeds_agent_node_id); memory is scope-only.
    assert {"source": "trigger-1", "target": "stream-1"} in [
        {k: e[k] for k in ("source", "target")} for e in graph["edges"]
    ]
    assert {"source": "stream-1", "target": "review"} in [
        {k: e[k] for k in ("source", "target")} for e in graph["edges"]
    ]
    assert any(
        e.get("source") == "memory-1"
        and e.get("target") == "review"
        and e.get("wire_role") == "scope"
        for e in graph["edges"]
    )
    assert not any(
        e.get("source") == "stream-1" and e.get("target") == "memory-1" for e in graph["edges"]
    )


def test_qg_scope_rejects_hitl():
    a = _agent("ocr", 5)
    rubric = QualityRubric(key="safe", rules_text="x", on_fail="hitl", id=1)

    @veeshtral.agent(a, task="Extract")
    def review(text: str) -> dict: ...

    @veeshtral.hitl_gate(condition=lambda ctx: ctx.flag)
    def escalate(r: dict): ...

    @veeshtral.workflow(name="qg-hitl")
    def flow(text: str):
        return escalate(review(text))

    wf = Workflow.from_flow(flow)
    wf.attach_quality_gate(rubric, scope=[review, escalate])
    with pytest.raises(CompileError, match="agent steps"):
        wf.compile()


def test_publish_respects_workflow_id_disambiguation():
    state = {"patched": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if path == "/api/workflows" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"id": 7, "name": "dup"},
                        {"id": 8, "name": "dup"},
                    ]
                },
            )
        if path == "/api/workflows/8" and request.method == "GET":
            return httpx.Response(200, json={"id": 8, "draft_revision": 2})
        if path == "/api/workflows/8/draft" and request.method == "PATCH":
            state["patched"] += 1
            return httpx.Response(200, json={"id": 8, "draft_revision": 3})
        if path == "/api/workflows/8/publish" and request.method == "POST":
            return httpx.Response(200, json={"workflow_version_id": 9, "version_number": 1})
        return httpx.Response(404, json={"detail": path})

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    a = _agent("ocr", 1)

    @veeshtral.agent(a, task="do")
    def review(invoice_text: str) -> dict: ...

    @veeshtral.workflow(name="dup")
    def flow(invoice_text: str):
        return review(invoice_text)

    with pytest.raises(CompileError, match="workflow_id"):
        Workflow.from_flow(flow, client=c).publish()

    pub = Workflow.from_flow(flow, client=c).publish(workflow_id=8)
    assert pub.workflow_id == 8
    assert state["patched"] == 1


def test_create_409_recovers_via_name_eq():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(f"{request.method} {path}")
        if path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if path == "/api/workflows" and request.method == "GET":
            # First lookup empty, after 409 return the winner.
            if any(c.startswith("POST") for c in calls):
                return httpx.Response(200, json={"items": [{"id": 42, "name": "race"}]})
            return httpx.Response(200, json={"items": []})
        if path == "/api/workflows" and request.method == "POST":
            return httpx.Response(
                409,
                json={"detail": {"error": "idempotency_request_in_progress", "message": "busy"}},
            )
        if path == "/api/workflows/42" and request.method == "GET":
            return httpx.Response(200, json={"id": 42, "draft_revision": 1})
        if path == "/api/workflows/42/draft" and request.method == "PATCH":
            return httpx.Response(200, json={"id": 42, "draft_revision": 2})
        if path == "/api/workflows/42/publish" and request.method == "POST":
            return httpx.Response(200, json={"workflow_version_id": 1, "version_number": 1})
        return httpx.Response(404, json={"detail": path})

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    a = _agent("ocr", 1)

    @veeshtral.agent(a, task="do")
    def review(invoice_text: str) -> dict: ...

    @veeshtral.workflow(name="race")
    def flow(invoice_text: str):
        return review(invoice_text)

    pub = Workflow.from_flow(flow, client=c).publish()
    assert pub.workflow_id == 42


def test_poll_timeout_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if path.endswith("/run") and request.method == "POST":
            return httpx.Response(200, json={"job_id": 99, "run_id": "r1", "status": "queued"})
        if path == "/api/jobs/99":
            return httpx.Response(200, json={"id": 99, "status": "running"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = veeshtral.Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    wf = Workflow(name="x", workflow_id=1, client=c, steps=[])
    # Bypass empty compile by setting id only and monkeypatching publish not called
    a = _agent("ocr", 1)

    @veeshtral.agent(a, task="do")
    def review(invoice_text: str) -> dict: ...

    wf._steps = [review]
    with pytest.raises(PollTimeout, match="still 'running'"):
        wf.run(poll_timeout_s=0.35)


def test_empty_workflow_errors_before_network():
    calls: list[str] = []

    class FakeClient:
        credentials = type("C", (), {"access_token": "t", "api_key": None})()

        def request(self, *a, **k):
            calls.append("hit")
            raise AssertionError("network should not be called")

    with pytest.raises(CompileError, match="flow or steps"):
        Workflow(name="empty", client=FakeClient()).compile()  # type: ignore[arg-type]
    assert calls == []


def test_duplicate_step_function_names_rejected():
    a = _agent("ocr", 1)
    b = _agent("match", 2)

    @veeshtral.agent(a, task="A")
    def process(x: str) -> dict: ...

    @veeshtral.agent(b, task="B")
    def process_b(x: dict) -> dict: ...

    # Force colliding StepMeta.name (factory smell) while keeping distinct callables.
    from veeshtral.decorators import StepMeta, _META_ATTR, get_meta

    meta_b = get_meta(process_b)
    assert isinstance(meta_b, StepMeta)
    setattr(
        process_b,
        _META_ATTR,
        StepMeta(
            kind="agent",
            fn=process_b,
            name="process",
            resource=b,
            task="B",
        ),
    )

    with pytest.raises(CompileError, match="duplicate step function name"):
        Workflow(name="dup", steps=[process, process_b]).compile()


def test_agent_above_branch_preserves_branch_fields():
    a = _agent("risk", 3)
    b = _agent("hitl-arm", 4)
    c = _agent("auto-arm", 5)

    @veeshtral.agent(a, task="Score")
    def score(text: str) -> dict: ...

    # Natural reading order: @agent then @branch (applied bottom-up as branch→agent).
    @veeshtral.agent(b, task="Escalate")
    @veeshtral.branch(condition=lambda ctx: ctx.high_risk)
    def escalate(s: dict) -> dict: ...

    @veeshtral.agent(c, task="Auto")
    @veeshtral.branch(is_default=True)
    def auto(s: dict) -> dict: ...

    @veeshtral.workflow(name="branch-order")
    def flow(text: str):
        s = score(text)
        if True:
            escalate(s)
        else:
            auto(s)

    graph = Workflow.from_flow(flow).compile()
    parent = next(n for n in graph["nodes"] if n["id"].startswith("score"))
    assert parent["config"]["branch_mode"] == "first_match"
    cond_edges = [e for e in graph["edges"] if e.get("condition")]
    assert any("ctx.high_risk" in str(e.get("condition")) for e in cond_edges)
    assert any(e.get("is_default") for e in graph["edges"])


def test_bad_key_trailing_separator():
    with pytest.raises(CompileError, match="alphanumeric"):
        Agent(key="bad-", api_endpoint="http://127.0.0.1:9/v1")
