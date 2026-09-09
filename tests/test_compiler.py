"""Compiler + decorator integration tests."""

from __future__ import annotations

import time

import pytest

import veeshtral
from veeshtral.errors import CompileError
from veeshtral.resources import Agent


def _agent(key: str, aid: int) -> Agent:
    return Agent(key=key, name=key, id=aid, api_endpoint="http://127.0.0.1:9/v1")


def test_compile_sequential_flow():
    a = _agent("ocr", 11)
    b = _agent("match", 12)

    @veeshtral.agent(a, task="Extract fields")
    def review(invoice_text: str) -> dict: ...

    @veeshtral.agent(b, task="Match PO")
    def three_way_match(extracted: dict) -> dict: ...

    @veeshtral.hitl_gate(condition=lambda ctx: ctx.amount_over_limit or ctx.match_failed)
    def escalate(match_result: dict): ...

    @veeshtral.workflow_fn(name="ap-invoice-review")
    def ap_flow(invoice_text: str):
        x = review(invoice_text)
        y = three_way_match(x)
        return escalate(y)

    wf = veeshtral.Workflow.from_flow(ap_flow)
    graph = wf.compile()
    types = [n["type"] for n in graph["nodes"]]
    assert types.count("trigger") == 1
    assert types.count("agent") == 2
    assert types.count("hitl") == 1
    assert types.count("end") == 1
    agent_nodes = [n for n in graph["nodes"] if n["type"] == "agent"]
    assert all("planner_assigned_task" in n["config"] for n in agent_nodes)
    hitl = next(n for n in graph["nodes"] if n["type"] == "hitl")
    assert hitl["config"]["gate_expression"] == "ctx.amount_over_limit or ctx.match_failed"
    assert hitl["config"]["gate_timing"] == "post_output_step"


def test_default_hitl_gate_uses_step_timing_with_expression():
    """Bare @hitl_gate() must not emit row_level + expression (platform 422)."""
    a = _agent("ocr", 11)

    @veeshtral.agent(a, task="Extract")
    def review(text: str) -> dict: ...

    @veeshtral.hitl_gate()
    def escalate(result: dict): ...

    @veeshtral.workflow_fn(name="default-hitl")
    def flow(text: str):
        return escalate(review(text))

    graph = veeshtral.Workflow.from_flow(flow).compile()
    hitl = next(n for n in graph["nodes"] if n["type"] == "hitl")
    assert hitl["config"]["gate_timing"] == "post_output_step"
    assert "ctx.approval_required" in hitl["config"]["gate_expression"]


def test_quality_gate_scope_wires():
    a = _agent("ocr", 11)
    b = _agent("match", 12)
    rubric = veeshtral.QualityRubric(
        key="ap-safety",
        rules_text="flag unsafe payments",
        on_fail="hitl",
        id=99,
    )

    @veeshtral.agent(a, task="Extract")
    def review(invoice_text: str) -> dict: ...

    @veeshtral.agent(b, task="Match")
    def three_way_match(extracted: dict) -> dict: ...

    @veeshtral.workflow_fn(name="qg-demo")
    def flow(invoice_text: str):
        return three_way_match(review(invoice_text))

    wf = veeshtral.Workflow.from_flow(flow)
    wf.attach_quality_gate(rubric, scope=[review, three_way_match])
    graph = wf.compile()
    qg = next(n for n in graph["nodes"] if n["type"] == "quality_gate")
    scope_edges = [e for e in graph["edges"] if e.get("wire_role") == "scope"]
    assert len(scope_edges) == 2
    assert all(e["source"] == qg["id"] for e in scope_edges)


def test_marketplace_never_create():
    calls: list[str] = []

    class FakeClient:
        credentials = type("C", (), {"access_token": "t", "api_key": None, "email": None, "password": None})()

        def ensure_jwt(self):
            return "t"

        def request(self, method, path, **kwargs):
            calls.append(f"{method} {path}")
            if path == "/api/agents":
                return []
            raise AssertionError(f"unexpected {method} {path}")

    with pytest.raises(CompileError, match="marketplace"):
        veeshtral.marketplace.install("missing-agent", client=FakeClient())  # type: ignore[arg-type]
    assert not any("/api/business/agents" in c and c.startswith("POST") for c in calls)


def test_compile_50_nodes_performance():
    agents = [_agent(f"a{i}", i + 1) for i in range(48)]
    steps = []
    for i, ag in enumerate(agents):

        @veeshtral.agent(ag, task=f"task {i}")
        def step(x, _ag=ag, _i=i):  # noqa: ANN001
            ...

        step.__name__ = f"step_{i}"
        # Re-stamp meta with correct name
        from veeshtral.decorators import StepMeta, _META_ATTR

        setattr(
            step,
            _META_ATTR,
            StepMeta(kind="agent", fn=step, name=f"step_{i}", resource=ag, task=f"task {i}"),
        )
        steps.append(step)

    t0 = time.perf_counter()
    graph = veeshtral.Workflow(name="perf", steps=steps).compile()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 100
    assert len([n for n in graph["nodes"] if n["type"] == "agent"]) == 48
