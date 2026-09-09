"""Branching patterns — fan_out diamond and first_match arms (#552).

Set credentials via env (VEESHTRAL_API_KEY or email/password) before running.
"""

from __future__ import annotations

import os

import veeshtral


def build_agents():
    endpoint = os.environ.get("VEESHTRAL_BYO_ENDPOINT", "http://127.0.0.1:9/v1")
    key = os.environ.get("VEESHTRAL_BYO_API_KEY", "local")
    return (
        veeshtral.Agent(
            key="triage-ingest",
            name="Ingest",
            api_endpoint=endpoint,
            api_key=key,
            id=int(os.environ.get("VEESHTRAL_AGENT_INGEST_ID", "0")) or None,
        ),
        veeshtral.Agent(
            key="triage-sentiment",
            name="Sentiment",
            api_endpoint=endpoint,
            api_key=key,
            id=int(os.environ.get("VEESHTRAL_AGENT_SENT_ID", "0")) or None,
        ),
        veeshtral.Agent(
            key="triage-priority",
            name="Priority",
            api_endpoint=endpoint,
            api_key=key,
            id=int(os.environ.get("VEESHTRAL_AGENT_PRIO_ID", "0")) or None,
        ),
        veeshtral.Agent(
            key="triage-merge",
            name="Merge",
            api_endpoint=endpoint,
            api_key=key,
            id=int(os.environ.get("VEESHTRAL_AGENT_MERGE_ID", "0")) or None,
        ),
        veeshtral.Agent(
            key="triage-escalate",
            name="Escalate",
            api_endpoint=endpoint,
            api_key=key,
            id=int(os.environ.get("VEESHTRAL_AGENT_ESC_ID", "0")) or None,
        ),
        veeshtral.Agent(
            key="triage-auto",
            name="Auto",
            api_endpoint=endpoint,
            api_key=key,
            id=int(os.environ.get("VEESHTRAL_AGENT_AUTO_ID", "0")) or None,
        ),
    )


def fan_out_example():
    ingest, sent, prio, merge, _, _ = build_agents()
    # Pre-resolved ids for offline compile demo
    for a, i in zip((ingest, sent, prio, merge), (1, 2, 3, 4)):
        if a.id is None:
            a.id = i

    @veeshtral.agent(ingest, task="Ingest ticket text")
    def ingest_step(text: str) -> dict: ...

    @veeshtral.agent(sent, task="Score sentiment")
    def sentiment(x: dict) -> dict: ...

    @veeshtral.agent(prio, task="Score priority")
    def priority(x: dict) -> dict: ...

    @veeshtral.agent(merge, task="Merge signals")
    def merge_step(a: dict, b: dict) -> dict: ...

    @veeshtral.workflow(name="ticket-fanout-demo")
    def flow(text: str):
        root = ingest_step(text)
        return merge_step(sentiment(root), priority(root))

    graph = veeshtral.Workflow.from_flow(flow).compile()
    parent = next(n for n in graph["nodes"] if n["id"].startswith("ingest"))
    assert parent["config"]["branch_mode"] == "fan_out"
    return graph


def first_match_example():
    _, _, _, _, esc, auto = build_agents()
    classify = veeshtral.Agent(key="triage-classify", name="Classify", id=10, api_endpoint="http://127.0.0.1:9/v1")
    if esc.id is None:
        esc.id = 11
    if auto.id is None:
        auto.id = 12

    @veeshtral.agent(classify, task="Classify urgency")
    def classify_step(text: str) -> dict: ...

    @veeshtral.branch(condition=lambda ctx: ctx.needs_escalation)
    @veeshtral.agent(esc, task="Human escalate")
    def escalate(c: dict) -> dict: ...

    @veeshtral.branch(is_default=True)
    @veeshtral.agent(auto, task="Auto resolve")
    def auto_step(c: dict) -> dict: ...

    @veeshtral.workflow(name="ticket-first-match-demo")
    def flow(text: str):
        c = classify_step(text)
        escalate(c)
        auto_step(c)

    graph = veeshtral.Workflow.from_flow(flow).compile()
    parent = next(n for n in graph["nodes"] if n["id"].startswith("classify"))
    assert parent["config"]["branch_mode"] == "first_match"
    return graph


if __name__ == "__main__":
    print("fan_out nodes", len(fan_out_example()["nodes"]))
    print("first_match nodes", len(first_match_example()["nodes"]))
