"""Complex AP invoice workflow — business-expert graph for load/robustness demos.

Compiles offline (pre-set agent ids). For live publish/run, configure credentials
and omit the id= fields so Agent.ensure creates/looks up by key.
"""

from __future__ import annotations

import veeshtral


def main() -> None:
    ocr = veeshtral.Agent(key="ap-ocr", id=1, api_endpoint="http://127.0.0.1:9/v1")
    match = veeshtral.Agent(key="ap-match", id=2, api_endpoint="http://127.0.0.1:9/v1")
    risk = veeshtral.Agent(key="ap-risk", id=3, api_endpoint="http://127.0.0.1:9/v1")
    esc = veeshtral.Agent(key="ap-escalate", id=4, api_endpoint="http://127.0.0.1:9/v1")
    auto = veeshtral.Agent(key="ap-auto", id=5, api_endpoint="http://127.0.0.1:9/v1")
    rubric = veeshtral.QualityRubric(
        key="ap-enterprise-safety",
        rules_text="Block duplicate payments; escalate over-limit amounts.",
        on_fail="hitl",
        id=99,
    )

    @veeshtral.agent(ocr, task="Extract line items, vendor, tax, and PO references.")
    def extract(invoice_text: str) -> dict: ...

    @veeshtral.agent(match, task="Three-way match against PO and goods receipt.")
    def three_way(extracted: dict) -> dict: ...

    @veeshtral.agent(risk, task="Score fraud / duplicate / policy risk.")
    def risk_score(matched: dict) -> dict: ...

    @veeshtral.branch(condition=lambda ctx: ctx.needs_escalation or ctx.amount_over_limit)
    @veeshtral.agent(esc, task="Build finance escalation package.")
    def escalate(c: dict) -> dict: ...

    @veeshtral.hitl_gate(condition=lambda ctx: ctx.amount_over_limit or ctx.match_failed)
    def finance_review(packet: dict): ...

    @veeshtral.branch(is_default=True)
    @veeshtral.agent(auto, task="Prepare straight-through payment proposal.")
    def auto_pay(c: dict) -> dict: ...

    @veeshtral.workflow(
        name="ap-enterprise-invoice-review",
        description="Complex AP: OCR → match → risk → first_match → HITL with QG + memory.",
    )
    def flow(invoice_text: str):
        x = extract(invoice_text)
        m = three_way(x)
        r = risk_score(m)
        e = escalate(r)
        finance_review(e)
        auto_pay(r)

    wf = veeshtral.Workflow.from_flow(flow)
    wf.attach_quality_gate(rubric, scope=[extract, three_way, risk_score])
    wf.attach_memory(veeshtral.Memory())
    graph = wf.compile()
    modes = {
        n["id"]: n["config"].get("branch_mode")
        for n in graph["nodes"]
        if n["type"] == "agent" and n["config"].get("branch_mode")
    }
    print("nodes", len(graph["nodes"]), "edges", len(graph["edges"]), "branch_modes", modes)
    assert any(v == "first_match" for v in modes.values())
    assert any(n["type"] == "quality_gate" for n in graph["nodes"])
    assert any(n["type"] == "memory" for n in graph["nodes"])
    print("OK complex AP graph")


if __name__ == "__main__":
    main()
