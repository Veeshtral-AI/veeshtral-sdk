"""Invoice review example — AP workflow with QG + HITL (#552).

Configure credentials via environment or veeshtral.configure(...):

  VEESHTRAL_BASE_URL=http://127.0.0.1:8000
  VEESHTRAL_API_KEY=...          # preferred for code/CI authoring
  # or JWT:
  VEESHTRAL_EMAIL=...
  VEESHTRAL_PASSWORD=...

API-key creates/runs send Idempotency-Key automatically.
"""

from __future__ import annotations

import os

import veeshtral


def main() -> None:
    veeshtral.configure(
        base_url=os.environ.get("VEESHTRAL_BASE_URL", "http://127.0.0.1:8000"),
        email=os.environ.get("VEESHTRAL_EMAIL"),
        password=os.environ.get("VEESHTRAL_PASSWORD"),
        api_key=os.environ.get("VEESHTRAL_API_KEY"),
    )

    ocr_agent = veeshtral.Agent(
        key="invoice-ocr-review",
        name="Invoice OCR Review",
        llm_provider="openai_compatible",
        llm_model="gpt-4o-mini",
        api_endpoint=os.environ["VEESHTRAL_BYO_ENDPOINT"],
        api_key=os.environ.get("VEESHTRAL_BYO_API_KEY", "configured-locally"),
    )
    match_agent = veeshtral.Agent(
        key="po-three-way-match",
        name="PO Three-Way Match",
        llm_provider="openai_compatible",
        llm_model="gpt-4o-mini",
        api_endpoint=os.environ["VEESHTRAL_BYO_ENDPOINT"],
        api_key=os.environ.get("VEESHTRAL_BYO_API_KEY", "configured-locally"),
    )

    safety_gate = veeshtral.QualityRubric(
        key="ap-invoice-safety",
        rules_text="Never approve payment without a matching PO and three-way match.",
        on_fail="hitl",
    )

    @veeshtral.agent(ocr_agent, task="Extract vendor, amount, and PO number from the invoice text.")
    def review(invoice_text: str) -> dict: ...

    @veeshtral.agent(match_agent, task="Match the extracted invoice fields against the PO and receipt.")
    def three_way_match(extracted: dict) -> dict: ...

    @veeshtral.hitl_gate(condition=lambda ctx: ctx.amount_over_limit or ctx.match_failed)
    def escalate_or_approve(match_result: dict): ...

    @veeshtral.workflow(name="ap-invoice-review")
    def ap_invoice_review(invoice_text: str):
        extracted = review(invoice_text)
        matched = three_way_match(extracted)
        return escalate_or_approve(matched)

    workflow = veeshtral.Workflow.from_flow(ap_invoice_review)
    workflow.attach_quality_gate(safety_gate, scope=[review, three_way_match])
    published = workflow.publish()
    print("published", published)

    result = workflow.run(
        invoice_text="Invoice #INV-2091 ... Amount: $7,850.00 ...",
        features={"amount_over_limit": True},
    )
    print("status", result.status)
    print("quality_gate_verdict", result.quality_gate_verdict)


if __name__ == "__main__":
    main()
