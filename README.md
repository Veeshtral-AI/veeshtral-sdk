# Veeshtral Python SDK

Native, decorator-based Python SDK for authoring **Veeshtral Workflow Studio** graphs in code and running them on the existing execution engine (Quality Gates, HITL, Skills, Memory, Live sources, Marketplace agents).

This repository is the **release surface** for the `veeshtral` package ([MIT](LICENSE)). Platform API hooks (`agent_key`, workflow `name_eq`) live in the core Veeshtral backend.

## Install

```bash
pip install veeshtral
```

Requires Python ≥ 3.11. Optional LangGraph adapter (future): `pip install "veeshtral[langgraph]"` — see [#2](https://github.com/Veeshtral-AI/veeshtral-sdk/issues/2).

From a clone (contributors):

```bash
pip install -e ".[dev]"
```

Release / PyPI publish: see [`CONTRIBUTING.md`](CONTRIBUTING.md#releasing-to-pypi).

## Auth

| Operation | Auth |
|-----------|------|
| Create / draft / publish / resource CRUD | **JWT** *or* **`X-API-Key`** |
| Run a published workflow | JWT **or** `X-API-Key` |

API-key **creates** and **runs** automatically send an `Idempotency-Key` (required by the platform). Prefer API keys for CI/CD and headless authoring; JWT for interactive Studio.

```python
import veeshtral

# Option A — session JWT
veeshtral.configure(
    base_url="http://127.0.0.1:8000",
    email="you@company.com",
    password="...",
)

# Option B — business API key only (no login)
veeshtral.configure(
    base_url="http://127.0.0.1:8000",
    api_key="vsk_...",
)
```

Environment: `VEESHTRAL_BASE_URL`, `VEESHTRAL_EMAIL`, `VEESHTRAL_PASSWORD`, `VEESHTRAL_ACCESS_TOKEN`, `VEESHTRAL_API_KEY`.

## Quick start

```python
import veeshtral

ocr = veeshtral.Agent(
    key="invoice-ocr-review",
    name="Invoice OCR Review",
    llm_provider="openai_compatible",
    llm_model="gpt-4o-mini",
    api_endpoint="https://your-llm.example/v1",  # required on first create
    api_key="...",
)

@veeshtral.agent(ocr, task="Extract vendor and amount.")
def review(invoice_text: str) -> dict: ...

@veeshtral.hitl_gate(condition=lambda ctx: ctx.amount_over_limit)
def escalate(result: dict): ...

@veeshtral.workflow(name="ap-invoice-review")
def flow(invoice_text: str):
    return escalate(review(invoice_text))

wf = veeshtral.Workflow.from_flow(flow)
wf.publish()   # upsert by exact name (name_eq); republish updates draft
result = wf.run(invoice_text="...", features={"amount_over_limit": True})
print(result.status, result.quality_gate_verdict)
```

### Branching

**Parallel fan-out** (diamond) — call multiple children with the same parent; no `@branch` needed:

```python
root = ingest(text)
left = sentiment(root)
right = priority(root)
return merge(left, right)  # parent gets branch_mode=fan_out
```

**First-match** — mark arms with `@branch` (stack `@branch` *above* `@agent`):

```python
@veeshtral.branch(condition=lambda ctx: ctx.needs_escalation)
@veeshtral.agent(human, task="Escalate")
def escalate(c: dict) -> dict: ...

@veeshtral.branch(is_default=True)
@veeshtral.agent(bot, task="Auto")
def auto(c: dict) -> dict: ...
```

### Duplicate workflow names

If several workflows share a name, `publish()` raises with their ids. Disambiguate:

```python
wf = veeshtral.Workflow.from_flow(flow)
wf.publish(workflow_id=42)
# or: Workflow(name="ap-invoice-review", workflow_id=42, flow_fn=flow)
```

Marketplace agents are **link-only** (never creates a BYO agent):

```python
support = veeshtral.marketplace.install("acme-support-triage-v2")
```

### Live sources (stream or voice)

Attach exactly one live ingress before publish:

```python
# Webhook / Kafka / SSE / WebSocket → stream_source node
wf.attach_live_source(veeshtral.LiveSource(kind="inbound_webhook", max_events=50))

# Phone or browser voice → voice_channel node
wf.attach_live_source(
    veeshtral.LiveSource(
        kind="inbound_phone",  # or outbound_phone / browser
        max_hold_minutes=5,    # HITL hold cap (1–30)
        barge_in=True,
    )
)
```

`outbound_phone` also requires `phone_number` and `outbound_number_consented=True`.
A workflow cannot mix `stream_source` and `voice_channel`.

## Design notes

- Compiles to the same `nodes`/`edges` JSON the canvas uses (`planner_assigned_task`, `wire_role=scope` for QG).
- Local graph lint runs **before** create-if-missing resource calls (invalid graphs do not mutate the tenant).
- HTTP client retries `429` (honors `Retry-After`), `502`/`503`, and `idempotency_request_in_progress`.
- No planner / autosplit — Studio jobs already use `workflow_origin=standalone_workflow`.
- Strict DAG: cycles are compile errors. No LangGraph adapter in v1.
- HITL / branch `condition=lambda ctx: ...` is compiled via AST only (**never** `eval`).
- Quality gate `scope=` must list **agent** steps only (not HITL).
- `Workflow.run(..., poll_timeout_s=...)` raises `veeshtral.PollTimeout` if the job stays non-terminal.
- `RunResult.ok` / `.is_pending_hitl` / `.raise_for_status()` for clean call-site handling.
- Inactive BYO agents matched by key raise a clear error (reactivate or change key).

## Acceptance criteria coverage

1. Decorator sequential + fan_out / first_match + HITL/QG → publish → Studio graph.
2. `attach_quality_gate(scope=[...])` scope-wires agents.
3. `.run()` uses the same engine path as canvas runs.
4. Lambda conditions match `ctx.*` grammar.
5. Republish same name updates draft; Agent/QualityRubric by key create-if-missing (409 races recover).
6. No execute-time replan for SDK-published workflows (platform regression).

## Develop

```bash
python -m pytest tests -c pytest.ini --cov=veeshtral --cov-config=.coveragerc --cov-fail-under=80
```

**PRs only:** do not push commits directly to `main`. Branch → PR → CI → merge. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

See [`examples/invoice_review.py`](examples/invoice_review.py),
[`examples/branching.py`](examples/branching.py), and
[`examples/complex_ap_invoice.py`](examples/complex_ap_invoice.py).
