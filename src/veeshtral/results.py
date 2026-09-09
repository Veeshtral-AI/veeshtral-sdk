"""Run / publish result objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from veeshtral.errors import VeeshtralError

_HITL_STATUSES = frozenset(
    {
        "pending_review",
        "awaiting_human_approval",
        "pending_human",
    }
)
_HITL_STEP_STATUSES = frozenset(
    {
        "awaiting_human_approval",
        "pending_human",
        "pending_review",
    }
)
_FAILURE_STATUSES = frozenset({"failed", "cancelled"})
_JOB_DONE_STATUSES = frozenset({"completed", "failed", "cancelled", "paused"})


def job_awaiting_hitl(raw_job: dict[str, Any]) -> bool:
    """True when JobResponse shows a HITL pause (job may still be in_progress)."""
    if raw_job.get("has_pending_human_approval"):
        return True
    steps = raw_job.get("workflow_steps") or []
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("status") or "").lower() in _HITL_STEP_STATUSES:
            return True
    return False


def job_poll_terminal(raw_job: dict[str, Any]) -> bool:
    status = str(raw_job.get("status") or "").lower()
    if status in _JOB_DONE_STATUSES:
        return True
    return job_awaiting_hitl(raw_job)


def normalize_run_status(raw_job: dict[str, Any]) -> str:
    """Map JobResponse into SDK run status (HITL → awaiting_human_approval)."""
    status = str(raw_job.get("status") or "").lower()
    if status in ("completed", "failed", "cancelled"):
        return status
    if job_awaiting_hitl(raw_job):
        return "awaiting_human_approval"
    return status or "unknown"


def _maybe_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    # Bound hostile / huge step payloads (client-side DoS).
    if len(text) > 1_000_000:
        return value
    if text[0] not in "{[":
        return value
    try:
        return json.loads(text)
    except (TypeError, ValueError, RecursionError):
        return value


def _qg_from_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    qg = payload.get("quality_gate_verdict") or payload.get("quality_gate")
    if qg is not None:
        return qg
    for value in payload.values():
        if isinstance(value, dict):
            nested = value.get("quality_gate_verdict") or value.get("quality_gate")
            if nested is not None:
                return nested
    return None


def extract_step_outputs_and_qg(
    raw_job: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Build step_outputs + QG from real JobResponse (workflow_steps[].output_data)."""
    legacy_steps = raw_job.get("step_outputs")
    qg = raw_job.get("quality_gate_verdict") or raw_job.get("quality_gate")
    if isinstance(legacy_steps, dict) and legacy_steps:
        if qg is None:
            for value in legacy_steps.values():
                qg = _qg_from_payload(value)
                if qg is not None:
                    break
        return legacy_steps, qg

    out: dict[str, Any] = {}
    steps = raw_job.get("workflow_steps") or []
    if not isinstance(steps, list):
        return out, qg
    for step in steps:
        if not isinstance(step, dict):
            continue
        key = step.get("graph_node_id") or f"step_{step.get('step_order')}"
        parsed = _maybe_json(step.get("output_data"))
        out[str(key)] = parsed
        if qg is None:
            qg = _qg_from_payload(parsed)
    return out, qg


@dataclass
class PublishResult:
    workflow_id: int
    workflow_version_id: int | None = None
    version_number: int | None = None
    draft_revision: int | None = None


@dataclass
class RunResult:
    status: str
    job_id: int | None = None
    run_id: str | None = None
    workflow_id: int | None = None
    quality_gate_verdict: Any = None
    step_outputs: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the job finished successfully (not HITL-paused / operator-paused)."""
        return str(self.status or "").lower() == "completed"

    @property
    def is_pending_hitl(self) -> bool:
        if str(self.status or "").lower() in _HITL_STATUSES:
            return True
        return job_awaiting_hitl(self.raw if isinstance(self.raw, dict) else {})

    @property
    def is_paused(self) -> bool:
        """True for non-HITL job pause (operator/billing hold). Not success."""
        return str(self.status or "").lower() == "paused" and not self.is_pending_hitl

    @property
    def failed(self) -> bool:
        return str(self.status or "").lower() in _FAILURE_STATUSES

    def raise_for_status(self) -> "RunResult":
        """Raise if the run failed or was cancelled. HITL-paused runs are not errors."""
        status = str(self.status or "").lower()
        if status in _FAILURE_STATUSES:
            raise VeeshtralError(
                f"workflow run {status}: job_id={self.job_id} run_id={self.run_id}"
            )
        return self
