"""Workflow object — attach, compile, publish, run."""

from __future__ import annotations

import hashlib
import random
import time
from typing import Any, Callable

from veeshtral.client import Client, get_client
from veeshtral.compiler import compile_flow, compile_linear_steps
from veeshtral.decorators import FlowMeta, StepMeta, get_meta
from veeshtral.errors import ApiError, CompileError, PollTimeout
from veeshtral.resources import Agent, LiveSource, Memory, QualityRubric, Skill
from veeshtral.results import (
    PublishResult,
    RunResult,
    extract_step_outputs_and_qg,
    job_poll_terminal,
    normalize_run_status,
)


class Workflow:
    def __init__(
        self,
        name: str,
        *,
        description: str = "",
        flow_fn: Callable[..., Any] | None = None,
        steps: list[Any] | None = None,
        workflow_id: int | None = None,
        client: Client | None = None,
    ) -> None:
        if not name or not str(name).strip():
            raise CompileError("workflow name is required", code="bad_name")
        self.name = str(name).strip()
        self.description = description or ""
        if len(self.description.encode("utf-8")) > 2_097_152:
            raise CompileError(
                "workflow description exceeds 2 MiB UTF-8",
                code="description_too_large",
            )
        self._flow_fn = flow_fn
        self._steps = list(steps or [])
        self._client = client
        self._quality_gates: list[tuple[QualityRubric, list[Any]]] = []
        self._memory: Memory | None = None
        self._live: LiveSource | None = None
        self.workflow_id: int | None = int(workflow_id) if workflow_id is not None else None
        self.draft_revision: int | None = None
        self._last_graph: dict[str, Any] | None = None

    @classmethod
    def from_flow(cls, flow_fn: Callable[..., Any], *, client: Client | None = None) -> "Workflow":
        meta = get_meta(flow_fn)
        if not isinstance(meta, FlowMeta):
            raise CompileError("from_flow requires @workflow decorated function", code="missing_workflow")
        return cls(meta.name, description=meta.description, flow_fn=flow_fn, client=client)

    def attach_quality_gate(self, rubric: QualityRubric, scope: list[Any]) -> "Workflow":
        if not scope:
            raise CompileError("attach_quality_gate scope cannot be empty", code="qg_empty_scope")
        self._quality_gates.append((rubric, list(scope)))
        return self

    def attach_memory(self, memory: Memory) -> "Workflow":
        self._memory = memory
        return self

    def attach_live_source(self, source: LiveSource) -> "Workflow":
        self._live = source
        return self

    def _client_or_default(self) -> Client:
        return self._client or get_client()

    def _ensure_resources(self, client: Client) -> None:
        # Match compile_flow discovery (globals + nonlocals) so nested @agent factories resolve.
        from veeshtral.compiler import _discover_candidates

        metas: list[StepMeta] = []
        if self._flow_fn:
            metas.extend(_discover_candidates(self._flow_fn).values())
        for s in self._steps:
            m = get_meta(s)
            if isinstance(m, StepMeta):
                metas.append(m)
        seen: set[int] = set()
        for m in metas:
            mid = id(m)
            if mid in seen:
                continue
            seen.add(mid)
            if isinstance(m.resource, Agent) and not m.resource._marketplace:
                m.resource.ensure(client)
            for sk in m.skills:
                if isinstance(sk, Skill):
                    sk.ensure(client)
        for rubric, _ in self._quality_gates:
            rubric.ensure(client)

    def _compile_impl(self, *, allow_unresolved: bool) -> dict[str, Any]:
        if self._flow_fn is not None:
            return compile_flow(
                self._flow_fn,
                quality_gates=self._quality_gates,
                memory=self._memory,
                live_source=self._live,
                allow_unresolved=allow_unresolved,
            )
        if self._steps:
            return compile_linear_steps(
                self._steps,
                quality_gates=self._quality_gates,
                memory=self._memory,
                live_source=self._live,
                allow_unresolved=allow_unresolved,
            )
        raise CompileError(
            "Workflow needs a @workflow flow or steps=. "
            "Example: Workflow.from_flow(my_flow) or Workflow(name=..., steps=[a, b])",
            code="empty_workflow",
        )

    def compile(self) -> dict[str, Any]:
        """Validate locally, ensure resources, then emit resolved graph JSON."""
        client = self._client_or_default()
        # Local lint before create-if-missing network side effects.
        self._compile_impl(allow_unresolved=True)
        self._ensure_resources(client)
        graph = self._compile_impl(allow_unresolved=False)
        self._last_graph = graph
        return graph

    def _find_by_name(self, client: Client) -> list[dict[str, Any]]:
        data = client.request(
            "GET",
            "/api/workflows",
            # include_total=False avoids COUNT + quota on the hot name_eq upsert path.
            params={"name_eq": self.name, "limit": 10, "include_total": False},
        )
        if isinstance(data, dict):
            return list(data.get("items") or [])
        return list(data or [])

    def _patch_draft(
        self,
        client: Client,
        wid: int,
        graph: dict[str, Any],
        *,
        draft_revision: int | None = None,
    ) -> dict[str, Any]:
        if draft_revision is not None:
            rev = int(draft_revision)
        else:
            detail = client.request("GET", f"/api/workflows/{wid}")
            rev = int(detail.get("draft_revision") or 1)
        last_exc: ApiError | None = None
        for attempt in range(6):
            try:
                return client.request(
                    "PATCH",
                    f"/api/workflows/{wid}/draft",
                    json={"graph": graph, "expected_draft_revision": rev},
                )
            except ApiError as exc:
                last_exc = exc
                if exc.status_code != 409 or attempt >= 5:
                    raise
                body = exc.body if isinstance(exc.body, dict) else {}
                detail_body = body.get("detail") if isinstance(body.get("detail"), dict) else body
                current = None
                if isinstance(detail_body, dict):
                    current = detail_body.get("current_draft_revision")
                if current is not None:
                    rev = int(current)
                else:
                    detail = client.request("GET", f"/api/workflows/{wid}")
                    rev = int(detail.get("draft_revision") or rev)
                time.sleep(0.12 * (attempt + 1) + random.uniform(0, 0.05))
        raise CompileError(
            f"draft still conflicted after retries (last: {last_exc})",
            code="draft_conflict",
        )

    def _create_workflow(self, client: Client, graph: dict[str, Any]) -> dict[str, Any]:
        idem = client.content_idempotency_key(
            "wf-" + hashlib.sha256(self.name.encode()).hexdigest()[:12],
            self.name,
            self.description,
            graph,
        )
        last_exc: ApiError | None = None
        for attempt in range(6):
            try:
                return client.request(
                    "POST",
                    "/api/workflows",
                    json={"name": self.name, "description": self.description, "graph": graph},
                    idempotency_key=idem,
                )
            except ApiError as exc:
                last_exc = exc
                typ = str(exc.typ or "")
                if typ == "idempotency_key_mismatch":
                    raise
                # Concurrent create / idempotency lock — recover via name_eq.
                if exc.status_code != 409 and typ != "idempotency_request_in_progress":
                    raise
                time.sleep(0.15 * (attempt + 1))
                matches = self._find_by_name(client)
                if len(matches) == 1:
                    wid = int(matches[0]["id"])
                    list_rev = matches[0].get("draft_revision")
                    patched = self._patch_draft(
                        client,
                        wid,
                        graph,
                        draft_revision=int(list_rev) if list_rev is not None else None,
                    )
                    patched = dict(patched)
                    patched.setdefault("id", wid)
                    return patched
                if len(matches) > 1:
                    raise CompileError(
                        f"multiple workflows named {self.name!r} after create race; "
                        f"pass workflow_id= to disambiguate (ids={[m.get('id') for m in matches]})",
                        code="ambiguous_name",
                    ) from exc
        raise CompileError(
            f"workflow create still racing after retries (last: {last_exc})",
            code="create_race",
        )

    def publish(self, *, workflow_id: int | None = None) -> PublishResult:
        client = self._client_or_default()
        graph = self.compile()
        if workflow_id is not None:
            self.workflow_id = int(workflow_id)

        if self.workflow_id is not None:
            wid = int(self.workflow_id)
            detail = self._patch_draft(client, wid, graph)
            self.draft_revision = int(detail.get("draft_revision") or 1)
        else:
            matches = self._find_by_name(client)
            if len(matches) > 1:
                raise CompileError(
                    f"multiple workflows named {self.name!r}; pass workflow_id= to disambiguate "
                    f"(ids={[m.get('id') for m in matches]}). "
                    f"Example: Workflow(name={self.name!r}, workflow_id={matches[0].get('id')}, ...)",
                    code="ambiguous_name",
                )
            if len(matches) == 1:
                wid = int(matches[0]["id"])
                list_rev = matches[0].get("draft_revision")
                detail = self._patch_draft(
                    client,
                    wid,
                    graph,
                    draft_revision=int(list_rev) if list_rev is not None else None,
                )
                self.workflow_id = wid
                self.draft_revision = int(detail.get("draft_revision") or 1)
            else:
                created = self._create_workflow(client, graph)
                self.workflow_id = int(created["id"])
                self.draft_revision = int(created.get("draft_revision") or 1)

        assert self.workflow_id is not None
        pub: dict[str, Any] | None = None
        for attempt in range(4):
            try:
                pub = client.request("POST", f"/api/workflows/{self.workflow_id}/publish")
                break
            except ApiError as exc:
                typ = str(exc.typ or "")
                # Name/BRD changed without a draft bump — re-patch will not help.
                if typ == "publish_context_changed":
                    raise
                if exc.status_code == 409 and attempt < 3:
                    # Another writer may have bumped draft — re-patch then republish.
                    detail = self._patch_draft(client, int(self.workflow_id), graph)
                    self.draft_revision = int(
                        detail.get("draft_revision") or self.draft_revision or 1
                    )
                    time.sleep(0.15 * (attempt + 1))
                    continue
                raise
        assert pub is not None
        return PublishResult(
            workflow_id=int(self.workflow_id),
            workflow_version_id=pub.get("workflow_version_id"),
            version_number=pub.get("version_number"),
            draft_revision=self.draft_revision,
        )

    def run(
        self,
        *,
        features: dict[str, Any] | None = None,
        poll_timeout_s: float = 300.0,
        use_api_key: bool = False,
        idempotency_key: str | None = None,
        **fields: Any,
    ) -> RunResult:
        client = self._client_or_default()
        if self.workflow_id is None:
            self.publish()
        assert self.workflow_id is not None

        body = {
            "transaction_context": {
                "features": features or {},
                "inputs": {"fields": fields},
            }
        }
        if use_api_key:
            if not client.credentials.api_key:
                raise CompileError("use_api_key=True requires api_key on Client", code="missing_api_key")
            key = idempotency_key or client.new_idempotency_key(prefix="run")
            run = client.request(
                "POST",
                f"/api/workflows/{self.workflow_id}/run",
                json=body,
                prefer_api_key=True,
                idempotency_key=key,
            )
        else:
            run = client.request(
                "POST",
                f"/api/workflows/{self.workflow_id}/run",
                json=body,
                idempotency_key=idempotency_key,
            )

        job_id = run.get("job_id")
        status = "queued"
        raw_job: dict[str, Any] = {}
        deadline = time.time() + poll_timeout_s
        delay = 0.2
        while job_id and time.time() < deadline:
            raw_job = client.request("GET", f"/api/jobs/{job_id}")
            if not isinstance(raw_job, dict):
                raw_job = {}
            status = str(raw_job.get("status") or "").lower()
            # HITL pauses the *step*; job often stays in_progress with
            # has_pending_human_approval=True — treat that as terminal for poll.
            if job_poll_terminal(raw_job):
                status = normalize_run_status(raw_job)
                break
            time.sleep(delay + random.uniform(0, delay * 0.25))
            delay = min(2.0, delay * 1.5)
        else:
            if job_id and not (isinstance(raw_job, dict) and job_poll_terminal(raw_job)):
                raise PollTimeout(
                    f"job {job_id} still {status!r} after {poll_timeout_s:g}s; "
                    "increase poll_timeout_s or poll GET /api/jobs/{id} yourself",
                    job_id=int(job_id),
                    status=status,
                    poll_timeout_s=poll_timeout_s,
                )
            if isinstance(raw_job, dict) and raw_job:
                status = normalize_run_status(raw_job)

        step_outputs, qg = extract_step_outputs_and_qg(
            raw_job if isinstance(raw_job, dict) else {}
        )
        return RunResult(
            status=status or str(run.get("status") or "unknown"),
            job_id=int(job_id) if job_id else None,
            run_id=run.get("run_id"),
            workflow_id=self.workflow_id,
            quality_gate_verdict=qg,
            step_outputs=step_outputs,
            raw=raw_job or run,
        )
