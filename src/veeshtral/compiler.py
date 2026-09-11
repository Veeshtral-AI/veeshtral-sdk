"""Compile decorated TaskFlow call graph → Veeshtral nodes/edges JSON."""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from typing import Any, Callable

from veeshtral.decorators import FlowMeta, StepMeta, get_meta
from veeshtral.errors import CompileError
from veeshtral.graph_lint import lint_graph


def _slug(name: str) -> str:
    raw = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    return (raw or "step")[:120]


def _resolve_resource_id(resource: Any, *, allow_unresolved: bool = False) -> int:
    if resource is None:
        raise CompileError("agent resource is required", code="missing_resource")
    rid = getattr(resource, "id", None)
    if rid is None:
        if isinstance(resource, int):
            rid = int(resource)
        elif allow_unresolved:
            return 0
        else:
            raise CompileError(
                f"resource {getattr(resource, 'key', resource)!r} is not resolved (call ensure() first)",
                code="unresolved_resource",
            )
    else:
        rid = int(rid)
    if rid <= 0:
        if allow_unresolved:
            return 0
        raise CompileError("agent id must be a positive integer", code="bad_agent_id")
    return rid


def _resolve_skill_ids(skills: list[Any], *, allow_unresolved: bool = False) -> list[int]:
    out: list[int] = []
    for s in skills:
        if isinstance(s, int):
            sid = int(s)
            if sid <= 0 and not allow_unresolved:
                raise CompileError("skill id must be a positive integer", code="bad_skill_id")
            if sid > 0:
                out.append(sid)
            elif allow_unresolved:
                continue
            else:
                raise CompileError("skill id must be a positive integer", code="bad_skill_id")
        else:
            sid = getattr(s, "id", None)
            if sid is None:
                if allow_unresolved:
                    continue
                raise CompileError("skill not resolved", code="unresolved_skill")
            sid_i = int(sid)
            if sid_i <= 0:
                if allow_unresolved:
                    continue
                raise CompileError("skill id must be a positive integer", code="bad_skill_id")
            out.append(sid_i)
    if len(out) > 8:
        raise CompileError("at most 8 skills per agent", code="skill_limit")
    return out


def _discover_candidates(flow_fn: Callable[..., Any]) -> dict[str, StepMeta]:
    candidates: dict[str, StepMeta] = {}
    closure = inspect.getclosurevars(flow_fn)
    for name, obj in {**closure.globals, **closure.nonlocals}.items():
        meta = get_meta(obj)
        if isinstance(meta, StepMeta):
            candidates[name] = meta
    for name, obj in flow_fn.__globals__.items():
        meta = get_meta(obj)
        if isinstance(meta, StepMeta):
            candidates.setdefault(name, meta)
    return candidates


def _trace_call_graph(
    flow_fn: Callable[..., Any], candidates: dict[str, StepMeta]
) -> tuple[list[StepMeta], list[tuple[StepMeta, StepMeta]]]:
    """AST-trace the workflow body for calls to decorated steps (no execution)."""
    try:
        src = textwrap.dedent(inspect.getsource(flow_fn))
    except (OSError, TypeError) as exc:
        raise CompileError(f"cannot read workflow source: {exc}", code="trace_no_source") from exc
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise CompileError(f"cannot parse workflow source: {exc}", code="trace_parse") from exc

    func = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == flow_fn.__name__:
            func = node
            break
    if func is None:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = node
                break
    if func is None:
        raise CompileError("workflow function not found in source", code="trace_no_fn")

    order: list[StepMeta] = []
    edges: list[tuple[StepMeta, StepMeta]] = []
    seen: set[str] = set()
    bindings: dict[str, StepMeta] = {}

    def _call_target(call: ast.Call) -> StepMeta | None:
        if isinstance(call.func, ast.Name) and call.func.id in candidates:
            return candidates[call.func.id]
        return None

    def _handle_call(call: ast.Call) -> StepMeta | None:
        meta = _call_target(call)
        if meta is None:
            return None
        if meta.name not in seen:
            order.append(meta)
            seen.add(meta.name)
        elif meta not in order:
            order.append(meta)
        for arg in list(call.args) + [kw.value for kw in call.keywords]:
            if isinstance(arg, ast.Name) and arg.id in bindings:
                edges.append((bindings[arg.id], meta))
            elif isinstance(arg, ast.Call):
                parent = _handle_call(arg)
                if parent is not None:
                    edges.append((parent, meta))
        return meta

    def _walk_stmts(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                meta = _handle_call(stmt.value)
                if meta is not None:
                    for t in stmt.targets:
                        if isinstance(t, ast.Name):
                            bindings[t.id] = meta
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call):
                meta = _handle_call(stmt.value)
                if meta is not None and isinstance(stmt.target, ast.Name):
                    bindings[stmt.target.id] = meta
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                if isinstance(stmt.value, ast.Call):
                    _handle_call(stmt.value)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                _handle_call(stmt.value)
            elif isinstance(stmt, ast.If):
                # Both arms are traced — use @branch conditions on targets, not the if test.
                _walk_stmts(stmt.body)
                _walk_stmts(stmt.orelse)
            elif isinstance(stmt, (ast.For, ast.While, ast.With, ast.AsyncWith)):
                _walk_stmts(list(stmt.body))
            elif isinstance(stmt, ast.Try):
                _walk_stmts(list(stmt.body))
                for handler in stmt.handlers:
                    _walk_stmts(list(handler.body))
                _walk_stmts(list(stmt.orelse))
                _walk_stmts(list(stmt.finalbody))

    _walk_stmts(list(func.body))

    if not order:
        raise CompileError("traced workflow produced no steps", code="empty_trace")
    uniq_edges: list[tuple[StepMeta, StepMeta]] = []
    seen_e: set[tuple[str, str]] = set()
    for a, b in edges:
        key = (a.name, b.name)
        if key not in seen_e:
            seen_e.add(key)
            uniq_edges.append((a, b))
    return order, uniq_edges


def _infer_branch_mode(outs: list[StepMeta]) -> str | None:
    """first_match when any arm is marked; otherwise fan_out for multi-out."""
    if len(outs) < 2:
        return None
    if any(
        o.branch_mode == "first_match" or o.edge_condition or o.is_default for o in outs
    ):
        return "first_match"
    return "fan_out"


def _build_graph_from_steps(
    steps_in_order: list[StepMeta],
    edges_pairs: list[tuple[StepMeta, StepMeta]],
    *,
    quality_gates: list[tuple[Any, list[Any]]] | None = None,
    memory: Any | None = None,
    live_source: Any | None = None,
    allow_unresolved: bool = False,
) -> dict[str, Any]:
    if len(steps_in_order) > 1 and not edges_pairs:
        raise CompileError(
            "ambiguous flow: steps were called but no data dependencies were traced; "
            "pass prior step outputs as arguments",
            code="ambiguous_flow",
        )

    nodes: list[dict[str, Any]] = [{"id": "trigger-1", "type": "trigger", "config": {}}]
    step_node_id: dict[str, str] = {}
    used_ids: set[str] = {"trigger-1", "end-1"}
    seen_names: dict[str, StepMeta] = {}
    for meta in steps_in_order:
        prev = seen_names.get(meta.name)
        if prev is not None and prev is not meta:
            raise CompileError(
                f"duplicate step function name {meta.name!r}; rename one of the "
                "@agent/@hitl_gate functions so graph edges stay unambiguous",
                code="duplicate_step_name",
            )
        seen_names[meta.name] = meta
        nid = _slug(meta.name)
        base = nid
        i = 2
        while nid in used_ids:
            nid = f"{base}-{i}"
            i += 1
        used_ids.add(nid)
        step_node_id[meta.name] = nid
        if meta.kind == "hitl":
            # Platform forbids gate_expression with post_output_row_level.
            # Default (no condition) still emits common-flag OR → must be step timing.
            gate_expression = meta.condition or (
                "ctx.approval_required or ctx.human_approval_required or "
                "ctx.human_review_required or ctx.requires_human_review or "
                "ctx.requires_review or ctx.is_negative"
            )
            cfg: dict[str, Any] = {
                "gate_timing": "post_output_step",
                "gate_expression": gate_expression,
            }
            nodes.append({"id": nid, "type": "hitl", "config": cfg})
        else:
            cfg = {
                "agent_id": _resolve_resource_id(meta.resource, allow_unresolved=allow_unresolved),
                "planner_assigned_task": meta.task,
            }
            skill_ids = _resolve_skill_ids(meta.skills, allow_unresolved=allow_unresolved)
            if skill_ids:
                cfg["skill_ids"] = skill_ids
                cfg["skill_mode"] = meta.skill_mode
            nodes.append({"id": nid, "type": "agent", "config": cfg})

    nodes.append({"id": "end-1", "type": "end", "config": {}})

    outs_by_src: dict[str, list[StepMeta]] = {}
    for src, tgt in edges_pairs:
        outs_by_src.setdefault(src.name, []).append(tgt)
    for src_name, outs in outs_by_src.items():
        mode = _infer_branch_mode(outs)
        if mode is None:
            continue
        for n in nodes:
            if n["id"] == step_node_id[src_name] and n["type"] == "agent":
                n["config"]["branch_mode"] = mode
                break

    edges: list[dict[str, Any]] = []
    roots = [s for s in steps_in_order if not any(e[1] is s for e in edges_pairs)]
    if not roots:
        raise CompileError("flow has no root step", code="no_root")
    if (memory is not None or live_source is not None) and len(roots) > 1:
        raise CompileError(
            "memory/live attachments require a single root step; "
            "fan-out after the first agent instead of multiple roots",
            code="multi_root_attachment",
        )
    for r in roots:
        edges.append({"source": "trigger-1", "target": step_node_id[r.name]})

    for src, tgt in edges_pairs:
        edge: dict[str, Any] = {
            "source": step_node_id[src.name],
            "target": step_node_id[tgt.name],
        }
        if tgt.edge_condition:
            edge["condition"] = str(tgt.edge_condition)
        if tgt.is_default:
            edge["is_default"] = True
        edges.append(edge)

    terminals = [s for s in steps_in_order if not any(e[0] is s for e in edges_pairs)]
    for t in terminals:
        edges.append({"source": step_node_id[t.name], "target": "end-1"})

    first = roots[0]
    first_id = step_node_id[first.name]
    spine_source = "trigger-1"
    spine_target = first_id

    if live_source is not None:
        kind = str(getattr(live_source, "kind", None) or "inbound_webhook").strip().lower()
        unsupported = {"meeting_source", "meeting"}
        if kind in unsupported:
            raise CompileError(
                f"live source kind {kind!r} is not supported yet",
                code="live_kind_unsupported",
            )
        stream_kinds = frozenset({"inbound_webhook", "kafka", "sse_pull", "websocket_pull"})
        voice_kinds = frozenset({"inbound_phone", "outbound_phone", "browser"})
        if kind not in stream_kinds and kind not in voice_kinds:
            raise CompileError(f"unknown live source kind: {kind}", code="live_kind")

        if kind in voice_kinds:
            if kind == "outbound_phone":
                phone = str(getattr(live_source, "phone_number", None) or "").strip()
                if not phone:
                    raise CompileError(
                        "outbound_phone requires phone_number",
                        code="voice_phone_required",
                    )
                if not bool(getattr(live_source, "outbound_number_consented", False)):
                    raise CompileError(
                        "outbound_phone requires outbound_number_consented=True",
                        code="voice_outbound_consent_required",
                    )
            src_id = "voice-1"
            hold = int(getattr(live_source, "max_hold_minutes", 5) or 5)
            hold = max(1, min(30, hold))
            voice_cfg: dict[str, Any] = {
                "channel": kind,
                "feeds_agent_node_id": first_id,
                "max_events": getattr(live_source, "max_events", 10) or 10,
                "max_in_flight": 2,
                "max_runtime_hours": 1,
                "max_hold_minutes": hold,
                "barge_in": bool(getattr(live_source, "barge_in", True)),
                "max_silence_ms": int(getattr(live_source, "max_silence_ms", 12000) or 0),
                "stt_provider": str(getattr(live_source, "stt_provider", None) or "deepgram"),
                "tts_provider": str(getattr(live_source, "tts_provider", None) or "elevenlabs"),
                "hitl_stream_behavior": "pause_source",
                "live_stream_fallback": "fail_step",
                "outbound_number_consented": bool(
                    getattr(live_source, "outbound_number_consented", False)
                ),
            }
            phone = str(getattr(live_source, "phone_number", None) or "").strip()
            if phone:
                voice_cfg["phone_number"] = phone
            vid = str(getattr(live_source, "voice_id", None) or "").strip()
            if vid:
                voice_cfg["voice_id"] = vid
            nodes.insert(
                1,
                {
                    "id": src_id,
                    "type": "voice_channel",
                    "config": voice_cfg,
                },
            )
        else:
            src_id = "stream-1"
            nodes.insert(
                1,
                {
                    "id": src_id,
                    "type": "stream_source",
                    "config": {
                        "source_kind": kind,
                        "feeds_agent_node_id": first_id,
                        "max_events": getattr(live_source, "max_events", 10) or 10,
                        "max_in_flight": 2,
                        "max_runtime_hours": 1,
                    },
                },
            )
        edges = [
            e
            for e in edges
            if not (e.get("source") == "trigger-1" and e.get("target") == first_id)
        ]
        edges.insert(0, {"source": "trigger-1", "target": src_id})
        edges.insert(1, {"source": src_id, "target": first_id})
        spine_source = src_id
        spine_target = first_id

    if memory is not None:
        mem_id = "memory-1"
        insert_at = 1
        live_ids = {"stream-1", "voice-1"}
        if any(n.get("id") in live_ids for n in nodes):
            insert_at = (
                next(i for i, n in enumerate(nodes) if n.get("id") in live_ids) + 1
            )
        nodes.insert(
            insert_at,
            {
                "id": mem_id,
                "type": "memory",
                "config": {
                    "memory_scope": getattr(memory, "scope", None) or "workflow",
                    "retention_class": getattr(memory, "retention_class", None) or "standard",
                    "retrieve_enabled": True,
                    "write_enabled": True,
                },
            },
        )
        if live_source is not None:
            # Live engine jumps ingress → agent via feeds_agent_node_id. Keep that flow
            # edge and attach memory as scope only (Studio-compatible under load).
            for s in steps_in_order:
                if s.kind != "agent":
                    continue
                edges.append(
                    {"source": mem_id, "target": step_node_id[s.name], "wire_role": "scope"}
                )
        else:
            edges = [
                e
                for e in edges
                if not (
                    e.get("source") == spine_source
                    and e.get("target") == spine_target
                    and not e.get("wire_role")
                )
            ]
            edges = [
                e
                for e in edges
                if not (e.get("source") == "trigger-1" and e.get("target") == first_id)
            ]
            edges.insert(0, {"source": spine_source, "target": mem_id})
            edges.insert(1, {"source": mem_id, "target": first_id})
            for s in steps_in_order:
                if s is first or s.kind != "agent":
                    continue
                edges.append(
                    {"source": mem_id, "target": step_node_id[s.name], "wire_role": "scope"}
                )

    for idx, (rubric, scope) in enumerate(quality_gates or []):
        rid = getattr(rubric, "id", None)
        if rid is None:
            if allow_unresolved:
                rid = 0
            else:
                raise CompileError("quality rubric not resolved", code="unresolved_rubric")
        else:
            rid = int(rid)
            if rid <= 0 and not allow_unresolved:
                raise CompileError("rubric id must be a positive integer", code="bad_rubric_id")
        qid = f"qg-{idx + 1}"
        snapshot = {
            "rules_text": getattr(rubric, "rules_text", "") or "",
            "on_fail": getattr(rubric, "on_fail", "continue") or "continue",
            "flags_enabled": getattr(rubric, "flags_enabled", None)
            or {"hallucination": True, "policy_violation": True, "confidence": True},
            "name": getattr(rubric, "name", None) or getattr(rubric, "key", "rubric"),
            "version": getattr(rubric, "version", 1) or 1,
        }
        nodes.append(
            {
                "id": qid,
                "type": "quality_gate",
                "config": {
                    "display_name": snapshot["name"],
                    "rubric_id": int(rid),
                    "rubric_snapshot": snapshot,
                },
            }
        )
        if not scope:
            raise CompileError("attach_quality_gate scope cannot be empty", code="qg_empty_scope")
        for item in scope:
            meta = get_meta(item) if callable(item) else None
            if not isinstance(meta, StepMeta):
                raise CompileError("quality gate scope item is not a workflow step", code="qg_scope")
            target = step_node_id.get(meta.name)
            if not target:
                raise CompileError(
                    f"quality gate scope step {meta.name!r} is not in this workflow",
                    code="qg_scope",
                )
            ntype = next(n["type"] for n in nodes if n["id"] == target)
            if ntype != "agent":
                raise CompileError(
                    f"quality gate scope only supports agent steps; {meta.name!r} is {ntype}. "
                    "HITL gates are not QG-scoped — put the agent that feeds the HITL in scope instead.",
                    code="qg_scope_agent_only",
                )
            edges.append({"source": qid, "target": target, "wire_role": "scope"})

    return lint_graph({"nodes": nodes, "edges": edges})


def compile_flow(
    flow_fn: Callable[..., Any],
    *,
    quality_gates: list[tuple[Any, list[Any]]] | None = None,
    memory: Any | None = None,
    live_source: Any | None = None,
    allow_unresolved: bool = False,
) -> dict[str, Any]:
    """Trace ``@workflow`` function and emit graph JSON."""
    flow_meta = get_meta(flow_fn)
    if not isinstance(flow_meta, FlowMeta):
        raise CompileError("@workflow decorator required", code="missing_workflow")

    candidates = _discover_candidates(flow_fn)
    if not candidates:
        raise CompileError("workflow references no @agent/@hitl_gate steps", code="empty_flow")

    steps_in_order, edges_pairs = _trace_call_graph(flow_fn, candidates)
    return _build_graph_from_steps(
        steps_in_order,
        edges_pairs,
        quality_gates=quality_gates,
        memory=memory,
        live_source=live_source,
        allow_unresolved=allow_unresolved,
    )


def compile_linear_steps(
    steps: list[Any],
    *,
    quality_gates: list[tuple[Any, list[Any]]] | None = None,
    memory: Any | None = None,
    live_source: Any | None = None,
    allow_unresolved: bool = False,
) -> dict[str, Any]:
    """Compile an explicit ordered list of decorated step functions."""
    metas: list[StepMeta] = []
    for s in steps:
        meta = get_meta(s)
        if not isinstance(meta, StepMeta):
            raise CompileError("linear steps must be @agent/@hitl_gate functions", code="bad_step")
        metas.append(meta)
    edges = [(metas[i], metas[i + 1]) for i in range(len(metas) - 1)]
    return _build_graph_from_steps(
        metas,
        edges,
        quality_gates=quality_gates,
        memory=memory,
        live_source=live_source,
        allow_unresolved=allow_unresolved,
    )
