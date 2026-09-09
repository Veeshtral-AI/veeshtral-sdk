"""Local graph lint mirroring engine limits (standalone_workflow_graph)."""

from __future__ import annotations

import json
from typing import Any

from veeshtral.errors import CompileError

MAX_NODES = 200
MAX_EDGES = 400
MAX_GRAPH_BYTES = 512 * 1024
MAX_CONDITION_LEN = 4096
MAX_NODE_ID_LEN = 128

ALLOWED_NODE_TYPES = frozenset(
    {
        "trigger",
        "agent",
        "hitl",
        "memory",
        "rag",
        "tool",
        "mcp",
        "stream_source",
        "quality_gate",
        "end",
    }
)
FLOW_TYPES = frozenset({"agent", "hitl", "stream_source", "end"})
AUX_TYPES = frozenset({"tool", "mcp", "memory", "rag", "quality_gate"})
FORBIDDEN_AGENT_KEYS = frozenset(
    {
        "veeshtral_skills",
        "veeshtral_memory",
        "veeshtral_rag",
        "veeshtral_tools",
    }
)


def _is_scope(edge: dict[str, Any]) -> bool:
    return str(edge.get("wire_role") or "").strip().lower() == "scope"


def lint_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Validate and return a cleaned graph. Raises CompileError on failure."""
    if not isinstance(graph, dict):
        raise CompileError("graph must be a dict", code="graph_type")

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise CompileError("graph must have nodes and edges lists", code="graph_shape")
    if len(nodes) > MAX_NODES:
        raise CompileError(
            f"graph has {len(nodes)} nodes; max is {MAX_NODES}",
            code="graph_limits",
        )
    if len(edges) > MAX_EDGES:
        raise CompileError(
            f"graph has {len(edges)} edges; max is {MAX_EDGES}",
            code="graph_limits",
        )

    raw = json.dumps(graph, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_GRAPH_BYTES:
        raise CompileError(
            f"graph is {len(raw)} bytes; max is {MAX_GRAPH_BYTES}",
            code="graph_too_large",
        )

    ids: set[str] = set()
    triggers = 0
    by_id: dict[str, dict[str, Any]] = {}
    for n in nodes:
        if not isinstance(n, dict):
            raise CompileError("invalid node", code="node_type")
        nid = str(n.get("id") or "")
        if not nid or len(nid) > MAX_NODE_ID_LEN:
            raise CompileError(f"invalid node id: {nid!r}", code="node_id")
        if nid in ids:
            raise CompileError(f"duplicate node id: {nid}", code="node_dup")
        ntype = str(n.get("type") or "")
        if ntype not in ALLOWED_NODE_TYPES:
            raise CompileError(f"unknown node type: {ntype}", code="node_type")
        ids.add(nid)
        by_id[nid] = n
        if ntype == "trigger":
            triggers += 1
        cfg = n.get("config") if isinstance(n.get("config"), dict) else {}
        if ntype == "agent":
            for bad in FORBIDDEN_AGENT_KEYS:
                if bad in cfg:
                    raise CompileError(f"forbidden agent config key: {bad}", code="forbidden_key")
            task = cfg.get("planner_assigned_task") or cfg.get("assigned_task") or ""
            if isinstance(task, str) and len(task) > 8000:
                raise CompileError("planner_assigned_task too long", code="task_too_long")

    if triggers != 1:
        raise CompileError("graph must have exactly one trigger", code="trigger_count")
    if not any(str(n.get("type")) == "end" for n in nodes):
        raise CompileError("graph must have at least one end node", code="missing_end")

    # Edge checks + adjacency for flow DAG
    adj: dict[str, list[str]] = {i: [] for i in ids}
    indeg: dict[str, int] = {i: 0 for i in ids}
    for e in edges:
        if not isinstance(e, dict):
            raise CompileError("invalid edge", code="edge_type")
        src, tgt = str(e.get("source") or ""), str(e.get("target") or "")
        if src not in ids or tgt not in ids:
            raise CompileError(f"edge endpoint missing: {src}->{tgt}", code="edge_endpoint")
        if src == tgt:
            raise CompileError("self-loops forbidden", code="self_loop")
        cond = e.get("condition")
        if cond is not None and len(str(cond)) > MAX_CONDITION_LEN:
            raise CompileError("condition too long", code="condition_len")
        src_type = str(by_id[src].get("type"))
        tgt_type = str(by_id[tgt].get("type"))
        if {src_type, tgt_type} == {"agent", "quality_gate"} and not _is_scope(e):
            raise CompileError(
                "quality_gate must attach with wire_role=scope",
                code="quality_gate_scope_required",
            )
        if not _is_scope(e):
            adj[src].append(tgt)
            indeg[tgt] += 1

    # Trigger indegree must be 0 on flow edges
    trigger_id = next(n["id"] for n in nodes if n["type"] == "trigger")
    if indeg[str(trigger_id)] != 0:
        raise CompileError("trigger must have no incoming flow edges", code="trigger_indegree")

    # Kahn cycle detection on flow edges
    queue = [i for i, d in indeg.items() if d == 0]
    seen = 0
    while queue:
        cur = queue.pop()
        seen += 1
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if seen != len(ids):
        raise CompileError("Graph contains a cycle (DAG required)", code="graph_cycle")

    # first_match validation
    for n in nodes:
        cfg = n.get("config") if isinstance(n.get("config"), dict) else {}
        if str(cfg.get("branch_mode") or "").lower() != "first_match":
            continue
        outs = [
            e
            for e in edges
            if str(e.get("source")) == str(n["id"]) and not _is_scope(e)
        ]
        if len(outs) < 2:
            continue
        defaults = [e for e in outs if e.get("is_default") is True]
        if len(defaults) != 1:
            raise CompileError(
                f"first_match node {n['id']} needs exactly one is_default else arm",
                code="branch_first_match_invalid",
            )
        for e in outs:
            if e.get("is_default") is True:
                continue
            if not str(e.get("condition") or "").strip():
                raise CompileError(
                    f"first_match edge from {n['id']} missing condition",
                    code="branch_first_match_invalid",
                )

    # Reachability of flow-chain nodes from trigger
    reachable: set[str] = set()
    stack = [str(trigger_id)]
    flow_adj = {
        i: [t for t in adj[i]]
        for i in ids
    }
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        stack.extend(flow_adj.get(cur, []))
    for n in nodes:
        ntype = str(n.get("type"))
        if ntype in FLOW_TYPES or ntype == "trigger":
            if str(n["id"]) not in reachable:
                raise CompileError(
                    f"unreachable flow node: {n['id']}",
                    code="unreachable",
                )

    for n in nodes:
        if str(n.get("type")) == "end":
            outs = [e for e in edges if str(e.get("source")) == str(n["id"]) and not _is_scope(e)]
            if outs:
                raise CompileError("end node cannot have outgoing flow edges", code="end_outgoing")

    return graph
