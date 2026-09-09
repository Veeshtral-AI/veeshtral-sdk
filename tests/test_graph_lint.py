"""Unit tests: graph lint limits and DAG rules."""

from __future__ import annotations

import pytest

from veeshtral.errors import CompileError
from veeshtral.graph_lint import lint_graph


def _linear():
    return {
        "nodes": [
            {"id": "trigger-1", "type": "trigger", "config": {}},
            {"id": "a1", "type": "agent", "config": {"agent_id": 1, "planner_assigned_task": "t"}},
            {"id": "end-1", "type": "end", "config": {}},
        ],
        "edges": [
            {"source": "trigger-1", "target": "a1"},
            {"source": "a1", "target": "end-1"},
        ],
    }


def test_valid_linear():
    g = lint_graph(_linear())
    assert len(g["nodes"]) == 3


def test_cycle_rejected():
    g = _linear()
    g["edges"].append({"source": "end-1", "target": "a1"})
    # end outgoing also fails — use agent cycle
    g = {
        "nodes": [
            {"id": "trigger-1", "type": "trigger", "config": {}},
            {"id": "a1", "type": "agent", "config": {"agent_id": 1, "planner_assigned_task": "t"}},
            {"id": "a2", "type": "agent", "config": {"agent_id": 1, "planner_assigned_task": "t"}},
            {"id": "end-1", "type": "end", "config": {}},
        ],
        "edges": [
            {"source": "trigger-1", "target": "a1"},
            {"source": "a1", "target": "a2"},
            {"source": "a2", "target": "a1"},
            {"source": "a2", "target": "end-1"},
        ],
    }
    with pytest.raises(CompileError, match="cycle"):
        lint_graph(g)


def test_qg_requires_scope():
    g = _linear()
    g["nodes"].append(
        {"id": "qg-1", "type": "quality_gate", "config": {"rubric_id": 1}}
    )
    g["edges"].append({"source": "qg-1", "target": "a1"})  # missing scope
    with pytest.raises(CompileError, match="scope"):
        lint_graph(g)


def test_node_limit():
    nodes = [{"id": "trigger-1", "type": "trigger", "config": {}}]
    edges = []
    prev = "trigger-1"
    for i in range(201):
        nid = f"a{i}"
        nodes.append(
            {"id": nid, "type": "agent", "config": {"agent_id": 1, "planner_assigned_task": "t"}}
        )
        edges.append({"source": prev, "target": nid})
        prev = nid
    nodes.append({"id": "end-1", "type": "end", "config": {}})
    edges.append({"source": prev, "target": "end-1"})
    with pytest.raises(CompileError, match="nodes"):
        lint_graph({"nodes": nodes, "edges": edges})
