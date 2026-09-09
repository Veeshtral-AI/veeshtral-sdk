"""Veeshtral Python SDK — decorator workflows on the existing Studio graph engine."""

from __future__ import annotations

from veeshtral.client import Client, configure, get_client
from veeshtral.decorators import agent, branch, hitl_gate, workflow as workflow_fn
from veeshtral.errors import (
    ApiError,
    AuthError,
    CompileError,
    GraphRejected,
    PollTimeout,
    VeeshtralError,
)
from veeshtral.marketplace import Marketplace
from veeshtral.resources import Agent, LiveSource, Memory, QualityRubric, Skill
from veeshtral.results import PublishResult, RunResult
from veeshtral.auth import Credentials
from veeshtral.workflow import Workflow

marketplace = Marketplace()
# Issue #552 / docs alias — same decorator as workflow_fn.
workflow = workflow_fn

__all__ = [
    "Agent",
    "ApiError",
    "AuthError",
    "Client",
    "CompileError",
    "Credentials",
    "GraphRejected",
    "LiveSource",
    "Memory",
    "PollTimeout",
    "PublishResult",
    "QualityRubric",
    "RunResult",
    "Skill",
    "VeeshtralError",
    "Workflow",
    "agent",
    "branch",
    "configure",
    "get_client",
    "hitl_gate",
    "marketplace",
    "workflow",
    "workflow_fn",
    "__version__",
]

__version__ = "0.1.1"
