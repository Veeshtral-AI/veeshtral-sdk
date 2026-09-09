"""TaskFlow-style decorators that stamp metadata for the graph compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from veeshtral.errors import CompileError
from veeshtral.lambda_compile import compile_condition_lambda

_META_ATTR = "__veeshtral_meta__"

ConditionInput = Union[str, Callable[..., Any], None]


@dataclass
class StepMeta:
    kind: str  # agent | hitl | branch
    fn: Callable[..., Any]
    name: str
    resource: Any = None
    task: str = ""
    skills: list[Any] = field(default_factory=list)
    skill_mode: str = "merge"
    condition: str | None = None
    branch_mode: str | None = None
    is_default: bool = False
    edge_condition: str | None = None


@dataclass
class FlowMeta:
    name: str
    description: str = ""
    fn: Callable[..., Any] | None = None


def get_meta(fn: Any) -> StepMeta | FlowMeta | None:
    return getattr(fn, _META_ATTR, None)


def agent(
    resource: Any,
    *,
    task: str,
    skills: list[Any] | None = None,
    skill_mode: str = "merge",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if not task or not str(task).strip():
        raise CompileError("task= is required on @agent", code="missing_task")
    if len(str(task)) > 8000:
        raise CompileError("task exceeds 8000 characters", code="task_too_long")

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        existing = get_meta(fn)
        meta = StepMeta(
            kind="agent",
            fn=fn,
            name=fn.__name__,
            resource=resource,
            task=str(task).strip(),
            skills=list(skills or []),
            skill_mode=skill_mode if skill_mode in ("merge", "replace") else "merge",
        )
        # Decorators apply bottom-up: @agent above @branch would otherwise wipe branch fields.
        if isinstance(existing, StepMeta):
            if existing.kind == "hitl":
                raise CompileError(
                    "cannot stack @agent on @hitl_gate; use separate steps",
                    code="decorator_conflict",
                )
            meta.edge_condition = existing.edge_condition
            meta.is_default = existing.is_default
            meta.branch_mode = existing.branch_mode
        setattr(fn, _META_ATTR, meta)
        return fn

    return deco


def hitl_gate(
    *,
    condition: ConditionInput = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    expr = _normalize_condition(condition)

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        existing = get_meta(fn)
        if isinstance(existing, StepMeta) and existing.kind == "agent":
            raise CompileError(
                "cannot stack @hitl_gate on @agent; use separate steps "
                "(HITL must be its own node after the agent)",
                code="decorator_conflict",
            )
        meta = StepMeta(
            kind="hitl",
            fn=fn,
            name=fn.__name__,
            condition=expr,
        )
        setattr(fn, _META_ATTR, meta)
        return fn

    return deco


def _normalize_condition(condition: ConditionInput) -> str | None:
    if condition is None:
        return None
    if callable(condition):
        return compile_condition_lambda(condition)
    from veeshtral.lambda_compile import compile_condition_string

    return compile_condition_string(str(condition))


def branch(
    *,
    condition: ConditionInput = None,
    is_default: bool = False,
    mode: str = "first_match",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a step as a branching arm when called from a multi-out upstream.

    Use ``condition=lambda ctx: ...`` (or a ``ctx.*`` string) on if-arms and
    ``is_default=True`` on the else arm. Multi-out parents without any ``@branch``
    arms compile as ``fan_out`` (parallel).
    """
    if mode not in ("first_match", "fan_out"):
        raise CompileError("branch mode must be first_match|fan_out", code="bad_branch_mode")
    if is_default and condition is not None:
        raise CompileError(
            "default branch arm cannot also set condition=",
            code="branch_default_and_condition",
        )
    if mode == "first_match" and not is_default and condition is None:
        raise CompileError(
            "first_match arm requires condition= or is_default=True",
            code="branch_missing_condition",
        )
    expr = None if is_default else _normalize_condition(condition)

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        existing = get_meta(fn)
        if isinstance(existing, StepMeta):
            existing.edge_condition = expr
            existing.is_default = is_default
            existing.branch_mode = mode
            return fn
        meta = StepMeta(
            kind="agent",
            fn=fn,
            name=fn.__name__,
            edge_condition=expr,
            is_default=is_default,
            branch_mode=mode,
        )
        setattr(fn, _META_ATTR, meta)
        return fn

    return deco


def workflow(name: str, *, description: str = "") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    clean = str(name or "").strip()
    if not clean:
        raise CompileError("workflow name is required", code="bad_name")
    desc = description or ""
    if len(desc.encode("utf-8")) > 2_097_152:
        raise CompileError(
            "workflow description exceeds 2 MiB UTF-8",
            code="description_too_large",
        )

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        meta = FlowMeta(name=clean, description=desc, fn=fn)
        setattr(fn, _META_ATTR, meta)
        return fn

    return deco
