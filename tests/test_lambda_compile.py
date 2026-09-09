"""Unit tests: lambda → ctx.* compiler."""

from __future__ import annotations

import pytest

from veeshtral.errors import CompileError
from veeshtral.lambda_compile import compile_condition_lambda


def test_simple_or_expression():
    expr = compile_condition_lambda(lambda ctx: ctx.amount_over_limit or ctx.match_failed)
    assert expr == "ctx.amount_over_limit or ctx.match_failed"


def test_comparison():
    expr = compile_condition_lambda(lambda ctx: ctx.risk_score > 0.7)
    assert "ctx.risk_score" in expr
    assert ">" in expr


def test_rejects_calls():
    with pytest.raises(CompileError, match="calls"):
        compile_condition_lambda(lambda ctx: len(ctx.output))  # type: ignore[arg-type]


def test_rejects_subscript():
    with pytest.raises(CompileError, match="subscript"):
        compile_condition_lambda(lambda ctx: ctx["x"])  # type: ignore[index]


def test_rejects_closure():
    flag = True

    def make():
        return lambda ctx: ctx.ok or flag

    with pytest.raises(CompileError, match="closure"):
        compile_condition_lambda(make())


def test_none_unconditional():
    assert compile_condition_lambda(None) is None


def test_rejects_dunder_and_calls():
    with pytest.raises(CompileError):
        compile_condition_lambda(lambda ctx: ctx.__class__)  # type: ignore[attr-defined]
    with pytest.raises(CompileError, match="calls"):
        compile_condition_lambda(lambda ctx: abs(ctx.risk_score))  # type: ignore[arg-type]


def test_renames_non_ctx_param_without_corrupting_string_literals():
    expr = compile_condition_lambda(lambda c: c.status == "c")  # type: ignore[misc]
    assert expr == "ctx.status == 'c'"


def test_rejects_ambiguous_multiple_lambdas_same_line():
    # Two single-arg lambdas on one physical line → must fail closed.
    with pytest.raises(CompileError, match="ambiguous"):
        compile_condition_lambda(
            (lambda ctx: ctx.a, lambda ctx: ctx.b)[0]  # type: ignore[misc]
        )