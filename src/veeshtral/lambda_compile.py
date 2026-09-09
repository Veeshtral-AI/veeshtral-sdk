"""Compile ``lambda ctx: ...`` to a restricted ``ctx.*`` expression string.

Never eval/exec the callable — source is parsed via AST only.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from typing import Any, Callable

from veeshtral.errors import CompileError

_MAX_CONDITION_LEN = 4096


class _LambdaValidator(ast.NodeVisitor):
    def __init__(self, ctx_name: str) -> None:
        self.ctx_name = ctx_name
        self.errors: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.errors.append("function calls are not allowed in gate conditions")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.errors.append("subscripts are not allowed; use ctx.field")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.errors.append(f"dunder attribute not allowed: {node.attr}")
        if isinstance(node.value, ast.Name) and node.value.id != self.ctx_name:
            self.errors.append(f"only '{self.ctx_name}.*' attributes are allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.ctx_name:
            return
        if isinstance(node.ctx, ast.Load):
            self.errors.append(f"name '{node.id}' is not allowed; use {self.ctx_name}.field")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # Align with platform gate evaluator (no Pow / MatMult).
        if isinstance(node.op, (ast.Pow, ast.MatMult)):
            self.errors.append(f"operator {type(node.op).__name__} is not allowed in gate conditions")
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.errors.append("comprehensions are not allowed")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.errors.append("comprehensions are not allowed")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.errors.append("comprehensions are not allowed")


def compile_condition_string(text: str) -> str | None:
    """AST-validate a ctx.* string the same way as lambdas (defense-in-depth)."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    if len(cleaned) > _MAX_CONDITION_LEN:
        raise CompileError(
            f"condition exceeds {_MAX_CONDITION_LEN} characters",
            code="condition_too_long",
        )
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise CompileError(
            f"invalid condition expression: {exc.msg}",
            code="condition_syntax",
        ) from exc
    validator = _LambdaValidator("ctx")
    validator.visit(tree)
    if validator.errors:
        raise CompileError("; ".join(validator.errors), code="condition_forbidden")
    body_src = ast.unparse(tree)
    if "ctx." not in body_src and body_src.strip() != "ctx":
        raise CompileError(
            "branch/HITL condition must be a ctx.* expression (or lambda ctx: ...)",
            code="condition_no_ctx",
        )
    return body_src


def _find_lambda_node(fn: Callable[..., Any]) -> ast.Lambda:
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as exc:
        raise CompileError(
            "cannot read source for condition lambda (define it in a .py file, not REPL/exec)",
            code="lambda_no_source",
        ) from exc
    if "lambda" not in src:
        raise CompileError("condition must be a lambda expression", code="lambda_required")

    trees: list[ast.AST] = []
    try:
        trees.append(ast.parse(src))
    except SyntaxError:
        pass
    # Statement may be incomplete when getsource returns a decorator fragment;
    # also try from the first ``lambda`` token wrapped as an expression.
    idx = src.find("lambda")
    fragment = src[idx:]
    for end in range(len(fragment), max(0, len(fragment) - 200), -1):
        chunk = fragment[:end].rstrip().rstrip(",").rstrip(")")
        try:
            trees.append(ast.parse(chunk, mode="eval"))
            break
        except SyntaxError:
            continue
    # Last resort: wrap whole line as module with pass
    try:
        trees.append(ast.parse(f"__veeshtral_lambda = ({fragment.splitlines()[0]})"))
    except SyntaxError:
        pass

    lambdas: list[ast.Lambda] = []
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                lambdas.append(node)
    if not lambdas:
        raise CompileError("could not parse condition lambda source", code="lambda_parse")
    # Prefer the lambda whose argcount matches the callable
    argc = fn.__code__.co_argcount
    for lam in lambdas:
        if len(lam.args.args) == argc:
            return lam
    return lambdas[0]


def compile_condition_lambda(fn: Callable[..., Any] | None) -> str | None:
    """Return normalized ``ctx.*`` expression string, or None if unconditional."""
    if fn is None:
        return None
    if not callable(fn):
        raise CompileError("condition must be callable", code="lambda_type")

    if getattr(fn, "__code__", None) is not None and fn.__code__.co_freevars:
        raise CompileError(
            f"closures are not allowed in gate conditions: {fn.__code__.co_freevars}",
            code="lambda_closure",
        )

    lam = _find_lambda_node(fn)
    args = [a.arg for a in lam.args.args]
    if len(args) != 1:
        raise CompileError("condition lambda must take exactly one argument (ctx)", code="lambda_args")
    ctx_name = args[0]

    validator = _LambdaValidator(ctx_name)
    validator.visit(lam.body)
    if validator.errors:
        raise CompileError("; ".join(validator.errors), code="lambda_forbidden")

    body_src = ast.unparse(lam.body)
    if ctx_name != "ctx":
        body_src = re.sub(rf"\b{re.escape(ctx_name)}\b", "ctx", body_src)

    if "ctx." not in body_src and body_src.strip() != "ctx":
        raise CompileError("condition must reference ctx.* fields", code="lambda_no_ctx")

    if len(body_src) > _MAX_CONDITION_LEN:
        raise CompileError(
            f"condition exceeds {_MAX_CONDITION_LEN} characters",
            code="lambda_too_long",
        )
    return body_src
