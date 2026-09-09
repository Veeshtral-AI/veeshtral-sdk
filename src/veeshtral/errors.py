"""Typed errors for the Veeshtral SDK."""

from __future__ import annotations

from typing import Any


class VeeshtralError(Exception):
    """Base SDK error."""


class CompileError(VeeshtralError):
    """Raised when decorator/graph compilation fails locally."""

    def __init__(self, message: str, *, code: str = "compile_error") -> None:
        super().__init__(message)
        self.code = code


class AuthError(VeeshtralError):
    """Authentication / authorization failures."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ApiError(VeeshtralError):
    """HTTP API error from Veeshtral."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        typ: str | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.typ = typ
        self.body = body


class GraphRejected(ApiError):
    """Server rejected the graph (422 validation)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 422,
        typ: str | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message, status_code=status_code, typ=typ, body=body)


class PollTimeout(VeeshtralError):
    """Raised when Workflow.run() exhausts poll_timeout_s before a terminal job status."""

    def __init__(
        self,
        message: str,
        *,
        job_id: int | None = None,
        status: str | None = None,
        poll_timeout_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.job_id = job_id
        self.status = status
        self.poll_timeout_s = poll_timeout_s
