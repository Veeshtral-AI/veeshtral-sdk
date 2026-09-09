"""HTTP client — JWT or X-API-Key for authoring and runs."""

from __future__ import annotations

import logging
import os
import random
import time
import uuid
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from veeshtral.auth import Credentials, credentials_from_env, validate_base_url
from veeshtral.errors import ApiError, AuthError, GraphRejected

_RETRY_STATUSES = frozenset({429, 502, 503})
_DEFAULT_TIMEOUT = 60.0
# Bound hostile / oversized API responses before JSON parse (Snyk: CWE-400).
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_IDEMPOTENCY_IN_PROGRESS = frozenset(
    {"idempotency_request_in_progress", "idempotency_in_progress"}
)
logger = logging.getLogger("veeshtral.client")


def _retry_after_seconds(resp: httpx.Response, *, fallback: float) -> float:
    raw = (resp.headers.get("Retry-After") or "").strip()
    if not raw:
        return fallback
    try:
        return max(fallback, min(120.0, float(raw)))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            delay = (dt.timestamp() - time.time())
            return max(fallback, min(120.0, delay))
    except (TypeError, ValueError, OverflowError):
        pass
    return fallback


def _error_typ(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    detail = body.get("detail", body)
    if isinstance(detail, dict):
        typ = detail.get("type") or detail.get("error") or detail.get("typ")
        return str(typ) if typ else None
    return None


class Client:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000",
        credentials: Credentials | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        debug: bool | None = None,
    ) -> None:
        self.base_url = validate_base_url(base_url)
        self.credentials = credentials or Credentials()
        self._timeout = timeout
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        self._refreshed_once = False
        if debug is None:
            debug = os.environ.get("VEESHTRAL_DEBUG", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
        self.debug = bool(debug)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def uses_api_key(self) -> bool:
        """True when requests will authenticate with X-API-Key (no JWT available)."""
        if self.credentials.access_token:
            return False
        if self.credentials.email and self.credentials.password:
            return False
        return bool(self.credentials.api_key)

    def ensure_jwt(self) -> str:
        if self.credentials.access_token:
            return self.credentials.access_token
        if not self.credentials.email or not self.credentials.password:
            raise AuthError(
                "JWT required. Provide access_token or email/password, "
                "or use api_key for API-key auth."
            )
        resp = self._http.post(
            "/api/auth/login",
            json={"email": self.credentials.email, "password": self.credentials.password},
            follow_redirects=False,
        )
        content = resp.content or b""
        if len(content) > _MAX_RESPONSE_BYTES:
            raise AuthError(
                f"login response exceeds {_MAX_RESPONSE_BYTES} bytes; refusing to parse",
                status_code=resp.status_code,
            )
        if resp.status_code >= 400:
            raise AuthError(
                f"login failed: {resp.status_code}",
                status_code=resp.status_code,
            )
        try:
            data = resp.json() if content else {}
        except Exception as exc:
            raise AuthError("login response is not JSON") from exc
        if not isinstance(data, dict):
            raise AuthError("login response is not a JSON object")
        token = data.get("access_token") or data.get("token")
        if not token:
            raise AuthError("login response missing access_token")
        self.credentials.access_token = str(token)
        return self.credentials.access_token

    def _auth_headers(
        self,
        *,
        prefer_api_key: bool = False,
        idempotency_key: str | None = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        use_key = prefer_api_key or self.uses_api_key
        if use_key and self.credentials.api_key:
            headers["X-API-Key"] = self.credentials.api_key
        elif self.credentials.access_token or (
            self.credentials.email and self.credentials.password
        ):
            headers["Authorization"] = f"Bearer {self.ensure_jwt()}"
        elif self.credentials.api_key:
            headers["X-API-Key"] = self.credentials.api_key
        else:
            raise AuthError(
                "No credentials configured. Set access_token, email/password, or api_key."
            )
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        require_jwt: bool = False,
        prefer_api_key: bool = False,
        json: Any = None,
        params: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        retries: int = 5,
    ) -> Any:
        """HTTP helper.

        Auth preference: JWT when available, else X-API-Key.
        Retries 429/502/503 (honors Retry-After) and ``idempotency_request_in_progress``.
        """
        if require_jwt and self.uses_api_key:
            raise AuthError(
                f"{method} {path} requires JWT; configure email/password or access_token"
            )

        # Path-only requests — absolute URLs would bypass base_url (SSRF via caller).
        path_s = str(path or "")
        if "://" in path_s or path_s.startswith("//"):
            raise AuthError(
                "request path must be relative (e.g. /api/workflows); absolute URLs are rejected",
            )
        if not path_s.startswith("/"):
            raise AuthError("request path must start with '/'")

        if (
            (prefer_api_key or self.uses_api_key)
            and method.upper() == "POST"
            and path.rstrip("/").endswith("/workflows")
            and not path.rstrip("/").endswith("/run")
            and not idempotency_key
        ):
            idempotency_key = self.new_idempotency_key(prefix="wf-create")

        # Always mint a run Idempotency-Key so 502/503 retries cannot double-execute
        # (JWT authoring path previously omitted the header — HIGH duplicate-run risk).
        if (
            method.upper() == "POST"
            and path.rstrip("/").endswith("/run")
            and not idempotency_key
        ):
            idempotency_key = self.new_idempotency_key(prefix="run")

        attempt = 0
        while True:
            attempt += 1
            headers = self._auth_headers(
                prefer_api_key=prefer_api_key,
                idempotency_key=idempotency_key,
            )
            resp = self._http.request(method, path, headers=headers, json=json, params=params)
            if self.debug:
                logger.debug("%s %s -> %s", method.upper(), path, resp.status_code)

            if (
                resp.status_code == 401
                and not (prefer_api_key or self.uses_api_key)
                and not self._refreshed_once
            ):
                self._refreshed_once = True
                self.credentials.access_token = None
                if self.credentials.email and self.credentials.password:
                    self.ensure_jwt()
                    continue
                raise AuthError("unauthorized", status_code=401)

            backoff = min(2.0, 0.2 * (2 ** (attempt - 1))) + random.random() * 0.1
            if resp.status_code in _RETRY_STATUSES and attempt <= retries:
                delay = (
                    _retry_after_seconds(resp, fallback=backoff)
                    if resp.status_code == 429
                    else backoff
                )
                time.sleep(delay)
                continue

            # Idempotency lock held by a concurrent twin request — retry same key.
            if resp.status_code == 409 and attempt <= retries:
                content = resp.content or b""
                if len(content) <= _MAX_RESPONSE_BYTES:
                    try:
                        body = resp.json() if content else None
                    except Exception:
                        body = None
                else:
                    body = None
                typ = _error_typ(body)
                if typ in _IDEMPOTENCY_IN_PROGRESS:
                    time.sleep(backoff)
                    continue

            parsed = self._parse(resp)
            # Successful request clears the one-shot refresh latch so long-lived
            # clients can re-login again on a later token expiry.
            if resp.status_code < 400:
                self._refreshed_once = False
            return parsed

    def _parse(self, resp: httpx.Response) -> Any:
        if resp.status_code == 204:
            return None
        content = resp.content or b""
        if len(content) > _MAX_RESPONSE_BYTES:
            raise ApiError(
                f"response body exceeds {_MAX_RESPONSE_BYTES} bytes "
                f"({len(content)} observed); refusing to parse",
                status_code=resp.status_code,
                typ="response_too_large",
                body=None,
            )
        try:
            body = resp.json() if content else None
        except Exception:
            body = resp.text
        if resp.status_code >= 400:
            typ = None
            message = f"HTTP {resp.status_code}"
            if isinstance(body, dict):
                detail = body.get("detail", body)
                if isinstance(detail, dict):
                    typ = detail.get("type") or detail.get("error") or detail.get("typ")
                    message = str(detail.get("message") or detail)
                elif isinstance(detail, list) and detail:
                    first = detail[0]
                    if isinstance(first, dict):
                        typ = first.get("type") or first.get("typ")
                        message = str(first.get("msg") or first)
                    else:
                        message = str(detail)
                else:
                    message = str(detail)
            if resp.status_code == 422:
                raise GraphRejected(message, status_code=422, typ=typ, body=body)
            if resp.status_code in (401, 403):
                raise AuthError(message, status_code=resp.status_code)
            raise ApiError(message, status_code=resp.status_code, typ=typ, body=body)
        return body

    @staticmethod
    def new_idempotency_key(prefix: str = "sdk") -> str:
        return f"{prefix}-{uuid.uuid4().hex}"

    @staticmethod
    def content_idempotency_key(prefix: str, *parts: Any) -> str:
        """Stable key for create-if-missing coalescing (same content → same key)."""
        import hashlib
        import json

        payload = json.dumps(
            parts,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"{prefix}-{digest}"


_default_client: Client | None = None


def configure(
    *,
    base_url: str = "http://127.0.0.1:8000",
    email: str | None = None,
    password: str | None = None,
    access_token: str | None = None,
    api_key: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    debug: bool | None = None,
) -> Client:
    """Configure the process-wide default client. Supports ``with configure(...) as c:``."""
    global _default_client
    env = credentials_from_env()
    creds = Credentials(
        access_token=access_token or env.access_token,
        api_key=api_key or env.api_key,
        email=email or env.email,
        password=password or env.password,
    )
    if _default_client is not None:
        _default_client.close()
    _default_client = Client(
        base_url=base_url, credentials=creds, timeout=timeout, debug=debug
    )
    return _default_client


def get_client() -> Client:
    global _default_client
    if _default_client is None:
        env = credentials_from_env()
        base = os.environ.get("VEESHTRAL_BASE_URL", "http://127.0.0.1:8000")
        _default_client = Client(base_url=base, credentials=env)
    return _default_client
