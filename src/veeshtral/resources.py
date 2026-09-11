"""Resource objects with create-if-missing-by-key semantics."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from veeshtral.client import Client, get_client
from veeshtral.errors import ApiError, CompileError

_KEY_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$")
_INACTIVE = frozenset({"inactive", "pending"})
_INACTIVE_SKILL = frozenset({"draft", "archived", "inactive"})
_INACTIVE_RUBRIC = frozenset({"draft", "archived"})
_CONFLICT_TYPS = frozenset(
    {
        "byo_agent_key_conflict",
        "developer_agent_key_conflict",
        "idempotency_request_in_progress",
        "skill_key_conflict",
        "rubric_key_conflict",
    }
)


def _normalize_key(raw: str) -> str:
    key = str(raw or "").strip().lower()
    if not _KEY_RE.fullmatch(key):
        raise CompileError(
            "key must be 1–63 chars: lowercase alphanumeric with optional - or _ "
            "(must start/end with alphanumeric)",
            code="bad_key",
        )
    return key


def _is_conflict(exc: ApiError) -> bool:
    """Only typed create-if-missing races — not every 409/422 (validation, entitlements)."""
    typ = str(exc.typ or "")
    if typ in _CONFLICT_TYPS:
        return True
    # Some deployments omit `type` on race 409s; still recover via keyed lookup.
    if exc.status_code == 409 and not typ:
        return True
    # Skills #412: duplicate key is 422 (not 409). Prefer typed detail; fall back to message.
    if exc.status_code == 422:
        if typ == "skill_key_conflict":
            return True
        msg = str(exc).lower()
        if "unavailable skill_key" in msg:
            return True
    return False


def _retry_lookup(lookup: Callable[[], Any | None], *, attempts: int = 5) -> Any | None:
    for i in range(attempts):
        found = lookup()
        if found is not None:
            return found
        if i < attempts - 1:
            time.sleep(0.15 * (i + 1))
    return None


def _reject_inactive_row(row: dict[str, Any], *, key: str) -> None:
    status = str(row.get("status") or "").lower()
    if status in _INACTIVE:
        raise CompileError(
            f"Agent(key={key!r}) exists but is {status!r}. "
            "Reactivate it in My Agents or use a different agent_key.",
            code="agent_inactive",
        )


def _reject_inactive_skill(row: dict[str, Any], *, key: str) -> None:
    status = str(row.get("status") or "").lower()
    if status in _INACTIVE_SKILL:
        raise CompileError(
            f"Skill(key={key!r}) exists but is {status!r}. "
            "Publish/activate it in Skills or use a different skill_key.",
            code="skill_inactive",
        )


def _reject_inactive_rubric(row: dict[str, Any], *, key: str) -> None:
    status = str(row.get("status") or "").lower()
    if status in _INACTIVE_RUBRIC:
        raise CompileError(
            f"QualityRubric(key={key!r}) exists but is {status!r}. "
            "Activate it in Quality Rubrics or use a different rubric_key.",
            code="rubric_inactive",
        )


@dataclass
class Agent:
    key: str
    name: str | None = None
    llm_provider: str = "openai_compatible"
    llm_model: str | None = None
    api_endpoint: str | None = None
    api_key: str | None = None
    a2a_enabled: bool = False
    description: str | None = None
    id: int | None = None
    _marketplace: bool = False

    def __post_init__(self) -> None:
        self.key = _normalize_key(self.key)
        if self.id is not None and int(self.id) <= 0:
            raise CompileError("agent id must be a positive integer", code="bad_agent_id")

    def __repr__(self) -> str:  # pragma: no cover
        return f"Agent(key={self.key!r}, id={self.id}, api_key={'***' if self.api_key else None})"

    def _lookup(self, c: Client) -> dict[str, Any] | None:
        # Single GET including inactive — reject inactive in-process (avoids 2× RTT under load).
        found = c.request(
            "GET",
            "/api/business/agents",
            params={
                "agent_key": self.key,
                "limit": 1,
                "include_inactive": True,
                "include_total": False,
            },
        )
        items = (found or {}).get("items") if isinstance(found, dict) else found
        if isinstance(items, list) and items:
            row = items[0]
            if str(row.get("agent_key") or "").lower() == self.key:
                return row
            # Misbehaving/old API — never adopt the wrong agent into a graph.
            return None
        return None

    def _adopt(self, row: dict[str, Any]) -> "Agent":
        _reject_inactive_row(row, key=self.key)
        self.id = int(row["id"])
        self.name = row.get("name") or self.name
        return self

    def ensure(self, client: Client | None = None) -> "Agent":
        if self._marketplace:
            raise CompileError(
                "marketplace agents are resolved via marketplace.install, not Agent.ensure",
                code="marketplace_vs_create",
            )
        if self.id is not None:
            return self
        c = client or get_client()
        row = self._lookup(c)
        if row is not None:
            self.api_key = None
            return self._adopt(row)
        if not self.api_endpoint and not self.a2a_enabled:
            raise CompileError(
                f"Agent(key={self.key!r}) create requires api_endpoint (or a2a_enabled=True)",
                code="missing_api_endpoint",
            )
        payload = {
            "name": self.name or self.key,
            "agent_key": self.key,
            "description": self.description,
            "api_endpoint": self.api_endpoint,
            "api_key": self.api_key,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "a2a_enabled": self.a2a_enabled,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        try:
            created = c.request("POST", "/api/business/agents", json=payload)
        except ApiError as exc:
            if _is_conflict(exc):
                recovered = _retry_lookup(lambda: self._lookup(c))
                if recovered is not None:
                    # Do not keep a secret that was never needed after adopt.
                    self.api_key = None
                    return self._adopt(recovered)
            raise
        self.id = int(created["id"])
        # Write-once: clear plaintext after successful create (CWE-312).
        self.api_key = None
        return self


@dataclass
class Skill:
    key: str
    name: str | None = None
    body_markdown: str = ""
    id: int | None = None

    def __post_init__(self) -> None:
        self.key = _normalize_key(self.key)
        if self.id is not None and int(self.id) <= 0:
            raise CompileError("skill id must be a positive integer", code="bad_skill_id")
        body = self.body_markdown or ""
        if len(body.encode("utf-8")) > 1_048_576:
            raise CompileError(
                "skill body_markdown exceeds 1 MiB UTF-8",
                code="skill_body_too_large",
            )

    def _lookup(self, c: Client) -> dict[str, Any] | None:
        # Prefer ACTIVE tenant-owned exact key (never adopt catalog or draft twins).
        listed = c.request(
            "GET",
            "/api/business/skills",
            params={
                "skill_key": self.key,
                "source": "tenant",
                "status": "active",
                "limit": 5,
                "offset": 0,
            },
        )
        items = (listed or {}).get("items") if isinstance(listed, dict) else listed
        if not isinstance(items, list):
            return None
        for row in items:
            if bool(row.get("is_platform_catalog")):
                continue
            if str(row.get("skill_key") or "").lower() != self.key:
                continue
            _reject_inactive_skill(row, key=self.key)
            return row
        return None

    def ensure(self, client: Client | None = None) -> "Skill":
        if self.id is not None:
            return self
        c = client or get_client()
        row = self._lookup(c)
        if row is not None:
            self.id = int(row["id"])
            return self
        payload = {
            "skill_key": self.key,
            "name": self.name or self.key,
            "body_markdown": self.body_markdown or f"# {self.key}",
            "publish": True,
        }
        try:
            created = c.request("POST", "/api/business/skills", json=payload)
        except ApiError as exc:
            if _is_conflict(exc):
                recovered = _retry_lookup(lambda: self._lookup(c))
                if recovered is not None:
                    self.id = int(recovered["id"])
                    return self
                # Conflict with a draft/archived twin — surface a clear activation error.
                any_status = c.request(
                    "GET",
                    "/api/business/skills",
                    params={
                        "skill_key": self.key,
                        "source": "tenant",
                        "limit": 5,
                        "offset": 0,
                    },
                )
                items = (any_status or {}).get("items") if isinstance(any_status, dict) else any_status
                if isinstance(items, list):
                    for cand in items:
                        if str(cand.get("skill_key") or "").lower() == self.key:
                            _reject_inactive_skill(cand, key=self.key)
            raise
        self.id = int(created["id"])
        return self


@dataclass
class QualityRubric:
    key: str
    name: str | None = None
    rules_text: str = ""
    on_fail: str = "continue"
    flags_enabled: dict[str, bool] | None = None
    id: int | None = None
    version: int = 1

    def __post_init__(self) -> None:
        self.key = _normalize_key(self.key)
        if self.id is not None and int(self.id) <= 0:
            raise CompileError("rubric id must be a positive integer", code="bad_rubric_id")
        if self.on_fail not in ("continue", "hitl", "block"):
            raise CompileError("on_fail must be continue|hitl|block", code="bad_on_fail")

    def _lookup(self, c: Client) -> dict[str, Any] | None:
        # Prefer ACTIVE exact rubric_key (list is fuzzy ILIKE — never adopt first hit).
        offset = 0
        page = 50
        while True:
            listed = c.request(
                "GET",
                "/api/business/rubrics",
                params={
                    "search": self.key,
                    "status": "active",
                    "limit": page,
                    "offset": offset,
                },
            )
            items = (listed or {}).get("items") if isinstance(listed, dict) else listed
            if not isinstance(items, list) or not items:
                return None
            for row in items:
                if str(row.get("rubric_key") or "").lower() != self.key:
                    continue
                _reject_inactive_rubric(row, key=self.key)
                return row
            if len(items) < page:
                return None
            offset += page
            if offset > 2000:
                return None

    def _adopt(self, row: dict[str, Any]) -> "QualityRubric":
        _reject_inactive_rubric(row, key=self.key)
        self.id = int(row["id"])
        self.rules_text = row.get("rules_text") or self.rules_text
        self.on_fail = row.get("on_fail") or self.on_fail
        self.version = int(row.get("version") or 1)
        return self

    def ensure(self, client: Client | None = None) -> "QualityRubric":
        if self.id is not None:
            return self
        c = client or get_client()
        row = self._lookup(c)
        if row is not None:
            return self._adopt(row)
        payload = {
            "rubric_key": self.key,
            "name": self.name or self.key,
            "rules_text": self.rules_text,
            "on_fail": self.on_fail,
            "flags_enabled": self.flags_enabled
            or {"hallucination": True, "policy_violation": True, "confidence": True},
            "status": "active",
        }
        try:
            created = c.request("POST", "/api/business/rubrics", json=payload)
        except ApiError as exc:
            if _is_conflict(exc):
                recovered = _retry_lookup(lambda: self._lookup(c))
                if recovered is not None:
                    return self._adopt(recovered)
            raise
        self.id = int(created["id"])
        self.version = int(created.get("version") or 1)
        return self


@dataclass
class Memory:
    scope: str = "workflow"
    retention_class: str = "standard"


@dataclass
class LiveSource:
    """Live ingress for a workflow: webhook/Kafka/SSE/WebSocket or phone/browser voice.

    Stream kinds emit a ``stream_source`` node. Voice kinds
    (``inbound_phone`` / ``outbound_phone`` / ``browser``) emit a ``voice_channel``
    node. A graph may include at most one live ingress.
    """

    kind: str = "inbound_webhook"
    max_events: int = 10
    # Voice-channel fields (ignored for stream kinds)
    phone_number: str | None = None
    outbound_number_consented: bool = False
    max_hold_minutes: int = 5
    barge_in: bool = True
    max_silence_ms: int = 12000
    stt_provider: str = "deepgram"
    tts_provider: str = "elevenlabs"
    voice_id: str | None = None

    def __post_init__(self) -> None:
        try:
            n = int(self.max_events)
        except (TypeError, ValueError) as exc:
            raise CompileError("max_events must be an integer", code="bad_max_events") from exc
        if n < 1 or n > 10_000:
            raise CompileError(
                "max_events must be between 1 and 10000",
                code="bad_max_events",
            )
        self.max_events = n
        try:
            hold = int(self.max_hold_minutes)
        except (TypeError, ValueError) as exc:
            raise CompileError("max_hold_minutes must be an integer", code="bad_max_hold") from exc
        if hold < 1 or hold > 30:
            raise CompileError(
                "max_hold_minutes must be between 1 and 30",
                code="bad_max_hold",
            )
        self.max_hold_minutes = hold
        try:
            silence = int(self.max_silence_ms)
        except (TypeError, ValueError) as exc:
            raise CompileError("max_silence_ms must be an integer", code="bad_max_silence") from exc
        if silence < 0:
            raise CompileError("max_silence_ms must be >= 0", code="bad_max_silence")
        self.max_silence_ms = silence
        self.kind = str(self.kind or "inbound_webhook").strip().lower()
        # Aliases → canonical voice channel
        if self.kind in ("voice", "voice_channel"):
            self.kind = "inbound_phone"
        self.stt_provider = str(self.stt_provider or "deepgram").strip().lower() or "deepgram"
        self.tts_provider = str(self.tts_provider or "elevenlabs").strip().lower() or "elevenlabs"
        if self.phone_number is not None:
            self.phone_number = str(self.phone_number).strip() or None
        if self.voice_id is not None:
            self.voice_id = str(self.voice_id).strip() or None
        self.outbound_number_consented = bool(self.outbound_number_consented)
        self.barge_in = bool(self.barge_in)

    def is_voice(self) -> bool:
        return self.kind in ("inbound_phone", "outbound_phone", "browser")

