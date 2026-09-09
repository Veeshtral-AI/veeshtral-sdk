"""Auth helpers — JWT or X-API-Key for authoring and runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from veeshtral.errors import AuthError


@dataclass
class Credentials:
    access_token: Optional[str] = None
    api_key: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None

    def __repr__(self) -> str:  # pragma: no cover - secret hygiene
        return (
            f"Credentials(access_token={'***' if self.access_token else None}, "
            f"api_key={'***' if self.api_key else None}, email={self.email!r})"
        )


def validate_base_url(base_url: str) -> str:
    raw = (base_url or "").strip().rstrip("/")
    if not raw:
        raise AuthError("base_url is required")
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise AuthError("base_url must be http or https")
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "http" and host not in ("localhost", "127.0.0.1", "::1"):
        raise AuthError("base_url must use https outside localhost")
    return raw


def credentials_from_env() -> Credentials:
    return Credentials(
        access_token=os.environ.get("VEESHTRAL_ACCESS_TOKEN") or None,
        api_key=os.environ.get("VEESHTRAL_API_KEY") or None,
        email=os.environ.get("VEESHTRAL_EMAIL") or None,
        password=os.environ.get("VEESHTRAL_PASSWORD") or None,
    )
