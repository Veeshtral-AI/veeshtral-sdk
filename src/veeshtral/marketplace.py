"""Marketplace install/find — link only, never creates BYO agents."""

from __future__ import annotations

from veeshtral.client import Client, get_client
from veeshtral.errors import CompileError
from veeshtral.resources import Agent, _normalize_key


class Marketplace:
    def find_all(self, agent_key: str, *, client: Client | None = None) -> list[Agent]:
        """Return all approved marketplace agents matching ``agent_key`` exactly."""
        key = _normalize_key(agent_key)
        c = client or get_client()
        rows = c.request("GET", "/api/agents", params={"agent_key": key, "limit": 20})
        if not isinstance(rows, list):
            rows = (rows or {}).get("items") or []
        out: list[Agent] = []
        for row in rows:
            if str(row.get("agent_key") or "").lower() != key:
                continue
            status = str(row.get("status") or "active").lower()
            if status in ("inactive", "pending"):
                continue
            out.append(
                Agent(
                    key=key,
                    name=row.get("name"),
                    id=int(row["id"]),
                    _marketplace=True,
                )
            )
        return out

    def find(self, agent_key: str, *, client: Client | None = None) -> Agent | None:
        matches = self.find_all(agent_key, client=client)
        if not matches:
            return None
        if len(matches) > 1:
            raise CompileError(
                f"multiple marketplace agents with agent_key={agent_key!r}; "
                f"ids={[m.id for m in matches]}. Contact support or use a unique key.",
                code="marketplace_ambiguous",
            )
        return matches[0]

    def install(self, agent_key: str, *, client: Client | None = None) -> Agent:
        """Resolve an approved marketplace agent by key. Never POSTs /api/business/agents."""
        found = self.find(agent_key, client=client)
        if found is None:
            raise CompileError(
                f"marketplace agent not found or not approved: {agent_key}. "
                "Confirm the key is published and active in the marketplace catalog.",
                code="marketplace_not_found",
            )
        return found
