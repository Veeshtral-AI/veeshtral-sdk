"""Extra coverage for publish upsert and auth helpers."""

from __future__ import annotations

import httpx

from veeshtral.client import Client
from veeshtral.resources import QualityRubric
from veeshtral.workflow import Workflow
import veeshtral
from veeshtral.resources import Agent


def test_publish_upsert_by_name():
    state = {"rev": 1, "patched": 0, "published": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if path == "/api/workflows" and request.method == "GET":
            return httpx.Response(
                200,
                json={"items": [{"id": 7, "name": "ap-invoice-review"}], "total": 1},
            )
        if path == "/api/workflows/7" and request.method == "GET":
            return httpx.Response(200, json={"id": 7, "draft_revision": state["rev"]})
        if path == "/api/workflows/7/draft" and request.method == "PATCH":
            state["patched"] += 1
            state["rev"] += 1
            return httpx.Response(200, json={"id": 7, "draft_revision": state["rev"]})
        if path == "/api/workflows/7/publish" and request.method == "POST":
            state["published"] += 1
            return httpx.Response(200, json={"workflow_version_id": 3, "version_number": 2})
        return httpx.Response(404, json={"detail": path})

    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )

    a = Agent(key="ocr", name="ocr", id=1, api_endpoint="http://127.0.0.1:9/v1")

    @veeshtral.agent(a, task="do")
    def review(invoice_text: str) -> dict: ...

    @veeshtral.workflow_fn(name="ap-invoice-review")
    def flow(invoice_text: str):
        return review(invoice_text)

    wf = Workflow.from_flow(flow, client=c)
    pub = wf.publish()
    assert pub.workflow_id == 7
    assert state["patched"] == 1
    assert state["published"] == 1


def test_ensure_resources_discovers_nonlocal_agents():
    """Nested @agent factories live in nonlocals — must still Agent.ensure before compile."""
    looked_up: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/business/agents" and request.method == "GET":
            looked_up.append(str(request.url.params.get("agent_key") or ""))
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": 42,
                            "agent_key": "nested-ocr",
                            "name": "nested",
                            "status": "active",
                        }
                    ]
                },
            )
        if request.url.path == "/api/workflows" and request.method == "GET":
            return httpx.Response(200, json={"items": [], "total": 0})
        if request.url.path == "/api/workflows" and request.method == "POST":
            return httpx.Response(201, json={"id": 9, "draft_revision": 1})
        if request.url.path.endswith("/draft") and request.method == "PATCH":
            return httpx.Response(200, json={"id": 9, "draft_revision": 2})
        if request.url.path.endswith("/publish") and request.method == "POST":
            return httpx.Response(200, json={"workflow_version_id": 1, "version_number": 1})
        return httpx.Response(404, json={"detail": request.url.path})

    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )

    def build_flow():
        nested = Agent(key="nested-ocr", api_endpoint="http://127.0.0.1:9/v1")

        @veeshtral.agent(nested, task="Extract")
        def review(text: str) -> dict: ...

        @veeshtral.workflow_fn(name="nested-wf")
        def flow(text: str):
            return review(text)

        return flow

    wf = Workflow.from_flow(build_flow(), client=c)
    pub = wf.publish()
    assert pub.workflow_id == 9
    assert "nested-ocr" in looked_up


def test_rubric_ensure_create():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/business/rubrics" and request.method == "GET":
            return httpx.Response(200, json={"items": []})
        if request.url.path == "/api/business/rubrics" and request.method == "POST":
            return httpx.Response(
                201, json={"id": 5, "rubric_key": "ap-safety", "version": 1}
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://127.0.0.1:8000")
    c.credentials.email = "a@b.com"
    c.credentials.password = "x"
    c._http = httpx.Client(
        base_url="http://127.0.0.1:8000", transport=transport, follow_redirects=False
    )
    r = QualityRubric(key="ap-safety", rules_text="rule one", on_fail="hitl")
    r.ensure(c)
    assert r.id == 5
