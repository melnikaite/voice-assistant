"""wwyd integration: connection storage, OAuth, tools, automatic capture.

wwyd is mocked over HTTP (``app.wwyd._TRANSPORT``) with a small fake that
speaks its discovery, DCR, token, revoke and MCP endpoints.
"""
from __future__ import annotations

import json
import logging
from urllib.parse import parse_qs, urlsplit

import httpx
import numpy as np
import pytest

from app import wwyd, wwyd_capture
from app.storage import (
    get_wwyd_connection,
    save_speaker_profile,
    save_wwyd_connection,
    set_wwyd_audience_map,
)

BASE = "http://127.0.0.1:8749"
SECRET_ACCESS = "acc-SECRET-1"
SECRET_REFRESH = "ref-SECRET-1"


class FakeWwyd:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []  # (tool, arguments)
        self.revoked: list[str] = []
        self.predict_replies: list[dict] = []
        self.access = SECRET_ACCESS
        self.reject_access_once = False
        self.token_forms: list[dict] = []

    def _tool(self, name: str, args: dict) -> dict:
        self.calls.append((name, args))
        if name == "list_audiences":
            return {"audiences": [
                {"name": "everyone", "description": "anyone", "is_default": True},
                {"name": "clients", "description": "customers, formal", "is_default": False},
                {"name": "family", "description": "close family", "is_default": False},
            ]}
        if name == "predict":
            return self.predict_replies.pop(0)
        if name == "add_claims":
            return {"created": ["clm_1"], "absorbed": [], "dropped": []}
        if name == "ingest_raw":
            return {"job_id": "job_1", "accepted": len(args["records"])}
        raise AssertionError(name)

    def handler(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/.well-known/oauth-protected-resource/mcp":
            return httpx.Response(200, json={"resource": f"{BASE}/mcp", "authorization_servers": [BASE]})
        if path == "/.well-known/oauth-authorization-server":
            return httpx.Response(200, json={
                "issuer": BASE, "authorization_endpoint": f"{BASE}/authorize",
                "token_endpoint": f"{BASE}/token", "registration_endpoint": f"{BASE}/register",
                "revocation_endpoint": f"{BASE}/revoke",
            })
        if path == "/register":
            body = json.loads(req.content)
            assert body["token_endpoint_auth_method"] == "none"
            return httpx.Response(201, json={"client_id": "cli_1", **body})
        if path == "/token":
            form = {k: v[0] for k, v in parse_qs(req.content.decode()).items()}
            self.token_forms.append(form)
            if form["grant_type"] == "authorization_code":
                assert form["code"] == "code-1" and form["code_verifier"]
            self.access = f"acc-SECRET-t{len(self.token_forms)}"
            return httpx.Response(200, json={
                "access_token": self.access, "refresh_token": f"ref-SECRET-t{len(self.token_forms)}",
                "expires_in": 3600, "token_type": "Bearer",
            })
        if path == "/revoke":
            self.revoked.append(parse_qs(req.content.decode())["token"][0])
            return httpx.Response(200)
        if path == "/mcp":
            if req.method == "DELETE":
                return httpx.Response(200)
            if req.headers.get("authorization") != f"Bearer {self.access}" or self.reject_access_once:
                self.reject_access_once = False
                return httpx.Response(401)
            msg = json.loads(req.content)
            if msg["method"] == "initialize":
                return httpx.Response(
                    200, headers={"mcp-session-id": "s1"},
                    json={"jsonrpc": "2.0", "id": msg["id"], "result": {"protocolVersion": "2025-11-25"}},
                )
            if "id" not in msg:
                return httpx.Response(202)
            if msg["method"] == "tools/list":
                result = {"tools": [{"name": n} for n in ("ingest_raw", "add_claims", "predict")]}
            else:
                payload = self._tool(msg["params"]["name"], msg["params"]["arguments"])
                result = {"content": [{"type": "text", "text": json.dumps(payload)}],
                          "structuredContent": payload, "isError": False}
            # Streamable HTTP may answer with SSE.
            body = f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result})}\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
        return httpx.Response(404)


@pytest.fixture
def fake(monkeypatch):
    f = FakeWwyd()
    monkeypatch.setattr(wwyd, "_TRANSPORT", httpx.MockTransport(f.handler))
    return f


async def _profile(name: str) -> int:
    return await save_speaker_profile("browser-1", name, np.zeros(256, np.float32).tobytes())


async def _connect(pid: int, **extra) -> None:
    await save_wwyd_connection(
        pid, base_url=BASE, resource=f"{BASE}/mcp", auth_kind="oauth", client_id="cli_1",
        access_token=SECRET_ACCESS, refresh_token=SECRET_REFRESH, expires_at=None,
        token_endpoint=f"{BASE}/token", revocation_endpoint=f"{BASE}/revoke", **extra,
    )


class _Ctx:
    def __init__(self, profile_id, services=frozenset(), mixed=False):
        self.client_id = "browser-1"
        self.profile_id = profile_id
        self.user_lang = "en"
        self.services = services
        self.mixed = mixed


# ── Connection: OAuth, API key, disconnect, tokens never exposed ───────


async def test_oauth_connect_callback_and_disconnect(fake, caplog):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.storage import create_session

    caplog.set_level(logging.DEBUG)
    pid = await _profile("Ann")
    with TestClient(app) as c:
        c.cookies.set("va_session", await create_session(pid))
        r = c.post("/api/wwyd/connect", json={"url": f"{BASE}/mcp"})
        assert r.status_code == 200
        auth = urlsplit(r.json()["authorize_url"])
        q = {k: v[0] for k, v in parse_qs(auth.query).items()}
        assert auth.path == "/authorize" and q["code_challenge_method"] == "S256"
        assert q["scope"] == "owner" and q["resource"] == f"{BASE}/mcp"

        bad = c.get("/api/wwyd/callback", params={"state": "nope", "code": "code-1"},
                    follow_redirects=False)
        assert bad.headers["location"] == "/?wwyd=invalid"
        ok = c.get("/api/wwyd/callback", params={"state": q["state"], "code": "code-1"},
                   follow_redirects=False)
        assert ok.headers["location"] == "/?wwyd=connected"
        # The state is single use.
        again = c.get("/api/wwyd/callback", params={"state": q["state"], "code": "code-1"},
                      follow_redirects=False)
        assert again.headers["location"] == "/?wwyd=invalid"

        status = c.get("/api/wwyd").json()
        assert status["connection"]["url"] == BASE
        assert [a["name"] for a in status["audiences"]] == ["everyone", "clients", "family"]
        assert "SECRET" not in json.dumps(status)

        conn = await get_wwyd_connection(pid)
        assert conn["access_token"].startswith("acc-SECRET")

        assert c.delete("/api/wwyd").json() == {"ok": True, "removed": True, "revoked": True}
        assert fake.revoked == [conn["refresh_token"]]
        assert await get_wwyd_connection(pid) is None
    assert "SECRET" not in caplog.text


async def test_api_key_connect_requires_owner_scope(fake, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.storage import create_session

    pid = await _profile("Ann")
    fake.access = "wwyd_abcdefgh_" + "k" * 24
    with TestClient(app) as c:
        c.cookies.set("va_session", await create_session(pid))
        r = c.post("/api/wwyd/api-key", json={"url": BASE, "key": fake.access})
        assert r.status_code == 200
        conn = await get_wwyd_connection(pid)
        assert conn["auth_kind"] == "api_key" and conn["refresh_token"] is None
        bad = c.post("/api/wwyd/api-key", json={"url": BASE, "key": "wwyd_wrongkey_" + "x" * 24})
        assert bad.status_code == 400


def test_plain_http_only_on_loopback():
    assert wwyd.normalize_base_url("http://localhost:8740/mcp/") == "http://localhost:8740"
    with pytest.raises(wwyd.WwydError):
        wwyd.normalize_base_url("http://wwyd.example.org")


async def test_expired_access_token_is_refreshed_once_on_401(fake):
    pid = await _profile("Ann")
    await _connect(pid)
    fake.reject_access_once = True
    out = await wwyd.call_tool(pid, "list_audiences", {})
    assert out["audiences"]
    assert fake.token_forms[-1]["grant_type"] == "refresh_token"
    conn = await get_wwyd_connection(pid)
    assert conn["access_token"] == fake.access and conn["refresh_token"] != SECRET_REFRESH


# ── Visibility ──────────────────────────────────────────────────────────


async def test_tools_hidden_without_connection():
    from app.agent import _schemas_with_clock
    from app.tools.wwyd import wwyd_services

    pid = await _profile("Ann")
    names = lambda svc: {s["function"]["name"] for s in _schemas_with_clock("now", None, svc)}  # noqa: E731
    svc = await wwyd_services(pid, False)
    assert svc == frozenset()
    assert not {"how_would_answer", "wwyd_remember"} & names(svc)

    await _connect(pid)
    assert {"how_would_answer", "wwyd_remember"} <= names(await wwyd_services(pid, False))
    # A guest may ask, but never teaches the persona; mixed voices get neither.
    guest = names(await wwyd_services(None, False))
    assert "how_would_answer" in guest and "wwyd_remember" not in guest
    assert not {"how_would_answer", "wwyd_remember"} & names(await wwyd_services(pid, True))


async def test_dispatch_refuses_api_tool_without_service():
    from app.agent import AgentContext, _execute_one
    from app.llm_utils import ParsedToolCall

    inv = await _execute_one(
        ParsedToolCall(id="c1", name="wwyd_remember", args={"text": "x", "kind": "style", "category": "food"}),
        AgentContext(profile_id=1),
    )
    assert inv.data == {"error": "service_not_connected"}


# ── Read path ───────────────────────────────────────────────────────────


async def test_guest_is_answered_as_mapped_audience_with_disclosure(fake):
    from app.tools.wwyd import how_would_answer

    owner = await _profile("Eugene")
    await _connect(owner)
    await set_wwyd_audience_map(owner, {"guest": "clients"})
    fake.predict_replies = [{"type": "answer", "answer": "Sure, Friday works.", "confidence": 0.8,
                             "disclosure": "This is Eugene's auto-responder:", "used_claims": []}]
    r = await how_would_answer("Can we meet Friday?", audience="family",
                               ctx=_Ctx(None, frozenset({"wwyd_ask"})))
    name, args = fake.calls[-1]
    assert name == "predict"
    # The guest's own audience choice is ignored.
    assert args["audience"] == "clients" and args["include_private"] is False
    assert r.text == "This is Eugene's auto-responder: Sure, Friday works."


async def test_owner_names_audience_and_sees_private(fake):
    from app.tools.wwyd import how_would_answer

    owner = await _profile("Eugene")
    await _connect(owner)
    fake.predict_replies = [{"type": "answer", "answer": "Dear client, …", "confidence": 0.9,
                             "disclosure": "x", "used_claims": ["clm_1"]}]
    r = await how_would_answer("client asks for a discount", audience="clients",
                               ctx=_Ctx(owner, frozenset({"wwyd", "wwyd_ask"})))
    args = fake.calls[-1][1]
    assert args["audience"] == "clients" and args["include_private"] is True
    assert r.text == "Dear client, …"


async def test_needs_only_calendar_is_looked_up(fake, monkeypatch):
    from app.tools import wwyd as tool_mod

    owner = await _profile("Eugene")
    await _connect(owner)
    looked: list[int] = []

    async def fake_busy(pid):
        looked.append(pid)
        return "Busy times …: Tue 2026-10-20 10:00-11:00"

    monkeypatch.setattr(tool_mod, "calendar_free_busy", fake_busy)
    fake.predict_replies = [
        {"type": "needs", "needs": [{"what": "free slots on Tuesday", "why": "decision_rule clm_2"}]},
        {"type": "answer", "answer": "Tuesday 14:00 works.", "confidence": 0.8, "disclosure": "d"},
    ]
    r = await tool_mod.how_would_answer("meet Tuesday?", ctx=_Ctx(owner, frozenset({"wwyd", "wwyd_ask"})))
    assert looked == [owner]
    facts = fake.calls[-1][1]["facts"]
    assert facts == [{"what": "free slots on Tuesday", "value": "Busy times …: Tue 2026-10-20 10:00-11:00"}]
    assert r.text == "Tuesday 14:00 works."

    # Anything else is never looked up: predict is asked to answer without it.
    looked.clear()
    fake.calls.clear()
    fake.predict_replies = [
        {"type": "needs", "needs": [{"what": "home address of the guest", "why": "r"}]},
        {"type": "needs", "needs": [{"what": "home address of the guest", "why": "r"}]},
    ]
    r = await tool_mod.how_would_answer("visit?", ctx=_Ctx(owner, frozenset({"wwyd", "wwyd_ask"})))
    assert looked == [] and [c[0] for c in fake.calls] == ["predict", "predict"]
    assert fake.calls[-1][1]["facts"] == [
        {"what": "home address of the guest", "value": tool_mod._UNAVAILABLE}]
    assert r.data["type"] == "needs" and "home address" in r.text


def test_free_busy_has_times_only():
    import datetime as dt
    from app.tools.wwyd import _format_busy

    out = _format_busy(
        "2026-10-20T10:00:00|2026-10-20T11:00:00\n2026-10-20T10:30:00|2026-10-20T12:00:00\njunk\n",
        dt.datetime(2026, 10, 19, 9, 0),
    )
    assert out.endswith("Tue 2026-10-20 10:00-12:00")


async def test_remember_adds_approved_claim(fake):
    from app.tools.wwyd import wwyd_remember

    owner = await _profile("Eugene")
    await _connect(owner)
    r = await wwyd_remember("Answers clients briefly and formally.", "style",
                            "communication/style", audience="clients",
                            ctx=_Ctx(owner, frozenset({"wwyd"})))
    name, args = fake.calls[-1]
    assert name == "add_claims"
    assert args["claims"] == [{"kind": "style", "category": "communication/style",
                               "text": "Answers clients briefly and formally.",
                               "status": "approved", "audiences": ["clients"]}]
    assert "Saved to wwyd" in r.text


# ── Automatic write path ────────────────────────────────────────────────


@pytest.fixture
def fake_classifier(monkeypatch):
    seen: list[list[str]] = []

    async def classify(texts):
        seen.append(texts)
        return [{"type": "message", "text": "I'll be late, sorry", "to": "Anna"},
                {"type": "note", "text": "I never take calls before 10", "to": ""}]

    monkeypatch.setattr(wwyd_capture, "classify", classify)
    return seen


async def test_capture_ingests_owner_words_with_mapped_audience(fake, fake_classifier):
    owner = await _profile("Eugene")
    anna = await _profile("Anna")
    await _connect(owner)
    await set_wwyd_audience_map(owner, {f"profile:{anna}": "family"})
    wwyd_capture.observe(owner, False, "tell Anna I'll be late, sorry")
    await wwyd_capture.wait_idle()
    name, args = fake.calls[-1]
    assert name == "ingest_raw"
    msg, note = args["records"]
    assert msg["type"] == "message" and msg["from_owner"] is True and msg["audience"] == "family"
    assert "Anna" not in json.dumps(args)
    assert note == {"type": "note", "text": "I never take calls before 10", "ts": note["ts"]}


async def test_capture_is_owner_only(fake, fake_classifier):
    owner = await _profile("Eugene")
    other = await _profile("Anna")
    await _connect(owner)
    wwyd_capture.observe(None, False, "I never take calls before 10")   # guest
    wwyd_capture.observe(owner, True, "I never take calls before 10")   # mixed voices
    wwyd_capture.observe(other, False, "I never take calls before 10")  # not connected
    wwyd_capture.observe(owner, False, "what's the weather in Berlin")  # no first person
    wwyd_capture.observe(owner, False, "I answer clients formally", tool_name="wwyd_remember")
    await wwyd_capture.wait_idle()
    assert fake_classifier == [] and fake.calls == []


async def test_capture_does_not_block_and_survives_wwyd_down(fake_classifier, monkeypatch):
    owner = await _profile("Eugene")
    await _connect(owner)

    def down(req):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(wwyd, "_TRANSPORT", httpx.MockTransport(down))
    wwyd_capture.observe(owner, False, "I never take calls before 10")  # returns at once
    await wwyd_capture.wait_idle()
    assert len(fake_classifier) == 1


async def test_classifier_prompt_parses_llm_json(monkeypatch):
    from app import llm_utils

    async def chat(messages, **kw):
        assert "OWN words" in messages[0]["content"]
        return {"message": {"content": 'Sure: {"items": [{"type": "note", "text": "I hate calls"},'
                                       ' {"type": "request", "text": "x"}]}'}}

    monkeypatch.setattr(llm_utils, "chat", chat)
    assert await wwyd_capture.classify(["I hate calls"]) == [
        {"type": "note", "text": "I hate calls", "to": ""}]
