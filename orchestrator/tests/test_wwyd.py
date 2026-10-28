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
        self.revoke_forms: list[dict] = []
        self.version = "0.2.0"
        self.predict_replies: list[dict] = []
        self.access = SECRET_ACCESS
        self.reject_access_once = False
        self.token_forms: list[dict] = []
        self.timezone = "UTC"
        self.job_statuses: list[dict] = []

    def _tool(self, name: str, args: dict) -> dict:
        self.calls.append((name, args))
        if name == "list_audiences":
            return {"audiences": [
                {"name": "everyone", "description": "anyone", "is_default": True},
                {"name": "clients", "description": "customers, formal", "is_default": False},
                {"name": "family", "description": "close family", "is_default": False},
            ]}
        if name == "list_categories":
            leaf = lambda p, *c: {"path": p, "name": p.rsplit("/", 1)[-1], "children": list(c)}  # noqa: E731
            return {"categories": [
                leaf("communication", leaf("communication/style")),
                leaf("work", leaf("work/meetings")), leaf("food"),
                leaf("garden", leaf("garden/roses")),  # owner-added
            ]}
        if name == "predict":
            return self.predict_replies.pop(0)
        if name == "add_claims":
            return {"created": ["clm_1"], "absorbed": [], "dropped": []}
        if name == "ingest_raw":
            return {"job_id": "job_1", "accepted": len(args["episodes"])}
        if name == "job_status":
            return self.job_statuses.pop(0) if self.job_statuses else {"status": "done"}
        if name == "get_settings":
            return {"timezone": self.timezone}
        if name == "update_settings":
            self.timezone = args["timezone"]
            return {}
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
            form = {k: v[0] for k, v in parse_qs(req.content.decode(), keep_blank_values=True).items()}
            self.revoke_forms.append(form)
            self.revoked.append(form["token"])
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
                    json={"jsonrpc": "2.0", "id": msg["id"], "result": {
                        "protocolVersion": "2025-11-25",
                        "serverInfo": {"name": "wwyd", "version": self.version}, "x-new": 1}},
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


def _ingested(f: FakeWwyd) -> list[dict]:
    return [a for n, a in f.calls if n == "ingest_raw"]


@pytest.fixture
def fake(monkeypatch):
    from app import llm_utils

    async def no_time(messages, **kw):
        return {"message": {"content": "When: NONE"}}

    f = FakeWwyd()
    monkeypatch.setattr(wwyd, "_TRANSPORT", httpx.MockTransport(f.handler))
    monkeypatch.setattr(wwyd, "_categories_cache", {})
    monkeypatch.setattr(wwyd, "local_timezone", lambda: "Europe/Berlin")
    monkeypatch.setattr(llm_utils, "chat", no_time)
    monkeypatch.setattr(wwyd_capture, "_POLL_S", 0)
    monkeypatch.setattr(wwyd_capture, "_RESEND_PAUSES_S", (0, 0, 0))
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
        # A persona on the default UTC gets this machine's zone.
        assert ("update_settings", {"timezone": "Europe/Berlin"}) in fake.calls

        assert c.delete("/api/wwyd").json() == {"ok": True, "removed": True, "revoked": True}
        assert fake.revoked == [conn["refresh_token"]]
        # Public client: client_id only, no empty client_secret.
        assert fake.revoke_forms[-1] == {"token": conn["refresh_token"],
                                         "token_type_hint": "refresh_token", "client_id": "cli_1"}
        assert await get_wwyd_connection(pid) is None
    assert "SECRET" not in caplog.text


async def test_api_key_connect_requires_owner_scope(fake, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.storage import create_session

    pid = await _profile("Ann")
    fake.access = "wwyd_abcdefgh_" + "k" * 24
    fake.timezone = "Asia/Tokyo"  # the owner's own choice stays
    with TestClient(app) as c:
        c.cookies.set("va_session", await create_session(pid))
        r = c.post("/api/wwyd/api-key", json={"url": BASE, "key": fake.access})
        assert r.status_code == 200
        conn = await get_wwyd_connection(pid)
        assert conn["auth_kind"] == "api_key" and conn["refresh_token"] is None
        assert fake.timezone == "Asia/Tokyo"
        bad = c.post("/api/wwyd/api-key", json={"url": BASE, "key": "wwyd_wrongkey_" + "x" * 24})
        assert bad.status_code == 400
        # Disconnect revokes the key itself, without client authentication.
        assert c.delete("/api/wwyd").json()["revoked"] is True
        assert fake.revoke_forms[-1] == {"token": fake.access}


@pytest.mark.parametrize("version", ["0.1.0", "1.0.0", "", "0.0.9"])
async def test_connect_refuses_incompatible_server(fake, version):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.storage import create_session

    pid = await _profile("Ann")
    fake.version = version
    fake.access = "wwyd_abcdefgh_" + "k" * 24
    with TestClient(app) as c:
        c.cookies.set("va_session", await create_session(pid))
        r = c.post("/api/wwyd/api-key", json={"url": BASE, "key": fake.access})
        assert r.status_code == 400 and "0.2.x" in r.json()["detail"]
        assert await get_wwyd_connection(pid) is None

        q = parse_qs(urlsplit(c.post("/api/wwyd/connect", json={"url": BASE}).json()["authorize_url"]).query)
        r = c.get("/api/wwyd/callback", params={"state": q["state"][0], "code": "code-1"},
                  follow_redirects=False)
        assert r.headers["location"] == "/?wwyd=version"
        assert await get_wwyd_connection(pid) is None
        assert fake.revoked  # the unused grant is given back


def test_version_check_accepts_0_2_series():
    for v in ("0.2.0", "0.2.7", "0.2.0-rc.1"):
        wwyd.check_version(v)
    for v in ("0.1.0", "0.3.0", "1.2.0", None, "garbage"):
        with pytest.raises(wwyd.WwydError):
            wwyd.check_version(v)


async def test_registration_refusal_explains_why(monkeypatch):
    def refuse(req):
        return httpx.Response(400, json={
            "error": "invalid_redirect_uri",
            "error_description": "http redirect URIs are only accepted for loopback hosts on this server"})

    monkeypatch.setattr(wwyd, "_TRANSPORT", httpx.MockTransport(refuse))
    with pytest.raises(wwyd.WwydError, match="loopback hosts"):
        await wwyd.register_client(f"{BASE}/register", "http://192.168.1.5:8080/api/wwyd/callback")


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


async def test_depends_on_is_settled_from_calendar_conclusion(fake, monkeypatch):
    import datetime as dt
    from app import llm_utils
    from app.tools import wwyd as tool_mod

    owner = await _profile("Eugene")
    await _connect(owner)
    tomorrow = dt.date.today() + dt.timedelta(days=1)
    spans = [(dt.datetime.combine(tomorrow, dt.time(13, 0)), dt.datetime.combine(tomorrow, dt.time(14, 0)))]
    seen: list[str] = []

    async def fake_busy(pid):
        return spans

    async def chat(messages, **kw):
        seen.append(messages[1]["content"])
        return {"message": {"content": f"When: {tomorrow} 14:00"}}

    monkeypatch.setattr(tool_mod, "calendar_busy", fake_busy)
    monkeypatch.setattr(llm_utils, "chat", chat)
    fake.predict_replies = [
        {"type": "answer", "answer": "let me check and get back to you", "confidence": 0.7,
         "depends_on": "Agrees if tomorrow around 14:00 is free.", "disclosure": "d"},
        {"type": "answer", "answer": "Tomorrow 14:00 works.", "confidence": 0.8,
         "depends_on": None, "disclosure": "d"},
    ]
    r = await tool_mod.how_would_answer("meet tomorrow?", ctx=_Ctx(owner, frozenset({"wwyd", "wwyd_ask"})))
    # The model only names the time; the schedule never reaches it or wwyd.
    assert "Agrees if tomorrow" in seen[0] and "13:00" not in seen[0]
    first, second = (c[1] for c in fake.calls if c[0] == "predict")
    assert "context" not in first
    assert second == {**first, "context": f"tomorrow ({tomorrow:%a %Y-%m-%d}) 14:00-14:30 is free"}
    assert r.text == "Tomorrow 14:00 works."


async def test_calendar_is_checked_when_depends_on_is_null(fake, monkeypatch):
    import datetime as dt
    from app import llm_utils
    from app.tools import wwyd as tool_mod

    owner = await _profile("Eugene")
    await _connect(owner)
    tomorrow = dt.date.today() + dt.timedelta(days=1)

    async def fake_busy(pid):
        return [(dt.datetime.combine(tomorrow, dt.time(15, 0)), dt.datetime.combine(tomorrow, dt.time(16, 0)))]

    async def chat(messages, **kw):
        return {"message": {"content": f"When: {tomorrow} 15:00"}}

    monkeypatch.setattr(tool_mod, "calendar_busy", fake_busy)
    monkeypatch.setattr(llm_utils, "chat", chat)
    fake.predict_replies = [
        {"type": "answer", "answer": "sure, 15:00 works", "depends_on": None, "disclosure": "d"},
        {"type": "answer", "answer": "can't at 15, I'm busy", "depends_on": None, "disclosure": "d"},
    ]
    r = await tool_mod.how_would_answer("call tomorrow at 15?", ctx=_Ctx(owner, frozenset({"wwyd", "wwyd_ask"})))
    second = [c[1] for c in fake.calls if c[0] == "predict"][1]
    assert second["context"] == f"tomorrow ({tomorrow:%a %Y-%m-%d}) 15:00-15:30 is taken"
    assert r.text == "can't at 15, I'm busy"


async def test_depends_on_without_conclusion_keeps_first_answer(fake, monkeypatch):
    from app import llm_utils
    from app.tools import wwyd as tool_mod

    owner = await _profile("Eugene")
    await _connect(owner)

    async def fake_busy(pid):
        return []

    async def chat(messages, **kw):
        return {"message": {"content": "When: NONE"}}

    monkeypatch.setattr(tool_mod, "calendar_busy", fake_busy)
    monkeypatch.setattr(llm_utils, "chat", chat)
    fake.predict_replies = [
        {"type": "answer", "answer": "sounds great, I'll look at the weather", "confidence": 0.7,
         "depends_on": "Goes to the lake only in warm weather.", "disclosure": "d"}]
    r = await tool_mod.how_would_answer("lake?", ctx=_Ctx(owner, frozenset({"wwyd", "wwyd_ask"})))
    assert [c[0] for c in fake.calls].count("predict") == 1
    assert r.text == "sounds great, I'll look at the weather"
    assert r.data["depends_on"] == "Goes to the lake only in warm weather."

    # A low-confidence escalation still sends the asker to the person.
    fake.predict_replies = [{"type": "escalate", "escalate": "low_confidence", "reason": "x"}]
    r = await tool_mod.how_would_answer("visit?", ctx=_Ctx(owner, frozenset({"wwyd", "wwyd_ask"})))
    assert "ask Eugene directly" in r.text


def test_busy_spans_have_times_only_and_merge():
    import datetime as dt
    from app.tools.wwyd import _busy_spans

    out = _busy_spans(
        "2026-10-20T10:00:00|2026-10-20T11:00:00\n2026-10-20T10:30:00|2026-10-20T12:00:00\njunk\n")
    assert out == [(dt.datetime(2026, 10, 20, 10), dt.datetime(2026, 10, 20, 12))]


def test_conclusion_answers_the_window_only():
    import datetime as dt
    from app.tools.wwyd import _conclude

    now = dt.datetime(2026, 10, 19, 9, 0)
    d = lambda h, m=0: dt.datetime(2026, 10, 20, h, m)
    spans = [(d(14), d(16)), (d(18), d(20))]
    assert _conclude(spans, d(15), d(15, 30), now) == "tomorrow (Tue 2026-10-20) 15:00-15:30 is taken"
    assert _conclude(spans, d(17), d(17, 30), now) == "tomorrow (Tue 2026-10-20) 17:00-17:30 is free"
    assert _conclude(spans, d(17), d(22), now) == (
        "Within tomorrow (Tue 2026-10-20) 17:00-22:00 only 17:00-18:00, 20:00-22:00 is free")


def test_match_name_across_scripts_and_cases():
    from app.tools.wwyd import _match_name

    profiles = [(1, "Anna"), (2, "Маша"), (3, "Eugene")]
    assert _match_name("Анне", profiles) == (1, "Anna")
    assert _match_name("Маше", profiles) == (2, "Маша")
    assert _match_name("Tom", profiles) is None


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


@pytest.mark.parametrize("spoken,expected", [
    ("garden/roses", "garden/roses"),   # owner-added category
    ("Meetings", "work/meetings"),      # leaf name
    ("food/vegan", "food"),             # unknown child → existing parent
    ("hobbies", "communication/style"),  # not in this persona → default
])
async def test_remember_category_from_persona(fake, spoken, expected):
    from app.tools.wwyd import wwyd_remember

    owner = await _profile("Eugene")
    await _connect(owner)
    await wwyd_remember("x", "habit", spoken, ctx=_Ctx(owner, frozenset({"wwyd"})))
    await wwyd_remember("y", "habit", spoken, ctx=_Ctx(owner, frozenset({"wwyd"})))
    assert fake.calls[-1][1]["claims"][0]["category"] == expected
    # Cached per profile: one list_categories for two claims.
    assert [c[0] for c in fake.calls].count("list_categories") == 1


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
    args = _ingested(fake)[-1]
    msg, note = args["episodes"]
    assert msg == {"when": msg["when"], "said": "I'll be late, sorry", "audience": "family"}
    assert "Anna" not in json.dumps(args)
    assert note == {"when": note["when"], "said": "I never take calls before 10"}


async def test_capture_unmapped_recipient_stays_an_outgoing_message(fake, fake_classifier):
    owner = await _profile("Eugene")
    await _connect(owner)
    wwyd_capture.observe(owner, False, "tell Anna I'll be late, sorry")
    await wwyd_capture.wait_idle()
    msg, _ = _ingested(fake)[-1]["episodes"]
    assert msg["audience"] == "default" and "situation" not in msg


async def test_capture_resends_what_a_busy_llm_left(fake, fake_classifier):
    owner = await _profile("Eugene")
    await _connect(owner)
    fake.job_statuses = [{"status": "running"},
                         {"status": "partial", "error_code": "llm_unavailable", "remaining_records": [1]},
                         {"status": "done"}]
    wwyd_capture.observe(owner, False, "tell Anna I'll be late, sorry")
    await wwyd_capture.wait_idle()
    first, second = _ingested(fake)
    assert second["episodes"] == [first["episodes"][1]]


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
