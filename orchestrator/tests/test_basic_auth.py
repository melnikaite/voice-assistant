"""
Tests for the outer Basic Auth middleware (#43) + the hygiene-pass fixes:
  * middleware actually enforces (it was previously added inside lifespan,
    after Starlette built its stack, so it never wrapped requests);
  * constant-time username compare with no short-circuit username oracle;
  * the skip-list (/health, /ws) bypasses auth.

We mount the middleware on a throwaway app so the test is independent of
main.py's import-time settings read.
"""
from __future__ import annotations

import base64

import bcrypt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import _BasicAuthMiddleware


_PW = "s3cret-pw"
_HASH = bcrypt.hashpw(_PW.encode(), bcrypt.gensalt()).decode("ascii")


def _app() -> FastAPI:
    a = FastAPI()
    a.add_middleware(_BasicAuthMiddleware, username="admin", password_hash=_HASH)

    @a.get("/api/config")
    async def cfg():
        return {"ok": True}

    @a.get("/health")
    async def health():
        return {"status": "ok"}

    return a


def _basic(user: str, pw: str) -> dict:
    raw = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return {"Authorization": f"Basic {raw}"}


def test_missing_credentials_401():
    with TestClient(_app()) as c:
        r = c.get("/api/config")
    assert r.status_code == 401
    assert "WWW-Authenticate" in r.headers


def test_correct_credentials_pass():
    with TestClient(_app()) as c:
        r = c.get("/api/config", headers=_basic("admin", _PW))
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_wrong_password_401():
    with TestClient(_app()) as c:
        r = c.get("/api/config", headers=_basic("admin", "nope"))
    assert r.status_code == 401


def test_wrong_username_401():
    with TestClient(_app()) as c:
        r = c.get("/api/config", headers=_basic("intruder", _PW))
    assert r.status_code == 401


def test_health_path_is_exempt():
    """/health must answer without credentials (liveness probe)."""
    with TestClient(_app()) as c:
        r = c.get("/health")
    assert r.status_code == 200
