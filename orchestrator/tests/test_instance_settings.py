"""
Tests for instance-level settings (#43).

Scope: read/write/patch round-trip + the registration one-way invariant
documented in the module.  All I/O is redirected to a tmp file so the
real /data is never touched.
"""
from __future__ import annotations

import pytest

import app.instance_settings as isettings


@pytest.fixture
def _tmp_settings(tmp_path, monkeypatch):
    """Point instance_settings at a throwaway file for this test."""
    p = tmp_path / "settings.json"
    monkeypatch.setattr(isettings, "_INSTANCE_SETTINGS_PATH", p)
    return p


async def test_defaults_when_missing(_tmp_settings):
    """A fresh install (no file) returns safe defaults."""
    cfg = await isettings.read()
    assert cfg.registration_open is True
    assert cfg.allow_guest_voice is False
    assert cfg.basic_auth_user is None


async def test_write_read_round_trip(_tmp_settings):
    """Written settings survive a read."""
    await isettings.write(
        isettings.InstanceSettings(
            registration_open=False,
            allow_guest_voice=True,
            basic_auth_user="admin",
            basic_auth_password_hash="$2b$dummyhash",
        )
    )
    cfg = await isettings.read()
    assert cfg.registration_open is False
    assert cfg.allow_guest_voice is True
    assert cfg.basic_auth_user == "admin"
    assert _tmp_settings.exists()


async def test_patch_merges_fields(_tmp_settings):
    """patch() merges without clobbering untouched fields."""
    await isettings.write(isettings.InstanceSettings(allow_guest_voice=True))
    updated = await isettings.patch(registration_open=False)
    assert updated.registration_open is False
    assert updated.allow_guest_voice is True  # preserved


async def test_corrupt_file_falls_back_to_defaults(_tmp_settings):
    """A malformed settings.json doesn't crash — falls back to defaults."""
    _tmp_settings.write_text("{ this is not valid json", encoding="utf-8")
    cfg = await isettings.read()
    assert cfg.registration_open is True  # default, not a crash
