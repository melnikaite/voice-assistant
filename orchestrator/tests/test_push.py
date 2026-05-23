"""
Web Push pipeline — storage CRUD + send_to_profile happy/error paths.

What we pin here:

  • The storage layer upserts on endpoint (same browser re-subscribing
    doesn't duplicate the row).
  • Listing scopes by profile and ordering is newest-first.
  • Endpoint deletion returns the right boolean signalling whether a
    row was actually removed.
  • init_vapid() generates a fresh keypair on first call and reuses
    the persisted one on the second.
  • send_to_profile() fans out across every subscription for one
    profile, counts only 2xx as success, and auto-deletes rows after a
    410 Gone from the push service.
  • send_to_profile() returns 0 (and doesn't try to import pywebpush)
    when no subscriptions exist.

Network calls are mocked at the pywebpush layer — we never hit a real
push service from tests.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app import push
from app.storage import (
    delete_push_subscription,
    list_push_subscriptions,
    upsert_push_subscription,
)


# ── Helpers ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_push_state(tmp_path, monkeypatch):
    """Per-test: point VAPID storage at a tmp file + clear the in-process cache.

    We can't touch the production /data path from tests; conftest.py
    already redirects DATA_DIR_CONTAINER to /tmp/voice-assistant-test-data
    but the cached private key in :mod:`app.push` outlives the fixture
    sweep.  Reset both so each test sees a clean slate.
    """
    # Re-anchor the pem path inside tmp so a generated key from one
    # test can't leak into the next.
    pem_path = tmp_path / "vapid_private.pem"
    monkeypatch.setattr(push, "VAPID_PRIVATE_PEM_PATH", pem_path)
    push._reset_for_tests()
    yield
    push._reset_for_tests()


# ── Storage CRUD ───────────────────────────────────────────────────────


async def test_upsert_then_list():
    """Two distinct endpoints under the same profile → two rows, newest first."""
    profile_id = 7
    await upsert_push_subscription(
        profile_id=profile_id,
        endpoint="https://fcm.googleapis.com/fcm/send/AAA",
        p256dh_key="p1", auth_key="a1",
        user_agent="ua1",
    )
    await upsert_push_subscription(
        profile_id=profile_id,
        endpoint="https://updates.push.services.mozilla.com/wpush/v2/BBB",
        p256dh_key="p2", auth_key="a2",
        user_agent="ua2",
    )
    rows = await list_push_subscriptions(profile_id)
    assert len(rows) == 2
    endpoints = {r["endpoint"] for r in rows}
    assert endpoints == {
        "https://fcm.googleapis.com/fcm/send/AAA",
        "https://updates.push.services.mozilla.com/wpush/v2/BBB",
    }
    # Newest-first ordering — the second upsert wins.
    assert rows[0]["endpoint"].endswith("BBB")


async def test_upsert_is_idempotent_on_endpoint():
    """Re-subscribing from the same browser refreshes the row, doesn't duplicate."""
    profile_id = 7
    id1 = await upsert_push_subscription(
        profile_id=profile_id,
        endpoint="https://fcm.googleapis.com/fcm/send/XXX",
        p256dh_key="p1", auth_key="a1",
        user_agent="ua1",
    )
    id2 = await upsert_push_subscription(
        profile_id=profile_id,
        endpoint="https://fcm.googleapis.com/fcm/send/XXX",
        p256dh_key="p2-rotated", auth_key="a2-rotated",
        user_agent="ua2",
    )
    rows = await list_push_subscriptions(profile_id)
    assert len(rows) == 1, "endpoint UNIQUE should suppress duplicates"
    # Same row id — ON CONFLICT updates in place rather than re-inserting.
    assert id1 == id2
    # Keys + UA refreshed to the second call's values.
    assert rows[0]["p256dh_key"] == "p2-rotated"
    assert rows[0]["user_agent"] == "ua2"


async def test_delete_by_endpoint_returns_flag():
    """delete_push_subscription's return signals whether a row existed."""
    profile_id = 7
    await upsert_push_subscription(
        profile_id=profile_id,
        endpoint="https://example.test/sub-1",
        p256dh_key="p1", auth_key="a1",
    )
    assert await delete_push_subscription("https://example.test/sub-1") is True
    # Second delete is a no-op — row already gone.
    assert await delete_push_subscription("https://example.test/sub-1") is False
    # Unknown endpoint also returns False.
    assert await delete_push_subscription("https://nope.test/sub-x") is False
    # And the listing reflects the deletion.
    assert await list_push_subscriptions(profile_id) == []


# ── VAPID lifecycle ────────────────────────────────────────────────────


def test_init_vapid_generates_then_reuses(tmp_path):
    """First call mints a key; second call must reuse it (same public key)."""
    # First boot — file doesn't exist yet, init_vapid generates.
    assert not push.VAPID_PRIVATE_PEM_PATH.exists()
    push.init_vapid()
    assert push.VAPID_PRIVATE_PEM_PATH.exists(), "PEM must be persisted"
    first_pub = push.get_public_key()
    assert isinstance(first_pub, str) and len(first_pub) > 40

    # Drop the in-process cache, re-call: should LOAD the same key
    # rather than generating a fresh one.
    push._reset_for_tests()
    push.init_vapid()
    assert push.get_public_key() == first_pub


# ── send_to_profile ────────────────────────────────────────────────────


async def test_send_to_profile_no_subscriptions_is_zero():
    """No rows for the profile → returns 0 without touching the network."""
    push.init_vapid()
    sent = await push.send_to_profile(
        profile_id=42, payload={"title": "x", "body": "y"}
    )
    assert sent == 0


async def test_send_to_profile_happy_path_counts_2xx():
    """Two subs, both push services return 201 → sent == 2 and both rows survive."""
    push.init_vapid()
    profile_id = 9
    await upsert_push_subscription(
        profile_id=profile_id,
        endpoint="https://fcm.googleapis.com/fcm/send/OK1",
        p256dh_key="p1", auth_key="a1",
    )
    await upsert_push_subscription(
        profile_id=profile_id,
        endpoint="https://updates.push.services.mozilla.com/wpush/v2/OK2",
        p256dh_key="p2", auth_key="a2",
    )

    # Stub the inner sync sender — same shape the real one returns
    # (the HTTP status code from the push service).
    def _fake_send(**_kwargs):
        return 201

    with patch.object(push, "_send_one_sync", side_effect=_fake_send) as m:
        sent = await push.send_to_profile(
            profile_id, {"title": "Voicemail", "body": "From Alice", "voicemail_id": 5},
        )
    assert sent == 2
    assert m.call_count == 2
    # Subscriptions are intact (no GC on 2xx).
    rows = await list_push_subscriptions(profile_id)
    assert len(rows) == 2


async def test_send_to_profile_gc_on_410_gone():
    """A 410 Gone from the push service auto-deletes the dead row."""
    push.init_vapid()
    profile_id = 11
    await upsert_push_subscription(
        profile_id=profile_id,
        endpoint="https://fcm.googleapis.com/fcm/send/GONE",
        p256dh_key="p1", auth_key="a1",
    )
    await upsert_push_subscription(
        profile_id=profile_id,
        endpoint="https://updates.push.services.mozilla.com/wpush/v2/LIVE",
        p256dh_key="p2", auth_key="a2",
    )

    # Build a fake exception that carries .response.status_code = 410.
    class _Resp:
        status_code = 410

    class _FakeWebPushError(Exception):
        def __init__(self):
            super().__init__("subscription gone")
            self.response = _Resp()

    def _fake_send(*, subscription_info, **_kwargs):
        if "GONE" in subscription_info["endpoint"]:
            raise _FakeWebPushError()
        return 201

    with patch.object(push, "_send_one_sync", side_effect=_fake_send):
        sent = await push.send_to_profile(
            profile_id, {"title": "x", "body": "y", "voicemail_id": 1},
        )
    # Only the live one counts as sent.
    assert sent == 1
    # GC removed the GONE row but kept the LIVE one.
    rows = await list_push_subscriptions(profile_id)
    assert len(rows) == 1
    assert "LIVE" in rows[0]["endpoint"]


async def test_send_to_profile_transient_failure_keeps_row():
    """A non-410 error (e.g. network blip) doesn't GC the subscription."""
    push.init_vapid()
    profile_id = 12
    await upsert_push_subscription(
        profile_id=profile_id,
        endpoint="https://fcm.googleapis.com/fcm/send/FLAKY",
        p256dh_key="p1", auth_key="a1",
    )

    class _Resp:
        status_code = 500  # push service had a hiccup, not a permanent failure

    class _FakeWebPushError(Exception):
        def __init__(self):
            super().__init__("upstream 500")
            self.response = _Resp()

    def _fake_send(**_kwargs):
        raise _FakeWebPushError()

    with patch.object(push, "_send_one_sync", side_effect=_fake_send):
        sent = await push.send_to_profile(
            profile_id, {"title": "x", "body": "y"},
        )
    assert sent == 0
    # Row must survive — we'll retry on the next voicemail.
    rows = await list_push_subscriptions(profile_id)
    assert len(rows) == 1


# ── register_subscription validation ──────────────────────────────────


async def test_register_subscription_rejects_missing_keys():
    """Browser-shape validation: endpoint + keys.p256dh + keys.auth all required."""
    with pytest.raises(ValueError):
        await push.register_subscription(
            profile_id=1,
            subscription={"endpoint": "https://x.test/sub"},  # no keys
            user_agent=None,
        )
    with pytest.raises(ValueError):
        await push.register_subscription(
            profile_id=1,
            subscription={
                "endpoint": "https://x.test/sub",
                "keys": {"p256dh": "p"},  # auth missing
            },
            user_agent=None,
        )

