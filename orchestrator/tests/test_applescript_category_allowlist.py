"""
AppleScript category-strict read-only verb allowlist.

The orchestrator's ``computer_use`` tool passes a category
(mail / calendar / files / browser) when it routes an intent through
the ``desktop`` tool's AppleScript path.  When a category is set, the
risk classifier enforces a STRICT read-only verb policy: any mutation
verb in the script (delete / send / move / …) forces an outright
reject via the ``DENIED_READONLY_VIOLATION`` sentinel.

The existing destructive-pattern detector (which UPGRADES risk to
high_write) is preserved.  These tests pin both behaviours.
"""
from __future__ import annotations

from app.tools.desktop import (
    DENIED_READONLY_VIOLATION,
    _classify_applescript_risk,
)


def test_mail_category_rejects_delete_verb():
    """``delete`` in a mail-category script → DENIED_READONLY_VIOLATION.

    Even with risk="read" the classifier rejects — defence-in-depth
    against an LLM that hand-waves itself into a destructive call.
    """
    script = 'tell application "Mail" to delete first message of inbox'
    out = _classify_applescript_risk(script, "read", category="mail")
    assert out == DENIED_READONLY_VIOLATION


def test_mail_category_allows_get_verb():
    """``get`` in a mail-category script → declared risk passes through.

    Pure read-only operation, no upgrade, no reject.  The legacy
    destructive-pattern detector doesn't trigger either (no delete /
    send / new event verbs).
    """
    script = 'tell application "Mail" to get count of inbox'
    out = _classify_applescript_risk(script, "read", category="mail")
    assert out == "read"


def test_no_category_keeps_existing_pattern_detector_behavior():
    """Without ``category`` set, the legacy detector still upgrades risk.

    Back-compat: free-form ``desktop`` tool calls (LLM directly,
    without computer_use) keep the original "destructive verbs upgrade
    to high_write" semantics.  Only category-mediated calls get the
    strict reject.
    """
    script = 'tell application "Mail" to send outgoing message 1'
    # No category → no strict reject, but destructive pattern → upgrade.
    out = _classify_applescript_risk(script, "read", category=None)
    assert out == "high_write"


def test_files_category_rejects_do_shell_script():
    """``do shell script`` in any category is rejected (shell escape)."""
    script = 'do shell script "rm -rf ~/Documents/foo"'
    out = _classify_applescript_risk(script, "read", category="files")
    assert out == DENIED_READONLY_VIOLATION


def test_calendar_category_allows_event_listing():
    """Reading events from Calendar → read passes through unchanged."""
    script = (
        'tell application "Calendar" to '
        'get summary of every event of calendar 1'
    )
    out = _classify_applescript_risk(script, "read", category="calendar")
    assert out == "read"
