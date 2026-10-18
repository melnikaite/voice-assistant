"""
Per-channel conversation state: recent turns, a running summary, and focus.

One ``Conversation`` per WS session / Telegram chat.  The agent sees:

  * ``messages()`` — a summary pseudo-turn (when older turns were folded)
    followed by the last ``KEEP_TURNS`` turns verbatim;
  * ``focus_block()`` — one line naming what "there" / "it" / "туда" most
    likely refer to, built deterministically from previous tool calls.

Why focus rather than more history: past turns are stored as text only, so
the entity a referential follow-up needs lives in tool arguments that the
model never sees again, and topically varied turns in between bury it.
Measured with ``scripts/context_probe.py`` (architecture.md §13).

Slots come from two places: ``@tool(focus={arg: slot})`` maps call arguments,
and a tool may return ``ToolResult.data["focus"] = {slot: value}`` for things
only its output knows (e.g. which reminder ``list_reminders`` found).

``held`` is a partial result a tool asked to keep until the user answers
(``ToolResult.data["hold"]``; ``None`` clears it) — e.g. a dataset entry
missing a required field.  The pipeline hands it back through
``AgentContext.held`` for ``HELD_TTL_TURNS`` turns.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

log = logging.getLogger(__name__)

KEEP_TURNS = int(os.environ.get("CONVERSATION_KEEP_TURNS", "6"))
FOCUS_TTL_TURNS = 10
HELD_TTL_TURNS = 3

# A list of subjects rather than prose: prose summaries kept rewriting the
# oldest subject away after a few merges.
SUMMARY_PROMPT = (
    "You keep the list of subjects discussed earlier in a conversation between "
    "a user and a voice assistant.  Add the subjects of the new turns to the "
    "previous list: one item per subject, a few words each with its named "
    "entities (place, person, list, date, time), in the conversation's "
    "language, separated by '; '.  Never drop an item from the previous list "
    "unless the list exceeds 12 items — then drop the oldest.  Output only "
    "the list."
)


@dataclass
class FocusEntry:
    value: str
    tool: str
    turn: int  # Conversation.turn when it was set


@dataclass
class Conversation:
    history: list[dict] = field(default_factory=list)  # verbatim, role/content
    summary: str = ""
    focus: dict[str, FocusEntry] = field(default_factory=dict)
    turn: int = 0
    _held: "tuple[dict, int] | None" = field(default=None, repr=False)
    keep_turns: int = KEEP_TURNS
    _unfolded: list[dict] = field(default_factory=list, repr=False)
    _fold_task: "asyncio.Task | None" = field(default=None, repr=False)

    # ── Write ─────────────────────────────────────────────────────────────

    def record(self, transcript: str, reply: str, invocations: Iterable[Any] = ()) -> None:
        """Append one finished turn; ``invocations`` are ``ToolInvocation``s."""
        self.turn += 1
        for inv in invocations:
            self._absorb(inv.name, inv.args or {}, inv.data)
        self.history += [
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": reply},
        ]
        overflow = len(self.history) - self.keep_turns * 2
        if overflow > 0:
            self._unfolded += self.history[:overflow]
            del self.history[:overflow]
            self._schedule_fold()

    def reset(self) -> None:
        self.history.clear()
        self._unfolded.clear()
        self.summary = ""
        self.focus.clear()
        self._held = None
        if self._fold_task is not None:
            self._fold_task.cancel()
            self._fold_task = None

    def _absorb(self, tool: str, args: dict, data: dict | None) -> None:
        from .tools import TOOL_REGISTRY  # late: tools import half the app

        if isinstance(data, dict) and data.get("error"):
            return
        if isinstance(data, dict) and "hold" in data:
            self._held = (data["hold"], self.turn) if data["hold"] else None
        meta = TOOL_REGISTRY.get(tool) or {}
        for arg, slot in (meta.get("focus") or {}).items():
            val = args.get(arg)
            if isinstance(val, str) and val.strip():
                self.focus[slot] = FocusEntry(val.strip(), tool, self.turn)
        for slot, val in ((data or {}).get("focus") or {}).items():
            if isinstance(val, str) and val.strip():
                self.focus[slot] = FocusEntry(val.strip(), tool, self.turn)

    # ── Read ──────────────────────────────────────────────────────────────

    def messages(self) -> list[dict]:
        if not self.summary:
            return list(self.history)
        return [
            {"role": "user", "content": f"(Earlier in this conversation, oldest first: {self.summary})"},
            {"role": "assistant", "content": "OK."},
        ] + self.history

    @property
    def held(self) -> dict | None:
        if self._held and self.turn - self._held[1] < HELD_TTL_TURNS:
            return self._held[0]
        return None

    @property
    def turns(self) -> int:
        """Turns the model sees, for UI counters."""
        return len(self.history) // 2 + (1 if self.summary else 0)

    def focus_block(self, fallback: dict[str, str] | None = None) -> str:
        """``fallback`` fills slots this conversation has not set (e.g. the
        profile's persisted last category)."""
        live = {
            s: e for s, e in self.focus.items()
            if self.turn - e.turn < FOCUS_TTL_TURNS
        }
        parts = []
        for slot, e in sorted(live.items(), key=lambda kv: -kv[1].turn):
            ago = self.turn - e.turn + 1
            parts.append(f"{slot}={e.value!r} ({e.tool}, {ago} turn{'s' if ago > 1 else ''} ago)")
        for slot, val in (fallback or {}).items():
            if slot not in live and val:
                parts.append(f"{slot}={val!r} (last used)")
        if not parts:
            return ""
        return (
            "[Conversation focus — if the message says 'there/it/this/туда/там/"
            "об этом' or names no target, put these values into the tool "
            "arguments instead of the pronoun: "
            + "; ".join(parts) + "]"
        )

    # ── Restore ───────────────────────────────────────────────────────────

    @classmethod
    def from_rows(cls, rows: Iterable[tuple]) -> "Conversation":
        """Rebuild from ``(transcript, response_text, tool_name, tool_args_json)``
        rows, oldest first.  Only the last tool's arguments were persisted, so
        output-derived slots are lost across reconnects."""
        conv = cls()
        for transcript, reply, tool, args_json in rows:
            args: dict = {}
            if tool and args_json:
                try:
                    args = json.loads(args_json) or {}
                except ValueError:
                    pass
            inv = _Inv(tool, args) if tool else None
            conv.record(transcript, reply, [inv] if inv else ())
        return conv

    # ── Summary (background) ──────────────────────────────────────────────

    def _schedule_fold(self) -> None:
        if self._fold_task is not None and not self._fold_task.done():
            return  # the running fold picks up what was queued meanwhile
        try:
            self._fold_task = asyncio.get_running_loop().create_task(self._fold())
        except RuntimeError:
            pass  # no loop (sync caller) — folded on the next async record

    async def _fold(self) -> None:
        from .llm_utils import chat, extract_text

        while self._unfolded:
            batch, self._unfolded = self._unfolded, []
            convo = "\n".join(f"{m['role']}: {m['content']}" for m in batch)
            focus = self.focus_block()
            prompt = (
                f"Previous list: {self.summary or '(empty)'}\n"
                + (f"{focus}\n" if focus else "")
                + "New turns:\n" + convo
            )
            try:
                choice = await chat(
                    [{"role": "system", "content": SUMMARY_PROMPT},
                     {"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=200, tool_name="<summary>",
                )
                text = extract_text(choice["message"]).strip()
            except Exception as e:  # keep the old summary; turns are dropped
                log.warning("conversation summary failed: %s", e)
                continue
            if text:
                self.summary = text

    async def wait_summary(self) -> None:
        if self._fold_task is not None:
            await asyncio.shield(self._fold_task)


@dataclass
class _Inv:
    name: str
    args: dict
    data: dict | None = None
