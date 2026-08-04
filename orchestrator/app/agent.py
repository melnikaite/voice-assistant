"""
Agent context + multi-step tool-using loop.

`AgentContext` carries the per-request state that side-effect tools need
(currently just `client_id`, room for more). It's passed explicitly through
the agent loop and injected into tool handlers that declare it as a kwarg —
no ContextVars, no globals, no `from .. import scheduler` inside hot paths.

`run_agent` is the OpenAI-style tool-use loop: ask the LLM, dispatch any
tool calls it asks for, feed results back, repeat. A tool can short-circuit
the loop by being marked `terminal=True` — its first call ends the turn
with its `text` as the spoken reply, with no follow-up LLM round-trip.
Non-terminal tools (currently just `general_answer`) leave the loop running
so the LLM can chain (e.g. unknown answer → web_search).
"""
from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import desktop_client
from .i18n import t
from .llm_utils import ParsedToolCall, chat, extract_text, parse_tool_calls
from .storage import enqueue_action
from .tools import TOOL_REGISTRY, dispatch, schemas

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentContext:
    """
    Per-request context handed to side-effect tools.

    The agent loop reads `inspect.signature(handler)` and injects this object
    only into tools that declare a `ctx: AgentContext` parameter. Value-style
    tools (general_answer) don't, and stay pure functions of their
    schema-described args.

    Fields:
      client_id        Stable per-browser ID, needed by side-effect tools
                       (reminders).
      profile_id       ID of the recognised speaker_profile row, set by
                       the pipeline after resemblyzer match.  None for an
                       unidentified speaker — Sprint-2 tier-2 features
                       degrade gracefully in that case (memory writes are
                       refused, reminders still work).
      is_authenticated True iff the speaker has supplied a valid passphrase
                       within the current auth window.  Read by
                       ``_execute_one`` to gate ``risk='high_write'`` tools
                       — fails closed by default.
      user_lang        Resolved locale for THIS turn's user-facing strings
                       — one of ``"en" | "ru" | "de"``.  Picked by the
                       pipeline from the speaker's ``settings.language``
                       falling back to Whisper's per-turn detection, then
                       to English.  Tools should call ``i18n.t(key, lang
                       =ctx.user_lang)`` rather than embedding any literal
                       user-facing string in their source.
      stream_sink      Async callback for incremental text emission.
                       Tools that generate their response via streaming
                       (web_search) call this per-sentence so the WS
                       layer can start TTS while the LLM is still
                       producing the rest.  None on entry paths that
                       don't have a TTS pipe (the /dev/respond HTTP
                       endpoint).
      progress_sink    Async callback for "what stage are we in" UI
                       updates.  Tools call this with step names
                       ("localize", "search", "fetch", "summarize", …)
                       so the user sees granular progress instead of an
                       opaque yellow status.  None on entry paths
                       without UI (dev endpoint).
    """
    client_id: str | None = None
    profile_id: int | None = None
    is_authenticated: bool = False
    user_lang: str = "en"
    stream_sink: "Any | None" = None      # Callable[[str], Awaitable[None]] | None
    progress_sink: "Any | None" = None    # Callable[[str, str | None], Awaitable[None]] | None
    media_sink: "Any | None" = None       # Callable[[str, str], Awaitable[None]] | None
    #   media_sink(url: str, source: str) — notifies the WS layer that a
    #   live stream has started.  ``url`` is the relative URL the browser
    #   should load in an <img> tag (e.g. "/api/stream/camera"); ``source``
    #   is a short identifier ("camera" / "tab") for the UI label.
    # Device kind for this WS session — e.g. "web", "macos_agent",
    # "linux_agent", or None for unclassified sessions (e.g. /dev/respond).
    # Used by the schema filter to hide device-tier tools that don't match,
    # and by the tier gate in _execute_one as a defence-in-depth check.
    device_kind: str | None = None
    # True when the speaker has a live step-up grant — they tapped the
    # Web Push approval notification for this session within the grant TTL.
    # Read by _execute_one to gate private=True tools.  Fails closed
    # (False = no grant).
    step_up_auth: bool = False
    # True when this utterance contained more than one voice (#59).
    # profile_id is None on such turns; device-tier tools ask whose
    # device to use instead of guessing, memory stays shared, and the
    # auth window is not inherited.
    mixed: bool = False
    # Recognised household members on a mixed turn — used to phrase the
    # clarifying question.  Empty on single-speaker turns.
    participants: "tuple[str, ...]" = ()


@dataclass
class ToolInvocation:
    """A single tool execution recorded for logging/persistence."""
    name: str
    args: dict
    text: str
    data: dict | None
    terminal: bool
    # Optional XTTS voice override for THIS tool's reply.  Currently
    # used only by ``inbox_summary`` to read a voicemail summary in
    # the message author's voice (different from the recipient who is
    # speaking).  ``None`` means "use the session's latched voice".
    tts_voice_override: str | None = None


@dataclass
class AgentResult:
    """Outcome of one `run_agent` turn — what the WS layer surfaces."""
    response_text: str
    invocations: list[ToolInvocation] = field(default_factory=list)
    elapsed_ms: int = 0
    error: str | None = None
    # Inherited from the terminal tool's ToolResult.tts_voice_override.
    # The pipeline forwards this to the WS layer via PipelineOutcome
    # so a single reply can be voiced with a non-default speaker.
    tts_voice_override: str | None = None

    @property
    def last_tool(self) -> ToolInvocation | None:
        return self.invocations[-1] if self.invocations else None


# How many LLM round-trips we allow within one user turn. Each non-terminal
# tool call costs one extra LLM call (the model re-prompts after seeing the
# tool result). 4 is enough for the realistic chains we have today
# (general_answer → web_search), with headroom for one more hop.
MAX_AGENT_STEPS = 4


SYSTEM_PROMPT = (
    "You are a voice assistant. Replies are spoken aloud — keep them "
    "terse: 1–2 sentences, no markdown, no lists. Reply in the user's "
    "language; honour explicit requests to switch.\n"
    "Use prior conversation to resolve pronouns and references. When "
    "a question references something earlier, expand the topic in the "
    "tool arguments so the tool receives a self-contained query.\n"
    "You MUST respond by calling exactly one tool from the catalog — "
    "never free text. After a tool returns, stop, UNLESS the tool "
    "explicitly signals a retry (e.g. `general_answer` with "
    "`unknown=true` → follow up with `web_search` for the same "
    "question). Prefer `general_answer` for timeless knowledge."
)


# Tool schemas are static after first import — every voice turn used to
# deep-copy the full catalog just to splice in the current time on the
# scheduling tool.  We now build the catalog ONCE at module load, leaving a
# ``{now}`` placeholder in that one description.  The per-turn cost drops to
# one ``str.format()``.
#
# Only ``set_reminder_at`` gets the clock: it has to resolve "tomorrow at 9"
# into an absolute datetime.  ``set_timer`` takes an ISO-8601 duration and is
# independent of what time it is now.
_CLOCK_TOOL = "set_reminder_at"
_CLOCK_TEMPLATE = (
    "\n\nCurrent local time for THIS call: {now}. "
    "Use it ONLY to resolve relative phrasing into the `fire_at` "
    "ISO-8601 datetime. Do NOT use this timestamp to answer questions "
    "about the date or time — such questions must go through `web_search`."
)


def _build_cached_schemas() -> list[dict]:
    """Snapshot the registry once, embedding the clock placeholder."""
    out: list[dict] = []
    for s in schemas():
        if s.get("function", {}).get("name") == _CLOCK_TOOL:
            fn = dict(s["function"])
            fn["description"] = fn["description"] + _CLOCK_TEMPLATE
            out.append({**s, "function": fn})
        else:
            out.append(s)
    return out


_TOOL_SCHEMAS_CACHED: list[dict] | None = None


def _tool_visible_for_session(schema: dict, device_kind: str | None) -> bool:
    """Return True if a tool schema should be advertised to the LLM for a
    given session ``device_kind``.

    Rules:
    * tier="system"  → always visible.
    * tier="device"  → visible only when session.device_kind matches
                        tool.device_kind exactly.  Hidden on "web" sessions
                        and on sessions that didn't declare a device_kind.
    * tier="api"     → always visible (token gating happens at dispatch,
                        not at schema-advertisement time).
    """
    name = schema.get("function", {}).get("name")
    if not name:
        return True
    entry = TOOL_REGISTRY.get(name)
    if not entry:
        return True
    tier = entry.get("tier", "system")
    if tier == "device":
        return entry.get("device_kind") == device_kind
    # system or api — always advertise
    return True


def _schemas_with_clock(now_str: str, device_kind: str | None = None) -> list[dict]:
    """Return the tool catalog filtered for ``device_kind``, with ``{now}``
    stamped into the reminders entry.

    The global cache holds ALL schemas; we filter it here per-call (cheap
    O(n) pass over a ~15-entry list) so device-tier tools are invisible
    to sessions that don't have a matching desktop-agent.  Filtering
    happens AFTER the reminders clock stamp so the logic is in one place.
    """
    global _TOOL_SCHEMAS_CACHED
    if _TOOL_SCHEMAS_CACHED is None:
        _TOOL_SCHEMAS_CACHED = _build_cached_schemas()
    cached = _TOOL_SCHEMAS_CACHED
    # Splice the clock into the scheduling entry's description.  We
    # rebuild ONLY that one entry's wrapper so the rest of the list
    # stays the same object identity — no allocation churn.
    out = list(cached)
    for i, s in enumerate(out):
        if s.get("function", {}).get("name") == _CLOCK_TOOL:
            base = s["function"]
            # The placeholder appears at the tail of the description
            # (see _CLOCK_TEMPLATE above).  Find it once at startup
            # via partition; at call time we just str.format() the
            # cached string.
            new_desc = base["description"].format(now=now_str) if "{now}" in base["description"] else base["description"].replace("{now}", now_str)
            out[i] = {**s, "function": {**base, "description": new_desc}}
            break
    # Filter device-tier tools that don't match this session's device_kind.
    # system/api tools pass through unconditionally.
    out = [s for s in out if _tool_visible_for_session(s, device_kind)]
    return out


def invalidate_schemas_cache() -> None:
    """Test hook: force the schema catalog to rebuild on next call.

    Useful in unit tests that register a fresh tool decorator after
    the catalog snapshot — without this they'd see the cached list.
    """
    global _TOOL_SCHEMAS_CACHED
    _TOOL_SCHEMAS_CACHED = None


def _format_tool_message(inv: ToolInvocation) -> str:
    """
    Turn a tool result into the text the LLM sees on the next loop iteration.

    For value-tools the natural language reply is enough. For tools that
    expose a structured `unknown=True` signal (general_answer), we prepend
    an explicit hint so the model knows to retry with a different tool —
    Gemma doesn't always pick that up just from an empty content string.
    """
    if inv.data and inv.data.get("unknown"):
        return (
            f"Tool `{inv.name}` was NOT confident in its answer. "
            f"You should call a different tool (most likely `web_search` "
            f"with the same question) to actually answer the user."
        )
    return inv.text or "(empty tool result)"


async def run_agent(
    transcript: str,
    *,
    ctx: AgentContext,
    history: list[dict] | None = None,
    memory_context: str = "",
) -> AgentResult:
    """
    Drive a single user turn through the tool-using LLM until it either:
      - emits a final text response (no tool_calls), or
      - calls a `terminal` tool whose reply IS the final response, or
      - exceeds MAX_AGENT_STEPS (safety cap).

    Never raises — degrades to a fixed fallback message so the user always
    hears something.
    """
    system = SYSTEM_PROMPT
    if memory_context:
        system = system + "\n" + memory_context

    # The LLM needs a clock reference to convert "in an hour" / "tomorrow at 9"
    # into reminder arguments. We inject the timestamp ONLY into the
    # `reminders` tool description — NOT into the system prompt — so it
    # doesn't bleed into free-form answers (general_answer must NOT know
    # the date; date queries belong to web_search like any other fresh data).
    now_str = datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M%z")
    tool_schemas = _schemas_with_clock(now_str, device_kind=ctx.device_kind)

    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": transcript})

    result = AgentResult(response_text="")
    t0 = time.monotonic()

    for step in range(MAX_AGENT_STEPS):
        try:
            choice = await chat(
                messages,
                temperature=0.3,
                tools=tool_schemas,
                tool_choice="auto",
                # Tool selection is structural ("which of these 3-4 functions
                # best answers this question?") and rarely benefits from
                # extensive chain-of-thought.  "low" cuts ~1-2 s off this
                # step.  Bump back to "medium" if you start seeing wrong
                # tool picks on ambiguous queries.
                reasoning_effort="low",
                # Token usage attribution: the outer agent loop itself
                # spends tokens picking a tool, which we record under the
                # synthetic name "<agent_loop>" so /api/stats can show
                # how much of the budget is selection overhead vs.
                # actual tool work.
                client_id=ctx.client_id,
                tool_name="<agent_loop>",
            )
        except httpx.TimeoutException:
            log.warning("agent: LLM timeout at step %d", step)
            result.response_text = t("llm.timeout", ctx.user_lang)
            result.error = "llm_timeout"
            result.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return result
        except httpx.HTTPError as e:
            log.warning("agent: LLM http error at step %d: %s", step, e)
            result.response_text = t("llm.unreachable", ctx.user_lang)
            result.error = f"{e.__class__.__name__}: {e}"
            result.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return result

        msg = choice["message"]
        tool_calls = parse_tool_calls(msg)

        # No tool requested — the LLM responded with text. End of turn.
        if not tool_calls:
            content = extract_text(msg)
            if not content:
                if choice.get("finish_reason") == "length":
                    content = t("llm.hit_max_tokens", ctx.user_lang)
                else:
                    content = t("llm.empty", ctx.user_lang)
            log.info(
                "agent: step %d, no tool, finish=%s, %d chars",
                step,
                choice.get("finish_reason"),
                len(content),
            )
            result.response_text = content
            result.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return result

        # Persist the assistant's tool-call request so the next LLM iteration
        # sees a well-formed `assistant → tool → assistant` chat history.
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args, ensure_ascii=False),
                    },
                }
                for tc in tool_calls
            ],
        })

        # Execute every tool the LLM requested in this batch. Note all of them
        # for the record, but exit the loop as soon as a `terminal` tool runs
        # (its `text` IS the final spoken reply — no need to ping the LLM
        # again just to read it back).
        terminal_inv: ToolInvocation | None = None
        for tc in tool_calls:
            # Fallback progress signal: shown briefly before the tool emits
            # its own more-specific step.  If a tool emits nothing (new
            # custom tools, fast ones), the user at least sees the tool
            # name instead of the stale "Picking a tool" placeholder.
            if ctx.progress_sink is not None:
                try:
                    await ctx.progress_sink("tool", tc.name)
                except Exception:
                    pass
            inv = await _execute_one(tc, ctx)
            result.invocations.append(inv)
            log.info(
                "agent: step %d, tool=%r terminal=%s text_chars=%d",
                step,
                inv.name,
                inv.terminal,
                len(inv.text),
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": _format_tool_message(inv),
            })
            if inv.terminal and terminal_inv is None:
                terminal_inv = inv

        if terminal_inv is not None:
            result.response_text = terminal_inv.text
            result.tts_voice_override = terminal_inv.tts_voice_override
            result.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return result
        # Loop: LLM will see the tool messages and decide whether to call
        # another tool or wrap up with free text.

    # Hit the step cap. Use the last tool's text if we have one, otherwise
    # the generic fallback. This is a safety net — should be rare in practice.
    log.warning("agent: hit MAX_AGENT_STEPS=%d", MAX_AGENT_STEPS)
    if result.last_tool and result.last_tool.text:
        result.response_text = result.last_tool.text
    else:
        result.response_text = t("agent.max_steps", ctx.user_lang)
    result.error = "max_steps"
    result.elapsed_ms = int((time.monotonic() - t0) * 1000)
    return result


async def _execute_one(tc: ParsedToolCall, ctx: AgentContext) -> ToolInvocation:
    """
    Dispatch a single tool call and decide whether it ended the turn.

    Three things happen here, in order:

      1. **Risk gating** — if the tool is declared ``risk="high_write"``
         and the speaker hasn't supplied a passphrase in this auth
         window, we don't run it.  We enqueue the requested call in
         ``pending_actions`` and return a deferred ToolResult — the
         user is told the action was parked, they can approve later by
         saying the passphrase or clicking through in the UI.
      2. **Dispatch** — normal tool execution.
      3. **Terminal decision** — same as before; ``data["unknown"]``
         downgrades a normally-terminal tool to allow chaining.
    """
    meta = TOOL_REGISTRY.get(tc.name)
    static_terminal: bool = bool(meta.get("terminal", True)) if meta else True
    risk_raw = meta.get("risk", "read") if meta else "read"
    # Per-action risk: the tool may register a callable instead of a
    # fixed string so that a multi-action tool (e.g. ``items``) can
    # return "read" for list/search and "high_write" for auto_sort.
    risk: str = risk_raw(tc.args) if callable(risk_raw) else risk_raw

    # ── Device-tier routing + lock gate ───────────────────────────────
    tool_tier = meta.get("tier", "system") if meta else "system"
    if tool_tier == "device":
        required_device_kind = meta.get("device_kind") if meta else None

        # Does the tool schema accept an explicit agent_id?
        tool_params = (
            (meta.get("schema") or {})
            .get("function", {})
            .get("parameters") or {}
        )
        tool_accepts_agent_id = "agent_id" in (
            tool_params.get("properties") or {}
        )

        # ── Mixed turn (#59): ask whose device instead of guessing ─────
        # More than one voice spoke and neither the LLM (no explicit
        # agent_id) nor speaker-ID (no profile on mixed turns) can pin
        # the owner.  The question ends the turn; the continuation
        # window catches the answer, which re-enters as a clean
        # single-speaker turn and routes through the normal profile path.
        if (
            ctx.mixed
            and ctx.participants
            and tool_accepts_agent_id
            and not tc.args.get("agent_id")
        ):
            return ToolInvocation(
                name=tc.name,
                args=tc.args,
                text=t(
                    "clarify.which_device",
                    ctx.user_lang,
                    names=", ".join(ctx.participants),
                ),
                data={"clarify": "device"},
                terminal=True,
            )

        if ctx.profile_id is not None:
            # ── Profile-aware path: route to the speaker's personal device ──
            # Import lazily to avoid a circular dep (device_router → storage
            # → db, which is fine; the guard is for test isolation).
            from . import device_router

            # Only inject when the LLM didn't pick an agent explicitly.
            if tool_accepts_agent_id and not tc.args.get("agent_id"):
                resolved_agent = await device_router.resolve_agent_id_for_profile(
                    ctx.profile_id, required_device_kind
                )
                if resolved_agent is None:
                    return ToolInvocation(
                        name=tc.name,
                        args=tc.args,
                        text=t("tool.no_personal_device", ctx.user_lang),
                        data={
                            "error": "no_personal_device",
                            "device_kind": required_device_kind,
                        },
                        terminal=True,
                    )
                # Inject the resolved agent_id into the args for dispatch.
                tc = ParsedToolCall(
                    id=tc.id,
                    name=tc.name,
                    args={**tc.args, "agent_id": resolved_agent},
                )

            # Lock-state gate: only blocks when the tool explicitly
            # declares ``needs_unlock`` AND we KNOW the device is locked.
            # Unknown lock state (None) fails open — agent hasn't reported
            # yet, most likely cold boot.
            locked_compat = meta.get("locked_compat", "always") if meta else "always"
            if locked_compat == "needs_unlock":
                effective_agent_id = tc.args.get("agent_id")
                agent_info = desktop_client.get_agent(effective_agent_id)
                if agent_info is not None and agent_info.locked is True:
                    return ToolInvocation(
                        name=tc.name,
                        args=tc.args,
                        text=t("tool.device_locked", ctx.user_lang),
                        data={
                            "error": "device_locked",
                            "agent_id": effective_agent_id,
                        },
                        terminal=True,
                    )

        else:
            # ── Session-device path: no identified speaker ──────────────────
            # Defence-in-depth: the schema filter already hid device tools
            # from non-matching sessions, but catch any residual call here.
            if required_device_kind != ctx.device_kind:
                return ToolInvocation(
                    name=tc.name,
                    args=tc.args,
                    text=t("tool.device_not_available", ctx.user_lang),
                    data={
                        "error": "device_mismatch",
                        "required_device_kind": required_device_kind,
                        "session_device_kind": ctx.device_kind,
                    },
                    terminal=True,
                )

    # ── Step-up gating: private tools require push-to-device approval ─
    is_private = bool(meta.get("private", False)) if meta else False
    if is_private and not ctx.step_up_auth:
        if ctx.profile_id is None:
            # Can't send a push without a known profile.
            return ToolInvocation(
                name=tc.name,
                args=tc.args,
                text=t("step_up.no_profile", ctx.user_lang),
                data={"error": "step_up_no_profile"},
                terminal=True,
            )
        # Create a challenge token and send a push notification.  The
        # user taps the notification → SW POSTs /api/step-up/approve →
        # orchestrator broadcasts step_up_granted WS event → session
        # sets step_up_auth=True → user re-asks and tool executes.
        try:
            from .storage import create_step_up_grant, GRANT_TTL_S
            from . import push as _push
            challenge_token = await create_step_up_grant(
                profile_id=ctx.profile_id,
                client_id=ctx.client_id,
            )
            push_payload = {
                "type": "step_up_request",
                "title": t("step_up.push_title", ctx.user_lang),
                "body": t("step_up.push_body", ctx.user_lang, tool=tc.name),
                "tag": f"step_up_{ctx.client_id}",
                # The SW reads ``token`` from the notification data and
                # POSTs it to /api/step-up/approve on tap.  Token stays in
                # the (encrypted) push payload + POST body only — never in
                # a URL — so it can't leak via logs / history / Referer.
                "token": challenge_token,
                "tool": tc.name,
                "expires_in": GRANT_TTL_S,
            }
            await _push.send_to_profile(ctx.profile_id, push_payload, ttl_s=GRANT_TTL_S + 10)
            log.info(
                "agent: step-up challenge sent for %r (profile=%d, token=%.8s…)",
                tc.name, ctx.profile_id, challenge_token,
            )
        except Exception:
            log.exception("agent: step-up push failed for %r", tc.name)
        return ToolInvocation(
            name=tc.name,
            args=tc.args,
            text=t("step_up.pending", ctx.user_lang, tool=tc.name),
            data={"step_up_pending": True, "tool": tc.name},
            terminal=True,
        )

    # ── Tier-2 gating: defer high-write calls when no passphrase auth ─
    if risk == "high_write" and not ctx.is_authenticated:
        # Best-effort summary for the UI / readback.  Just join the
        # most semantically-loaded arg values; the LLM can produce
        # something cleaner later if we add a `summary` arg convention.
        summary_bits = [
            f"{k}={v}" for k, v in tc.args.items()
            if isinstance(v, (str, int, float)) and len(str(v)) <= 80
        ]
        summary = f"{tc.name}: " + ", ".join(summary_bits[:4]) if summary_bits else tc.name
        try:
            action_id = await enqueue_action(
                profile_id=ctx.profile_id,
                client_id=ctx.client_id,
                tool_name=tc.name,
                tool_args=tc.args,
                summary=summary,
            )
        except Exception:
            log.exception("agent: enqueue_action failed for %r", tc.name)
            return ToolInvocation(
                name=tc.name,
                args=tc.args,
                text=t("auth.queue_error", ctx.user_lang),
                data={"error": "enqueue_failed", "risk": risk},
                terminal=True,
            )
        log.info(
            "agent: deferred %r → pending_actions id=%d (no auth)",
            tc.name, action_id,
        )
        return ToolInvocation(
            name=tc.name,
            args=tc.args,
            text=t("auth.action_deferred", ctx.user_lang, summary=summary),
            data={
                "deferred": True,
                "pending_action_id": action_id,
                "risk": risk,
            },
            # Terminal: don't loop the LLM again — we already told the
            # user "I deferred it", further LLM thought would just
            # double-explain.
            terminal=True,
        )

    result = await dispatch(tc.name, tc.args, ctx=ctx)
    requested_followup = bool((result.data or {}).get("unknown"))
    return ToolInvocation(
        name=tc.name,
        args=tc.args,
        text=result.text,
        data=result.data,
        terminal=static_terminal and not requested_followup,
        tts_voice_override=result.tts_voice_override,
    )
