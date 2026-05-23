"""
Tool decorator + registry.

A tool is just an async function with a JSON-schema describing its
arguments. The decorator registers it; the registry exposes a flat
``schemas()`` list (what the LLM sees) and a ``dispatch()`` function
(how the agent loop runs them).

Three flags control how the agent loop treats a tool:

* ``terminal`` — default ``True``. The tool's ``text`` IS the final
  spoken reply; one call ends the turn. Set ``terminal=False`` on
  tools that benefit from an LLM follow-up (e.g. ``general_answer``
  signalling ``unknown=True`` so the LLM can chain into ``web_search``).

* ``risk`` — security classification used by the agent loop to gate
  invasive actions behind the spoken passphrase:

    - ``read`` (default) — answer-only tools: weather, calculator,
      translate, general_answer, web_search, list-anything.  Always
      execute on voice ID alone.
    - ``low_write`` — write actions that are reversible and low-stakes:
      timers, reminders, volume, music control, "like" actions, brief
      desktop tweaks.  Voice ID is enough; never queued.
    - ``high_write`` — invasive / destructive / hard-to-reverse: edit
      memory, change settings, create or delete calendar events, send
      mail/messages, run arbitrary AppleScript, factory-reset.  Require
      either an active passphrase auth window OR get deferred into the
      ``pending_actions`` queue for later approval.

  The classification is the *tool's* responsibility — pick the strictest
  level that's still honest.  The agent never escalates a ``read``
  call's risk, and never demotes a ``high_write`` one.

Tools may optionally accept ``ctx: AgentContext`` as a keyword argument —
the dispatch layer detects this via ``inspect.signature`` and injects
the current request's context (client_id etc.). Tools that don't take
ctx stay pure functions of their LLM-visible args.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

if TYPE_CHECKING:
    from ..agent import AgentContext

log = logging.getLogger(__name__)


# Risk levels — see module docstring for definitions.  Used by
# ``agent._execute_one`` to decide whether a tool call may run as-is,
# must wait on a passphrase, or should be enqueued for later approval.
RiskLevel = Literal["read", "low_write", "high_write"]


@dataclass
class ToolResult:
    text: str  # what the assistant should say to the user
    data: dict | None = None  # optional structured payload (for logs / UI)
    # Optional per-call TTS voice override.  When set, the WS layer
    # speaks ``text`` with this XTTS voice instead of the current
    # speaker's latched voice.  Used by the inbox tools to read a
    # voicemail summary in the SENDER's voice (the LLM author is the
    # household owner, but the voice we want is the message author's).
    # ``None`` means "keep the session's current speaker voice".
    tts_voice_override: str | None = None


ToolHandler = Callable[..., Awaitable[ToolResult]]

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


# ──────────────────────────────────────────────────────────────────────
# Tool-side helpers
# ──────────────────────────────────────────────────────────────────────
#
# Almost every tool repeats the same three patterns:
#
#   progress = getattr(ctx, "progress_sink", None) if ctx else None
#   client_id = getattr(ctx, "client_id", None) if ctx else None
#   lang = getattr(ctx, "user_lang", None) if ctx else None
#
#   async def _progress(step, detail=None):
#       if progress is not None:
#           await progress(step, detail)
#
# That's ~6 lines of boilerplate per tool, and the same pattern was
# diverging in subtle ways (some tools forgot the None-check on ctx,
# some forgot the second arg on progress, etc.).  ``unwrap_ctx``
# collapses it to one call returning a tiny object with safe accessors.


class ToolCtx:
    """Tool-side context handle with safe accessors + a progress helper.

    Build via :func:`unwrap_ctx(ctx)` at the top of every tool.  Holds
    the resolved ``client_id``, ``user_lang``, and an always-callable
    ``progress(step, detail=None)`` coroutine that no-ops when the
    upstream sink is missing.

    Tools that legitimately need the full :class:`~app.agent.AgentContext`
    (very rare — only pending.py needs ``profile_id``) can still take
    ``ctx`` themselves.  This helper is the 90% case.
    """

    __slots__ = ("client_id", "user_lang", "profile_id", "is_authenticated", "_progress_sink", "_stream_sink")

    def __init__(self, ctx: "Any | None"):
        self.client_id = getattr(ctx, "client_id", None) if ctx else None
        self.user_lang = getattr(ctx, "user_lang", None) if ctx else None
        self.profile_id = getattr(ctx, "profile_id", None) if ctx else None
        self.is_authenticated = bool(getattr(ctx, "is_authenticated", False)) if ctx else False
        self._progress_sink = getattr(ctx, "progress_sink", None) if ctx else None
        self._stream_sink = getattr(ctx, "stream_sink", None) if ctx else None

    async def progress(self, step: str, detail: str | None = None) -> None:
        """No-op when there's no upstream sink (e.g. /dev/respond)."""
        sink = self._progress_sink
        if sink is None:
            return
        try:
            await sink(step, detail)
        except Exception:
            log.debug("ToolCtx.progress: sink raised", exc_info=True)

    @property
    def stream_sink(self):
        return self._stream_sink


def unwrap_ctx(ctx: "Any | None") -> ToolCtx:
    """Build a :class:`ToolCtx` from an :class:`AgentContext` (or ``None``).

    Tools call ``cx = unwrap_ctx(ctx)`` at the top, then use
    ``await cx.progress(...)``, ``cx.client_id``, ``cx.user_lang``
    without further None-checks.
    """
    return ToolCtx(ctx)


def tool(
    name: str,
    description: str,
    params_schema: dict,
    *,
    terminal: bool = True,
    risk: "RiskLevel | Callable[[dict], RiskLevel]" = "read",
):
    """Decorator: register an async function as an LLM-callable tool.

    ``risk`` is the security classification (see module docstring).
    Default is ``"read"`` — explicitly opt into ``low_write`` or
    ``high_write`` when defining tools that mutate user state.

    For multi-action tools pass a callable ``(args: dict) -> RiskLevel``
    instead of a fixed string.  The agent loop calls it with the actual
    args at dispatch time so per-action gating is fully dynamic.
    """

    def deco(fn: ToolHandler):
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"tool {name!r} must be async")
        sig = inspect.signature(fn)
        TOOL_REGISTRY[name] = {
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": params_schema,
                },
            },
            "handler": fn,
            "terminal": terminal,
            "risk": risk,
            "wants_ctx": "ctx" in sig.parameters,
        }
        return fn

    return deco


def schemas() -> list[dict]:
    """The list of tools to advertise to the LLM (OpenAI-style)."""
    return [v["schema"] for v in TOOL_REGISTRY.values()]


async def dispatch(
    name: str,
    args: dict,
    *,
    ctx: "AgentContext | None" = None,
) -> ToolResult:
    # Late import to avoid a circular dep: i18n loads cleanly but tools
    # pull base.py at module import time, before app.i18n is initialised
    # in some test paths.
    from ..i18n import t

    lang = getattr(ctx, "user_lang", None) if ctx else None
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        log.warning("unknown tool: %r", name)
        return ToolResult(
            text=t("tools.unknown_tool", lang, name=name),
            data={"error": "unknown_tool"},
        )
    handler: ToolHandler = entry["handler"]
    call_kwargs = dict(args)
    if entry["wants_ctx"]:
        if ctx is None:
            log.warning("tool %r needs ctx but none was provided", name)
            return ToolResult(
                text=t("tools.internal_no_ctx", lang),
                data={"error": "missing_ctx"},
            )
        call_kwargs["ctx"] = ctx
    try:
        return await handler(**call_kwargs)
    except TypeError as e:
        log.warning("tool %r bad args %r: %s", name, args, e)
        return ToolResult(text=t("tools.bad_args", lang), data={"error": str(e)})
    except Exception as e:
        log.exception("tool %r crashed", name)
        return ToolResult(text=t("tools.crashed", lang), data={"error": str(e)})
