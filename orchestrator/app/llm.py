"""
Thin compatibility facade over the agent loop.

Historically `respond()` did the whole job: build messages → one LLM call →
dispatch one tool → return text. With the multi-step agent loop in
``agent.py`` that logic is gone — `respond()` now just wraps `run_agent`
and projects its richer `AgentResult` onto the legacy `LlmDecision` shape
that the dev `/dev/respond` HTTP endpoint and persistence in `ws.py` still
expect.

New code should call `run_agent` directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .agent import AgentContext, run_agent

log = logging.getLogger(__name__)


@dataclass
class LlmDecision:
    response_text: str
    tool_name: str | None
    tool_args: dict | None
    tool_data: dict | None
    elapsed_ms: int


async def respond(
    transcript: str,
    history: list[dict] | None = None,
    memory_context: str = "",
    *,
    ctx: AgentContext | None = None,
) -> LlmDecision:
    """
    Run one user turn through the agent loop and flatten the result.

    The legacy callers (the `/dev/respond` HTTP endpoint, and the
    persistence record in `ws.py`) need a single tool name + args to log.
    We surface the LAST invocation — for chained turns
    (general_answer→web_search) that's the one whose `response_text` the
    user actually heard, which is what's interesting in logs.
    """
    ctx = ctx or AgentContext()
    result = await run_agent(
        transcript,
        ctx=ctx,
        history=history,
        memory_context=memory_context,
    )
    last = result.last_tool
    return LlmDecision(
        response_text=result.response_text,
        tool_name=last.name if last else None,
        tool_args=last.args if last else None,
        tool_data=last.data if last else None,
        elapsed_ms=result.elapsed_ms,
    )
