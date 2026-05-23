# Adding a new LLM tool

One async function + a JSON-schema describing its arguments. The LLM sees the schema; the orchestrator dispatches by name. Two pieces of plumbing:

- `@tool(...)` decorator in `orchestrator/app/tools/base.py:145` — registers into `TOOL_REGISTRY` at import time.
- Auto-discovery in `orchestrator/app/tools/__init__.py:24-33` — every non-underscored module under `tools/` is imported at startup.

Drop `orchestrator/app/tools/my_tool.py`, restart, the LLM sees the new tool. No central list to edit.

## Recipe

1. Create `orchestrator/app/tools/my_tool.py`.
2. Decorate an `async def` with `@tool(name=..., description=..., params_schema=..., terminal=..., risk=...)`.
3. Return a `ToolResult(text=..., data=...)`.
4. For user-facing strings, add `en` / `ru` / `de` entries to `orchestrator/app/i18n.py::CATALOG`.
5. Add a test under `orchestrator/tests/test_my_tool.py`.
6. `docker compose restart orchestrator`.

## Worked example — `dice`

Roll N dice with M sides, return the sum. Pure-function, same shape as `calculator`.

### Tool file — `orchestrator/app/tools/dice.py`

```python
"""dice — roll N dice with M sides, return the sum."""
from __future__ import annotations

import logging
import random

from ..i18n import t
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)


@tool(
    name="dice",
    description=(
        "Roll N dice with M sides each and return the total + individual "
        "rolls. Use when the user asks «roll a die», «брось кубик», "
        "«roll 3d6», «throw two twenty-sided dice». Do NOT use for "
        "deterministic random numbers (use calculator or a generator "
        "tool); do NOT use for shuffling or sampling — this is dice only."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "count": {"type": "integer", "description": "How many dice. 1..20.", "minimum": 1, "maximum": 20},
            "sides": {"type": "integer", "description": "Sides per die. Common: 6, 20.", "minimum": 2, "maximum": 1000},
        },
        "required": ["count", "sides"],
    },
    risk="read",
)
async def dice(count: int, sides: int, *, ctx=None) -> ToolResult:
    cx = unwrap_ctx(ctx)
    await cx.progress("dice", f"{count}d{sides}")
    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls)
    text = t("dice.result", cx.user_lang, total=total, count=count, sides=sides)
    return ToolResult(text=text, data={"rolls": rolls, "total": total})
```

### i18n — `orchestrator/app/i18n.py::CATALOG`

```python
    "dice.result": {
        "en": "Rolled {count}d{sides} — total {total}.",
        "ru": "Бросил {count}d{sides} — выпало {total}.",
        "de": "{count}d{sides} gewürfelt — Summe {total}.",
    },
```

Missing translations fall back to `en` (`i18n.py:329-348`).

### Test — `orchestrator/tests/test_dice.py`

```python
"""dice tool — pure-function smoke test."""
import random

import pytest

from app.tools.dice import dice


async def test_dice_returns_total_and_individual_rolls():
    random.seed(42)
    res = await dice(count=3, sides=6)
    assert res.data["total"] == sum(res.data["rolls"])
    assert len(res.data["rolls"]) == 3
    assert all(1 <= r <= 6 for r in res.data["rolls"])


async def test_dice_text_uses_i18n_format():
    res = await dice(count=2, sides=20)
    assert str(res.data["total"]) in res.text
    assert "2d20" in res.text
```

Run with `pytest orchestrator/tests/test_dice.py`. The `_fresh_db` fixture in `conftest.py` runs around every test; pure tools inherit it harmlessly.

## Decorator parameters

| Param          | Type   | Default | Notes |
|----------------|--------|---------|-------|
| `name`         | str    | —       | Unique, snake_case. LLM dispatches by this; also the key in `TOOL_REGISTRY`. |
| `description`  | str    | —       | What the LLM reads. Spend effort here — see below. |
| `params_schema`| dict   | —       | JSON-schema in OpenAI tool-call shape. |
| `terminal`     | bool   | `True`  | If True, the tool's `text` is the spoken reply and the agent loop ends. If False, the LLM sees the result and may chain. |
| `risk`         | str    | `"read"`| `"read"` / `"low_write"` / `"high_write"`. Gates execution behind the passphrase. |

### `description` — write for the LLM

This single field determines whether your tool gets picked. Patterns that work:

- Positive triggers — what the user would say (`«какая погода», «roll 3d6», «timer on 10 minutes»`).
- Negative triggers — when NOT to use it (`do NOT use for shuffling`).
- 2-3 few-shot args inline. See `calculator.py:382-391`, `set_reminder.py:108-122`.
- Cross-reference sibling tools by name when there's overlap (`read_settings.description` ends with "for «о чём мы говорили» use `my_history`" — `settings.py:53-56`).

### `terminal=False` — chaining

`general_answer.py` is the canonical example (`tools/general_answer.py:80-100`). When the LLM signals it doesn't know, the tool returns `data={"unknown": True}` and the agent loop chains into `web_search`. Decision lives in `app/agent.py::_execute_one`. Most tools should stay `terminal=True`.

### `risk` — three levels

| Level         | When to pick it                              | What happens                                                  |
|---------------|----------------------------------------------|---------------------------------------------------------------|
| `"read"`      | Answer-only, no state change                 | Always runs on voice ID alone. `calculator`, `weather`, `web_search`, `my_history`. |
| `"low_write"` | Reversible writes the user clearly initiated | Runs on voice ID alone. `reminders` — every action has a matching cancel. |
| `"high_write"`| Invasive, destructive, hard-to-reverse       | Runs only if `is_authenticated=True`, else queued in `pending_actions`. `update_settings`, `update_memory`, calendar writes. |

Pick the strictest level that's still honest. The agent never escalates or demotes. `tests/test_risk_gate.py:60` asserts unauthenticated `high_write` calls are enqueued, not executed.

## The `ctx` parameter

Optional. The dispatcher detects it via `inspect.signature` and injects `AgentContext` when present (`tools/base.py:201-209`). At the top:

```python
cx = unwrap_ctx(ctx)
# now: cx.client_id, cx.user_lang, cx.profile_id, cx.is_authenticated
# and: await cx.progress(step, detail)
```

`unwrap_ctx` lives in `tools/base.py:135-142`. `progress` no-ops when no upstream sink — call it unconditionally.

### Why progress matters

The "Думаю" card subtext is driven by `progress` messages — see `frontend/main.js:110-126` (`PROGRESS_LABELS`). Without a `progress` call the user sees stale "Запускаю инструмент" until your tool returns. Pick one short snake_case step name per phase and add a label to `PROGRESS_LABELS` in `frontend/main.js`. Unknown step names are silently ignored.

## Returning a `ToolResult`

```python
@dataclass
class ToolResult:
    text: str                           # what the assistant speaks
    data: dict | None = None            # structured payload for logs / UI
    tts_voice_override: str | None = None
```

Defined in `tools/base.py:59-69`.

- `text` — spoken via XTTS. Keep it under ~80 words; long lists become unbearable. Speak a summary, dump detail to `data` (see `inbox.py`).
- `data` — written to `utterances.response_data` in the DB. Use this for anything you'll grep later or display in the UI.
- `tts_voice_override` — XTTS speaker name to override the user's latched voice for THIS reply. Used by `inbox.py` to play a voicemail summary in the SENDER's cloned voice. Most tools leave it `None`.

## Calling the LLM from inside a tool

Use `llm_utils.chat(...)` — single entry point for `/v1/chat/completions`.

```python
from ..llm_utils import chat, parse_tool_calls

choice = await chat(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ],
    temperature=0.6,
    tools=[...],                  # optional, OpenAI shape
    tool_choice="required",       # optional
    client_id=cx.client_id,       # for token accounting
    tool_name="my_tool",          # for token accounting
)
```

See `llm_utils.py:135-216` for the full signature. `client_id` + `tool_name` are rolled into `token_usage` so the dashboard tracks your tool's cost.

### Vision

```python
from ..llm_utils import chat, LLM_VISION_URL, LLM_VISION_MODEL

choice = await chat(
    messages,
    endpoint_url=LLM_VISION_URL,
    model=LLM_VISION_MODEL,
    client_id=cx.client_id,
    tool_name="my_vision_tool",
)
```

When both endpoints point at the same provider (default), these vars resolve to `LLM_URL` / `LLM_MODEL`. Splitting is opt-in via env (see `docker-compose.yml:46-54`).

## i18n notes

- Keys are dot-namespaced (`my_tool.error_empty_dice`). EN required; RU/DE recommended.
- Placeholders are Python `str.format` with NAMED args (`{total}`, not `{0}`) so translators can reorder.
- For "needs internet, we're offline" use `offline_for_tool("tool.weather", lang)` (`i18n.py:356-364`). Register a `tool.<name>` noun key (`i18n.py:65-77`).

The legacy `messages.py` module still holds Russian-only constants for pipeline paths. New code uses `i18n.t(key, lang)` directly; only touch `messages.py` when migrating a legacy call site.

## Common pitfalls

- **Forgetting `required` in `params_schema`.** The LLM sometimes skips required args, and `TypeError` from your handler comes back as a generic "Не понял параметры команды." (`base.py:213`).
- **Spoken text longer than ~80 words.** Speak the summary, dump the full list to `data`.
- **`print()` instead of `logging.getLogger(__name__)`.** Print isn't tagged with the tool name.
- **No offline degradation for HTTP-dependent tools.** Use `app.net.has_internet()` as a fast-path check — see `weather.py:184-190`, `calculator.py:478-484`. Without it your tool blocks the voice loop on a 5-10 s TCP timeout when offline.
- **Mutating global state in a tool module.** Tools import once at startup; module-level state survives across calls AND across users. Use the DB layer (`app/storage/`) for anything durable.
- **`terminal=True` with `data={"unknown": True}`.** Chain never happens. See `general_answer.py:166-170` for the contract.

## Checklist before opening a PR

- [ ] Tool file under `orchestrator/app/tools/` — auto-discovered.
- [ ] `description` includes positive + negative triggers.
- [ ] `params_schema.required` covers every non-optional argument.
- [ ] `risk` set to the strictest honest level.
- [ ] User-facing strings keyed in `i18n.py::CATALOG` with `en`/`ru`/`de`.
- [ ] Progress step registered in `frontend/main.js::PROGRESS_LABELS` if the tool takes >1 s.
- [ ] HTTP-dependent path guarded by `await has_internet()`.
- [ ] Test under `orchestrator/tests/test_<name>.py` — happy path + error path.
- [ ] No `print()`; no module-level mutable user state; English comments and identifiers throughout.
