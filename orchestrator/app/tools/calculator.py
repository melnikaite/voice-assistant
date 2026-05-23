"""
calculator — arithmetic evaluation + currency conversion.

Two modes selected by the ``mode`` argument:

  * ``arith``    — Evaluate a math expression.  Uses Python's ``ast``
                   module to parse and walk the tree; only BinOp / UnaryOp
                   / Constant / Num / Pow / USub nodes are allowed.  No
                   names, no calls, no attribute access — and absolutely
                   no ``eval``/``exec``.  This makes the path safe against
                   arbitrary code execution from the LLM-supplied string.

  * ``currency`` — Convert ``amount`` of source currency to destination.
                   Rates come from fawazahmed0/currency-api — a free,
                   CDN-hosted (jsdelivr + Cloudflare mirror) feed of
                   ~200 currencies + crypto + metals, refreshed daily.
                   We fetch with EUR as base so the conversion code can
                   keep its EUR-pivot shape, then cache in memory for
                   24 h.  The official jsdelivr endpoint is primary;
                   the Cloudflare mirror kicks in only if jsdelivr is
                   unreachable, giving us a near-100% uptime story
                   without depending on any single CDN.

Number-to-words and currency naming come from ``i18n`` (num2words /
locale JSON) so the spoken reply is in the speaker's language.  Floats
are emitted as digits with a comma decimal separator because spelling
out fractional parts is awkward in most languages.
"""
from __future__ import annotations

import ast
import logging
import operator
import time

import httpx

from ..i18n import currency_alias, currency_name, num_to_words, t
from ..net import has_internet
from .base import ToolResult, tool, unwrap_ctx

log = logging.getLogger(__name__)


# ── Arithmetic via AST whitelist ────────────────────────────────────────
#
# We parse the expression with ``ast.parse(..., mode="eval")`` and walk
# the resulting tree, rejecting any node not in the explicit allowlist.
# This is safer than `eval` even with a constrained namespace: the AST
# walk forbids attribute access, function calls, names, comprehensions,
# subscripting, and every other escape hatch in one place.

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    """Recursive whitelist-only evaluator. Raises ValueError on anything else."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {type(node.value).__name__}")
    # ast.Num was a separate node type pre-3.8 and removed in 3.14.
    # ``getattr`` keeps the branch alive on older interpreters that the
    # production Docker image might still be pinned to, without raising
    # ``AttributeError`` on newer ones.
    _Num = getattr(ast, "Num", None)
    if _Num is not None and isinstance(node, _Num):  # pragma: no cover
        return node.n
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported binary op: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        # Cap exponents — `10 ** 10000` would spin forever and consume RAM.
        if op is operator.pow and isinstance(right, (int, float)) and abs(right) > 100:
            raise ValueError("exponent too large")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"unsupported node: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Evaluate ``expression`` under the AST whitelist. Raises ValueError on reject."""
    expr = expression.replace(",", ".")  # tolerate Russian / German decimal commas
    # Replace common spoken-math operators with their Python equivalents.
    # The LLM should output canonical Python operators, but be forgiving:
    # '×' / '·' → '*', '÷' → '/'.
    expr = expr.replace("×", "*").replace("·", "*").replace("÷", "/")
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree)


# ── Rates cache (fawazahmed0/currency-api) ─────────────────────────────
#
# Source: https://github.com/fawazahmed0/exchange-api — a free, no-key,
# rate-limit-free feed of ~200 fiat + crypto + metals, refreshed daily.
# Served from two CDN mirrors (jsdelivr primary, Cloudflare Pages
# secondary) — we try them in order so one outage doesn't break us.
#
# We fetch with EUR as base so the result is already shaped as
# ``{<CODE>: per-EUR-rate}`` — same as the old ECB-only path — and the
# downstream ``_convert()`` (EUR pivot) stays untouched.
#
# Cache lives in-memory for 24 h; the feed itself updates around 00:00
# UTC so anything more frequent would be wasted requests.

_RATES_CACHE: dict[str, float] | None = None
_RATES_EXPIRES_AT: float = 0.0
_RATES_TTL_S = 86400  # 24 hours

# Two CDN mirrors of the SAME data.  Order = priority on a fresh fetch.
# Each URL must produce a payload of shape ``{"date": "...", "eur":
# {"usd": 1.087, "rub": 100.5, ...}}`` — i.e. ISO codes lowercased,
# values = rate per 1 EUR.
_RATES_URLS = (
    "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/eur.json",
    "https://latest.currency-api.pages.dev/v1/currencies/eur.json",
)


async def _fetch_currency_rates() -> dict[str, float] | None:
    """Fetch the EUR-base rates payload, tolerant to one CDN being down.

    Returns ``{ISO_CODE: per_eur_rate}`` (including the synthetic
    ``"EUR": 1.0``) on success, ``None`` if both mirrors fail.
    """
    payload: dict | None = None
    last_err: str | None = None
    async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
        for url in _RATES_URLS:
            try:
                r = await client.get(url)
                r.raise_for_status()
                payload = r.json()
                if payload:
                    break
            except (httpx.HTTPError, ValueError) as exc:
                last_err = f"{exc.__class__.__name__}: {exc}"
                log.warning("rates: %s failed (%s) — trying next mirror", url, last_err)
                payload = None
    if not payload:
        log.warning("rates: all CDN mirrors failed; last error: %s", last_err)
        return None

    # The feed nests its quotes under a lowercase base key.  Be defensive
    # if the schema ever shifts: tolerate either {"eur": {...}} or a flat
    # {...} object.
    quotes = payload.get("eur") if isinstance(payload.get("eur"), dict) else payload
    rates: dict[str, float] = {"EUR": 1.0}
    for code, rate in quotes.items():
        if not isinstance(code, str):
            continue
        try:
            rates[code.upper()] = float(rate)
        except (TypeError, ValueError):
            continue
    if len(rates) <= 1:
        log.warning("rates: parsed 0 currencies from payload")
        return None
    log.info("rates: parsed %d currencies from %s", len(rates), url)
    return rates


async def _fetch_rates() -> dict[str, float] | None:
    """24h-cached wrapper around :func:`_fetch_currency_rates`."""
    global _RATES_CACHE, _RATES_EXPIRES_AT
    now = time.time()
    if _RATES_CACHE is not None and _RATES_EXPIRES_AT > now:
        return _RATES_CACHE
    fetched = await _fetch_currency_rates()
    if not fetched:
        return None
    _RATES_CACHE = fetched
    _RATES_EXPIRES_AT = now + _RATES_TTL_S
    return fetched


def _convert(amount: float, src: str, dst: str, rates: dict[str, float]) -> float | None:
    """Convert ``amount`` from src to dst via EUR pivot. ``None`` on unknown ccy."""
    src = src.upper()
    dst = dst.upper()
    if src not in rates or dst not in rates:
        return None
    # rates[X] = how many X you get for 1 EUR.
    # To go src → dst: amount * (rates[dst] / rates[src]).
    return amount * (rates[dst] / rates[src])


# ── Number formatting for spoken reply ─────────────────────────────────
#
# Words for small enough integers; digit form for everything else and
# all floats (decimal commas read more naturally than spelling
# fractional parts).  ``num_to_words`` in i18n delegates to ``num2words``
# under the hood — handles English/Russian/German correctly.


def _format_number(value: float, lang: str | None) -> str:
    """Format a number for spoken output. Whole → words; fractional → digits."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        # Words for small enough integers, digits for huge.
        if -999_999 <= value <= 999_999:
            return num_to_words(value, lang)
        return f"{value}"
    # Floats: digits with comma separator, rounded to 2 decimals.
    return f"{value:.2f}".replace(".", ",")


# ── Currency name → ISO code (multi-locale) ───────────────────────────
#
# Aliases live in each locale's JSON.  ``currency_alias`` searches the
# user's locale first then falls back to English, so the matcher works
# regardless of which language the speaker happens to use for a noun.
# Anything not in the alias table falls through to uppercasing whatever
# the LLM produced (already the canonical ISO form 99% of the time).


def _normalize_currency(ccy: str, lang: str | None) -> str:
    """Resolve a spoken currency phrase to an ISO code via locale aliases."""
    resolved = currency_alias(ccy, lang)
    if resolved:
        return resolved
    return ccy.strip().upper()


@tool(
    name="calculator",
    description=(
        "Evaluate arithmetic expressions and convert currencies.\n"
        "  • mode='arith' + expression='12 * 4'   — math.  Operators: + - * / "
        "% ** and parentheses.  Use Python-syntax operators.\n"
        "  • mode='currency' + amount=500 + from_ccy='EUR' + to_ccy='USD' — "
        "convert between ~200 currencies including fiat (EUR/USD/GBP/JPY/"
        "CNY/CHF/TRY/PLN/...), crypto (BTC/ETH/USDT/SOL/...), and precious "
        "metals (XAU/XAG).  ISO 4217 codes preferred but natural-language "
        "names also accepted ('euro', 'dollar', 'rubles', 'bitcoin')."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["arith", "currency"],
                "description": "Which operation to perform.",
            },
            "expression": {
                "type": "string",
                "description": (
                    "Required for mode=arith. Python-style math expression, "
                    "e.g. '12 * 4', '(5 + 3) / 2', '2 ** 10'."
                ),
            },
            "amount": {
                "type": "number",
                "description": "Required for mode=currency. Numeric amount to convert.",
            },
            "from_ccy": {
                "type": "string",
                "description": (
                    "Required for mode=currency. Source currency — ISO 4217 "
                    "('EUR', 'USD') or a natural-language name ('euro', 'dollar')."
                ),
            },
            "to_ccy": {
                "type": "string",
                "description": (
                    "Required for mode=currency. Destination currency — same "
                    "format as `from_ccy`."
                ),
            },
        },
        "required": ["mode"],
    },
    risk="read",
)
async def calculator(
    *,
    mode: str,
    expression: str | None = None,
    amount: float | None = None,
    from_ccy: str | None = None,
    to_ccy: str | None = None,
    ctx=None,
) -> ToolResult:
    cx = unwrap_ctx(ctx)
    lang = cx.user_lang

    # ── arith ────────────────────────────────────────────────────────
    if mode == "arith":
        if not expression or not expression.strip():
            return ToolResult(
                text=t("calculator.no_expression", lang),
                data={"error": "no_expression"},
            )
        await cx.progress("calculate", None)
        try:
            value = safe_eval(expression)
        except ZeroDivisionError:
            return ToolResult(
                text=t("calculator.div_by_zero", lang),
                data={"expression": expression, "error": "div_zero"},
            )
        except (SyntaxError, ValueError, TypeError) as e:
            log.info("calculator arith: bad expression %r: %s", expression, e)
            return ToolResult(
                text=t("calculator.bad_expression", lang),
                data={"expression": expression, "error": f"{e.__class__.__name__}"},
            )
        return ToolResult(
            text=t("calculator.result", lang, value=_format_number(value, lang)),
            data={"expression": expression, "result": value},
        )

    # ── currency ─────────────────────────────────────────────────────
    if mode == "currency":
        if amount is None:
            return ToolResult(
                text=t("calculator.no_expression", lang), data={"error": "no_amount"},
            )
        if not from_ccy or not to_ccy:
            return ToolResult(
                text=t("calculator.unsupported_currency", lang),
                data={"error": "missing_currency"},
            )
        # Rates live on a CDN; short-circuit when offline so we don't
        # block the voice loop waiting on the TCP timeout.  arith mode
        # above stays fully local and unaffected.  When we have a
        # cached rates dict from a previous online call, fall through
        # and serve from cache even if we're temporarily offline.
        if not await has_internet() and _RATES_CACHE is None:
            return ToolResult(
                text=t("offline.for_tool", lang, what=t("tool.currency", lang)),
                data={"error": "offline", "from": from_ccy, "to": to_ccy},
            )
        await cx.progress("currency", f"{from_ccy}→{to_ccy}")

        src = _normalize_currency(from_ccy, lang)
        dst = _normalize_currency(to_ccy, lang)
        rates = await _fetch_rates()
        if rates is None:
            return ToolResult(
                text=t("calculator.rates_unavailable", lang),
                data={"error": "rates_unavailable"},
            )
        converted = _convert(float(amount), src, dst, rates)
        if converted is None:
            log.info(
                "calculator currency: unknown ccy %r/%r (have %d)",
                src, dst, len(rates),
            )
            return ToolResult(
                text=t("calculator.unsupported_currency", lang),
                data={"from": src, "to": dst, "error": "unknown_currency"},
            )
        # Rate displayed = how many dst for 1 src (useful for the user).
        rate_per_unit = rates[dst] / rates[src]
        amount_str = _format_number(float(amount), lang)
        result_str = _format_number(converted, lang)
        rate_str = f"{rate_per_unit:.4f}".rstrip("0").rstrip(".").replace(".", ",")
        return ToolResult(
            text=t(
                "calculator.currency_reply", lang,
                amount=amount_str,
                src=currency_name(src, lang),
                result=result_str,
                dst=currency_name(dst, lang),
                rate=rate_str,
            ),
            data={
                "amount": amount,
                "from": src,
                "to": dst,
                "result": converted,
                "rate": rate_per_unit,
            },
        )

    return ToolResult(
        text=t("calculator.bad_expression", lang),
        data={"error": f"unknown_mode:{mode!r}"},
    )
