"""
Structured datasets — schema building, entry validation, forecasting.

Pure functions; storage lives in ``storage/datasets.py`` and the voice
surface in ``tools/datasets.py``.  A dataset's schema is a small JSON
Schema subset: ``object`` with ``properties`` / ``required``, leaf types
``string | number | integer | boolean``, plus two string formats the
period logic understands (``month`` = ``YYYY-MM``, ``date`` =
``YYYY-MM-DD``) and an ``x-unit`` annotation.  Nested objects are
unrestricted; arrays are stored as given and not validated further.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
from typing import Any, Iterable

LEAF_TYPES = ("string", "number", "integer", "boolean", "month", "date", "array")
_NAME_RE = re.compile(r"^[^\W\d][\w]*$", re.UNICODE)
_AMOUNT_NAMES = ("amount", "sum", "total", "cost", "price", "value")
MIN_FORECAST_PERIODS = 12


class SchemaError(ValueError):
    pass


# ── Schema ────────────────────────────────────────────────────────────────

def build_schema(fields: Iterable[dict]) -> dict:
    """``[{name: "meter.from", type: "number", unit?, required?, description?}]``
    → JSON Schema.  Dotted names become nested objects; ``required``
    defaults to true."""
    root: dict = {"type": "object", "properties": {}, "required": []}
    for f in fields:
        if not isinstance(f, dict) or not f.get("name"):
            raise SchemaError("field without a name")
        parts = [p.strip() for p in str(f["name"]).split(".")]
        if not all(_NAME_RE.match(p) for p in parts):
            raise SchemaError(f"bad field name {f['name']!r}")
        typ = str(f.get("type") or "string").lower()
        if typ not in LEAF_TYPES:
            typ = {"text": "string", "float": "number", "int": "integer",
                   "bool": "boolean", "datetime": "date"}.get(typ, "string")
        node = root
        for p in parts[:-1]:
            child = node["properties"].setdefault(
                p, {"type": "object", "properties": {}, "required": []})
            if child.get("type") != "object":
                raise SchemaError(f"{p!r} is both a value and an object")
            if f.get("required", True) and p not in node["required"]:
                node["required"].append(p)
            node = child
        leaf: dict
        if typ in ("month", "date"):
            leaf = {"type": "string", "format": typ}
        else:
            leaf = {"type": typ}
        if f.get("unit"):
            leaf["x-unit"] = str(f["unit"])
        if f.get("description"):
            leaf["description"] = str(f["description"])
        node["properties"][parts[-1]] = leaf
        if f.get("required", True) and parts[-1] not in node["required"]:
            node["required"].append(parts[-1])
    if not root["properties"]:
        raise SchemaError("no fields")
    return root


def merge_schema(schema: dict, fields: Iterable[dict]) -> dict:
    """Add fields to an existing schema.  Added fields are optional:
    older entries don't have them."""
    extra = build_schema([{**f, "required": False} for f in fields])
    out = _copy(schema)

    def merge(dst: dict, src: dict) -> None:
        for k, v in src["properties"].items():
            if k in dst["properties"] and v.get("type") == "object":
                merge(dst["properties"][k], v)
            else:
                dst["properties"].setdefault(k, v)
    merge(out, extra)
    return out


def flatten(schema: dict, prefix: str = "") -> list[dict]:
    """Leaf fields as ``{path, type, unit, required}``, schema order."""
    out = []
    req = set(schema.get("required") or ())
    for k, v in (schema.get("properties") or {}).items():
        path = f"{prefix}{k}"
        if v.get("type") == "object":
            out += flatten(v, path + ".")
            continue
        out.append({
            "path": path,
            "type": v.get("format") or v.get("type", "string"),
            "unit": v.get("x-unit"),
            "required": k in req,
        })
    return out


def guess_hints(schema: dict) -> tuple[str | None, str | None]:
    """(amount_field, period_field) when the user didn't name them."""
    leaves = flatten(schema)
    numbers = [f["path"] for f in leaves if f["type"] in ("number", "integer")]
    amount = next((p for p in numbers if p.rsplit(".", 1)[-1].lower() in _AMOUNT_NAMES),
                  numbers[0] if numbers else None)
    period = next((f["path"] for f in leaves if f["type"] in ("month", "date")), None)
    return amount, period


def describe(ds: dict, max_fields: int = 12, values: dict[str, list] | None = None) -> str:
    """One compact line for the model: name, description, fields, and the
    stored ``values`` of some fields."""
    bits = []
    for f in flatten(ds["schema"])[:max_fields]:
        s = f["path"] + ("*" if f["required"] else "")
        typ = f["type"] if f["type"] != "string" else ""
        if f["unit"]:
            typ = f"{typ} {f['unit']}".strip()
        s = f"{s}:{typ}" if typ else s
        if (values or {}).get(f["path"]):
            s += " (" + "|".join(str(v) for v in values[f["path"]]) + ")"
        bits.append(s)
    line = f"«{ds['name']}»"
    if ds.get("description"):
        line += f" ({ds['description'][:80]})"
    return line + " — " + ", ".join(bits)


def flatten_names(schema: dict) -> set[str]:
    return {f["path"] for f in flatten(schema)}


def known_path(schema: dict, path: str) -> bool:
    return any(f["path"] == path for f in flatten(schema))


# ── Validation ────────────────────────────────────────────────────────────

def validate(entry: dict, schema: dict) -> tuple[dict, list[str], list[str], list[str]]:
    """→ (clean, missing, unknown, invalid), paths dotted.

    ``clean`` holds coerced known values only; the caller saves it iff all
    three lists are empty."""
    missing: list[str] = []
    unknown: list[str] = []
    invalid: list[str] = []

    def walk(val: Any, node: dict, path: str) -> Any:
        props = node.get("properties") or {}
        out = {}
        if not isinstance(val, dict):
            invalid.append(path.rstrip(".") or "entry")
            return out
        for k, v in val.items():
            if k not in props:
                if not _empty(v):
                    unknown.append(path + k)
                continue
            if _empty(v):
                continue
            sub = props[k]
            if sub.get("type") == "object":
                out[k] = walk(v, sub, path + k + ".")
                continue
            ok, coerced = _coerce(v, sub)
            if ok:
                out[k] = coerced
            else:
                invalid.append(path + k)
        for k in node.get("required") or ():
            if k not in out:
                sub = props.get(k, {})
                if sub.get("type") == "object":
                    walk({}, sub, path + k + ".")  # report leaves, not the object
                else:
                    missing.append(path + k)
        return out

    clean = walk(entry, schema, "")
    return clean, missing, unknown, invalid


def leaf_paths(d: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += leaf_paths(v, f"{prefix}{k}.")
        else:
            out.append(prefix + k)
    return out


def get_path(d: dict, path: str) -> Any:
    for p in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def deep_merge(a: dict, b: dict) -> dict:
    out = _copy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def unflatten(d: dict) -> dict:
    """``{"meter.from": 1}`` → ``{"meter": {"from": 1}}`` (models do both)."""
    out: dict = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            v = unflatten(v)
        node = out
        parts = str(k).split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                break
        else:
            if isinstance(v, dict) and isinstance(node.get(parts[-1]), dict):
                node[parts[-1]] = deep_merge(node[parts[-1]], v)
            else:
                node[parts[-1]] = v
    return out


def _empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip()) or v == {}


def _coerce(v: Any, node: dict) -> tuple[bool, Any]:
    typ = node.get("type")
    fmt = node.get("format")
    if typ in ("number", "integer"):
        if isinstance(v, bool):
            return False, None
        if isinstance(v, str):
            s = re.sub(r"[^\d,.\-]", "", v).replace(",", ".")
            try:
                v = float(s)
            except ValueError:
                return False, None
        if not isinstance(v, (int, float)) or math.isnan(v):
            return False, None
        if typ == "integer":
            return (True, int(v)) if float(v).is_integer() else (False, None)
        return True, int(v) if float(v).is_integer() else float(v)
    if typ == "boolean":
        if isinstance(v, bool):
            return True, v
        s = str(v).strip().lower()
        if s in ("true", "yes", "да", "ja", "1"):
            return True, True
        if s in ("false", "no", "нет", "nein", "0"):
            return True, False
        return False, None
    if typ == "array":
        return (True, v) if isinstance(v, list) else (True, [v])
    if fmt == "month":
        m = normalize_month(v)
        return (m is not None), m
    if fmt == "date":
        d = normalize_date(v)
        return (d is not None), d
    return True, str(v).strip() if not isinstance(v, (dict, list)) else v


# Month-name stems (en/ru/de) → number, for "сентябрь 2026" / "Sept 2026".
_MONTH_STEMS = [
    (1, ("jan", "янв", "jän")), (2, ("feb", "фев")), (3, ("mar", "mär", "мар")),
    (4, ("apr", "апр")), (5, ("may", "mai", "мая", "май", "мае")), (6, ("jun", "июн")),
    (7, ("jul", "июл")), (8, ("aug", "авг")), (9, ("sep", "сен")),
    (10, ("oct", "okt", "окт")), (11, ("nov", "ноя")), (12, ("dec", "dez", "дек")),
]


def _month_from_name(word: str) -> int | None:
    w = word.lower()
    for num, stems in _MONTH_STEMS:
        if any(w.startswith(st) for st in stems):
            return num
    return None


def normalize_month(v: Any) -> str | None:
    s = str(v).strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-\d{1,2})?", s) or None
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
    elif m := re.fullmatch(r"(\d{1,2})[./](\d{4})", s):
        mo, y = int(m.group(1)), int(m.group(2))
    elif m := (re.fullmatch(r"([^\W\d]+)\.?\s+(\d{4})", s)
               or re.fullmatch(r"(\d{4})\s+([^\W\d]+)\.?", s)):
        a, b = m.groups()
        word, y = (a, int(b)) if b.isdigit() else (b, int(a))
        mo = _month_from_name(word)
        if mo is None:
            return None
    else:
        return None
    return f"{y:04d}-{mo:02d}" if 1 <= mo <= 12 else None


_PERIOD_CUE = re.compile(
    r"\b(?:19|20)\d\d\b|\d{1,2}[./]\d{1,2}|month|месяц|monat|today|yesterday|сегодня|вчера|heute|gestern|year|год|jahr"
    r"|week|недел|woche|quarter|квартал|quartal",
    re.IGNORECASE,
)


def has_period_cue(text: str) -> bool:
    """Whether an utterance names a time at all.  A model that fills a
    month/date the user never said is guessing — treat it as missing."""
    if _PERIOD_CUE.search(text):
        return True
    return any(_month_from_name(w) for w in re.findall(r"[^\W\d]{3,}", text))


_REL = [
    # (pattern, unit, offset) — models get relative periods wrong, so code
    # resolves them from the utterance.
    (r"last month|previous month|прошл\w* месяц|предыдущ\w* месяц|letzt\w* monat|vormonat|vorig\w* monat", "month", -1),
    (r"this month|current month|эт\w* месяц|текущ\w* месяц|dies\w* monat|aktuell\w* monat", "month", 0),
    (r"yesterday|вчера|gestern", "day", -1),
    (r"today|сегодня|heute", "day", 0),
]


def relative_period(text: str, today: _dt.date | None = None) -> dict[str, str]:
    """``{"month": "YYYY-MM"}`` and/or ``{"date": "YYYY-MM-DD"}`` named
    relatively in ``text`` ("за прошлый месяц", "gestern")."""
    d = today or _dt.date.today()
    out: dict[str, str] = {}
    for pat, unit, off in _REL:
        if not re.search(pat, text, re.IGNORECASE):
            continue
        if unit == "month" and "month" not in out:
            y, m = divmod(d.year * 12 + d.month - 1 + off, 12)
            out["month"] = f"{y:04d}-{m + 1:02d}"
        elif unit == "day" and "date" not in out:
            day = d + _dt.timedelta(days=off)
            out["date"] = day.isoformat()
            out.setdefault("month", f"{day:%Y-%m}")
    return out


def grounded(value: Any, text: str) -> bool:
    """Loose check that a string value comes from the utterance: some word
    of it shares a 4-letter stem with a word the user said."""
    said = {w[:4] for w in re.findall(r"\w+", text.lower()) if len(w) >= 3}
    words = [w for w in re.findall(r"\w+", str(value).lower()) if len(w) >= 3]
    return not words or any(w[:4] in said for w in words)


def normalize_date(v: Any) -> str | None:
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return _dt.datetime.strptime(s[:19], fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _copy(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _copy(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_copy(v) for v in o]
    return o


# ── Periods ───────────────────────────────────────────────────────────────

PERIOD_KEYWORDS = ("this_year", "last_year", "this_month", "last_month",
                   "this_quarter", "last_12_months")


def period_bounds(keyword: str | None, today: _dt.date | None = None) -> tuple[str | None, str | None]:
    """Keyword → inclusive (from, to) prefixes comparable with YYYY-MM[-DD]."""
    if not keyword:
        return None, None
    d = today or _dt.date.today()
    if keyword == "this_year":
        return f"{d.year}", f"{d.year}"
    if keyword == "last_year":
        return f"{d.year - 1}", f"{d.year - 1}"
    if keyword == "this_month":
        return f"{d:%Y-%m}", f"{d:%Y-%m}"
    if keyword == "last_month":
        y, m = (d.year, d.month - 1) if d.month > 1 else (d.year - 1, 12)
        return f"{y:04d}-{m:02d}", f"{y:04d}-{m:02d}"
    if keyword == "this_quarter":
        q0 = (d.month - 1) // 3 * 3 + 1
        return f"{d.year}-{q0:02d}", f"{d.year}-{q0 + 2:02d}"
    if keyword == "last_12_months":
        y, m = d.year - 1, d.month + 1
        if m > 12:
            y, m = y + 1, 1
        return f"{y:04d}-{m:02d}", f"{d:%Y-%m}"
    return None, None


# ── Forecast ──────────────────────────────────────────────────────────────

def _month_index(p: str) -> int:
    y, m = p.split("-")[:2]
    return int(y) * 12 + int(m) - 1


def _month_label(i: int) -> str:
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def forecast_monthly(series: list[tuple[str, float]], horizon: int = 1) -> dict:
    """Linear trend + month-of-year seasonality over ``[(YYYY-MM, value)]``.

    Seasonal indices are mean detrended residuals per calendar month,
    shrunk by k/(k+1) for k observations so a single year doesn't
    reproduce itself exactly.  The range is ±1.645 σ of what's left
    (≈90 %), floored at 5 % of the point value.  Fewer than
    ``MIN_FORECAST_PERIODS`` distinct months → ``{"enough": False}``."""
    pts = sorted((_month_index(p), float(v)) for p, v in series if v is not None)
    n = len(pts)
    if n < MIN_FORECAST_PERIODS:
        return {"enough": False, "periods": n, "needed": MIN_FORECAST_PERIODS}
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs) or 1.0
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
    icpt = my - slope * mx
    resid = {x: y - (icpt + slope * x) for x, y in pts}
    by_moy: dict[int, list[float]] = {}
    for x, r in resid.items():
        by_moy.setdefault(x % 12, []).append(r)
    season = {k: (sum(v) / len(v)) * len(v) / (len(v) + 1) for k, v in by_moy.items()}
    left = [r - season.get(x % 12, 0.0) for x, r in resid.items()]
    sd = math.sqrt(sum(e * e for e in left) / max(n - 2, 1))
    out = []
    last = xs[-1]
    for h in range(1, max(1, horizon) + 1):
        x = last + h
        point = icpt + slope * x + season.get(x % 12, 0.0)
        half = max(1.645 * sd * math.sqrt(1 + 1 / n), abs(point) * 0.05)
        out.append({"period": _month_label(x), "value": point,
                    "low": point - half, "high": point + half})
    return {"enough": True, "periods": n, "trend_per_period": slope, "points": out}
