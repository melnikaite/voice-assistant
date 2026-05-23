"""Tool registry — auto-discovers every tool module in this package.

Each tool module uses the ``@tool(...)`` decorator at import time, which
mutates ``TOOL_REGISTRY`` as a side effect.  So the act of importing
the module IS the registration — there's no central list to keep in
sync.  Adding a new tool: drop ``tools/my_tool.py``, decorate the
handler with ``@tool(...)``, restart the orchestrator.  No edits here.

Module names starting with an underscore are skipped (private helpers
or test fixtures); ``base`` is imported explicitly above for the
re-exports.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil

from .base import TOOL_REGISTRY, ToolResult, dispatch, schemas, tool

log = logging.getLogger(__name__)


for _mod_info in pkgutil.iter_modules(__path__):
    if _mod_info.name.startswith("_") or _mod_info.name == "base":
        continue
    try:
        importlib.import_module(f"{__name__}.{_mod_info.name}")
    except Exception:  # pragma: no cover — module import failure is loud
        log.exception(
            "tools: failed to import %r — tool will be unavailable",
            _mod_info.name,
        )


__all__ = ["TOOL_REGISTRY", "ToolResult", "dispatch", "schemas", "tool"]
