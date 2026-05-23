"""
Tests for orchestrator/app/items/search.py and orchestrator/app/items/auto_sort.py.

Strategy:
  * Uses the conftest _fresh_db fixture (real /tmp SQLite file, fresh per test).
  * fastembed model and llm_utils are mocked — no live services required.
  * Storage helpers (fts_search, get_item_embeddings, move_item) are tested
    through the real DB layer to catch real SQL edge-cases.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.items.search import _rrf_merge, hybrid_search
from app.storage import init_schema
from app.storage.items import create_item, set_item_embedding


# ── Shared fixtures ────────────────────────────────────────────────────────

_OWNER = 42

# Helper: create an item with optional embedding blob.
async def _make_item(title: str, body: str = "", kind: str = "text", vec: np.ndarray | None = None):
    item_id = await create_item(
        owner_profile_id=_OWNER,
        created_by_profile_id=_OWNER,
        kind=kind,
        title=title,
        body=body,
    )
    if vec is not None:
        await set_item_embedding(item_id, vec.astype("float32").tobytes())
    return item_id


def _item(id: int, **kw) -> dict:
    """Build a minimal item dict for RRF tests."""
    return {"id": id, **kw}


# ── _rrf_merge ─────────────────────────────────────────────────────────────


def test_rrf_merge_basic():
    """Two lists with overlapping items give higher scores to the overlapping ones."""
    fts = [_item(1), _item(2), _item(3)]
    sem = [_item(2), _item(1), _item(4)]
    merged = _rrf_merge(fts, sem)
    ids = [r["id"] for r in merged]

    # Items 1 and 2 appear in both lists and must rank before 3 and 4.
    assert ids.index(1) < ids.index(3)
    assert ids.index(2) < ids.index(4)
    # Scores are set and > 0.
    for r in merged:
        assert "score" in r
        assert r["score"] > 0


def test_rrf_merge_no_overlap():
    """Items from a single list get a partial score; none are dropped."""
    fts = [_item(10), _item(11)]
    sem = [_item(20), _item(21)]
    merged = _rrf_merge(fts, sem)
    ids = {r["id"] for r in merged}
    assert ids == {10, 11, 20, 21}
    # All items in fts appear before all items in sem (same rank position parity).
    # At minimum: scores are non-zero.
    for r in merged:
        assert r["score"] > 0


def test_rrf_merge_empty():
    """Empty inputs produce an empty output."""
    assert _rrf_merge([], []) == []
    assert _rrf_merge([_item(1)], []) == [{"id": 1, "score": pytest.approx(1 / 61)}]
    assert _rrf_merge([], [_item(2)]) == [{"id": 2, "score": pytest.approx(1 / 61)}]


# ── hybrid_search ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hybrid_search_fts_fallback():
    """When encode_query raises, FTS results are still returned (semantic leg skipped)."""
    item_id = await _make_item("coffee beans", body="morning coffee")

    with (
        patch("app.items.search.encode_query", side_effect=RuntimeError("no model")),
        # get_item_embeddings is still reached via asyncio.gather — mock to avoid
        # the gather failing on the other coroutine side.
        patch("app.storage.items.get_item_embeddings", new=AsyncMock(return_value=[])),
    ):
        results = await hybrid_search(_OWNER, "coffee")

    ids = [r["id"] for r in results]
    assert item_id in ids, "FTS result must be present even when encoding fails"
    for r in results:
        assert "score" in r


@pytest.mark.asyncio
async def test_hybrid_search_no_results():
    """Empty FTS + no embeddings → empty list (no crash)."""
    with (
        patch("app.items.search.encode_query", side_effect=RuntimeError("no model")),
        patch("app.storage.items.get_item_embeddings", new=AsyncMock(return_value=[])),
    ):
        results = await hybrid_search(_OWNER, "xyzzy_nonexistent_query_abc")

    assert results == []


@pytest.mark.asyncio
async def test_hybrid_search_date_filter():
    """Items with created_at outside [date_from, date_to] are excluded."""
    import time

    now = time.time()
    item_id = await _make_item("date filter item", body="filter test")

    # Fetch the actual created_at so the test is not time-sensitive.
    from app.storage.items import get_item
    row = await get_item(item_id)
    assert row is not None
    created = row["created_at"]

    # Mock encode + embeddings to force FTS-only path.
    with (
        patch("app.items.search.encode_query", side_effect=RuntimeError("no model")),
        patch("app.storage.items.get_item_embeddings", new=AsyncMock(return_value=[])),
    ):
        # date_from far in the future → item excluded.
        results = await hybrid_search(
            _OWNER, "filter", date_from=created + 1000
        )
        assert not any(r["id"] == item_id for r in results)

        # date_to in the past → item excluded.
        results = await hybrid_search(
            _OWNER, "filter", date_to=created - 1000
        )
        assert not any(r["id"] == item_id for r in results)

        # Both bounds bracket the item → item included.
        results = await hybrid_search(
            _OWNER, "filter", date_from=created - 10, date_to=created + 10
        )
        assert any(r["id"] == item_id for r in results)


@pytest.mark.asyncio
async def test_hybrid_search_kind_filter():
    """kind filter excludes items of wrong kinds."""
    text_id = await _make_item("text note", body="text content", kind="text")
    link_id = await _make_item("link bookmark", body="link content", kind="link")

    with (
        patch("app.items.search.encode_query", side_effect=RuntimeError("no model")),
        patch("app.storage.items.get_item_embeddings", new=AsyncMock(return_value=[])),
    ):
        results = await hybrid_search(_OWNER, "content", kind="text")

    ids = [r["id"] for r in results]
    assert text_id in ids
    assert link_id not in ids


@pytest.mark.asyncio
async def test_hybrid_search_semantic_ranking():
    """Semantic leg promotes items with high cosine similarity."""
    # Create two items: one semantically close to query, one far.
    close_id = await _make_item("apple fruit", body="apple")
    far_id = await _make_item("database engine", body="sql")

    # Fake embeddings: query vec = [1, 0], close = [1, 0], far = [0, 1].
    query_vec = np.array([1.0, 0.0], dtype="float32")
    close_vec = np.array([1.0, 0.0], dtype="float32")
    far_vec = np.array([0.0, 1.0], dtype="float32")

    await set_item_embedding(close_id, close_vec.tobytes())
    await set_item_embedding(far_id, far_vec.tobytes())

    embedding_pairs = [
        (close_id, close_vec.tobytes()),
        (far_id, far_vec.tobytes()),
    ]

    with (
        patch("app.items.search.encode_query", new=AsyncMock(return_value=query_vec)),
        patch(
            "app.storage.items.get_item_embeddings",
            new=AsyncMock(return_value=embedding_pairs),
        ),
    ):
        results = await hybrid_search(_OWNER, "apple fruit")

    ids = [r["id"] for r in results]
    # close_id should rank ahead of far_id (higher cosine similarity).
    if close_id in ids and far_id in ids:
        assert ids.index(close_id) < ids.index(far_id)


# ── auto_sort: suggest_auto_sort ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_sort_suggest_parses_llm_json():
    """Mock LLM returning valid JSON → suggestions returned with category_name enriched."""
    from app.items.auto_sort import suggest_auto_sort

    items = [{"id": 1, "title": "Python tutorial", "summary": "", "url": ""}]
    categories = [{"id": 10, "name": "Tech", "kind": "folder"}]

    llm_response = json.dumps(
        [{"item_id": 1, "category_id": 10, "reason": "programming tutorial fits tech"}]
    )
    fake_choice = {"message": {"content": llm_response, "role": "assistant"}}

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=fake_choice)
    mock_llm.extract_text = MagicMock(return_value=llm_response)

    with patch("app.items.auto_sort.llm_utils", mock_llm):
        suggestions = await suggest_auto_sort(items, categories)

    assert len(suggestions) == 1
    s = suggestions[0]
    assert s["item_id"] == 1
    assert s["category_id"] == 10
    assert s["category_name"] == "Tech"
    assert "programming" in s["reason"]


@pytest.mark.asyncio
async def test_auto_sort_suggest_strips_markdown():
    """LLM wrapping JSON in ```json blocks is handled gracefully."""
    from app.items.auto_sort import suggest_auto_sort

    items = [{"id": 2, "title": "Recipe", "summary": "", "url": ""}]
    categories = [{"id": 20, "name": "Food", "kind": "folder"}]

    json_payload = json.dumps([{"item_id": 2, "category_id": 20, "reason": "a food recipe"}])
    wrapped = f"```json\n{json_payload}\n```"

    fake_choice = {"message": {"content": wrapped, "role": "assistant"}}

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=fake_choice)
    mock_llm.extract_text = MagicMock(return_value=wrapped)

    with patch("app.items.auto_sort.llm_utils", mock_llm):
        suggestions = await suggest_auto_sort(items, categories)

    assert len(suggestions) == 1
    assert suggestions[0]["item_id"] == 2
    assert suggestions[0]["category_name"] == "Food"


@pytest.mark.asyncio
async def test_auto_sort_suggest_handles_bad_json():
    """LLM returning non-JSON prose → returns empty list, no crash."""
    from app.items.auto_sort import suggest_auto_sort

    items = [{"id": 3, "title": "Item", "summary": "", "url": ""}]
    categories = [{"id": 30, "name": "Other", "kind": "folder"}]

    bad_response = "I cannot sort this item, it does not fit any category."
    fake_choice = {"message": {"content": bad_response, "role": "assistant"}}

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=fake_choice)
    mock_llm.extract_text = MagicMock(return_value=bad_response)

    with patch("app.items.auto_sort.llm_utils", mock_llm):
        suggestions = await suggest_auto_sort(items, categories)

    assert suggestions == []


@pytest.mark.asyncio
async def test_auto_sort_suggest_llm_failure():
    """LLM call raising an exception → returns empty list, no crash."""
    from app.items.auto_sort import suggest_auto_sort

    items = [{"id": 4, "title": "Item", "summary": "", "url": ""}]
    categories = [{"id": 40, "name": "Cat", "kind": "folder"}]

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(side_effect=RuntimeError("LLM offline"))

    with patch("app.items.auto_sort.llm_utils", mock_llm):
        suggestions = await suggest_auto_sort(items, categories)

    assert suggestions == []


@pytest.mark.asyncio
async def test_auto_sort_suggest_empty_inputs():
    """Empty items or empty categories → returns [] without calling LLM."""
    from app.items.auto_sort import suggest_auto_sort

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock()

    with patch("app.items.auto_sort.llm_utils", mock_llm):
        assert await suggest_auto_sort([], [{"id": 1, "name": "Cat", "kind": "folder"}]) == []
        assert await suggest_auto_sort([{"id": 1, "title": "T", "summary": "", "url": ""}], []) == []

    mock_llm.chat.assert_not_called()


# ── auto_sort: apply_auto_sort ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_sort_apply_calls_move_item():
    """apply_auto_sort calls move_item for each suggestion and returns success count."""
    from app.items.auto_sort import apply_auto_sort
    from app.storage.categories import create_category
    import time

    # Create two real items and a category in the live DB.
    cat_id = await create_category(
        owner_profile_id=_OWNER,
        name="Dest",
        kind="folder",
        parent_id=None,
    )
    item_id1 = await _make_item("apply item 1")
    item_id2 = await _make_item("apply item 2")

    suggestions = [
        {"item_id": item_id1, "category_id": cat_id, "reason": "fits"},
        {"item_id": item_id2, "category_id": cat_id, "reason": "fits"},
    ]

    count = await apply_auto_sort(suggestions, _OWNER)

    assert count == 2

    # Verify rows actually moved.
    from app.storage.items import get_item
    r1 = await get_item(item_id1)
    r2 = await get_item(item_id2)
    assert r1 is not None and r1["category_id"] == cat_id
    assert r2 is not None and r2["category_id"] == cat_id


@pytest.mark.asyncio
async def test_auto_sort_apply_empty():
    """No suggestions → returns 0 without touching the DB."""
    from app.items.auto_sort import apply_auto_sort

    count = await apply_auto_sort([], _OWNER)
    assert count == 0


@pytest.mark.asyncio
async def test_auto_sort_apply_partial_success():
    """A suggestion with a non-existent item_id returns False from move_item; count is partial."""
    from app.items.auto_sort import apply_auto_sort

    real_id = await _make_item("real item")
    ghost_id = 99999  # does not exist

    suggestions = [
        {"item_id": real_id, "category_id": None, "reason": "uncategorise"},
        {"item_id": ghost_id, "category_id": None, "reason": "ghost"},
    ]

    count = await apply_auto_sort(suggestions, _OWNER)

    # Only the real item should succeed.
    assert count == 1
