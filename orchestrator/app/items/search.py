"""
Hybrid item search — BM25 (FTS5) + semantic (fastembed cosine) with RRF fusion.

Algorithm:
  1. Run FTS5 BM25 search → up to ``fts_k`` candidates.
  2. Encode the query with the same model used for item embeddings.
  3. Fetch all item embeddings for the profile (or scope to category).
  4. Score cosine similarity → top ``sem_k`` candidates.
  5. Merge via Reciprocal Rank Fusion (RRF) with k=60.
  6. Apply any remaining filters (date range, kind, source).
  7. Return top ``limit`` results, each with a ``score`` field.

Why RRF instead of weighted sum:
  The two signals live on different scales (BM25 is unbounded, cosine is
  [-1,1]).  RRF normalises by rank position rather than raw score, which
  avoids fiddly calibration and performs well in practice.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

# RRF smoothing constant (Cormack et al. 2009 recommend 60).
_RRF_K = 60


async def encode_query(text: str) -> "Any":
    """Encode a query string with the fastembed model.

    Returns a numpy float32 array of the same dimensionality as item
    embeddings.  Reuses the model instance from app.memory via the
    module-level ``_embed_sync`` helper — wrapped in asyncio.to_thread
    so the synchronous ONNX call doesn't block the event loop.

    Raises if the model has not been loaded (EMBEDDING_ENABLED is False
    or init_embedding_model() was never called).  Caller should catch
    and fall back to FTS-only.
    """
    from app import memory as mem

    return await asyncio.to_thread(mem._embed_sync, text, True)


async def hybrid_search(
    owner_profile_id: int,
    query: str,
    *,
    category_id: int | None = None,
    include_subtree: bool = False,
    kind: str | None = None,
    date_from: float | None = None,
    date_to: float | None = None,
    limit: int = 20,
    sort: str = "relevance",    # 'relevance' | 'date_desc' | 'date_asc'
    fts_k: int = 50,            # BM25 candidate pool
    sem_k: int = 50,            # semantic candidate pool
) -> list[dict]:
    """Hybrid BM25 + semantic search with RRF fusion.

    Returns items dicts sorted by ``sort`` with an extra ``score`` field
    (float, higher = better match).  Items without embeddings are still
    included via FTS5 only — they appear lower in the ranking because
    the semantic leg gives no signal.

    ``date_from`` / ``date_to`` are Unix timestamps (filter on created_at).
    ``kind`` filters to a specific item kind ('text', 'link', etc.).
    """
    from app.storage import items as storage_items

    # ── Step 1: BM25 candidates via FTS5 ────────────────────────────────
    fts_results: list[dict] = []
    try:
        fts_results = await storage_items.fts_search(
            owner_profile_id,
            query,
            category_id=category_id,
            include_subtree=include_subtree,
            kind=kind,
            limit=fts_k,
        )
    except Exception as exc:
        log.warning("hybrid_search: FTS5 failed for query %r: %s", query, exc)

    # ── Step 2 + 3: encode query and fetch embeddings concurrently ───────
    sem_results: list[dict] = []
    try:
        query_vec, embedding_pairs = await asyncio.gather(
            encode_query(query),
            storage_items.get_item_embeddings(owner_profile_id, limit=sem_k * 4),
        )
    except Exception as exc:
        log.warning(
            "hybrid_search: semantic leg disabled (encode or embedding fetch failed): %s", exc
        )
        # Fall back to FTS-only: still apply filters and return.
        embedding_pairs = []
        query_vec = None

    # ── Step 4: cosine ranking ───────────────────────────────────────────
    if query_vec is not None and embedding_pairs:
        import numpy as np
        from app import memory as mem

        scored: list[tuple[float, int]] = []
        for item_id, blob in embedding_pairs:
            if not blob:
                continue
            item_vec = mem.decode(blob)
            sim = mem._cosine_sim(query_vec, item_vec)
            scored.append((sim, item_id))

        scored.sort(reverse=True)
        top_ids = [item_id for _, item_id in scored[:sem_k]]

        # ── Step 5 (partial): fetch full rows for ids not already in FTS5 ──
        # Batch the missing ids into one ``WHERE id IN (...)`` query.
        # Without this, semantic-only matches would cost ``len(missing)``
        # round-trips through asyncio.to_thread — up to 50 small queries
        # for a typical top-50 candidate pool.
        fts_ids = {r["id"] for r in fts_results}
        missing_ids = [iid for iid in top_ids if iid not in fts_ids]

        fetched: dict[int, dict] = {r["id"]: r for r in fts_results}
        if missing_ids:
            fetched.update(await storage_items.batch_get_items(missing_ids))

        # Build sem_results in cosine-rank order.
        sem_results = [fetched[iid] for iid in top_ids if iid in fetched]

    # ── Step 6: RRF fusion ───────────────────────────────────────────────
    merged = _rrf_merge(fts_results, sem_results)

    if not merged:
        return []

    # ── Step 7: post-filters ─────────────────────────────────────────────
    if date_from is not None:
        merged = [r for r in merged if (r.get("created_at") or 0) >= date_from]
    if date_to is not None:
        merged = [r for r in merged if (r.get("created_at") or 0) <= date_to]
    if kind is not None:
        # kind may already be filtered at FTS level but semantic leg doesn't filter.
        merged = [r for r in merged if r.get("kind") == kind]

    # ── Step 8: sort ─────────────────────────────────────────────────────
    if sort == "date_desc":
        merged.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    elif sort == "date_asc":
        merged.sort(key=lambda r: r.get("created_at") or 0)
    # "relevance" keeps the RRF order already established.

    log.info(
        "hybrid_search: query=%r owner=%d → %d results (fts=%d sem=%d)",
        query, owner_profile_id, len(merged), len(fts_results), len(sem_results),
    )

    return merged[:limit]


def _rrf_merge(
    fts_results: list[dict],
    sem_results: list[dict],
    *,
    k: int = _RRF_K,
) -> list[dict]:
    """Merge two ranked lists via Reciprocal Rank Fusion.

    Each item id accumulates ``1 / (k + rank)`` from each list it appears
    in.  Items that appear in both lists score higher than items in only one.
    Returns items sorted by descending RRF score with ``score`` field set.
    """
    scores: dict[int, float] = {}
    item_by_id: dict[int, dict] = {}

    for rank, item in enumerate(fts_results, 1):
        iid = item["id"]
        scores[iid] = scores.get(iid, 0.0) + 1.0 / (k + rank)
        item_by_id.setdefault(iid, item)

    for rank, item in enumerate(sem_results, 1):
        iid = item["id"]
        scores[iid] = scores.get(iid, 0.0) + 1.0 / (k + rank)
        item_by_id.setdefault(iid, item)

    merged = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    result = []
    for iid in merged:
        row = dict(item_by_id[iid])
        row["score"] = scores[iid]
        result.append(row)
    return result
