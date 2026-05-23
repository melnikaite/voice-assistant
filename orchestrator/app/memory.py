"""
Semantic memory using fastembed.

Default model: ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``
(multilingual, 384-dim).  It is the smallest multilingual encoder shipped by
fastembed 0.4.x; override via ``EMBEDDING_MODEL`` if you swap.

On every utterance we compute an embedding of (transcript + response) and
store it in SQLite.  On the next query we embed the incoming transcript,
score all stored embeddings with cosine similarity, and inject the top-K
matches as extra context into the system prompt — so the LLM can reference
past conversations even after a tab reload or a completely new WS session.

Graceful degradation: if the model fails to load (e.g. first run before the
Docker image is rebuilt), EMBEDDING_ENABLED stays False and the rest of the
pipeline works normally without semantic memory.
"""
import asyncio
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

EMBEDDING_MODEL: str = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
MEMORY_TOP_K: int = int(os.environ.get("MEMORY_TOP_K", "3"))
MEMORY_MAX_AGE_DAYS: int = int(os.environ.get("MEMORY_MAX_AGE_DAYS", "30"))
MEMORY_SIMILARITY_THRESHOLD: float = float(
    os.environ.get("MEMORY_SIMILARITY_THRESHOLD", "0.72")
)

EMBEDDING_ENABLED: bool = False  # flipped to True once init_embedding_model() succeeds
_model = None


def init_embedding_model() -> None:
    """
    Load the fastembed model.  Safe to call multiple times (no-op after first).
    Sets EMBEDDING_ENABLED = True on success.
    """
    global _model, EMBEDDING_ENABLED
    if _model is not None:
        return
    try:
        from fastembed import TextEmbedding  # type: ignore[import]

        log.info("loading embedding model %r …", EMBEDDING_MODEL)
        _model = TextEmbedding(EMBEDDING_MODEL)
        # Warm-up: first inference triggers ONNX compilation.
        _embed_sync("warmup")
        EMBEDDING_ENABLED = True
        log.info("semantic memory ready (model=%s)", EMBEDDING_MODEL)
    except Exception as exc:
        log.warning(
            "semantic memory disabled — could not load embedding model: %s", exc
        )
        EMBEDDING_ENABLED = False


# ---------------------------------------------------------------------------
# Low-level sync helpers (run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _embed_sync(text: str, query: bool = False) -> "np.ndarray":
    """
    Embed a single string.  ``query`` is kept in the signature so call-sites
    (embed_query / embed_passage) read symmetrically with E5-style models;
    for MiniLM it is ignored because the model has no query/passage modes.

    Note: ``query`` must be a regular positional/keyword arg, NOT keyword-only.
    The callers route through ``asyncio.to_thread(_embed_sync, text, False)``,
    which passes args positionally — a ``*,`` here breaks them with
    ``_embed_sync() takes 1 positional argument but 2 were given``.
    """
    import numpy as np  # noqa: F401 — only imported when needed

    assert _model is not None, "call init_embedding_model() first"
    _ = query  # MiniLM is symmetric — no special query/passage prefix
    vec = list(_model.embed([text]))[0]
    return vec.astype("float32")


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------

async def embed_passage(text: str) -> "np.ndarray":
    """Embed a passage (utterance text).  Returns a float32 numpy array."""
    return await asyncio.to_thread(_embed_sync, text, False)


async def embed_query(text: str) -> "np.ndarray":
    """Embed a query (incoming transcript).  Returns a float32 numpy array."""
    return await asyncio.to_thread(_embed_sync, text, True)


# ---------------------------------------------------------------------------
# Encoding helpers for SQLite BLOB storage
# ---------------------------------------------------------------------------

def encode(vec: "np.ndarray") -> bytes:
    return vec.astype("float32").tobytes()


def decode(blob: bytes) -> "np.ndarray":
    import numpy as np

    return np.frombuffer(blob, dtype="float32")


# ---------------------------------------------------------------------------
# Similarity & ranking
# ---------------------------------------------------------------------------

def _cosine_sim(a: "np.ndarray", b: "np.ndarray") -> float:
    import numpy as np

    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-9 or b_norm < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def retrieve(
    query_vec: "np.ndarray",
    candidates: list[tuple[str, str, bytes]],  # (transcript, response_text, blob)
    *,
    top_k: int = MEMORY_TOP_K,
    threshold: float = MEMORY_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Score each candidate by cosine similarity, filter by threshold, return top-K.

    Returns a list of dicts with keys: transcript, response_text, similarity.
    """
    results: list[tuple[float, str, str]] = []
    for transcript, response_text, blob in candidates:
        if not blob:
            continue
        sim = _cosine_sim(query_vec, decode(blob))
        if sim >= threshold:
            results.append((sim, transcript, response_text))
    results.sort(reverse=True)
    return [
        {"transcript": t, "response_text": r, "similarity": s}
        for s, t, r in results[:top_k]
    ]


# ---------------------------------------------------------------------------
# Background task: compute & persist embedding after utterance saved
# ---------------------------------------------------------------------------

async def compute_and_save_embedding(
    utterance_id: int, transcript: str, response_text: str
) -> None:
    """
    Background task: embed ``transcript + response_text`` and store the
    resulting blob in the utterances table.  Errors are logged but swallowed
    so they never affect the user-facing flow.
    """
    if not EMBEDDING_ENABLED:
        return
    try:
        combined = f"{transcript} {response_text}"
        vec = await embed_passage(combined)
        blob = encode(vec)
        from .storage import update_utterance_embedding  # late import — no circular dep

        await update_utterance_embedding(utterance_id, blob)
        log.debug("stored embedding for utterance %d", utterance_id)
    except Exception as exc:
        log.warning(
            "failed to compute/store embedding for utterance %d: %s",
            utterance_id,
            exc,
        )
