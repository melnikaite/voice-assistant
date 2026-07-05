"""
Tests for url_summarizer text chunking (#54).

Scope: ``_split_text`` — the pure map-reduce splitter.  The LLM map/reduce
phases need a live model, so we only unit-test the deterministic splitter
here (offline rule: no real HTTP / model calls).
"""
from __future__ import annotations

from app.url_summarizer import _split_text


def test_split_short_text_single_chunk():
    """Text under the chunk size stays in one chunk."""
    text = "First sentence. Second sentence. Third one!"
    chunks = _split_text(text, chunk_size=10_000)
    assert chunks == [text]


def test_split_empty_text_no_chunks():
    """Empty / whitespace text yields no chunks."""
    assert _split_text("", chunk_size=100) == []
    assert _split_text("   ", chunk_size=100) == []


def test_split_long_text_multiple_chunks_at_sentence_boundaries():
    """A long multi-sentence text splits into several chunks, each at a
    sentence boundary and (for short sentences) within the size budget."""
    sentence = "This is sentence number {}."
    text = " ".join(sentence.format(i) for i in range(200))
    chunk_size = 200
    chunks = _split_text(text, chunk_size)
    assert len(chunks) > 1
    # Every chunk is non-empty and ends at a sentence boundary.
    for c in chunks:
        assert c.strip()
        assert c.rstrip().endswith(".")
    # Reassembling preserves all sentences (no content dropped).
    assert "sentence number 0." in chunks[0]
    assert "sentence number 199." in chunks[-1]


def test_split_oversize_single_sentence_not_dropped():
    """A single sentence longer than chunk_size is still emitted (the
    splitter never silently drops content)."""
    big = "x" * 500 + "."
    chunks = _split_text(big, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == big
