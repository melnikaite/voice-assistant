"""
Personal item store — higher-level operations on top of storage/items.py.

Modules:
  ingest     — create items from different source kinds (text, link,
               video, screenshot).  Each ingester calls storage.items.*
               and then fires async background tasks (embedding, summary).
  search     — hybrid BM25 + semantic search with Reciprocal Rank Fusion.
  auto_sort  — LLM-powered "suggest category moves for these items".
"""
