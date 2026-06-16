"""
aeon/cache.py — Aeon's key/value cache.

The per-token forward threads a single cache object through every layer so each
layer accumulates its K/V left-to-right (this is what makes attention causal
across the sequence). We use a thin Aeon-owned subclass of the transformers
dynamic cache so the rest of the codebase imports from `aeon.cache` rather than
reaching into the library directly, and so future cache changes are localized
here.
"""
from transformers.cache_utils import DynamicCache


class AeonCache(DynamicCache):
    """Dynamic (growing) KV cache. Same contract as the library's DynamicCache:
    `.update(key, value, layer_idx, cache_kwargs)` appends and returns the full
    key/value for the layer; `.get_seq_length()` reports cached length."""
    pass


def make_cache():
    return AeonCache()
