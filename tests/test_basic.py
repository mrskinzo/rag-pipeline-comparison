"""
Basic tests for the RAG pipeline components.
These tests do not require API keys or a running database.
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrape_articles import is_article_url
from rag_core import CONFIG_PRESETS, stable_id


# ── scrape_articles helpers ────────────────────────────────────────────────

def test_is_article_url_valid():
    assert is_article_url("https://www.godaddy.com/help/turn-off-auto-renew-4562")
    assert is_article_url("/help/some-title-999")


def test_is_article_url_invalid():
    assert not is_article_url("https://www.godaddy.com/help/domains-4562/")
    assert not is_article_url("https://www.godaddy.com/help/")
    assert not is_article_url("https://www.godaddy.com")
    assert not is_article_url("https://www.godaddy.com/help/no-number-here")


# ── chunk ID stability ─────────────────────────────────────────────────────

def test_chunk_id_no_special_chars():
    """Chunk IDs must not contain characters that break ChromaDB."""
    url = "https://www.godaddy.com/help/some-article-1234"
    chunk_id = f"{stable_id(url)}_0"
    assert re.match(r'^[\w-]+$', chunk_id), f"Invalid chunk ID: {chunk_id}"


def test_chunk_id_is_deterministic():
    url = "https://www.godaddy.com/help/some-article-1234"
    assert f"{stable_id(url)}_0" == f"{stable_id(url)}_0"


def test_stable_id_known_value():
    """sha1-based IDs are stable across processes, unlike hash()."""
    import hashlib
    url = "https://www.godaddy.com/help/some-article-1234"
    expected = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    assert stable_id(url) == expected


# ── config presets ─────────────────────────────────────────────────────────

def test_preset_keys():
    assert list(CONFIG_PRESETS) == ["naive", "optimized", "mmr", "multiquery"]


def test_preset_values_match_readme():
    assert CONFIG_PRESETS["naive"]["chunk_size"] == 256
    assert CONFIG_PRESETS["naive"]["chunk_overlap"] == 32
    assert CONFIG_PRESETS["naive"]["k"] == 3
    assert CONFIG_PRESETS["optimized"]["chunk_size"] == 512
    assert CONFIG_PRESETS["mmr"]["retrieval"] == "mmr"
    assert CONFIG_PRESETS["multiquery"]["retrieval"] == "multiquery"


def test_preset_overlap_less_than_size():
    for preset in CONFIG_PRESETS.values():
        assert 0 <= preset["chunk_overlap"] < preset["chunk_size"]
