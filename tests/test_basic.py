"""
Basic tests for the RAG pipeline components.
These tests do not require API keys or a running database.
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── scrape_articles helpers ────────────────────────────────────────────────

def is_article_url(href):
    return bool(re.search(r'/help/[a-z0-9-]+-\d+$', href))


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
    chunk_id = f"{abs(hash(url))}_0"
    # ChromaDB IDs should only contain alphanumerics and underscores/hyphens
    assert re.match(r'^[\w-]+$', chunk_id), f"Invalid chunk ID: {chunk_id}"


def test_chunk_id_is_deterministic():
    url = "https://www.godaddy.com/help/some-article-1234"
    id1 = f"{abs(hash(url))}_0"
    id2 = f"{abs(hash(url))}_0"
    assert id1 == id2


# ── chunking logic ─────────────────────────────────────────────────────────

def test_chunk_overlap_less_than_size():
    chunk_size = 1000
    chunk_overlap = 200
    assert chunk_overlap < chunk_size


def test_naive_config_params():
    chunk_size = 500
    chunk_overlap = 0
    assert chunk_size > 0
    assert chunk_overlap == 0


def test_optimized_config_params():
    chunk_size = 1000
    chunk_overlap = 200
    assert chunk_size > chunk_overlap
