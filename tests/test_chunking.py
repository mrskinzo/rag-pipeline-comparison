import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from rag_core import chunk_articles


def test_chunk_articles_splits_text():
    # build a fake article with 1000 characters and chunk size 200
    art = {"url": "http://x", "title": "T", "content": "a" * 1000}
    chunks = chunk_articles([art], chunk_size=200, chunk_overlap=50)
    # with a 50 overlap, we expect more chunks than 5
    assert len(chunks) > 5
    # each chunk text should not be empty
    for c in chunks:
        assert c["text"]
        assert c["source"] == art["url"]


def test_chunk_ids_unique_and_stable():
    art = {"url": "http://x", "title": "T", "content": "b" * 600}
    chunks1 = chunk_articles([art], chunk_size=200, chunk_overlap=0)
    chunks2 = chunk_articles([art], chunk_size=200, chunk_overlap=0)
    ids1 = [c["id"] for c in chunks1]
    ids2 = [c["id"] for c in chunks2]
    assert ids1 == ids2
    assert len(set(ids1)) == len(ids1)
