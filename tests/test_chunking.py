import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from evaluate import chunk_articles


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
