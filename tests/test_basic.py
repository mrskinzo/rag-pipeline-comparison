"""URL filtering for the scraper.

These import the real is_article_url from scrape_articles. An earlier version
of this file pasted a copy of the regex into the test body, so it verified the
copy rather than the shipped function and would have passed even if the real
one were deleted. The chunk-config assertions it also carried (chunk_size > 0,
overlap < size) restated their own literals and are gone.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrape_articles import is_article_url  # noqa: E402


def test_accepts_article_urls():
    assert is_article_url("https://www.godaddy.com/help/turn-off-auto-renew-4562")
    assert is_article_url("/help/some-title-999")


def test_rejects_non_article_urls():
    assert not is_article_url("https://www.godaddy.com/help/domains-4562/")
    assert not is_article_url("https://www.godaddy.com/help/")
    assert not is_article_url("https://www.godaddy.com")
    assert not is_article_url("https://www.godaddy.com/help/no-number-here")
