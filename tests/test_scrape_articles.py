import json
import sys, os
# allow importing project modules when tests executed from workspace root
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from scrape_articles import is_article_url, get_article_links, scrape_article


def test_is_article_url_valid_and_invalid():
    assert is_article_url("https://www.godaddy.com/help/foo-bar-1234")
    assert not is_article_url("https://www.godaddy.com/help/foo-bar")
    assert not is_article_url("/help/notvalid")


def test_get_article_links_empty(monkeypatch):
    # simulate category page with no valid links
    class DummyResponse:
        status_code = 200
        text = "<html><body><a href='/not-an-article'>link</a></body></html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr("requests.get", lambda url, headers, timeout: DummyResponse())
    links = get_article_links(limit=5, category_urls=["http://example.com"])
    assert links == []


def test_scrape_article_short(monkeypatch):
    # stub a very simple html page
    class DummyResponse:
        status_code = 200
        text = "<html><h1>Title</h1><article><p>Hi</p></article></html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr("requests.get", lambda url, headers, timeout: DummyResponse())
    result = scrape_article("http://example.com/article-1")
    assert result is not None
    assert result["title"] == "Title"
    assert "Hi" in result["content"]
