"""Small scraper for GoDaddy help center articles.

The original script was written as a standalone; this refactor adds
command-line arguments, proper logging, and type annotations so that the
core helper functions can be imported by tests or other modules.
"""

import argparse
import json
import logging
import re
import time
from typing import List, Optional, Set

import requests
from bs4 import BeautifulSoup

# constants ------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36"
    )
}
BASE_URL = "https://www.godaddy.com"

DEFAULT_CATEGORY_URLS = [
    "https://www.godaddy.com/help/domains-4562",
    "https://www.godaddy.com/help/ssl-certificates-268",
    "https://www.godaddy.com/help/wordpress-hosting-1543",
    "https://www.godaddy.com/help/email-4568",
]


# helpers --------------------------------------------------------------------

def is_article_url(href: str) -> bool:
    """Return ``True`` if ``href`` looks like a real help article URL."""
    # Articles end with a dash and some digits, e.g. /help/title-1234
    return bool(re.search(r"/help/[a-z0-9-]+-\d+$", href))


def get_article_links(
    limit: int = 40, category_urls: Optional[List[str]] = None
) -> List[str]:
    """Scrape category pages and return up to ``limit`` distinct URLs."""
    urls = category_urls or DEFAULT_CATEGORY_URLS
    links: Set[str] = set()

    for cat_url in urls:
        logging.info("Scanning category: %s", cat_url)
        try:
            resp = requests.get(cat_url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if not href.startswith("http"):
                    href = BASE_URL + href
                if is_article_url(href):
                    links.add(href)
            time.sleep(1)
        except Exception as exc:
            logging.warning("failed to scan %s: %s", cat_url, exc)
    return list(links)[:limit]


def scrape_article(url: str) -> Optional[dict]:
    """Fetch a single article and return a normalized dictionary.

    ``None`` is returned if the article cannot be retrieved or is too
    short.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Unknown"

        content_area = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", {"id": "content"})
            or soup.find("div", class_=re.compile(r"content|article|help", re.I))
        )

        if content_area:
            for tag in content_area.find_all(
                ["nav", "footer", "header", "script", "style"]
            ):
                tag.decompose()
            text = content_area.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # collapse blank lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        return {"title": title, "url": url, "content": text}
    except Exception as exc:
        logging.warning("skipping %s due to %s", url, exc)
        return None


# command-line entrypoint ---------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl GoDaddy help categories and extract articles."
    )
    parser.add_argument("--limit", type=int, default=40,
                        help="max number of articles to collect")
    parser.add_argument("--output", type=str, default="articles.json",
                        help="output file for the article JSON")
    parser.add_argument("--categories", type=str, nargs="*",
                        help="override default category URLs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info("Starting scrape with limit=%d", args.limit)
    links = get_article_links(limit=args.limit, category_urls=args.categories)
    logging.info("found %d article links", len(links))

    articles = []
    for idx, url in enumerate(links, start=1):
        logging.info("[%d/%d] %s", idx, len(links), url)
        art = scrape_article(url)
        if art and len(art["content"]) > 300:
            articles.append(art)
            logging.info("  kept '%s' (%d chars)", art["title"], len(art["content"]))
        else:
            logging.info("  skipped (too short or failed)")
        time.sleep(1)

    with open(args.output, "w") as f:
        json.dump(articles, f, indent=2)

    logging.info("done – saved %d articles to %s", len(articles), args.output)


if __name__ == "__main__":
    main()
