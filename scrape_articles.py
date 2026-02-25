import requests
from bs4 import BeautifulSoup
import json
import time
import re

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
BASE_URL = "https://www.godaddy.com"

# These category pages contain links to real articles
CATEGORY_URLS = [
    "https://www.godaddy.com/help/domains-4562",
    "https://www.godaddy.com/help/ssl-certificates-268",
    "https://www.godaddy.com/help/wordpress-hosting-1543",
    "https://www.godaddy.com/help/email-4568",
]

def is_article_url(href):
    # Real articles end with a number, e.g. /help/some-title-1234
    return bool(re.search(r'/help/[a-z0-9-]+-\d+$', href))

def get_article_links(limit=40):
    links = set()
    for cat_url in CATEGORY_URLS:
        print(f"Scanning category: {cat_url}")
        try:
            response = requests.get(cat_url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if not href.startswith("http"):
                    href = BASE_URL + href
                if is_article_url(href):
                    links.add(href)
            time.sleep(1)
        except Exception as e:
            print(f"  Error: {e}")
    return list(links)[:limit]

def scrape_article(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("h1")
        title = title.get_text(strip=True) if title else "Unknown"

        content_area = (soup.find("article") or
                        soup.find("main") or
                        soup.find("div", {"id": "content"}) or
                        soup.find("div", class_=re.compile(r'content|article|help', re.I)))

        if content_area:
            for tag in content_area.find_all(["nav", "footer", "header", "script", "style"]):
                tag.decompose()
            text = content_area.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        text = "\n".join(lines)

        return {"title": title, "url": url, "content": text}
    except Exception as e:
        print(f"  Error: {e}")
        return None

def main():
    print("Finding article links from category pages...")
    links = get_article_links(limit=40)
    print(f"Found {len(links)} article links\n")

    articles = []
    for i, url in enumerate(links):
        print(f"[{i+1}/{len(links)}] {url}")
        article = scrape_article(url)
        if article and len(article["content"]) > 300:
            articles.append(article)
            print(f"  OK: '{article['title']}' — {len(article['content'])} chars")
        else:
            print(f"  Skipped (too short or failed)")
        time.sleep(1)

    with open("articles.json", "w") as f:
        json.dump(articles, f, indent=2)

    print(f"\nDone. Saved {len(articles)} articles to articles.json")

if __name__ == "__main__":
    main()