import argparse
import json
import statistics
from typing import List, Dict


def summarize_articles(articles: List[Dict[str, str]]) -> None:
    print(f"Total articles: {len(articles)}")
    print("\n--- Sample titles ---")
    for a in articles[:5]:
        print(f"  {a['title']}")

    print("\n--- Content length stats ---")
    lengths = [len(a['content']) for a in articles]
    if lengths:
        print(f"  Min:  {min(lengths)} chars")
        print(f"  Max:  {max(lengths)} chars")
        print(f"  Avg:  {int(statistics.mean(lengths))} chars")
    else:
        print("  (no articles)")

    if articles:
        print("\n--- First article preview ---")
        print(articles[0]["content"][:500])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a JSON file of articles.")
    parser.add_argument("--file", type=str, default="articles.json",
                        help="path to the articles JSON file")
    args = parser.parse_args()

    with open(args.file) as f:
        articles = json.load(f)
    summarize_articles(articles)


if __name__ == "__main__":
    main()
