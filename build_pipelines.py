"""Utility for quickly constructing and testing individual RAG configs.

This script is now mostly a thin wrapper around the functions defined in
:evaluate.py so that there isn't duplicate logic.
"""

import argparse
import json
import time
from typing import Any

from evaluate import build_collection, generate, retrieve, multi_query_retrieve


def run_rag(config_name: str, collection: Any, question: str, k: int, multi_query: bool = False) -> dict:
    start = time.time()
    chunks = multi_query_retrieve(collection, question, k) if multi_query else retrieve(collection, question, k)
    answer = generate(question, chunks)
    return {
        "config": config_name,
        "question": question,
        "answer": answer,
        "contexts": chunks,
        "latency": round(time.time() - start, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Build and test RAG configs")
    parser.add_argument("--question", type=str,
                        default="How do I turn off auto-renew for my domain?",
                        help="single test question to run")
    args = parser.parse_args()

    with open("articles.json") as f:
        articles = json.load(f)

    print("Building vector stores...")
    config1 = build_collection("naive_rag", articles, 500, 0)
    config2 = build_collection("optimized_rag", articles, 1000, 200)
    config3 = build_collection("multiquery_rag", articles, 1000, 200)

    r1 = run_rag("Config 1 - Naive", config1, args.question, k=3)
    r2 = run_rag("Config 2 - Optimized", config2, args.question, k=5)
    r3 = run_rag("Config 3 - MultiQuery", config3, args.question, k=5, multi_query=True)

    for r in [r1, r2, r3]:
        print(f"--- {r['config']} ({r['latency']}s) ---")
        print(r['answer'])
        print(f"Chunks retrieved: {len(r['contexts'])}\n")


if __name__ == "__main__":
    main()
