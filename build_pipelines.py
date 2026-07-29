"""Build the RAG vector stores for each configuration and smoke-test them.

All the heavy lifting (chunking, embedding, retrieval, generation) lives in
``rag_core``; this script just wires the config presets to a test question.
"""

import argparse

from rag_core import (
    CONFIG_PRESETS,
    build_collection,
    load_articles,
    run_rag,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build vector stores for each RAG configuration."
    )
    parser.add_argument("--configs", nargs="*", default=list(CONFIG_PRESETS),
                        choices=list(CONFIG_PRESETS),
                        help="which config presets to build")
    parser.add_argument("--question", type=str,
                        default="How do I turn off auto-renew for my domain?",
                        help="test question to run through each pipeline")
    parser.add_argument("--articles", type=str, default="articles.json",
                        help="path to the scraped articles JSON")
    parser.add_argument("--no-persist", action="store_true",
                        help="use an in-memory Chroma client instead of chroma_db/")
    parser.add_argument("--skip-test", action="store_true",
                        help="only build the collections, don't call the LLM")
    args = parser.parse_args()

    articles = load_articles(args.articles)
    persist = not args.no_persist

    print("Building vector stores...")
    collections = {}
    for key in args.configs:
        preset = CONFIG_PRESETS[key]
        collections[key] = build_collection(
            f"{key}_rag", articles,
            chunk_size=preset["chunk_size"],
            chunk_overlap=preset["chunk_overlap"],
            persist=persist,
        )

    if args.skip_test:
        return

    print(f"\nTest question: '{args.question}'\n")
    for key in args.configs:
        preset = CONFIG_PRESETS[key]
        result = run_rag(
            preset["label"], collections[key], args.question,
            k=preset["k"], strategy=preset["retrieval"],
        )
        print(f"--- {result['config']} ({result['latency']}s) ---")
        print(result["answer"])
        print(f"Chunks retrieved: {len(result['contexts'])}\n")


if __name__ == "__main__":
    main()
