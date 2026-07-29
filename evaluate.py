"""Evaluate RAG configurations across a fixed question set.

Metrics: faithfulness, answer relevancy, context precision,
answer correctness (all LLM-as-judge), plus end-to-end latency.
Results are written to a CSV (default: evaluation_results.csv).
"""

import argparse
import time

import pandas as pd

from rag_core import (
    CONFIG_PRESETS,
    build_collection,
    generate,
    load_articles,
    run_retrieval,
    score_answer_correctness,
    score_answer_relevancy,
    score_context_precision,
    score_faithfulness,
)

TEST_QUESTIONS = [
    "How do I turn off auto-renew for my domain?",
    "How do I cancel a domain transfer?",
    "What is domain protection?",
    "How do I close my GoDaddy account?",
    "How do I enable 2-step verification on my account?",
    "How do I request a refund from GoDaddy?",
    "What is a one-time password and when is it used?",
    "How do I revoke an SSL certificate?",
    "How do I update my GoDaddy account profile?",
    "How do I remove GoDaddy Payments from my account?",
]

METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_precision",
                  "answer_correctness", "latency"]


def load_questions(path: str) -> list:
    """Read one question per non-empty line."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG configurations.")
    parser.add_argument("--configs", nargs="*", default=list(CONFIG_PRESETS),
                        choices=list(CONFIG_PRESETS),
                        help="which config presets to evaluate")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N questions")
    parser.add_argument("--questions-file", type=str, default=None,
                        help="file with one question per line (default: built-in set)")
    parser.add_argument("--articles", type=str, default="articles.json",
                        help="path to the scraped articles JSON")
    parser.add_argument("--output", type=str, default="evaluation_results.csv",
                        help="output CSV path")
    parser.add_argument("--no-persist", action="store_true",
                        help="use an in-memory Chroma client instead of chroma_db/")
    args = parser.parse_args()

    questions = load_questions(args.questions_file) if args.questions_file else TEST_QUESTIONS
    if args.limit:
        questions = questions[:args.limit]
    persist = not args.no_persist

    articles = load_articles(args.articles)

    print("=" * 55)
    print("RAG PIPELINE EVALUATION")
    print("=" * 55)

    print("\nBuilding vector stores...")
    configs = []
    for key in args.configs:
        preset = CONFIG_PRESETS[key]
        collection = build_collection(
            f"{key}_rag", articles,
            chunk_size=preset["chunk_size"],
            chunk_overlap=preset["chunk_overlap"],
            persist=persist,
        )
        configs.append((key, preset, collection))

    total = len(questions) * len(configs)
    print(f"\nRunning {len(questions)} questions × {len(configs)} configs "
          f"= {total} evaluations\n")

    results = []
    count = 0
    for i, question in enumerate(questions):
        print(f"Q{i+1}/{len(questions)}: {question}")
        for key, preset, collection in configs:
            count += 1
            start = time.time()
            chunks = run_retrieval(collection, question, preset["k"],
                                   strategy=preset["retrieval"])
            answer = generate(question, chunks)
            latency = round(time.time() - start, 2)

            f_score = score_faithfulness(answer, chunks)
            time.sleep(0.3)
            r_score = score_answer_relevancy(question, answer)
            time.sleep(0.3)
            p_score = score_context_precision(question, chunks)
            time.sleep(0.3)
            c_score = score_answer_correctness(question, answer, chunks)

            results.append({
                "config": preset["label"],
                "config_key": key,
                "question": question,
                "answer": answer,
                "contexts": " ||| ".join(chunks),
                "latency": latency,
                "faithfulness": f_score,
                "answer_relevancy": r_score,
                "context_precision": p_score,
                "answer_correctness": c_score,
            })

            print(f"  [{count}/{total}] {preset['label']}: "
                  f"faith={f_score:.2f}  rel={r_score:.2f}  prec={p_score:.2f}  "
                  f"corr={c_score:.2f}  ({latency}s)")
            time.sleep(0.5)

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)

    print("\n" + "=" * 55)
    print("RESULTS SUMMARY (mean across all questions)")
    print("=" * 55)
    summary = df.groupby("config")[METRIC_COLUMNS].mean().round(3)
    print(summary.to_string())
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
