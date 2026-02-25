"""Evaluation harness for comparing multiple RAG pipeline configurations.

The original script executed a fixed set of questions over hard‑coded
configs.  This refactor adds command line flags, type annotations, and
utility functions so that pieces can be exercised from unit tests.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------- initialization --------------------------------------------------
load_dotenv()

with open("articles.json") as f:
    ARTICLES = json.load(f)

CLIENT = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
DB = chromadb.Client()

SYSTEM_PROMPT = (
    "You are a GoDaddy customer support assistant.\n"
    "Answer the question using only the provided context.\n"
    "If the context doesn't contain enough information, say so clearly.\n"
    "Be concise and direct."
)


# ---------- chunking --------------------------------------------------------

def chunk_articles(
    articles: List[dict], chunk_size: int, chunk_overlap: int
) -> List[dict]:
    """Split each article into chunks using the given parameters."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks: List[dict] = []
    for article in articles:
        for i, split in enumerate(splitter.split_text(article["content"])):
            chunks.append(
                {
                    "text": split,
                    "source": article["url"],
                    "title": article["title"],
                    "id": f"{abs(hash(article['url']))}_{i}",
                }
            )
    return chunks


def build_collection(
    name: str,
    articles: List[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> chromadb.api.models.Collection:
    """Create or rebuild a ChromaDB collection for the given parameters."""
    try:
        DB.delete_collection(name)
    except Exception:  # ignore if it doesn't exist
        pass
    collection = DB.create_collection(name)
    chunks = chunk_articles(articles, chunk_size, chunk_overlap)
    texts = [c["text"] for c in chunks]
    embeddings = EMBEDDER.encode(texts, show_progress_bar=False).tolist()
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"], "title": c["title"]} for c in chunks],
        ids=[c["id"] for c in chunks],
    )
    print(f"  {name}: {len(chunks)} chunks")
    return collection


# ---------- retrieval / generation -----------------------------------------

def retrieve(
    collection: chromadb.api.models.Collection, query: str, k: int
) -> List[str]:
    embedding = EMBEDDER.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=k)
    return results["documents"][0]


def multi_query_retrieve(
    collection: chromadb.api.models.Collection, query: str, k: int
) -> List[str]:
    response = CLIENT.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": (
                    "Write 2 alternative phrasings of this question. "
                    "Return only the questions, one per line:\n" + query
                ),
            }
        ],
    )
    variants = [query] + [v for v in response.content[0].text.strip().split("\n") if v.strip()][:2]
    seen, all_chunks = set(), []
    for q in variants:
        for chunk in retrieve(collection, q, k):
            if chunk not in seen:
                seen.add(chunk)
                all_chunks.append(chunk)
    return all_chunks[:k]


def generate(question: str, chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    response = CLIENT.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return response.content[0].text


# ---------- scoring --------------------------------------------------------

def call_claude_score(prompt: str) -> float:
    try:
        response = CLIENT.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        return max(0.0, min(1.0, float(response.content[0].text.strip())))
    except Exception:
        return 0.5


def score_faithfulness(answer: str, contexts: List[str]) -> float:
    context_text = "\n\n".join(contexts)[:2000]
    return call_claude_score(
        f"Rate how faithful this answer is to the context.\n"
        f"0 = answer contains unsupported claims, 1 = every claim is supported by context.\n"
        f"Context: {context_text}\nAnswer: {answer}\n"
        f"Return only a decimal number between 0 and 1."
    )


def score_answer_relevancy(question: str, answer: str) -> float:
    return call_claude_score(
        f"Rate how well this answer addresses the question.\n"
        f"0 = answer is off-topic, 1 = answer fully addresses the question.\n"
        f"Question: {question}\nAnswer: {answer}\n"
        f"Return only a decimal number between 0 and 1."
    )


def score_context_precision(question: str, contexts: List[str]) -> float:
    scores = []
    for ctx in contexts:
        scores.append(
            call_claude_score(
                f"Rate how relevant this retrieved chunk is to the question.\n"
                f"0 = irrelevant, 1 = highly relevant.\n"
                f"Question: {question}\nChunk: {ctx[:500]}\n"
                f"Return only a decimal number between 0 and 1."
            )
        )
        time.sleep(0.3)
    return round(sum(scores) / len(scores), 3) if scores else 0.0


# ---------- orchestration ---------------------------------------------------

def evaluate(
    questions: List[str],
    configs: List[Tuple[str, chromadb.api.models.Collection, int, bool]],
) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []
    total = len(questions) * len(configs)
    count = 0

    for i, question in enumerate(questions, start=1):
        print(f"Q{i}/{len(questions)}: {question}")
        for config_name, collection, k, use_mq in configs:
            count += 1
            start = time.time()
            chunks = (
                multi_query_retrieve(collection, question, k)
                if use_mq
                else retrieve(collection, question, k)
            )
            answer = generate(question, chunks)
            latency = round(time.time() - start, 2)

            f_score = score_faithfulness(answer, chunks)
            time.sleep(0.3)
            r_score = score_answer_relevancy(question, answer)
            time.sleep(0.3)
            p_score = score_context_precision(question, chunks)

            results.append(
                {
                    "config": config_name,
                    "question": question,
                    "answer": answer,
                    "contexts": " ||| ".join(chunks),
                    "latency": latency,
                    "faithfulness": f_score,
                    "answer_relevancy": r_score,
                    "context_precision": p_score,
                }
            )

            print(
                f"  [{count}/{total}] {config_name}: "
                f"faith={f_score:.2f}  rel={r_score:.2f}  prec={p_score:.2f}  ({latency}s)"
            )
            time.sleep(0.5)

    df = pd.DataFrame(results)
    return df


# ---------- command line ----------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG pipeline evaluation")
    parser.add_argument(
        "--questions-file",
        type=str,
        help="path to a JSON file containing a list of questions",
    )
    parser.add_argument("--output", type=str, default="evaluation_results.csv",
                        help="csv file to write results to")
    parser.add_argument("--config", action="append", nargs=3,
                        metavar=("NAME", "CHUNK_SIZE", "CHUNK_OVERLAP"),
                        help="add a config e.g. --config naive 500 0")
    args = parser.parse_args()

    questions = TEST_QUESTIONS if not args.questions_file else json.load(open(args.questions_file))
    # default configs if none provided
    if args.config:
        collections = []
        for name, size, overlap in args.config:
            coll = build_collection(name, ARTICLES, int(size), int(overlap))
            collections.append((name, coll, 3, False))
        CONFIGS = collections
    else:
        # same defaults as before
        c1 = build_collection("naive_rag", ARTICLES, 500, 0)
        c2 = build_collection("optimized_rag", ARTICLES, 1000, 200)
        c3 = build_collection("multiquery_rag", ARTICLES, 1000, 200)
        CONFIGS = [
            ("Config 1 - Naive", c1, 3, False),
            ("Config 2 - Optimized", c2, 5, False),
            ("Config 3 - MultiQuery", c3, 5, True),
        ]

    df = evaluate(questions, CONFIGS)
    df.to_csv(args.output, index=False)

    print("\nRESULTS SUMMARY (mean across all questions)")
    summary = df.groupby("config")[
        ["faithfulness", "answer_relevancy", "context_precision", "latency"]
    ].mean().round(3)
    print(summary.to_string())
    print(f"\nFull results saved to {args.output}")


# keep old default list so we can run without specifying anything
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


if __name__ == "__main__":
    main()
