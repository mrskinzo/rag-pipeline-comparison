"""
evaluate.py
Full evaluation of 3 RAG configurations across 10 questions.
Metrics: Faithfulness, Answer Relevancy, Context Precision, Latency
Results saved to evaluation_results.csv
Runtime: ~5-8 minutes
"""

import json
import time
import os
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

import retrieval

load_dotenv()

# ── SETUP ──────────────────────────────────────────────────────────────────
CLIENT = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.Client()

SYSTEM_PROMPT = """You are a GoDaddy customer support assistant.
Answer the question using only the provided context.
If the context doesn't contain enough information, say so clearly.
Be concise and direct."""

# ── TEST QUESTIONS ─────────────────────────────────────────────────────────
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

# ── CHUNKING ───────────────────────────────────────────────────────────────
def chunk_articles(articles: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for article in articles:
        for i, split in enumerate(splitter.split_text(article["content"])):
            chunks.append({
                "text": split,
                "source": article["url"],
                "title": article["title"],
                "id": f"{abs(hash(article['url']))}_{i}"
            })
    return chunks


def build_collection(name: str, chunks: List[Dict]):
    """Embed a precomputed list of chunks into a fresh ChromaDB collection.

    Chunks are passed in (rather than chunked here) so the caller can build a
    BM25 index over the exact same units for hybrid retrieval.
    """
    try:
        db.delete_collection(name)
    except Exception:
        pass
    collection = db.create_collection(name)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"], "title": c["title"]} for c in chunks],
        ids=[c["id"] for c in chunks]
    )
    print(f"  {name}: {len(chunks)} chunks")
    return collection

# ── RETRIEVAL ──────────────────────────────────────────────────────────────
def retrieve(collection, query: str, k: int) -> List[str]:
    """Dense (semantic) retrieval. Thin wrapper kept for backwards compatibility."""
    return retrieval.dense_retrieve(collection, embedder, query, k)


def multi_query_variants(query: str, n: int = 2) -> List[str]:
    """Original query plus up to n LLM-generated paraphrases."""
    response = CLIENT.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=150,
        messages=[{"role": "user", "content":
            f"Write {n} alternative phrasings of this question. "
            f"Return only the questions, one per line:\n{query}"}]
    )
    rewrites = [v for v in response.content[0].text.strip().split("\n") if v.strip()][:n]
    return [query] + rewrites


def multi_query_rerank(collection, cross_encoder, query: str, k: int,
                       pool_per_variant: int = 5) -> List[str]:
    """Config C arm: multi-query rewrite expands the candidate pool, then a
    cross-encoder re-ranker re-orders the pool and keeps the top-k."""
    seen, candidates = set(), []
    for variant in multi_query_variants(query):
        for chunk in retrieval.dense_retrieve(collection, embedder, variant, pool_per_variant):
            if chunk not in seen:
                seen.add(chunk)
                candidates.append(chunk)
    return retrieval.rerank_retrieve(candidates, cross_encoder, query, k)

# ── GENERATION ─────────────────────────────────────────────────────────────
def generate(question: str, chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    response = CLIENT.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    )
    return response.content[0].text

# ── METRICS ────────────────────────────────────────────────────────────────
def call_claude_score(prompt: str) -> float:
    """Ask Claude for a 0-1 score. Returns float."""
    try:
        response = CLIENT.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        return max(0.0, min(1.0, float(response.content[0].text.strip())))
    except Exception:
        return 0.5


def score_faithfulness(answer: str, contexts: List[str]) -> float:
    """Are all claims in the answer supported by the context?"""
    if not contexts:
        return 0.5
    context_text = "\n\n".join(contexts)[:2000]
    return call_claude_score(
        f"Rate how faithful this answer is to the context.\n"
        f"0 = answer contains unsupported claims, 1 = every claim is supported by context.\n"
        f"Context: {context_text}\nAnswer: {answer}\n"
        f"Return only a decimal number between 0 and 1."
    )


def score_answer_relevancy(question: str, answer: str) -> float:
    """Does the answer actually address what was asked?"""
    return call_claude_score(
        f"Rate how well this answer addresses the question.\n"
        f"0 = answer is off-topic, 1 = answer fully addresses the question.\n"
        f"Question: {question}\nAnswer: {answer}\n"
        f"Return only a decimal number between 0 and 1."
    )


def score_context_precision(question: str, contexts: List[str]) -> float:
    """Are the retrieved chunks relevant to the question?"""
    if not contexts:
        return 0.0
    scores = []
    for ctx in contexts:
        scores.append(call_claude_score(
            f"Rate how relevant this retrieved chunk is to the question.\n"
            f"0 = irrelevant, 1 = highly relevant.\n"
            f"Question: {question}\nChunk: {ctx[:500]}\n"
            f"Return only a decimal number between 0 and 1."
        ))
        time.sleep(0.3)
    return round(sum(scores) / len(scores), 3) if scores else 0.0

# ── MAIN EVALUATION ────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("articles.json") as f:
        articles = json.load(f)

    print("=" * 55)
    print("RAG PIPELINE EVALUATION")
    print("=" * 55)

    from sentence_transformers import CrossEncoder

    print("\nBuilding vector stores...")
    # Config A — small, non-overlapping chunks + single-pass hybrid retrieval
    #            (dense semantic + BM25 keyword, fused with reciprocal rank fusion).
    chunks_a = chunk_articles(articles, chunk_size=500,  chunk_overlap=0)
    coll_a = build_collection("config_a_hybrid", chunks_a)
    bm25_a = retrieval.BM25Index([c["text"] for c in chunks_a])

    # Config B — larger, overlapping chunks + semantic retrieval only (the
    #            pre-existing "optimized" dense baseline).
    chunks_b = chunk_articles(articles, chunk_size=1000, chunk_overlap=200)
    coll_b = build_collection("config_b_semantic", chunks_b)

    # Config C — small chunks + multi-query rewrite + cross-encoder re-ranker.
    chunks_c = chunk_articles(articles, chunk_size=500,  chunk_overlap=0)
    coll_c = build_collection("config_c_rerank", chunks_c)
    print("  loading cross-encoder re-ranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # k is tuned so each config feeds the generator a comparable amount of
    # context: ~6 small (500-char) chunks ≈ ~3 large (1000-char) chunks.
    CONFIGS = [
        ("Config A - Hybrid (dense+BM25 RRF)",
         lambda q, k: retrieval.hybrid_retrieve(coll_a, embedder, bm25_a, q, k), 6),
        ("Config B - Semantic only",
         lambda q, k: retrieval.dense_retrieve(coll_b, embedder, q, k), 3),
        ("Config C - MultiQuery + Re-ranker",
         lambda q, k: multi_query_rerank(coll_c, cross_encoder, q, k), 6),
    ]

    print(f"\nRunning {len(TEST_QUESTIONS)} questions × 3 configs = {len(TEST_QUESTIONS)*3} evaluations\n")

    results = []
    total = len(TEST_QUESTIONS) * len(CONFIGS)
    count = 0

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"Q{i+1}/{len(TEST_QUESTIONS)}: {question}")
        for config_name, retrieve_fn, k in CONFIGS:
            count += 1
            start = time.time()
            chunks = retrieve_fn(question, k)
            answer = generate(question, chunks)
            latency = round(time.time() - start, 2)

            f_score = score_faithfulness(answer, chunks)
            time.sleep(0.3)
            r_score = score_answer_relevancy(question, answer)
            time.sleep(0.3)
            p_score = score_context_precision(question, chunks)

            results.append({
                "config": config_name,
                "question": question,
                "answer": answer,
                "contexts": " ||| ".join(chunks),
                "latency": latency,
                "faithfulness": f_score,
                "answer_relevancy": r_score,
                "context_precision": p_score,
            })

            print(f"  [{count}/{total}] {config_name}: "
                  f"faith={f_score:.2f}  rel={r_score:.2f}  prec={p_score:.2f}  ({latency}s)")
            time.sleep(0.5)

    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)

    print("\n" + "=" * 55)
    print("RESULTS SUMMARY (mean across all questions)")
    print("=" * 55)
    summary = df.groupby("config")[
        ["faithfulness", "answer_relevancy", "context_precision", "latency"]
    ].mean().round(3)
    print(summary.to_string())
    print("\nFull results saved to evaluation_results.csv")
