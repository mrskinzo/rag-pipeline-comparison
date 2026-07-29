"""Shared core for the RAG pipeline comparison project.

Everything that used to be duplicated between ``build_pipelines.py`` and
``evaluate.py`` lives here: chunking, embedding (with a disk cache),
vector-store construction, the three retrieval strategies (cosine, MMR,
multi-query), answer generation, and the LLM-as-judge scoring helpers.

Heavy dependencies (sentence-transformers, the Anthropic client, Chroma)
are created lazily so this module can be imported — and most of it tested —
without an API key or a downloaded embedding model.
"""

import hashlib
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── constants ──────────────────────────────────────────────────────────────

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-3-haiku-20240307"
CHROMA_PATH = "chroma_db"
EMBED_CACHE_DIR = os.path.join(".cache", "embeddings")
CHROMA_BATCH_SIZE = 500

SYSTEM_PROMPT = """You are a GoDaddy customer support assistant.
Answer the question using only the provided context.
If the context doesn't contain enough information, say so clearly.
Be concise and direct."""

# Presets match the README table: key -> (label, chunk, overlap, k, retrieval)
CONFIG_PRESETS: Dict[str, Dict] = {
    "naive": {
        "label": "Config 1 - Naive",
        "chunk_size": 256,
        "chunk_overlap": 32,
        "k": 3,
        "retrieval": "cosine",
    },
    "optimized": {
        "label": "Config 2 - Optimized",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "k": 5,
        "retrieval": "cosine",
    },
    "mmr": {
        "label": "Config 3 - MMR",
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "k": 5,
        "retrieval": "mmr",
    },
    "multiquery": {
        "label": "Config 4 - MultiQuery",
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "k": 5,
        "retrieval": "multiquery",
    },
}

# ── lazy singletons ────────────────────────────────────────────────────────

_embedder = None
_anthropic_client = None
_chroma_client = None
_chroma_client_is_persistent = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client


def get_chroma_client(persist: bool = True):
    """Return a shared Chroma client, persistent under ``chroma_db/`` by default."""
    global _chroma_client, _chroma_client_is_persistent
    if _chroma_client is None or _chroma_client_is_persistent != persist:
        import chromadb

        _chroma_client = (
            chromadb.PersistentClient(path=CHROMA_PATH) if persist else chromadb.Client()
        )
        _chroma_client_is_persistent = persist
    return _chroma_client

# ── data loading & chunking ────────────────────────────────────────────────


def load_articles(path: str = "articles.json") -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def stable_id(text: str) -> str:
    """Deterministic ID for a string — unlike ``hash()``, stable across processes."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def chunk_articles(articles: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for article in articles:
        for i, split in enumerate(splitter.split_text(article["content"])):
            chunks.append({
                "text": split,
                "source": article["url"],
                "title": article["title"],
                "id": f"{stable_id(article['url'])}_{i}",
            })
    return chunks

# ── embeddings with disk cache ─────────────────────────────────────────────


def _embedding_cache_path(text: str) -> str:
    key = hashlib.sha1(f"{EMBED_MODEL_NAME}::{text}".encode("utf-8")).hexdigest()
    return os.path.join(EMBED_CACHE_DIR, f"{key}.npy")


def embed_texts(texts: List[str], use_cache: bool = True) -> np.ndarray:
    """Embed ``texts``, reading/writing per-text ``.npy`` files under ``.cache/``."""
    if not use_cache:
        return np.asarray(get_embedder().encode(texts, show_progress_bar=False))

    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    vectors: List[Optional[np.ndarray]] = [None] * len(texts)
    missing_idx = []
    for i, text in enumerate(texts):
        path = _embedding_cache_path(text)
        if os.path.exists(path):
            vectors[i] = np.load(path)
        else:
            missing_idx.append(i)

    if missing_idx:
        fresh = get_embedder().encode(
            [texts[i] for i in missing_idx], show_progress_bar=False
        )
        for j, i in enumerate(missing_idx):
            vec = np.asarray(fresh[j])
            np.save(_embedding_cache_path(texts[i]), vec)
            vectors[i] = vec

    return np.stack(vectors)


def embed_query(query: str) -> List[float]:
    return embed_texts([query])[0].tolist()

# ── vector store ───────────────────────────────────────────────────────────


def build_collection(
    name: str,
    articles: List[Dict],
    chunk_size: int,
    chunk_overlap: int,
    persist: bool = True,
    use_cache: bool = True,
):
    db = get_chroma_client(persist=persist)
    try:
        db.delete_collection(name)
    except Exception:
        pass
    collection = db.create_collection(name)
    chunks = chunk_articles(articles, chunk_size, chunk_overlap)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, use_cache=use_cache).tolist()
    for start in range(0, len(chunks), CHROMA_BATCH_SIZE):
        batch = slice(start, start + CHROMA_BATCH_SIZE)
        collection.add(
            embeddings=embeddings[batch],
            documents=texts[batch],
            metadatas=[{"source": c["source"], "title": c["title"]} for c in chunks[batch]],
            ids=[c["id"] for c in chunks[batch]],
        )
    print(f"  '{name}': {len(chunks)} chunks from {len(articles)} articles")
    return collection

# ── retrieval ──────────────────────────────────────────────────────────────


def retrieve(collection, query: str, k: int,
             query_embedding: Optional[List[float]] = None) -> List[str]:
    """Standard cosine-similarity dense retrieval."""
    embedding = query_embedding if query_embedding is not None else embed_query(query)
    results = collection.query(query_embeddings=[embedding], n_results=k)
    return results["documents"][0]


def mmr_select(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    k: int,
    lambda_mult: float = 0.5,
) -> List[int]:
    """Pick ``k`` candidate indices by Maximal Marginal Relevance.

    Balances similarity to the query against redundancy with already-selected
    candidates: score = λ·sim(query, c) − (1−λ)·max sim(c, selected).
    """
    def normalize(m):
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.clip(norms, 1e-10, None)

    q = normalize(np.asarray(query_embedding, dtype=float).reshape(1, -1))[0]
    cands = normalize(np.asarray(candidate_embeddings, dtype=float))
    query_sim = cands @ q
    cand_sim = cands @ cands.T

    selected: List[int] = []
    remaining = list(range(len(cands)))
    while remaining and len(selected) < k:
        if not selected:
            best = remaining[int(np.argmax(query_sim[remaining]))]
        else:
            best, best_score = remaining[0], -np.inf
            for idx in remaining:
                redundancy = cand_sim[idx, selected].max()
                score = lambda_mult * query_sim[idx] - (1 - lambda_mult) * redundancy
                if score > best_score:
                    best, best_score = idx, score
        selected.append(best)
        remaining.remove(best)
    return selected


def retrieve_mmr(
    collection,
    query: str,
    k: int,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    query_embedding: Optional[List[float]] = None,
) -> List[str]:
    """Fetch ``fetch_k`` candidates, then re-rank with MMR for diversity."""
    embedding = query_embedding if query_embedding is not None else embed_query(query)
    n_fetch = min(max(fetch_k, k), collection.count())
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_fetch,
        include=["documents", "embeddings"],
    )
    docs = results["documents"][0]
    cand_embeddings = np.asarray(results["embeddings"][0])
    if len(docs) <= k:
        return docs
    picked = mmr_select(np.asarray(embedding), cand_embeddings, k, lambda_mult)
    return [docs[i] for i in picked]


def multi_query_retrieve(collection, query: str, k: int) -> List[str]:
    """Retrieve with the original query plus 2 LLM-generated rephrasings."""
    response = get_anthropic_client().messages.create(
        model=LLM_MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content":
            f"Write 2 alternative phrasings of this question. "
            f"Return only the questions, one per line:\n{query}"}],
    )
    variants = [query] + [
        v for v in response.content[0].text.strip().split("\n") if v.strip()
    ][:2]
    seen, all_chunks = set(), []
    for q in variants:
        for chunk in retrieve(collection, q, k):
            if chunk not in seen:
                seen.add(chunk)
                all_chunks.append(chunk)
    return all_chunks[:k]


def run_retrieval(collection, query: str, k: int, strategy: str = "cosine") -> List[str]:
    if strategy == "cosine":
        return retrieve(collection, query, k)
    if strategy == "mmr":
        return retrieve_mmr(collection, query, k)
    if strategy == "multiquery":
        return multi_query_retrieve(collection, query, k)
    raise ValueError(f"Unknown retrieval strategy: {strategy!r}")

# ── generation ─────────────────────────────────────────────────────────────


def generate(question: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    response = get_anthropic_client().messages.create(
        model=LLM_MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return response.content[0].text


def run_rag(config_name: str, collection, question: str, k: int,
            strategy: str = "cosine") -> Dict:
    start = time.time()
    chunks = run_retrieval(collection, question, k, strategy=strategy)
    answer = generate(question, chunks)
    return {
        "config": config_name,
        "question": question,
        "answer": answer,
        "contexts": chunks,
        "latency": round(time.time() - start, 2),
    }

# ── LLM-as-judge scoring ───────────────────────────────────────────────────


def call_claude_score(prompt: str) -> float:
    """Ask Claude for a 0-1 score. Returns 0.5 on any API failure."""
    try:
        response = get_anthropic_client().messages.create(
            model=LLM_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
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


def score_answer_correctness(question: str, answer: str, contexts: List[str]) -> float:
    """Is the answer factually correct, judged against the retrieved context?"""
    context_text = "\n\n".join(contexts)[:2000] if contexts else "(no context)"
    return call_claude_score(
        f"You are grading a support answer for factual correctness.\n"
        f"Using the reference material below as ground truth, rate the answer:\n"
        f"0 = factually wrong or fabricated, 1 = fully correct and complete.\n"
        f"Reference material: {context_text}\n"
        f"Question: {question}\nAnswer: {answer}\n"
        f"Return only a decimal number between 0 and 1."
    )
