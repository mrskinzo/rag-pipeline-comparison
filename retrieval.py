"""
retrieval.py
Retrieval strategies for the RAG pipeline comparison.

Three retrieval arms are exposed here and shared by build_pipelines.py and
evaluate.py so the comparison is apples-to-apples:

  - dense_retrieve   : semantic (embedding) retrieval via a ChromaDB collection.
  - hybrid_retrieve  : dense + BM25 keyword retrieval, fused with Reciprocal
                       Rank Fusion (RRF). Catches exact technical terms and
                       brand/jargon tokens that the embedding model never
                       learned to encode, while keeping semantic recall.
  - rerank_retrieve  : re-orders an existing candidate pool with a
                       sentence-transformers CrossEncoder re-ranker and keeps
                       the top-k (used after multi-query expansion).

Models (the sentence-transformers embedder and CrossEncoder) are injected by
the caller so the weights are loaded once and reused across every config.
Nothing in this module loads a model at import time.
"""

from typing import List, Sequence

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Cheap whitespace/lowercase tokenizer for BM25. Good enough for a
    support corpus dominated by exact-match terms; deliberately dependency-free."""
    return text.lower().split()


class BM25Index:
    """A lightweight BM25 keyword index over a fixed chunk corpus.

    The corpus is the exact same list of chunk texts that was embedded into the
    ChromaDB collection, so dense and sparse rankings are over identical units.
    """

    def __init__(self, documents: Sequence[str]):
        self.documents = list(documents)
        # BM25Okapi divides by the corpus size, so guard the empty case.
        self._bm25 = BM25Okapi([_tokenize(d) for d in self.documents]) if self.documents else None

    def search(self, query: str, k: int) -> List[str]:
        if not self.documents:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in ranked[:k]]


def dense_retrieve(collection, embedder, query: str, k: int) -> List[str]:
    """Semantic retrieval: embed the query and pull the k nearest chunks."""
    embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=k)
    return results["documents"][0]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k: int, rrf_k: int = 60
) -> List[str]:
    """Fuse several ranked lists of documents with Reciprocal Rank Fusion.

    Each document's fused score is the sum over the lists of 1 / (rrf_k + rank),
    with rank starting at 1. RRF needs no score normalisation across retrievers,
    which is exactly why it is the standard way to combine dense + BM25.
    """
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank)
    fused = sorted(scores, key=lambda doc: scores[doc], reverse=True)
    return fused[:k]


def hybrid_retrieve(
    collection,
    embedder,
    bm25: BM25Index,
    query: str,
    k: int,
    candidate_k: int = None,
) -> List[str]:
    """Single-pass hybrid retrieval: dense + BM25 fused with RRF.

    Each retriever contributes a candidate pool (default 4*k, min 10); the two
    rankings are fused with RRF and the top-k fused chunks are returned.
    """
    pool = candidate_k or max(k * 4, 10)
    dense = dense_retrieve(collection, embedder, query, pool)
    sparse = bm25.search(query, pool)
    return reciprocal_rank_fusion([dense, sparse], k)


def rerank_retrieve(
    candidates: Sequence[str], cross_encoder, query: str, k: int
) -> List[str]:
    """Re-order a candidate pool with a cross-encoder and keep the top-k.

    The cross-encoder scores each (query, chunk) pair jointly, which is more
    precise than the bi-encoder cosine used for first-pass retrieval but too
    expensive to run over the whole corpus -- hence it runs only on a pool.
    """
    candidates = list(candidates)
    if not candidates:
        return []
    pairs = [(query, c) for c in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in ranked[:k]]
