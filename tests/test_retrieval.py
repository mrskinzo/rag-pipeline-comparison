"""
Offline tests for the new retrieval arms.

These exercise the BM25 index and reciprocal rank fusion directly, with no
embedding model, cross-encoder, or API key required — so they run in CI without
network access to a model hub.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import retrieval


CORPUS = [
    "Turn off auto-renew for your domain in the renewal settings.",
    "Two-step verification adds a one-time password to your login.",
    "Revoke an SSL certificate from the certificates dashboard.",
    "Auto renew billing keeps your domain from expiring automatically.",
]


def test_bm25_keyword_match_ranks_exact_term_first():
    index = retrieval.BM25Index(CORPUS)
    top = index.search("auto-renew domain", k=2)
    assert len(top) == 2
    # A document containing the exact keywords should surface in the top results.
    assert any("auto-renew" in d or "auto renew" in d.lower() for d in top)


def test_bm25_empty_corpus_returns_empty():
    assert retrieval.BM25Index([]).search("anything", k=3) == []


def test_rrf_rewards_agreement_across_lists():
    # A document ranked highly by both retrievers should beat one ranked
    # highly by only a single retriever.
    dense = ["doc_shared", "doc_dense_only", "doc_x"]
    sparse = ["doc_shared", "doc_sparse_only", "doc_y"]
    fused = retrieval.reciprocal_rank_fusion([dense, sparse], k=3)
    assert fused[0] == "doc_shared"
    assert len(fused) == 3


def test_rrf_dedupes_and_caps_at_k():
    fused = retrieval.reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]], k=2)
    assert len(fused) == 2
    assert len(set(fused)) == len(fused)


def test_rrf_preserves_single_list_order():
    fused = retrieval.reciprocal_rank_fusion([["a", "b", "c"]], k=3)
    assert fused == ["a", "b", "c"]


def test_rerank_handles_empty_candidates():
    # Should short-circuit before touching the (here, irrelevant) cross-encoder.
    assert retrieval.rerank_retrieve([], cross_encoder=None, query="q", k=3) == []


def test_rerank_orders_by_cross_encoder_score():
    class FakeCrossEncoder:
        # Score is the chunk's length; longer chunk should rank first.
        def predict(self, pairs):
            return [len(chunk) for _, chunk in pairs]

    candidates = ["short", "a much longer candidate chunk", "mid length"]
    ranked = retrieval.rerank_retrieve(candidates, FakeCrossEncoder(), "q", k=2)
    assert ranked[0] == "a much longer candidate chunk"
    assert len(ranked) == 2
