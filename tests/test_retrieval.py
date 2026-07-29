"""Retrieval tests against a tiny in-memory Chroma collection.

These use hand-crafted embeddings (no sentence-transformers download) and
pass query_embedding explicitly, so they need neither an API key nor network.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pytest

from rag_core import mmr_select, retrieve, retrieve_mmr


@pytest.fixture(scope="module")
def collection():
    import chromadb

    db = chromadb.Client()
    try:
        db.delete_collection("test_retrieval")
    except Exception:
        pass
    coll = db.create_collection("test_retrieval")
    # 4D toy space: docs a/b point one way, c another, d a third
    coll.add(
        ids=["a", "b", "c", "d"],
        documents=["doc about domains",
                   "doc about domain renewals",
                   "doc about ssl certificates",
                   "doc about email setup"],
        embeddings=[
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    )
    return coll


def test_cosine_retrieve_orders_by_similarity(collection):
    docs = retrieve(collection, "domains", k=2,
                    query_embedding=[1.0, 0.0, 0.0, 0.0])
    assert docs == ["doc about domains", "doc about domain renewals"]


def test_cosine_retrieve_k(collection):
    docs = retrieve(collection, "domains", k=3,
                    query_embedding=[1.0, 0.0, 0.0, 0.0])
    assert len(docs) == 3


def test_mmr_select_prefers_diversity():
    query = np.array([1.0, 0.0])
    candidates = np.array([
        [1.0, 0.0],    # most relevant
        [0.99, 0.01],  # near-duplicate of the first
        [0.5, 0.5],    # less relevant but diverse
    ])
    picked = mmr_select(query, candidates, k=2, lambda_mult=0.3)
    assert picked[0] == 0
    # with diversity weighted, the diverse doc beats the near-duplicate
    assert picked[1] == 2


def test_mmr_select_high_lambda_tracks_relevance():
    query = np.array([1.0, 0.0])
    candidates = np.array([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.5, 0.5],
    ])
    picked = mmr_select(query, candidates, k=2, lambda_mult=1.0)
    assert picked == [0, 1]


def test_retrieve_mmr_diversifies(collection):
    # cosine top-2 would be the two near-duplicate domain docs; MMR should
    # keep the best match and swap in something diverse
    docs = retrieve_mmr(collection, "domains", k=2, fetch_k=4, lambda_mult=0.3,
                        query_embedding=[1.0, 0.0, 0.0, 0.0])
    assert len(docs) == 2
    assert docs[0] == "doc about domains"
    assert docs[1] != "doc about domain renewals"


def test_retrieve_mmr_small_collection_returns_all(collection):
    docs = retrieve_mmr(collection, "domains", k=10, fetch_k=20,
                        query_embedding=[1.0, 0.0, 0.0, 0.0])
    assert len(docs) == 4
