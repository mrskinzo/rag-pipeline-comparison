"""Retrieval fusion.

Regression tests for the bug that made the multi-query config a no-op. The old
implementation concatenated each rewrite's hits, deduped, and truncated to k.
The original query alone already returns k chunks, so every rewrite was sliced
off and Config 3 retrieved byte-identical context to Config 2 on all 10
questions. These tests fail against that implementation.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from evaluate import reciprocal_rank_fusion, chunk_id  # noqa: E402


def test_fusion_can_surface_a_chunk_the_first_query_missed():
    # "d" is missed entirely by the original query but ranked top by both
    # rewrites. Fusion must be able to pull it into the top 3.
    original = ["a", "b", "c"]
    rewrite_1 = ["d", "a", "e"]
    rewrite_2 = ["d", "f", "a"]

    fused = reciprocal_rank_fusion([original, rewrite_1, rewrite_2], k=3)

    assert "d" in fused, "a chunk agreed on by both rewrites never surfaced"
    assert fused != original, "fusion returned the first query's hits unchanged"


def test_agreement_across_queries_beats_a_single_top_hit():
    # "b" is second everywhere; "a" is first once and absent elsewhere.
    # Consistent agreement should win.
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"], ["d", "b"]], k=1)
    assert fused == ["b"]


def test_identical_lists_are_a_passthrough():
    # When every rewrite agrees exactly, fusion changes nothing. This is the
    # one case where multi-query legitimately equals single-query.
    ranked = ["a", "b", "c"]
    assert reciprocal_rank_fusion([ranked, ranked, ranked], k=3) == ranked


def test_fusion_respects_k():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["d", "e", "f"]], k=2)
    assert len(fused) == 2


def test_fusion_of_nothing_is_empty():
    assert reciprocal_rank_fusion([], k=5) == []
    assert reciprocal_rank_fusion([[], []], k=5) == []


def test_chunk_ids_are_stable_across_processes():
    # The point of the sha1 switch. The builtin hash() is salted per process,
    # so this value would differ between runs. Pinning the literal is what
    # makes the test meaningful; asserting hash(x) == hash(x) in one process
    # proves nothing.
    assert chunk_id("https://www.godaddy.com/help/some-article-1234", 0) == (
        "2c7a25224a16aa94_0"
    )


def test_chunk_ids_are_chroma_safe():
    import re
    assert re.match(r"^[\w-]+$", chunk_id("https://x.test/a-1", 3))
