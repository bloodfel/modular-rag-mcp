"""Unit tests for retrieval evaluation metrics (nDCG@k, MRR@k, Hit@k).

These metrics are the scoring basis for the Stage J rerank benchmark
(BEIR scifact / nfcorpus). Semantics per spec:

- ``qrels`` maps ``doc_id -> relevance``; relevance > 0 counts as relevant.
- nDCG uses graded gain ``2**rel - 1`` with IDCG from the ideal ordering
  of ``qrels`` (top k).
- MRR is the reciprocal rank of the first relevant doc within top k.
- Hit@k is 1.0 when any of the top-k docs is relevant.
- Empty ``qrels`` yields 0.0 for every metric.
"""

import math

import pytest

from src.observability.evaluation.metrics import hit_at_k, mrr_at_k, ndcg_at_k

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def graded_qrels() -> dict:
    """Graded qrels: three relevant docs with descending relevance."""
    return {"d1": 3, "d2": 2, "d3": 1}


# =============================================================================
# nDCG@k Tests
# =============================================================================

class TestNDCGAtK:
    """Tests for ndcg_at_k."""

    def test_perfect_ranking_returns_one(self, graded_qrels):
        """Ranking docs in ideal relevance order must yield nDCG = 1.0."""
        ranked = ["d1", "d2", "d3"]
        assert ndcg_at_k(ranked, graded_qrels, 3) == pytest.approx(1.0)

    def test_reversed_ranking_below_one(self, graded_qrels):
        """Reversed (worst) ordering must score strictly below 1.0."""
        ranked = ["d3", "d2", "d1"]
        score = ndcg_at_k(ranked, graded_qrels, 3)

        ideal_dcg = (
            (2**3 - 1) / math.log2(2)
            + (2**2 - 1) / math.log2(3)
            + (2**1 - 1) / math.log2(4)
        )
        reversed_dcg = (
            (2**1 - 1) / math.log2(2)
            + (2**2 - 1) / math.log2(3)
            + (2**3 - 1) / math.log2(4)
        )
        assert score == pytest.approx(reversed_dcg / ideal_dcg)
        assert score < 1.0

    def test_only_irrelevant_docs_returns_zero(self, graded_qrels):
        """Ranking containing none of the relevant docs must yield 0.0."""
        ranked = ["x", "y", "z"]
        assert ndcg_at_k(ranked, graded_qrels, 3) == 0.0

    def test_k_truncation_ignores_tail(self, graded_qrels):
        """Docs beyond position k must not contribute to DCG."""
        # d2 (rel=2) sits inside top-2; d3 (rel=1) sits outside and is ignored.
        ranked = ["d2", "x", "d3"]
        score_at_2 = ndcg_at_k(ranked, graded_qrels, 2)

        idcg_at_2 = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
        expected_at_2 = (2**2 - 1) / math.log2(2) / idcg_at_2
        assert score_at_2 == pytest.approx(expected_at_2)

        # Raising k to 3 lets d3 count (and d1 joins the ideal), so score changes.
        score_at_3 = ndcg_at_k(ranked, graded_qrels, 3)
        idcg_at_3 = idcg_at_2 + (2**1 - 1) / math.log2(4)
        dcg_at_3 = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(4)
        assert score_at_3 == pytest.approx(dcg_at_3 / idcg_at_3)
        assert score_at_3 > score_at_2


# =============================================================================
# MRR@k Tests
# =============================================================================

class TestMRRAtK:
    """Tests for mrr_at_k."""

    def test_relevant_at_first_position(self, graded_qrels):
        """Relevant doc at rank 1 gives reciprocal rank 1.0."""
        ranked = ["d2", "x", "y"]
        assert mrr_at_k(ranked, graded_qrels, 3) == pytest.approx(1.0)

    def test_relevant_at_second_position(self, graded_qrels):
        """Relevant doc at rank 2 gives reciprocal rank 0.5."""
        ranked = ["x", "d2", "y"]
        assert mrr_at_k(ranked, graded_qrels, 3) == pytest.approx(0.5)

    def test_zero_relevance_not_counted(self):
        """A doc present in qrels with relevance 0 is NOT relevant."""
        qrels = {"x": 0, "d2": 2}
        ranked = ["x", "d2"]
        assert mrr_at_k(ranked, qrels, 2) == pytest.approx(0.5)

    def test_relevant_beyond_k_ignored(self, graded_qrels):
        """Relevant doc ranked outside top k must not count."""
        ranked = ["x", "y", "d1"]
        assert mrr_at_k(ranked, graded_qrels, 2) == 0.0
        assert mrr_at_k(ranked, graded_qrels, 3) == pytest.approx(1 / 3)


# =============================================================================
# Hit@k Tests
# =============================================================================

class TestHitAtK:
    """Tests for hit_at_k."""

    def test_hit_and_miss_in_top_k(self, graded_qrels):
        """Hit = 1.0 when a relevant doc is in top k, 0.0 otherwise."""
        assert hit_at_k(["x", "d3"], graded_qrels, 2) == pytest.approx(1.0)
        assert hit_at_k(["x", "y"], graded_qrels, 2) == 0.0


# =============================================================================
# Shared Edge Cases
# =============================================================================

class TestEmptyQrels:
    """Empty qrels must yield 0.0 for every metric."""

    def test_empty_qrels_returns_zero_for_all_metrics(self):
        ranked = ["d1", "d2", "d3"]
        empty_qrels: dict = {}

        assert ndcg_at_k(ranked, empty_qrels, 3) == 0.0
        assert mrr_at_k(ranked, empty_qrels, 3) == 0.0
        assert hit_at_k(ranked, empty_qrels, 3) == 0.0
