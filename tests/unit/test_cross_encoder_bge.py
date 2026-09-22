"""Tests for CrossEncoderReranker with BGE-style cross-encoder models.

BGE rerankers (e.g. BAAI/bge-reranker-v2-m3) emit raw logits from
``model.predict(pairs)``. Raw logits are monotonically rankable, so no
sigmoid normalization is required for ordering. These tests verify that
``CrossEncoderReranker`` handles such raw scores correctly:
correct descending order, float ``rerank_score`` passthrough (no
clipping/normalization), and proper (query, passage) pair construction.

A fake model (score = passage length) is injected so no real weights
are ever downloaded. Real-model smoke is deferred to Task 5.
"""

from typing import Any, Dict, List, Sequence, Tuple
from unittest.mock import Mock

from src.core.settings import RerankSettings, Settings
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker

Pair = Tuple[str, str]


class FakeBGEModel:
    """Fake BGE cross-encoder: returns raw logits equal to passage length.

    Mirrors the real BGE interface: ``predict(pairs) -> scores`` where
    scores are raw (possibly negative) logits, monotonically rankable.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.last_pairs: List[Pair] = []

    def predict(self, pairs: Sequence[Pair]) -> List[float]:
        self.call_count += 1
        self.last_pairs = list(pairs)
        return [float(len(passage)) for _, passage in pairs]


class FakeBGELogitModel:
    """Fake BGE cross-encoder with signed raw logits (can be negative).

    Score = passage length - offset, so short passages get negative
    logits as real BGE models often do.
    """

    def __init__(self, offset: float = 100.0) -> None:
        self.offset = offset
        self.call_count = 0
        self.last_pairs: List[Pair] = []

    def predict(self, pairs: Sequence[Pair]) -> List[float]:
        self.call_count += 1
        self.last_pairs = list(pairs)
        return [float(len(passage)) - self.offset for _, passage in pairs]


def make_settings() -> Mock:
    """Settings mock mirroring a BGE reranker config (model never loaded)."""
    settings = Mock(spec=Settings)
    settings.rerank = Mock(spec=RerankSettings)
    settings.rerank.model = "BAAI/bge-reranker-v2-m3"
    settings.rerank.enabled = True
    settings.rerank.provider = "cross_encoder"
    return settings


def make_reranker(model: Any) -> CrossEncoderReranker:
    return CrossEncoderReranker(settings=make_settings(), model=model)


class TestCrossEncoderBGEScoring:
    """Verify pair construction and raw-logit passthrough with BGE semantics."""

    def test_predict_receives_query_passage_pairs(self):
        model = FakeBGEModel()
        reranker = make_reranker(model)
        candidates = [
            {"id": "a", "text": "short"},
            {"id": "b", "text": "a much longer passage text"},
        ]

        reranker.rerank("what is rag", candidates)

        assert model.call_count == 1
        assert model.last_pairs == [
            ("what is rag", "short"),
            ("what is rag", "a much longer passage text"),
        ]

    def test_rerank_score_is_float_passthrough(self):
        model = FakeBGEModel()
        reranker = make_reranker(model)
        candidates = [{"id": "a", "text": "x" * 17}]

        result = reranker.rerank("query", candidates)

        assert isinstance(result[0]["rerank_score"], float)
        # Raw logit passed through unmodified (no sigmoid/clamping).
        assert result[0]["rerank_score"] == 17.0

    def test_negative_raw_logits_are_not_clipped(self):
        """BGE logits can be negative; ordering must still work untouched."""
        model = FakeBGELogitModel(offset=100.0)
        reranker = make_reranker(model)
        candidates = [
            {"id": "long", "text": "x" * 150},
            {"id": "short", "text": "x" * 20},
        ]

        result = reranker.rerank("query", candidates)

        assert result[0]["id"] == "long"
        assert result[0]["rerank_score"] == 50.0  # 150 - 100, kept negative-free
        assert result[1]["id"] == "short"
        assert result[1]["rerank_score"] == -80.0  # raw negative logit preserved
        assert isinstance(result[1]["rerank_score"], float)


class TestCrossEncoderBGESortOrder:
    """Verify descending sort over 5 candidates (the acceptance criterion)."""

    def test_five_candidates_sorted_descending(self):
        model = FakeBGEModel()
        reranker = make_reranker(model)
        # Text lengths deliberately shuffled relative to input order:
        # a=5, b=30, c=12, d=44, e=21.
        candidates: List[Dict[str, Any]] = [
            {"id": "a", "text": "x" * 5},
            {"id": "b", "text": "x" * 30},
            {"id": "c", "text": "x" * 12},
            {"id": "d", "text": "x" * 44},
            {"id": "e", "text": "x" * 21},
        ]

        result = reranker.rerank("query", candidates)

        assert [c["id"] for c in result] == ["d", "b", "e", "c", "a"]
        scores = [c["rerank_score"] for c in result]
        assert scores == sorted(scores, reverse=True)
        assert all(isinstance(s, float) for s in scores)

    def test_five_candidates_top_k_subset_keeps_descending(self):
        model = FakeBGEModel()
        reranker = make_reranker(model)
        candidates: List[Dict[str, Any]] = [
            {"id": "a", "text": "x" * 5},
            {"id": "b", "text": "x" * 30},
            {"id": "c", "text": "x" * 12},
            {"id": "d", "text": "x" * 44},
            {"id": "e", "text": "x" * 21},
        ]

        result = reranker.rerank("query", candidates, top_k=3)

        assert [c["id"] for c in result] == ["d", "b", "e"]
        scores = [c["rerank_score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_attach_scores_and_sort_descending_directly(self):
        """Direct check of _attach_scores_and_sort with raw BGE logits."""
        model = FakeBGEModel()
        reranker = make_reranker(model)
        candidates = [{"id": str(i), "text": "x"} for i in range(5)]
        # Deliberately shuffled signed logits.
        logits = [0.5, -1.25, 3.75, 0.0, -2.5]

        result = reranker._attach_scores_and_sort(candidates, logits, top_k=5)

        assert [c["id"] for c in result] == ["2", "0", "3", "1", "4"]
        assert [c["rerank_score"] for c in result] == [3.75, 0.5, 0.0, -1.25, -2.5]

    def test_original_candidates_not_mutated(self):
        model = FakeBGEModel()
        reranker = make_reranker(model)
        candidates = [{"id": "a", "text": "x" * 10}]

        reranker.rerank("query", candidates)

        assert "rerank_score" not in candidates[0]

    def test_content_field_fallback_with_bge_model(self):
        model = FakeBGEModel()
        reranker = make_reranker(model)
        candidates = [{"id": "a", "content": "x" * 9}]

        result = reranker.rerank("query", candidates)

        assert result[0]["rerank_score"] == 9.0
