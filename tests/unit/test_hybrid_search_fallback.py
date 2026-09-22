"""Unit tests for HybridSearch degradation behaviour.

When one retrieval path fails the other must carry the query, and the
degradation must land on the trace — otherwise the degradation rate is
unmeasurable from traces.jsonl, which is what the observability report
and the Langfuse fallback panel are built on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult


class BrokenRetriever:
    """Retriever whose backing service is unreachable."""

    provider_name = "broken"

    def __init__(self, message: str = "embedding service unreachable") -> None:
        self.message = message

    def retrieve(self, **kwargs: Any) -> List[RetrievalResult]:
        raise ConnectionError(self.message)


class FakeRetriever:
    """Retriever that returns one canned chunk."""

    provider_name = "fake"

    def __init__(self, chunk_id: str = "c1", count: int = 1) -> None:
        self.chunk_id = chunk_id
        self.count = count

    def retrieve(self, **kwargs: Any) -> List[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=f"{self.chunk_id}{i}" if i else self.chunk_id,
                score=1.0 - i * 0.1,
                text="hello",
                metadata={"source_path": "doc.md"},
            )
            for i in range(self.count)
        ]


def _config(**overrides: Any) -> HybridSearchConfig:
    base: Dict[str, Any] = {
        "dense_top_k": 5,
        "sparse_top_k": 5,
        "fusion_top_k": 5,
        "parallel_retrieval": False,
    }
    base.update(overrides)
    return HybridSearchConfig(**base)


def _stages(trace: TraceContext, name: str) -> List[Dict[str, Any]]:
    return [s for s in trace.stages if s["stage"] == name]


class TestSinglePathFailureDegrades:
    """One broken path must not fail the query."""

    def test_dense_failure_serves_sparse_results(self) -> None:
        search = HybridSearch(
            dense_retriever=BrokenRetriever(),
            sparse_retriever=FakeRetriever(),
            config=_config(),
        )

        result = search.search("hello world", top_k=5, return_details=True)

        assert result.used_fallback is True
        assert len(result.results) == 1
        assert "embedding service unreachable" in result.dense_error

    def test_sparse_failure_serves_dense_results(self) -> None:
        search = HybridSearch(
            dense_retriever=FakeRetriever(),
            sparse_retriever=BrokenRetriever("bm25 index missing"),
            config=_config(),
        )

        result = search.search("hello world", top_k=5, return_details=True)

        assert result.used_fallback is True
        assert len(result.results) == 1
        assert "bm25 index missing" in result.sparse_error


class TestFallbackIsRecorded:
    """Degradation must be visible on the trace, not just in the return value."""

    def test_dense_failure_records_retrieval_fallback_stage(self) -> None:
        trace = TraceContext()
        search = HybridSearch(
            dense_retriever=BrokenRetriever(),
            sparse_retriever=FakeRetriever(),
            config=_config(),
        )

        search.search("hello world", top_k=5, trace=trace, return_details=True)

        stages = _stages(trace, "retrieval_fallback")
        assert len(stages) == 1, "a degraded query must record retrieval_fallback"
        data = stages[0]["data"]
        assert data["dense_failed"] is True
        assert data["sparse_failed"] is False
        assert "embedding service unreachable" in data["dense_error"]
        assert data["degraded_result_count"] == 1

    def test_failed_path_also_records_its_own_stage(self) -> None:
        trace = TraceContext()
        search = HybridSearch(
            dense_retriever=BrokenRetriever(),
            sparse_retriever=FakeRetriever(),
            config=_config(),
        )

        search.search("hello world", top_k=5, trace=trace, return_details=True)

        dense = _stages(trace, "dense_retrieval")
        assert len(dense) == 1
        assert "error" in dense[0]["data"]

    def test_healthy_query_records_no_fallback_stage(self) -> None:
        trace = TraceContext()
        search = HybridSearch(
            dense_retriever=FakeRetriever(),
            sparse_retriever=FakeRetriever("c9"),
            config=_config(),
        )

        result = search.search("hello world", top_k=5, trace=trace, return_details=True)

        assert result.used_fallback is False
        assert _stages(trace, "retrieval_fallback") == []


class TestBothPathsFailed:
    """Losing both paths is not degradable and must surface."""

    def test_raises_when_both_paths_fail(self) -> None:
        search = HybridSearch(
            dense_retriever=BrokenRetriever("dense down"),
            sparse_retriever=BrokenRetriever("sparse down"),
            config=_config(),
        )

        with pytest.raises(RuntimeError, match="Both retrieval paths failed"):
            search.search("hello world", top_k=5)

    def test_trace_still_holds_both_errors_when_both_fail(self) -> None:
        trace = TraceContext()
        search = HybridSearch(
            dense_retriever=BrokenRetriever("dense down"),
            sparse_retriever=BrokenRetriever("sparse down"),
            config=_config(),
        )

        with pytest.raises(RuntimeError):
            search.search("hello world", top_k=5, trace=trace)

        assert "error" in _stages(trace, "dense_retrieval")[0]["data"]
        assert "error" in _stages(trace, "sparse_retrieval")[0]["data"]
