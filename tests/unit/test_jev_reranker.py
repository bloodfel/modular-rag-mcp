"""Tests for Jev (TypeSafe System One) based Reranker implementation.

All tests mock ``JevReranker._post`` (or ``httpx.post``) — no network access.
``TYPESAFE_API_KEY`` is only ever provided via monkeypatched environment
variables; it is never written to any file.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from src.core.settings import RerankSettings, Settings
from src.libs.reranker.base_reranker import BaseReranker
from src.libs.reranker.jev_reranker import JevRerankError, JevReranker
from src.libs.reranker.reranker_factory import RerankerFactory


DEFAULT_MODEL = "jev-1.13.0"
COST_PER_MTOK = 0.042


@pytest.fixture
def mock_settings():
    """Create mock settings with a rerank section exposing min_score."""
    settings = Mock(spec=Settings)
    settings.rerank = Mock(spec=RerankSettings)
    settings.rerank.min_score = 4.0
    return settings


@pytest.fixture
def sample_candidates():
    """Sample candidate list for reranking."""
    return [
        {"id": "chunk_1", "text": "Python is a programming language."},
        {"id": "chunk_2", "text": "Machine learning uses neural networks."},
        {"id": "chunk_3", "text": "RAG combines retrieval and generation."},
    ]


def make_api_response(
    scores: Dict[str, Any], input_tokens: int = 1500
) -> Dict[str, Any]:
    """Build a System One style API response body (answers keyed by question id)."""
    return {
        "model": DEFAULT_MODEL,
        "answers": {
            chunk_id: {"type": "score", "score": score}
            for chunk_id, score in scores.items()
        },
        "usage": {"input_tokens": input_tokens},
    }


@pytest.fixture
def jev_api_key(monkeypatch):
    """Provide a fake TYPESAFE_API_KEY via the environment only."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-api-key")
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    return "test-api-key"


def make_reranker(mock_settings, **kwargs: Any) -> JevReranker:
    """Construct a JevReranker with a fake api key (env var must be set)."""
    return JevReranker(settings=mock_settings, **kwargs)


class TestJevRerankerSortingAndFiltering:
    """Test ordering and min_score filtering behavior."""

    def test_rerank_sorts_by_score_descending(
        self, jev_api_key, mock_settings, sample_candidates
    ):
        """Candidates are returned sorted by Jev score, descending, with rerank_score attached."""
        reranker = make_reranker(mock_settings)
        reranker._post = Mock(
            return_value=make_api_response(
                {"chunk_1": 7, "chunk_2": 9, "chunk_3": 5}
            )
        )

        reranked = reranker.rerank("What is RAG?", sample_candidates)

        assert [c["id"] for c in reranked] == ["chunk_2", "chunk_1", "chunk_3"]
        assert [c["rerank_score"] for c in reranked] == [9.0, 7.0, 5.0]
        assert all(isinstance(c["rerank_score"], float) for c in reranked)
        assert reranker._post.call_count == 1

    def test_rerank_filters_below_min_score(
        self, jev_api_key, mock_settings, sample_candidates
    ):
        """Candidates scoring below min_score are dropped (min_score itself passes)."""
        reranker = make_reranker(mock_settings)
        reranker._post = Mock(
            return_value=make_api_response(
                {"chunk_1": 9, "chunk_2": 2, "chunk_3": 4}
            )
        )

        reranked = reranker.rerank("What is RAG?", sample_candidates)

        assert [c["id"] for c in reranked] == ["chunk_1", "chunk_3"]
        assert [c["rerank_score"] for c in reranked] == [9.0, 4.0]


class TestJevRerankerErrorPaths:
    """Test failure semantics: no silent drops, hard limits, wrapped errors."""

    def test_rerank_missing_answer_raises(
        self, jev_api_key, mock_settings, sample_candidates
    ):
        """A candidate without a matching answer raises instead of being dropped."""
        reranker = make_reranker(mock_settings)
        reranker._post = Mock(
            return_value=make_api_response({"chunk_1": 8, "chunk_3": 6})
        )

        with pytest.raises(JevRerankError, match="chunk_2"):
            reranker.rerank("What is RAG?", sample_candidates)

    def test_rerank_rejects_more_than_ten_candidates(
        self, jev_api_key, mock_settings
    ):
        """More than 10 candidates is a hard error, rejected before any API call."""
        reranker = make_reranker(mock_settings)
        reranker._post = Mock(return_value=make_api_response({}))
        candidates = [
            {"id": f"chunk_{i}", "text": f"text {i}"} for i in range(11)
        ]

        with pytest.raises(JevRerankError, match="10"):
            reranker.rerank("query", candidates)

        reranker._post.assert_not_called()

    def test_rerank_wraps_api_failure(self, jev_api_key, mock_settings, sample_candidates):
        """Transport-level failures surface as JevRerankError, never raw httpx errors."""
        # Part A: _post raising an httpx transport error is wrapped by rerank()
        reranker = make_reranker(mock_settings)
        reranker._post = Mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with pytest.raises(JevRerankError, match="connection refused"):
            reranker.rerank("query", sample_candidates)

        # Part B: the real _post wraps httpx failures, and sends the
        # expected URL / auth header / payload shape.
        reranker2 = make_reranker(mock_settings)
        with patch("src.libs.reranker.jev_reranker.httpx.post") as mock_http_post:
            mock_http_post.side_effect = httpx.ConnectTimeout("timed out")
            with pytest.raises(JevRerankError, match="timed out"):
                reranker2.rerank("query", sample_candidates)

        assert mock_http_post.call_count == 1
        call = mock_http_post.call_args
        assert call.args[0] == "https://api.typesafe.ai/v1/systemone"
        assert call.kwargs["headers"]["Authorization"] == f"Bearer {jev_api_key}"
        payload = call.kwargs["json"]
        assert payload["model"] == DEFAULT_MODEL
        assert payload["state"] == "query"
        assert list(payload["questions"].keys()) == [
            "chunk_1",
            "chunk_2",
            "chunk_3",
        ]
        assert all(q["type"] == "score" for q in payload["questions"].values())
        for question, candidate in zip(
            payload["questions"].values(), sample_candidates
        ):
            assert candidate["text"] in question["instructions"]
            # Score questions require ordered criteria levels (1-10 scale).
            assert len(question["criteria"]) == 10


class TestJevRerankerUsageMetadata:
    """Test token usage accounting surfaced via _jev_meta."""

    def test_rerank_records_usage_in_jev_meta(
        self, jev_api_key, mock_settings, sample_candidates
    ):
        """Passing candidates carry _jev_meta with tokens, cost estimate and model."""
        reranker = make_reranker(mock_settings)
        reranker._post = Mock(
            return_value=make_api_response(
                {"chunk_1": 8, "chunk_2": 3, "chunk_3": 6},
                input_tokens=1500,
            )
        )

        reranked = reranker.rerank("What is RAG?", sample_candidates)

        expected_cost = 1500 / 1e6 * COST_PER_MTOK
        assert [c["id"] for c in reranked] == ["chunk_1", "chunk_3"]
        for candidate in reranked:
            meta = candidate["_jev_meta"]
            assert meta["tokens_in"] == 1500
            assert meta["est_cost_usd"] == pytest.approx(expected_cost)
            assert meta["model"] == DEFAULT_MODEL


class TestJevRerankerCredentials:
    """Test API key handling (environment only, never files)."""

    def test_missing_api_key_rejected(self, mock_settings, monkeypatch):
        """Construction without TYPESAFE_API_KEY fails fast; explicit key overrides env."""
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

        with pytest.raises(JevRerankError, match="TYPESAFE_API_KEY"):
            JevReranker(settings=mock_settings)

        # An explicitly provided key still allows construction without env var.
        reranker = JevReranker(settings=mock_settings, api_key="explicit-key")
        assert reranker.api_key == "explicit-key"


class TestJevRerankerFactoryRegistration:
    """Test factory support for provider: 'jev'."""

    def setup_method(self) -> None:
        RerankerFactory._PROVIDERS.clear()

    def test_factory_creates_jev_reranker(self, jev_api_key, mock_settings):
        """Factory resolves provider 'jev' to JevReranker."""
        settings = MagicMock()
        settings.rerank.enabled = True
        settings.rerank.provider = "jev"
        settings.rerank.model = DEFAULT_MODEL
        settings.rerank.min_score = 4.0

        reranker = RerankerFactory.create(settings)

        assert isinstance(reranker, JevReranker)
        assert isinstance(reranker, BaseReranker)
        assert "jev" in RerankerFactory.list_providers()
