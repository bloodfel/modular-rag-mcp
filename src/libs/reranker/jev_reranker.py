"""TypeSafe Jev (System One) based Reranker implementation.

This module implements reranking by calling the TypeSafe System One REST API.
Unlike LLM-based rerankers, Jev emits typed judgments (Score with an ordinal
1-10 relevance rating), so there is no free-text parsing path: any missing or
non-numeric answer is a hard failure raised as :class:`JevRerankError`, which
upstream fallback logic handles.

Credentials: the ``TYPESAFE_API_KEY`` is read from the environment only
(never from settings files).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"
DEFAULT_MODEL = "jev-1.13.0"
DEFAULT_TIMEOUT = 15.0
DEFAULT_MIN_SCORE = 4.0

# Jev evaluates questions in a single call; more than 10 triggers context rot.
MAX_QUESTIONS_PER_CALL = 10

# Passages are truncated before being embedded in a question.
PASSAGE_MAX_CHARS = 1000

# Estimated input token price (output tokens are free for Score answers).
COST_PER_MTOK_USD = 0.042

# Ordered relevance levels for Score questions (the System One API requires
# 2-10 concrete level descriptions; 10 matches the ordinal 1-10 design).
_SCORE_CRITERIA = [
    "Completely irrelevant to the query",
    "Barely related to the query; no useful information",
    "Weakly related; touches the topic but adds little",
    "Some topical overlap; partially addresses the query",
    "Related; addresses part of the query",
    "Relevant; addresses the query with some detail",
    "Quite relevant; mostly answers the query",
    "Highly relevant; answers most of the query well",
    "Very highly relevant; direct, thorough answer material",
    "Exactly answers the query; ideal passage",
]


class JevRerankError(RuntimeError):
    """Raised when Jev (TypeSafe System One) reranking fails."""


class JevReranker(BaseReranker):
    """Reranker backed by the TypeSafe Jev (System One) typed Score API.

    Each candidate becomes one Score question in a single System One call
    (query as ``state``). Answers are strictly validated: every candidate
    must receive a numeric score or :class:`JevRerankError` is raised —
    candidates are never silently dropped due to parsing.

    Design Principles Applied:
    - Pluggable: Swapped in via RerankerFactory with provider 'jev'.
    - Config-Driven: base_url/model/min_score overridable via args, env and
      settings; credentials come from the environment only.
    - Typed Output: No free-text parsing; scores arrive as numbers.
    - Fail-Fast: Missing answers, non-numeric scores, transport errors and
      oversized batches all raise JevRerankError for upstream fallback.
    """

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Jev Reranker.

        Args:
            settings: Application settings containing rerank configuration
                (``settings.rerank.min_score`` is honored when present).
            api_key: Explicit TypeSafe API key. If None, reads the
                ``TYPESAFE_API_KEY`` environment variable.
            base_url: Optional API base URL. Falls back to the
                ``TYPESAFE_BASE_URL`` environment variable, then
                ``https://api.typesafe.ai/v1``.
            timeout: Request timeout in seconds. Default 15s.
            model: Optional model name override. Default ``jev-1.13.0``.
            **kwargs: Additional provider-specific parameters.

        Raises:
            JevRerankError: If no API key is available.
        """
        self.settings = settings
        self.kwargs = kwargs

        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevRerankError(
                "Missing API key: set the TYPESAFE_API_KEY environment variable "
                "(or pass api_key explicitly)."
            )

        env_base_url = os.environ.get("TYPESAFE_BASE_URL")
        self.base_url = (base_url or env_base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout)
        self.model = model or DEFAULT_MODEL
        self.min_score = self._get_min_score_from_settings(settings)

    def _get_min_score_from_settings(self, settings: Any) -> float:
        """Read ``min_score`` from settings, defaulting to 4.0.

        Args:
            settings: Application settings (may lack a rerank section).

        Returns:
            The minimum score for a candidate to survive filtering.
        """
        rerank = getattr(settings, "rerank", None)
        min_score = getattr(rerank, "min_score", DEFAULT_MIN_SCORE)
        try:
            return float(min_score)
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid rerank.min_score {min_score!r}; falling back to "
                f"{DEFAULT_MIN_SCORE}"
            )
            return DEFAULT_MIN_SCORE

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using Jev typed relevance scores.

        Args:
            query: The user query string.
            candidates: List of candidate records to rerank. Each must contain
                an identifier (``id``) and text (``text`` or ``content``).
                At most 10 candidates per call.
            trace: Optional TraceContext for observability (stage recording is
                handled upstream by CoreReranker).
            **kwargs: Additional parameters (unused).

        Returns:
            Candidates scoring >= min_score, sorted by score descending. Each
            carries ``rerank_score`` (float) and ``_jev_meta`` with
            ``tokens_in``, ``est_cost_usd`` and ``model``.

        Raises:
            ValueError: If query or candidates are invalid.
            JevRerankError: On oversized batches, API/transport failures, or
                missing/non-numeric answers.
        """
        self.validate_query(query)
        self.validate_candidates(candidates)

        if len(candidates) > MAX_QUESTIONS_PER_CALL:
            raise JevRerankError(
                f"Jev accepts at most {MAX_QUESTIONS_PER_CALL} candidates per "
                f"call, got {len(candidates)}."
            )

        payload = self._build_payload(query, candidates)

        try:
            response = self._post(payload)
        except JevRerankError:
            raise
        except Exception as e:
            raise JevRerankError(f"Jev API call failed: {e}") from e

        scores = self._parse_answers(response, candidates)
        reranked = self._filter_sort_and_annotate(candidates, scores, response)

        logger.debug(
            f"Jev rerank: query='{query[:50]}...', "
            f"input={len(candidates)}, output={len(reranked)}, "
            f"model={self.model}"
        )
        return reranked

    def _build_payload(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the System One request payload.

        The documented wire format (docs.typesafe.ai/api) is::

            {"model", "state", "questions": {<id>: {
                "type": "score", "instructions": str, "criteria": [str, ...]
            }, ...}}

        ``questions`` is a *map* keyed by question id (returned answers use
        the same keys), each Score question carries an ordered ``criteria``
        level list, and the query itself travels as ``state``.

        Args:
            query: The user query string (becomes ``state``).
            candidates: Candidate records to score.

        Returns:
            Payload dict ready for :meth:`_post`.
        """
        questions: Dict[str, Dict[str, Any]] = {}
        for i, candidate in enumerate(candidates):
            chunk_id = str(candidate.get("id", f"candidate_{i}"))
            text = candidate.get("text") or candidate.get("content", "")
            if not isinstance(text, str):
                text = str(text)
            instructions = (
                f"Passage: {text[:PASSAGE_MAX_CHARS]}\n\n"
                "Rate how relevant this passage is to the query given in "
                "the application state."
            )
            questions[chunk_id] = {
                "type": "score",
                "instructions": instructions,
                "criteria": list(_SCORE_CRITERIA),
            }
        return {"model": self.model, "state": query, "questions": questions}

    def _post(self, payload: Dict[str, Any]) -> Any:
        """Send a request to the System One REST endpoint.

        Args:
            payload: Request payload from :meth:`_build_payload`.

        Returns:
            Decoded JSON response body.

        Raises:
            JevRerankError: On transport errors, HTTP error statuses, or
                undecodable response bodies.
        """
        url = f"{self.base_url}/systemone"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise JevRerankError(f"Jev API request failed: {e}") from e
        except ValueError as e:
            raise JevRerankError(f"Jev API returned invalid JSON: {e}") from e

    def _parse_answers(
        self,
        response: Any,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Map ``answers`` (a map keyed by question id) to float scores.

        Args:
            response: Decoded API response body.
            candidates: Original candidate records (defines required ids).

        Returns:
            Mapping from candidate id to float score.

        Raises:
            JevRerankError: If the response shape is wrong, any candidate has
                no answer, or any score is non-numeric. Never silently drops.
        """
        if not isinstance(response, dict):
            raise JevRerankError(
                f"Jev API response must be a dict, got {type(response).__name__}"
            )
        answers = response.get("answers")
        if not isinstance(answers, dict):
            raise JevRerankError("Jev API response missing 'answers' map")

        id_to_score: Dict[str, float] = {}
        for answer_id, answer in answers.items():
            if not isinstance(answer, dict):
                raise JevRerankError(
                    f"Jev answer '{answer_id}' is not a dict "
                    f"(type: {type(answer).__name__})"
                )
            score = answer.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise JevRerankError(
                    f"Jev answer '{answer_id}' has non-numeric score: {score!r}"
                )
            id_to_score[str(answer_id)] = float(score)

        scores: Dict[str, float] = {}
        for i, candidate in enumerate(candidates):
            chunk_id = str(candidate.get("id", f"candidate_{i}"))
            if chunk_id not in id_to_score:
                raise JevRerankError(
                    f"Jev API returned no answer for candidate '{chunk_id}'"
                )
            scores[chunk_id] = id_to_score[chunk_id]
        return scores

    def _filter_sort_and_annotate(
        self,
        candidates: List[Dict[str, Any]],
        scores: Dict[str, float],
        response: Any,
    ) -> List[Dict[str, Any]]:
        """Filter by min_score, sort descending, and attach score metadata.

        Args:
            candidates: Original candidate records.
            scores: Candidate id to Jev score mapping (from _parse_answers).
            response: Decoded API response body (for usage accounting).

        Returns:
            New list of candidate copies with ``rerank_score`` and
            ``_jev_meta`` attached, filtered and sorted.
        """
        tokens_in = 0
        usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict):
            raw_tokens = usage.get("input_tokens", 0)
            if isinstance(raw_tokens, (int, float)) and not isinstance(raw_tokens, bool):
                tokens_in = int(raw_tokens)
        est_cost_usd = tokens_in / 1e6 * COST_PER_MTOK_USD
        jev_meta = {
            "tokens_in": tokens_in,
            "est_cost_usd": est_cost_usd,
            "model": self.model,
        }

        scored = []
        for i, candidate in enumerate(candidates):
            chunk_id = candidate.get("id", f"candidate_{i}")
            score = scores[chunk_id]
            if score < self.min_score:
                continue
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = score
            candidate_copy["_jev_meta"] = dict(jev_meta)
            scored.append(candidate_copy)

        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored
