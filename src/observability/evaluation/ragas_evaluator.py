"""Ragas-based evaluator for RAG quality assessment.

This evaluator wraps the Ragas framework to compute LLM-as-Judge metrics:
- Faithfulness: Does the answer stick to the retrieved context?
- Answer Relevancy: Is the answer relevant to the query?
- Context Precision: Are the retrieved chunks relevant and well-ordered?
- Context Recall: Do the retrieved chunks cover the reference answer?

Design Principles:
- Pluggable: Implements BaseEvaluator interface, swappable via factory.
- Config-Driven: LLM/Embedding backend read from settings.yaml.
- Graceful Degradation: Clear ImportError if ragas not installed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Metric name constants
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"
CONTEXT_RECALL = "context_recall"

SUPPORTED_METRICS = {
    FAITHFULNESS,
    ANSWER_RELEVANCY,
    CONTEXT_PRECISION,
    CONTEXT_RECALL,
}


def _import_ragas() -> None:
    """Validate that ragas is importable, raising a clear error if not."""
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ragas' package is required for RagasEvaluator. "
            "Install it with: pip install ragas datasets"
        ) from exc


class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses the Ragas framework for LLM-as-Judge metrics.

    Ragas does NOT require ground-truth labels.  It uses an LLM to judge
    the quality of the generated answer against the retrieved context.

    Supported metrics:
        - faithfulness: Measures factual consistency with context.
        - answer_relevancy: Measures how relevant the answer is to the query.
        - context_precision: Measures relevance/ordering of retrieved chunks.

    Example::

        evaluator = RagasEvaluator(settings=settings)
        metrics = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"id": "c1", "text": "RAG is ..."}],
            generated_answer="RAG stands for ...",
        )
        # metrics == {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}
    """

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RagasEvaluator.

        Args:
            settings: Application settings (used to configure LLM backend).
            metrics: Metric names to compute. Defaults to all supported.
            **kwargs: Additional parameters (reserved).

        Raises:
            ImportError: If ragas is not installed.
            ValueError: If unsupported metric names are requested.
        """
        _import_ragas()

        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalised = [m.strip().lower() for m in (metrics or [])]
        if not normalised:
            normalised = sorted(SUPPORTED_METRICS)

        unsupported = [m for m in normalised if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )

        self._metric_names = normalised

    # ── public API ────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate RAG quality using Ragas LLM-as-Judge metrics.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks (dicts with 'text' key or strings).
            generated_answer: The generated answer text. Required for Ragas.
            ground_truth: Reference answer; required when ``context_recall``
                is among the metrics, ignored by the others.
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters.

        Returns:
            Dictionary mapping metric names to float scores (0.0 – 1.0).

        Raises:
            ValueError: If query/chunks are invalid or generated_answer is missing.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        if not generated_answer or not generated_answer.strip():
            raise ValueError(
                "RagasEvaluator requires a non-empty 'generated_answer'. "
                "Ragas uses LLM-as-Judge and needs the answer text to evaluate."
            )

        contexts = self._extract_texts(retrieved_chunks)
        reference = self._extract_reference(ground_truth)

        if CONTEXT_RECALL in self._metric_names and not reference:
            raise ValueError(
                "context_recall needs a reference answer: pass it as the "
                "'ground_truth' argument (the golden set's reference_answer)."
            )

        try:
            result = self._run_ragas(query, contexts, generated_answer, reference)
        except Exception as exc:
            logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ragas evaluation failed: {exc}") from exc

        return result

    # ── private helpers ───────────────────────────────────────────

    @staticmethod
    def _extract_reference(ground_truth: Any) -> str:
        """Normalise the ground truth into a plain reference string."""
        if ground_truth is None:
            return ""
        if isinstance(ground_truth, str):
            return ground_truth.strip()
        if isinstance(ground_truth, dict):
            for field in ("reference_answer", "reference", "answer", "text"):
                if ground_truth.get(field):
                    return str(ground_truth[field]).strip()
            return ""
        if isinstance(ground_truth, (list, tuple)):
            return " ".join(str(item) for item in ground_truth).strip()
        return str(ground_truth).strip()

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        reference: str = "",
    ) -> Dict[str, float]:
        """Execute Ragas collections metrics and return normalised scores.

        Ragas 0.4+ collections metrics use per-metric ``score()`` instead of
        the legacy ``evaluate()`` pipeline.  Each metric has its own signature:
        - Faithfulness / ContextPrecision: (user_input, response, retrieved_contexts)
        - AnswerRelevancy: (user_input, response)
        - ContextRecall: (user_input, retrieved_contexts, reference)

        All metrics run inside ONE event loop via ``ascore()``: the shared
        OpenAI client binds its connection pool to the loop of its first
        request, and ragas's sync ``score()`` (``asyncio.run`` per call)
        closes that loop after every metric — the next metric then talks to
        a dead pool and hangs.
        """
        import asyncio

        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
            ContextRecall,
            Faithfulness,
        )

        # Build LLM / Embedding wrappers from settings
        llm, embeddings = self._build_wrappers()

        scores: Dict[str, float] = {}

        async def _score_all() -> None:
            for metric_name in self._metric_names:
                if metric_name == FAITHFULNESS:
                    metric = Faithfulness(llm=llm)
                    result = await metric.ascore(
                        user_input=query, response=answer, retrieved_contexts=contexts,
                    )
                elif metric_name == ANSWER_RELEVANCY:
                    metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
                    result = await metric.ascore(user_input=query, response=answer)
                elif metric_name == CONTEXT_PRECISION:
                    metric = ContextPrecisionWithoutReference(llm=llm)
                    result = await metric.ascore(
                        user_input=query, response=answer, retrieved_contexts=contexts,
                    )
                elif metric_name == CONTEXT_RECALL:
                    metric = ContextRecall(llm=llm)
                    result = await metric.ascore(
                        user_input=query,
                        retrieved_contexts=contexts,
                        reference=reference,
                    )
                else:
                    continue

                scores[metric_name] = (
                    float(result.value) if result.value is not None else 0.0
                )

        asyncio.run(_score_all())
        return scores

    def _build_wrappers(self) -> tuple:
        """Build Ragas LLM and Embedding wrappers from project settings.

        Uses Ragas 0.4+ native API (InstructorLLM + OpenAIEmbeddings) and the
        project's configured providers. Everything Ragas talks to must be
        OpenAI-compatible: DeepSeek and GLM expose compatible endpoints, and
        Ollama serves one at ``<base_url>/v1`` (the served model must support
        embeddings). Azure keeps its own client.

        Returns:
            Tuple of (llm_wrapper, embeddings_wrapper).
        """
        import os

        from openai import AsyncAzureOpenAI, AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory

        if self.settings is None:
            raise ValueError("Settings required to create LLM for Ragas evaluation")

        # ── LLM ──
        llm_cfg = self.settings.llm
        provider = llm_cfg.provider.lower()
        llm_azure_endpoint = getattr(llm_cfg, "azure_endpoint", None)

        _LLM_BASE_URLS = {
            "deepseek": "https://api.deepseek.com",
            "glm": "https://open.bigmodel.cn/api/paas/v4",
        }

        # Azure-compatible mode: if azure_endpoint is configured, use Azure
        # client even when provider is "openai" (matches project convention).
        use_azure_llm = (
            provider == "azure"
            or (provider == "openai" and llm_azure_endpoint)
        )

        if use_azure_llm:
            llm_client = AsyncAzureOpenAI(
                api_key=llm_cfg.api_key,
                azure_endpoint=llm_azure_endpoint or llm_cfg.azure_endpoint,
                api_version=getattr(llm_cfg, "api_version", None) or "2024-02-15-preview",
            )
        elif provider in ("openai", "deepseek", "glm"):
            llm_client = AsyncOpenAI(
                api_key=llm_cfg.api_key,
                base_url=getattr(llm_cfg, "base_url", None) or _LLM_BASE_URLS.get(provider),
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider for Ragas: '{provider}'. "
                "Supported: azure, openai, deepseek, glm"
            )

        llm = llm_factory(llm_cfg.model, client=llm_client, max_tokens=8192)

        # ── Embeddings ──
        emb_cfg = self.settings.embedding
        emb_provider = emb_cfg.provider.lower()
        emb_azure_endpoint = getattr(emb_cfg, "azure_endpoint", None)

        # Same Azure-compatible mode detection for embeddings
        use_azure_emb = (
            emb_provider == "azure"
            or (emb_provider == "openai" and emb_azure_endpoint)
        )

        if use_azure_emb:
            emb_client = AsyncAzureOpenAI(
                api_key=emb_cfg.api_key,
                azure_endpoint=emb_azure_endpoint or emb_cfg.azure_endpoint,
                api_version=getattr(emb_cfg, "api_version", None) or "2024-02-15-preview",
            )
        elif emb_provider in ("openai", "glm"):
            emb_client = AsyncOpenAI(
                api_key=emb_cfg.api_key,
                base_url=getattr(emb_cfg, "base_url", None) or _LLM_BASE_URLS.get(emb_provider),
            )
        elif emb_provider == "ollama":
            # Ollama exposes an OpenAI-compatible endpoint at /v1; any
            # non-empty key is accepted by the OpenAI client.
            base_url = (
                getattr(emb_cfg, "base_url", None)
                or os.environ.get("OLLAMA_BASE_URL")
                or "http://localhost:11434"
            ).rstrip("/")
            emb_client = AsyncOpenAI(api_key="ollama", base_url=f"{base_url}/v1")
        else:
            raise ValueError(
                f"Unsupported embedding provider for Ragas: '{emb_provider}'. "
                "Supported: azure, openai, glm, ollama"
            )

        embeddings = OpenAIEmbeddings(model=emb_cfg.model, client=emb_client)

        return llm, embeddings

    def _extract_texts(self, chunks: List[Any]) -> List[str]:
        """Extract text strings from various chunk representations.

        Args:
            chunks: List of chunk dicts, strings, or objects with .text.

        Returns:
            List of text strings.
        """
        texts: List[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content", "")
                texts.append(str(text))
            elif hasattr(chunk, "text"):
                texts.append(str(getattr(chunk, "text")))
            else:
                texts.append(str(chunk))
        return texts

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []
        raw_metrics = getattr(evaluation, "metrics", None)
        if raw_metrics is None:
            return []
        # Filter to only ragas-supported metrics
        return [m for m in raw_metrics if m.lower() in SUPPORTED_METRICS]
