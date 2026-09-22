#!/usr/bin/env python
"""Rerank benchmark runner over BEIR datasets (Stage J5).

For each ``dataset x retriever(bm25/dense/hybrid) x reranker(none/bge/jev/llm)``
combination this script:

1. Samples a fixed-seed set of BEIR **test split** queries that have qrels.
2. Retrieves the top-10 candidates with the project's existing retrieval
   components (SparseRetriever / DenseRetriever / HybridSearch RRF fusion).
3. Reranks the top-10 (depth=10, matching Jev's <=10 questions constraint;
   ``none`` keeps the original order).
4. Scores nDCG@10 / MRR@10 / Hit@10 against the dumped qrels and records
   rerank latency, Jev token usage and rerank failures.  Queries within a
   (dataset, retriever, reranker) cell run in a thread pool
   (``--workers``, default 6); transient rerank errors (HTTP 429, timeouts)
   are retried with exponential backoff (1s/2s, max 2 retries) and only
   exhausted retries / non-transient errors fall back to the original order
   and count as failures.

Outputs:
    reports/rerank_benchmark_<date>.md    - comparison tables
    reports/rerank_benchmark_<date>.json  - raw per-query values

Test split detection: query ids are taken from ``ir_datasets.load(
f"beir/<name>/test").qrels_iter()`` (the dumped qrels.json is a superset
across train/dev/test).  If ir_datasets is unavailable the sampler falls
back to "queries with any relevance >= 1 in qrels.json" and the report is
annotated accordingly.

Usage:
    # Smoke (no external API needed)
    python scripts/benchmark_rerank.py --datasets scifact --num-queries 5 --rerankers none

    # Full run
    python scripts/benchmark_rerank.py --datasets scifact nfcorpus
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.fusion import RRFFusion
from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
from src.core.query_engine.query_processor import QueryProcessor, QueryProcessorConfig
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.settings import Settings, load_settings, resolve_path
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.evaluation.metrics import hit_at_k, mrr_at_k, ndcg_at_k

TOP_K = 10
SUPPORTED_DATASETS = ("scifact", "nfcorpus")
ALL_RETRIEVERS = ("bm25", "dense", "hybrid")
ALL_RERANKERS = ("none", "bge", "jev", "llm")

BGE_MODEL = "BAAI/bge-reranker-base"
JEV_MODEL = "jev-1.13.0"
# 1.0 = 纯排序对比（与 bge/llm 同口径，不过滤）；过滤式用法（如 4.0）属"质量门"场景，见报告
JEV_MIN_SCORE = 1.0
# Must mirror JevReranker.COST_PER_MTOK_USD (kept literal here so the
# benchmark report can be regenerated without importing the module).
JEV_COST_PER_MTOK_USD = 0.042
# Approx. USD per MTok (input, output) for pricing the chat-LLM rerank column:
# glm-4-flash is on Zhipu's free tier; deepseek-chat at V3 cache-miss list price.
LLM_PRICE_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "glm-4-flash": (0.0, 0.0),
    "deepseek-chat": (0.28, 0.42),
    "deepseek-reasoner": (0.55, 2.19),
}

# Caveat mandated by the spec: prepare_beir truncates "title\n\nbody" to
# 5000 chars for embedding-model context safety.
CORPUS_TRUNCATION_CAVEAT = (
    "Caveat: BEIR corpus texts were truncated to 5000 chars (title+body) at "
    "ingest; this affects <0.25% of documents and may slightly depress recall "
    "on the longest documents."
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the rerank benchmark over BEIR datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "nfcorpus"],
        choices=SUPPORTED_DATASETS,
        help="BEIR datasets to benchmark (default: scifact nfcorpus)",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=list(ALL_RETRIEVERS),
        choices=ALL_RETRIEVERS,
        help="Retrieval paths to benchmark (default: bm25 dense hybrid)",
    )
    parser.add_argument(
        "--rerankers",
        nargs="+",
        default=list(ALL_RERANKERS),
        choices=ALL_RERANKERS,
        help="Rerankers to benchmark (default: none bge jev llm)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=50,
        help="Queries sampled per dataset (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query sampling (default: 42)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Concurrent worker threads for rerank calls within one "
        "dataset/retriever/reranker cell (default: 6; 1 = serial)",
    )
    parser.add_argument(
        "--jev-min-score",
        type=float,
        default=JEV_MIN_SCORE,
        help="Jev candidate-filter threshold (default: %(default)s; 1.0 = pure "
        "ranking, 4.0 = production default, 0 = no filtering)",
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "settings.yaml"),
        help="Path to configuration file (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--outdir",
        default=str(_REPO_ROOT / "reports"),
        help="Output directory (default: reports/)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading & sampling
# ---------------------------------------------------------------------------

def load_artifacts(name: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load queries and qrels dumped by scripts/prepare_beir.py.

    Args:
        name: BEIR dataset name.

    Returns:
        Tuple of (queries as {query_id: text}, qrels as {query_id: {doc_id: rel}}).
    """
    beir_dir = resolve_path(f"data/beir/{name}")
    queries_list = json.loads((beir_dir / "queries.json").read_text(encoding="utf-8"))
    queries = {str(q["query_id"]): q["text"] for q in queries_list}
    qrels_raw = json.loads((beir_dir / "qrels.json").read_text(encoding="utf-8"))
    qrels = {
        str(qid): {str(doc): int(rel) for doc, rel in docs.items()}
        for qid, docs in qrels_raw.items()
    }
    return queries, qrels


def test_split_query_ids(name: str) -> Tuple[Optional[set], str]:
    """Return the BEIR test split query ids and how they were determined.

    Args:
        name: BEIR dataset name.

    Returns:
        Tuple of (set of query ids or None, human-readable source note).
        ``None`` means ir_datasets could not provide the split and the caller
        must fall back to the qrels superset.
    """
    try:
        import ir_datasets

        ds = ir_datasets.load(f"beir/{name}/test")
        qids = {str(rec.query_id) for rec in ds.qrels_iter()}
        if qids:
            return qids, f"ir_datasets beir/{name}/test"
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[WARN] Could not resolve test split for '{name}': {e}")
    return None, "qrels superset (relevance>=1; train/dev/test not separated)"


def sample_queries(
    name: str,
    num_queries: int,
    seed: int,
) -> Tuple[List[str], Dict[str, str], Dict[str, Dict[str, int]], str]:
    """Sample fixed-seed test-split queries that have qrels and query text.

    Args:
        name: BEIR dataset name.
        num_queries: Number of queries to sample.
        seed: Random seed.

    Returns:
        Tuple of (sampled query ids, queries, qrels, split source note).
    """
    queries, qrels = load_artifacts(name)
    test_qids, split_note = test_split_query_ids(name)

    if test_qids is not None:
        eligible = sorted(test_qids & set(qrels) & set(queries))
    else:
        eligible = sorted(
            qid
            for qid, docs in qrels.items()
            if qid in queries and any(rel >= 1 for rel in docs.values())
        )

    if not eligible:
        raise RuntimeError(
            f"No eligible queries for '{name}' (test split ∩ qrels ∩ queries is empty)"
        )

    rng = random.Random(seed)
    if len(eligible) <= num_queries:
        sampled = list(eligible)
    else:
        sampled = sorted(rng.sample(eligible, num_queries))
    return sampled, queries, qrels, split_note


# ---------------------------------------------------------------------------
# Retrieval components
# ---------------------------------------------------------------------------

class DatasetRetrievers:
    """Retrieval bundle for one BEIR dataset (all three paths share stores).

    The dense and sparse paths are invoked directly (top-10 from a single
    route); the hybrid path goes through the existing HybridSearch component
    with RRF fusion.  ``enable_filter_parsing`` is disabled so natural
    language BEIR queries can never be mistaken for ``key:value`` filters.
    """

    def __init__(self, settings: Settings, name: str, retrievers: Sequence[str]) -> None:
        """Create the retrievers for collection ``beir-<name>``.

        Args:
            settings: Application settings.
            name: BEIR dataset name.
            retrievers: Requested retrieval paths (bm25/dense/hybrid).
        """
        beir_settings = replace(
            settings,
            vector_store=replace(
                settings.vector_store, collection_name=f"beir-{name}"
            ),
        )
        self.name = name
        self.processor = QueryProcessor(
            QueryProcessorConfig(enable_filter_parsing=False)
        )
        self.vector_store = VectorStoreFactory.create(beir_settings)
        # J4 layout: data/db/bm25/beir-<name>/beir-<name>_bm25.json
        # (BM25Indexer resolves <index_dir>/<collection>_bm25.json).
        self.sparse = SparseRetriever(
            settings=beir_settings,
            bm25_indexer=BM25Indexer(
                index_dir=str(resolve_path(f"data/db/bm25/beir-{name}"))
            ),
            vector_store=self.vector_store,
            default_collection=f"beir-{name}",
        )
        self.dense: Optional[DenseRetriever] = None
        self.hybrid: Optional[HybridSearch] = None
        if "dense" in retrievers or "hybrid" in retrievers:
            embedding = EmbeddingFactory.create(beir_settings)
            self.dense = DenseRetriever(
                settings=beir_settings,
                embedding_client=embedding,
                vector_store=self.vector_store,
                default_top_k=TOP_K,
            )
        if "hybrid" in retrievers:
            self.hybrid = HybridSearch(
                settings=beir_settings,
                query_processor=self.processor,
                dense_retriever=self.dense,
                sparse_retriever=self.sparse,
                fusion=RRFFusion(k=beir_settings.retrieval.rrf_k),
                config=HybridSearchConfig(
                    dense_top_k=TOP_K,
                    sparse_top_k=TOP_K,
                    fusion_top_k=TOP_K,
                    metadata_filter_post=False,
                ),
            )

    def retrieve(self, method: str, query: str) -> List[Any]:
        """Retrieve top-10 results via the requested path.

        Args:
            method: One of 'bm25', 'dense', 'hybrid'.
            query: Raw query text.

        Returns:
            List of RetrievalResult objects (may be empty).

        Raises:
            ValueError: If the method is unknown.
        """
        if method == "bm25":
            keywords = self.processor.process(query).keywords or query.split()
            return self.sparse.retrieve(keywords=keywords, top_k=TOP_K)
        if method == "dense":
            assert self.dense is not None, "dense retriever not built"
            return self.dense.retrieve(query=query, top_k=TOP_K)
        if method == "hybrid":
            assert self.hybrid is not None, "hybrid search not built"
            return list(self.hybrid.search(query=query, top_k=TOP_K))
        raise ValueError(f"Unknown retrieval method: {method}")


# ---------------------------------------------------------------------------
# Reranker construction
# ---------------------------------------------------------------------------

def build_rerankers(
    settings: Settings,
    requested: Sequence[str],
    jev_min_score: float = JEV_MIN_SCORE,
) -> Dict[str, Any]:
    """Instantiate the requested rerankers, skipping unavailable ones.

    Args:
        settings: Application settings.
        requested: Reranker names (none/bge/jev/llm).
        jev_min_score: Candidate filter threshold for the Jev reranker.

    Returns:
        Mapping from reranker name to instance ('none' maps to None, i.e.
        keep the original order).
    """
    available: Dict[str, Any] = {}
    for rer_name in requested:
        if rer_name == "none":
            available["none"] = None
            continue
        try:
            if rer_name == "bge":
                import torch
                from src.libs.reranker.cross_encoder_reranker import (
                    CrossEncoderReranker,
                )

                # Cap torch CPU intra-op threads.  Note: this alone does NOT
                # make concurrent predict() safe — the crash comes from torch's
                # MPS/Metal backend when multiple threads submit inference
                # (native abort, no Python traceback).  The actual guard is
                # running bge serially (see run_benchmark).
                torch.set_num_threads(1)

                bge_settings = replace(
                    settings,
                    rerank=replace(
                        settings.rerank,
                        enabled=True,
                        provider="cross_encoder",
                        model=BGE_MODEL,
                    ),
                )
                available["bge"] = CrossEncoderReranker(bge_settings)
            elif rer_name == "jev":
                # load_settings() already loaded .env into the environment.
                api_key = os.environ.get("TYPESAFE_API_KEY")
                if not api_key:
                    print("[skip] jev: TYPESAFE_API_KEY not set")
                    continue
                from src.libs.reranker.jev_reranker import JevReranker

                jev_settings = replace(
                    settings,
                    rerank=replace(
                        settings.rerank,
                        enabled=True,
                        provider="jev",
                        model=JEV_MODEL,
                        min_score=jev_min_score,
                    ),
                )
                available["jev"] = JevReranker(jev_settings, api_key=api_key)
            elif rer_name == "llm":
                from src.libs.reranker.llm_reranker import LLMReranker

                available["llm"] = LLMReranker(settings)
        except Exception as e:
            print(f"[skip] {rer_name}: {e}")
    return available


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def beir_doc_id(candidate: Dict[str, Any], name: str) -> str:
    """Map a candidate back to its raw BEIR doc id.

    Chroma chunk ids are upserter-generated hashes; the BEIR doc id is
    recovered from the ``beir://<name>/<doc_id>`` source_path metadata (with
    a ``beir-<name>-`` prefix fallback).

    Args:
        candidate: Candidate dict with 'id' and 'metadata'.
        name: BEIR dataset name.

    Returns:
        Raw BEIR doc id string.
    """
    metadata = candidate.get("metadata") or {}
    source_path = str(metadata.get("source_path") or "")
    prefix = f"beir://{name}/"
    if source_path.startswith(prefix):
        return source_path[len(prefix):]
    chunk_id = str(candidate.get("id") or "")
    alt_prefix = f"beir-{name}-"
    if chunk_id.startswith(alt_prefix):
        return chunk_id[len(alt_prefix):]
    return chunk_id


def result_to_candidates(results: Sequence[Any]) -> List[Dict[str, Any]]:
    """Convert RetrievalResult objects to reranker candidate dicts.

    Args:
        results: RetrievalResult list from a retriever.

    Returns:
        List of dicts with id/text/score/metadata keys.
    """
    return [
        {
            "id": r.chunk_id,
            "text": r.text,
            "score": r.score,
            "metadata": dict(r.metadata or {}),
        }
        for r in results
    ]


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of a sequence.

    Args:
        values: Sample values.
        q: Quantile in [0, 1].

    Returns:
        The percentile value (0.0 for an empty sample).
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

# Queries within one (dataset, retriever, reranker) cell run concurrently.
DEFAULT_WORKERS = 6

# Transient (rate-limit/timeout) failures are retried with exponential
# backoff before being counted as failures; max 2 retries.
RETRY_DELAYS_S = (1.0, 2.0)

# Lower-cased markers identifying retryable transport/API errors.  Rerankers
# wrap low-level exceptions (e.g. httpx) into their own error types, so the
# message text is the stable signal across providers.
_TRANSIENT_MARKERS = (
    "429",
    "529",
    "rate limit",
    "too many requests",
    "timeout",
    "timed out",
    "connection reset",
    "temporarily unavailable",
    "502",
    "503",
    "504",
)


def _is_transient(exc: Exception) -> bool:
    """Return True for rate-limit/timeout style errors worth retrying.

    Schema/parse errors (e.g. missing answers) are not retryable and return
    False here so they fail immediately.

    Args:
        exc: The exception raised by the reranker.

    Returns:
        True if the error looks transient (429/timeout/5xx).
    """
    try:
        import httpx

        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in (429, 502, 503, 504)
    except ImportError:  # pragma: no cover - httpx is a project dependency
        pass
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def rerank_with_retry(
    reranker: Any,
    query_text: str,
    candidates: List[Dict[str, Any]],
    log_prefix: str,
) -> Tuple[List[Dict[str, Any]], bool, float, int]:
    """Rerank with exponential-backoff retries on transient errors.

    Latency only accumulates the actual rerank attempts (backoff sleeps are
    excluded) so concurrent per-call timings stay comparable.

    Args:
        reranker: Reranker instance.
        query_text: Raw query text.
        candidates: Candidate dicts (copied per attempt).
        log_prefix: Identifier used in retry log lines.

    Returns:
        Tuple of (ranked candidates, fallback flag, latency in ms, retries).
        On retry exhaustion or non-transient errors the original order is
        returned with fallback=True.
    """
    latency_ms = 0.0
    for attempt in range(len(RETRY_DELAYS_S) + 1):
        started = time.perf_counter()
        try:
            ranked = reranker.rerank(
                query_text, [dict(c) for c in candidates]
            )
            latency_ms += (time.perf_counter() - started) * 1000.0
            return ranked, False, latency_ms, attempt
        except Exception as exc:
            latency_ms += (time.perf_counter() - started) * 1000.0
            if attempt < len(RETRY_DELAYS_S) and _is_transient(exc):
                delay = RETRY_DELAYS_S[attempt]
                print(
                    f"[retry] {log_prefix}: transient error ({exc}); "
                    f"retrying in {delay:.0f}s "
                    f"({attempt + 1}/{len(RETRY_DELAYS_S)})"
                )
                time.sleep(delay)
                continue
            return candidates, True, latency_ms, attempt
    return candidates, True, latency_ms, len(RETRY_DELAYS_S)  # unreachable


def benchmark_combination(
    reranker: Any,
    reranker_name: str,
    query_text: str,
    candidates: List[Dict[str, Any]],
    dataset_name: str,
    qrels: Dict[str, int],
) -> Dict[str, Any]:
    """Rerank one query's candidates and score the result.

    Transient (429/timeout) reranker errors are retried with backoff; only
    non-transient errors or exhausted retries fall back to the original
    order and count as failures.  ``none`` records zero rerank latency.

    Args:
        reranker: Reranker instance or None for 'none'.
        reranker_name: Reranker name (used for meta extraction).
        query_text: Raw query text.
        candidates: Top-10 candidates in retrieval order.
        dataset_name: BEIR dataset name (for doc id mapping).
        qrels: This query's {doc_id: relevance} map.

    Returns:
        Per-query record with metrics, latency, token usage, retry count and
        fallback flag.
    """
    ranked = candidates
    rerank_ms = 0.0
    fallback = False
    retries = 0
    tokens_in = 0
    tokens_out = 0

    if reranker is not None and candidates:
        log_prefix = f"{reranker_name} query='{query_text[:40]}...'"
        ranked, fallback, rerank_ms, retries = rerank_with_retry(
            reranker, query_text, candidates, log_prefix
        )
        # Rerankers that track per-call usage annotate every returned
        # candidate with the same meta dict (JevReranker / LLMReranker).
        meta_key = {"jev": "_jev_meta", "llm": "_llm_meta"}.get(reranker_name)
        if meta_key:
            metas = [
                c.get(meta_key)
                for c in ranked
                if isinstance(c, dict) and isinstance(c.get(meta_key), dict)
            ]
            if metas:
                tokens_in = max(int(m.get("tokens_in", 0)) for m in metas)
                tokens_out = max(int(m.get("tokens_out", 0)) for m in metas)

    ranked_ids = [beir_doc_id(c, dataset_name) for c in ranked]
    return {
        "ndcg": ndcg_at_k(ranked_ids, qrels, TOP_K),
        "mrr": mrr_at_k(ranked_ids, qrels, TOP_K),
        "hit": hit_at_k(ranked_ids, qrels, TOP_K),
        "rerank_ms": rerank_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "fallback": fallback,
        "retries": retries,
        "n_ranked": len(ranked),
    }


def run_benchmark(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Run the full benchmark grid.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Tuple of (per-combination aggregate records, report note lines).
    """
    settings = load_settings(args.config)
    llm_model_name = settings.llm.model
    rerankers = build_rerankers(settings, args.rerankers, args.jev_min_score)
    if not rerankers:
        raise RuntimeError("No rerankers available; nothing to benchmark")

    notes: List[str] = []
    combinations: List[Dict[str, Any]] = []

    for name in args.datasets:
        print(f"\n=== dataset: {name} ===")
        sampled, queries, qrels, split_note = sample_queries(
            name, args.num_queries, args.seed
        )
        print(f"[OK] sampled {len(sampled)}/{args.num_queries} test-split queries "
              f"(seed={args.seed}, split: {split_note})")
        if split_note not in notes:
            notes.append(split_note)
        if len(sampled) < args.num_queries:
            notes.append(
                f"{name}: only {len(sampled)} eligible queries available "
                f"(< {args.num_queries} requested)"
            )

        retrievers = DatasetRetrievers(settings, name, args.retrievers)

        # Retrieval runs once per (retriever, query); rerankers reuse it.
        base_rankings: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for method in args.retrievers:
            for qid in sampled:
                results = retrievers.retrieve(method, queries[qid])
                base_rankings[(method, qid)] = result_to_candidates(results)
            sizes = [len(base_rankings[(method, q)]) for q in sampled]
            print(f"[OK] {method}: top-{TOP_K} retrieved "
                  f"(avg {sum(sizes) / len(sizes):.1f} candidates/query)")

        rerank_started = time.monotonic()
        for rer_name, reranker in rerankers.items():
            for method in args.retrievers:
                jobs = [
                    (qid, base_rankings[(method, qid)], queries[qid])
                    for qid in sampled
                ]

                def _run_one(job: Tuple[str, List[Dict[str, Any]], str]) -> Dict[str, Any]:
                    """Score a single query (executed in the worker thread)."""
                    qid, cands, qtext = job
                    return benchmark_combination(
                        reranker, rer_name, qtext, cands, name, qrels.get(qid, {})
                    )

                # Query-level concurrency: rerank calls (jev/llm HTTP) dominate
                # wall time.  bge stays serial — concurrent predict() calls
                # from pool worker threads crash natively in torch's MPS/
                # Metal backend (exit 134/139, no Python traceback; even with
                # torch.set_num_threads(1)), and local inference gains nothing
                # from threads anyway.  Results are collected in submission
                # order so per-query ordering stays deterministic; each query
                # is independent.
                effective_workers = 1 if rer_name == "bge" else args.workers
                if effective_workers <= 1 or len(jobs) <= 1:
                    per_query = [_run_one(job) for job in jobs]
                else:
                    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                        futures = [pool.submit(_run_one, job) for job in jobs]
                        per_query = [f.result() for f in futures]
                for (qid, _, _), record in zip(jobs, per_query):
                    record["query_id"] = qid

                failures = sum(1 for r in per_query if r["fallback"])
                retries = sum(r["retries"] for r in per_query)
                latencies = [r["rerank_ms"] for r in per_query if not r["fallback"]]
                total_tokens = sum(r["tokens_in"] for r in per_query)
                total_tokens_out = sum(r["tokens_out"] for r in per_query)
                est_cost = total_tokens / 1e6 * JEV_COST_PER_MTOK_USD
                if rer_name == "llm":
                    in_price, out_price = LLM_PRICE_PER_MTOK.get(
                        llm_model_name, (0.0, 0.0)
                    )
                    est_cost = (
                        total_tokens / 1e6 * in_price
                        + total_tokens_out / 1e6 * out_price
                    )
                    notes.append(
                        f"llm rerank usage {name}/{method}: avg "
                        f"{total_tokens / len(per_query):.0f} in + "
                        f"{total_tokens_out / len(per_query):.0f} out tokens/call"
                    )
                combinations.append({
                    "dataset": name,
                    "retriever": method,
                    "reranker": rer_name,
                    "num_queries": len(per_query),
                    "ndcg_at_10": sum(r["ndcg"] for r in per_query) / len(per_query),
                    "mrr_at_10": sum(r["mrr"] for r in per_query) / len(per_query),
                    "hit_at_10": sum(r["hit"] for r in per_query) / len(per_query),
                    "rerank_ms_p50": percentile(latencies, 0.50),
                    "rerank_ms_p95": percentile(latencies, 0.95),
                    "rerank_failures": failures,
                    "rerank_retries": retries,
                    "rerank_tokens_in": total_tokens,
                    "rerank_tokens_out": total_tokens_out,
                    "est_cost_usd": est_cost,
                    "per_query": per_query,
                })
                retry_note = f" retries={retries}" if retries else ""
                print(
                    f"[OK] {name}/{method} x {rer_name}: "
                    f"nDCG@10={combinations[-1]['ndcg_at_10'] * 100:.2f} "
                    f"failures={failures}{retry_note}"
                )
                if failures:
                    notes.append(
                        f"{name}/{method} x {rer_name}: {failures} rerank "
                        f"failures fell back to the original order"
                    )

        rerank_wall = time.monotonic() - rerank_started
        wall_note = (
            f"{name}: rerank wall time {rerank_wall:.1f}s "
            f"({args.workers} workers, {len(sampled)} queries)"
        )
        notes.append(wall_note)
        print(f"[OK] {wall_note}")

    return combinations, notes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_cell(value: Optional[float], scale: float = 1.0, digits: int = 2) -> str:
    """Format a metric cell, rendering missing combinations as '-'.

    Args:
        value: Metric value or None.
        scale: Multiplier (e.g. 100 for percentages).
        digits: Decimal digits.

    Returns:
        Formatted cell text.
    """
    if value is None:
        return "-"
    return f"{value * scale:.{digits}f}"


def _grid(combinations: List[Dict[str, Any]],
          key: str) -> Dict[Tuple[str, str], Optional[float]]:
    """Index a metric by (dataset/retriever) x reranker for table rendering.

    Args:
        combinations: Aggregate records.
        key: Metric field name.

    Returns:
        Mapping from (dataset, retriever, reranker) to value (None if the
        combination was not run).
    """
    grid: Dict[Tuple[str, str], Optional[float]] = {}
    for combo in combinations:
        row = f"{combo['dataset']}/{combo['retriever']}"
        grid[(row, combo["reranker"])] = combo[key]
    return grid


def render_markdown(
    combinations: List[Dict[str, Any]],
    rerankers: Sequence[str],
    notes: List[str],
    args: argparse.Namespace,
) -> str:
    """Render the markdown report.

    Args:
        combinations: Aggregate records from run_benchmark.
        rerankers: Column order.
        notes: Caveat / warning lines.
        args: Parsed CLI arguments.

    Returns:
        Full markdown document.
    """
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    columns = list(rerankers)
    rows = sorted({(c["dataset"], c["retriever"]) for c in combinations})
    row_labels = [f"{d}/{m}" for d, m in rows]

    lines: List[str] = []
    lines.append(f"# Rerank Benchmark Report ({date})")
    lines.append("")
    lines.append(
        f"- Datasets: {', '.join(args.datasets)} | Retrievers: "
        f"{', '.join(args.retrievers)} | Rerankers: {', '.join(columns)}"
    )
    lines.append(
        f"- Sampling: {args.num_queries} BEIR **test split** queries per "
        f"dataset, seed={args.seed}, retrieval depth={TOP_K}, rerank depth={TOP_K}"
    )
    lines.append(f"- Generated: {date}")
    lines.append("")

    # Main quality table.
    ndcg_grid = _grid(combinations, "ndcg_at_10")
    lines.append("## Quality: nDCG@10 (x100, higher is better)")
    lines.append("")
    lines.append("| dataset/retriever | " + " | ".join(columns) + " |")
    lines.append("|---" * (len(columns) + 1) + "|")
    for label in row_labels:
        cells = [
            _fmt_cell(ndcg_grid.get((label, col)), scale=100.0) for col in columns
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # Secondary quality metrics.
    mrr_grid = _grid(combinations, "mrr_at_10")
    hit_grid = _grid(combinations, "hit_at_10")
    lines.append("## MRR@10 / Hit@10 (x100)")
    lines.append("")
    lines.append("| dataset/retriever | " + " | ".join(
        f"{col} MRR" for col in columns
    ) + " | " + " | ".join(f"{col} Hit" for col in columns) + " |")
    lines.append("|---" * (2 * len(columns) + 1) + "|")
    for label in row_labels:
        cells = [
            _fmt_cell(mrr_grid.get((label, col)), scale=100.0) for col in columns
        ]
        cells += [
            _fmt_cell(hit_grid.get((label, col)), scale=100.0) for col in columns
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # Speed table (rerank latency only; 'none' is 0 by definition).
    p50_grid = _grid(combinations, "rerank_ms_p50")
    p95_grid = _grid(combinations, "rerank_ms_p95")
    lines.append("## Speed: rerank latency (ms, p50 / p95)")
    lines.append("")
    lines.append("| dataset/retriever | " + " | ".join(columns) + " |")
    lines.append("|---" * (len(columns) + 1) + "|")
    for label in row_labels:
        cells = []
        for col in columns:
            p50 = p50_grid.get((label, col))
            p95 = p95_grid.get((label, col))
            cells.append("-" if p50 is None else f"{p50:.0f} / {p95:.0f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # Cost table.
    cost_grid = _grid(combinations, "est_cost_usd")
    nq_grid = _grid(combinations, "num_queries")
    lines.append("## Cost: estimated rerank cost ($ total / $ per 1k queries)")
    lines.append("")
    price_note = ", ".join(
        f"{model} ${pin}/${pout}"
        for model, (pin, pout) in LLM_PRICE_PER_MTOK.items()
    )
    lines.append(
        "Jev usage comes from `_jev_meta.tokens_in` at $0.042/MTok (output "
        "free); llm usage from `_llm_meta` priced per model, USD per MTok "
        f"(input/output): {price_note} — unlisted models count $0. "
        "bge runs locally ($0)."
    )
    lines.append("")
    lines.append("| dataset/retriever | " + " | ".join(columns) + " |")
    lines.append("|---" * (len(columns) + 1) + "|")
    for label in row_labels:
        cells = []
        for col in columns:
            cost = cost_grid.get((label, col))
            if cost is None:
                cells.append("-")
                continue
            per_1k = cost / max(1, (nq_grid.get((label, col)) or 1)) * 1000
            cells.append(f"${cost:.4f} / ${per_1k:.2f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # Reliability table.
    fail_grid = _grid(combinations, "rerank_failures")
    lines.append("## Reliability: rerank failures (fallback to original order)")
    lines.append("")
    lines.append("| dataset/retriever | " + " | ".join(columns) + " |")
    lines.append("|---" * (len(columns) + 1) + "|")
    for label in row_labels:
        cells = []
        for col in columns:
            failures = fail_grid.get((label, col))
            cells.append("-" if failures is None else str(int(failures)))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # Sample size warning + caveats.
    lines.append("## Notes & caveats")
    lines.append("")
    sample_note = (
        f"[WARN] Small sample: {args.num_queries} queries per dataset - "
        "differences under ~2 nDCG points are within noise at this size."
    )
    lines.append(f"- {sample_note}")
    lines.append(f"- {CORPUS_TRUNCATION_CAVEAT}")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")

    return "\n".join(lines)


def write_reports(
    combinations: List[Dict[str, Any]],
    notes: List[str],
    args: argparse.Namespace,
) -> Tuple[Path, Path]:
    """Write the markdown and JSON reports.

    Args:
        combinations: Aggregate records.
        notes: Caveat / warning lines.
        args: Parsed CLI arguments.

    Returns:
        Tuple of (markdown path, json path).
    """
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    md_path = outdir / f"rerank_benchmark_{stamp}.md"
    json_path = outdir / f"rerank_benchmark_{stamp}.json"

    md_path.write_text(
        render_markdown(combinations, args.rerankers, notes, args),
        encoding="utf-8",
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "datasets": list(args.datasets),
            "retrievers": list(args.retrievers),
            "rerankers": list(args.rerankers),
            "num_queries": args.num_queries,
            "seed": args.seed,
            "workers": args.workers,
            "retry_delays_s": list(RETRY_DELAYS_S),
            "top_k": TOP_K,
            "bge_model": BGE_MODEL,
            "jev_model": JEV_MODEL,
            "jev_min_score": args.jev_min_score,
        },
        "notes": notes,
        "results": combinations,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return md_path, json_path


def main() -> int:
    """Main entry point. Returns process exit code."""
    args = parse_args()

    print("[*] Rerank Benchmark Runner")
    print("=" * 60)
    print(f"datasets={args.datasets} retrievers={args.retrievers} "
          f"rerankers={args.rerankers} num_queries={args.num_queries} "
          f"seed={args.seed} workers={args.workers}")

    try:
        combinations, notes = run_benchmark(args)
    except Exception as e:
        print(f"[FAIL] Benchmark failed: {e}")
        return 2

    md_path, json_path = write_reports(combinations, notes, args)
    print("\n" + "=" * 60)
    print(f"[OK] Report: {md_path}")
    print(f"[OK] Raw data: {json_path}")

    best: Dict[Tuple[str, str], Tuple[float, str]] = {}
    for combo in combinations:
        key = (combo["dataset"], combo["retriever"])
        if key not in best or combo["ndcg_at_10"] > best[key][0]:
            best[key] = (combo["ndcg_at_10"], combo["reranker"])
    for (dataset, method), (score, rer_name) in sorted(best.items()):
        print(f"[BEST] {dataset}/{method}: {rer_name} (nDCG@10={score * 100:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
