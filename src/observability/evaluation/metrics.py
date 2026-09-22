"""Retrieval evaluation metrics: nDCG@k, MRR@k, Hit@k.

Pure, side-effect-free functions used as the scoring basis for the Stage J
rerank benchmark (BEIR scifact / nfcorpus) and reusable by any retrieval
evaluation. No I/O, no external dependencies.

Conventions (per spec docs/specs/2026-09-21-jev-rerank-benchmark-design.md):

- ``qrels`` maps ``doc_id -> relevance`` (graded relevance).
- Relevance ``> 0`` counts as relevant; ``<= 0`` or missing means non-relevant.
- All functions return ``0.0`` when ``qrels`` is empty.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

__all__ = ["ndcg_at_k", "mrr_at_k", "hit_at_k"]


def _dcg(gains: Sequence[float]) -> float:
    """Discounted Cumulative Gain from an ordered gain sequence (rank 1 first).

    Each gain is discounted by ``1 / log2(rank + 1)``.
    """
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranked_ids: Sequence[str], qrels: Dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k for a single query.

    Uses the graded gain ``2**rel - 1`` and position discount
    ``1 / log2(rank + 1)``. Only the top ``k`` of ``ranked_ids`` contribute.
    IDCG is computed from the ideal ordering of this query's ``qrels``
    (relevance values sorted descending, truncated to ``k``).

    Returns 0.0 when ``qrels`` is empty (or contains no relevant docs).
    """
    if not qrels or k <= 0:
        return 0.0

    ideal_gains = sorted((2**rel - 1 for rel in qrels.values()), reverse=True)[:k]
    ideal_dcg = _dcg(ideal_gains)
    if ideal_dcg <= 0:
        return 0.0

    gains = [2 ** qrels.get(doc_id, 0) - 1 for doc_id in ranked_ids[:k]]
    return _dcg(gains) / ideal_dcg


def mrr_at_k(ranked_ids: Sequence[str], qrels: Dict[str, int], k: int) -> float:
    """Reciprocal Rank at k for a single query (first relevant rank).

    Returns ``1 / rank`` of the first doc within top ``k`` whose qrels
    relevance is ``> 0``; 0.0 if no relevant doc appears within top ``k``
    or ``qrels`` is empty.
    """
    if not qrels or k <= 0:
        return 0.0

    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def hit_at_k(ranked_ids: Sequence[str], qrels: Dict[str, int], k: int) -> float:
    """Hit Rate at k for a single query.

    Returns 1.0 if any of the top ``k`` docs has qrels relevance ``> 0``,
    otherwise 0.0 (also for empty ``qrels``).
    """
    if not qrels or k <= 0:
        return 0.0

    if any(qrels.get(doc_id, 0) > 0 for doc_id in ranked_ids[:k]):
        return 1.0
    return 0.0
