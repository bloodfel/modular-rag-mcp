#!/usr/bin/env python
"""CI quality gate: run the offline golden-set evaluation and enforce thresholds.

The gate is the regression lock for the RAG pipeline: it rebuilds nothing and
calls no LLM. It searches the golden questions against an already-ingested
deterministic collection (built with config/settings.eval.yaml), computes
hit_rate / MRR / nDCG@k against the labelled ground truth, and exits non-zero
when any metric falls below its threshold — which is what blocks the merge in
CI.

Ground truth is addressed portably: `expected_locations` entries are
(repo-relative source path, chunk_index) pairs, translated at runtime into the
current collection's chunk ids. Chunk ids themselves are machine-specific
(they hash the absolute source path and the post-transform text), which is
exactly why this translation exists.

Usage:
    # Locally, against the deterministic corpus
    .venv/bin/python scripts/quality_gate.py --collection project_docs_ci

    # Custom thresholds
    .venv/bin/python scripts/quality_gate.py --min-ndcg 0.5

Exit codes:
    0 - all metrics at or above threshold
    1 - gate FAILED (metrics below threshold, or ground truth unusable)
    2 - configuration error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def _relative(source: str) -> str:
    """Repo-relative path for a stored source_path, on any machine."""
    try:
        rel = Path(source).resolve().relative_to(_REPO_ROOT)
    except ValueError:
        rel = Path(source)
    return rel.as_posix()

if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Golden-set quality gate for CI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set.json",
        help="Golden set JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--collection",
        default="project_docs_ci",
        help="Deterministic collection to search (default: %(default)s)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval depth (default: 10)")
    parser.add_argument(
        "--min-hit-rate", type=float, default=1.0,
        help="Gate threshold for hit_rate (default: %(default)s)",
    )
    parser.add_argument(
        "--min-mrr", type=float, default=0.45,
        help="Gate threshold for MRR (default: %(default)s)",
    )
    parser.add_argument(
        "--min-ndcg", type=float, default=0.48,
        help="Gate threshold for nDCG@k (default: %(default)s)",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    return parser.parse_args()


def _load_collection_map(collection: str) -> Dict[Tuple[str, int], str]:
    """Map (repo-relative source, chunk_index) -> chunk id for the collection.

    Also indexes (file basename, chunk_index): stored source paths are
    absolute, so after the repo folder is renamed the exact relative match
    fails and the basename keeps the gate working without a re-ingest.
    """
    from src.core.settings import load_settings
    from src.libs.vector_store.chroma_store import ChromaStore

    store = ChromaStore(settings=load_settings(), collection_name=collection)
    data = store.client.get_collection(collection).get(include=["metadatas"])
    mapping: Dict[Tuple[str, int], str] = {}
    for cid, meta in zip(data["ids"], data["metadatas"]):
        rel = _relative(str(meta.get("source_path", "")))
        idx = meta.get("chunk_index")
        mapping[(rel, idx)] = cid
        mapping[(Path(rel).name, idx)] = cid
    return mapping


def main() -> int:
    args = parse_args()

    try:
        from src.core.settings import load_settings
        from src.core.query_engine.bootstrap import build_query_components
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        settings = load_settings("config/settings.eval.yaml")
        hybrid_search, _ = build_query_components(settings, args.collection)
    except Exception as exc:
        print(f"[FAIL] Configuration error: {exc}", file=sys.stderr)
        return 2

    cases = json.loads(Path(args.test_set).read_text(encoding="utf-8")).get("test_cases", [])
    if not cases:
        print("[FAIL] Golden set has no test cases.", file=sys.stderr)
        return 1

    location_map = _load_collection_map(args.collection)

    evaluator = CustomEvaluator(metrics=["hit_rate", "mrr", "ndcg"])
    rows: List[Dict[str, Any]] = []
    stale = 0

    for case in cases:
        query = case["query"]
        expected_ids: List[str] = []
        for loc in case.get("expected_locations", []):
            key = (loc["source"], loc["chunk_index"])
            cid = location_map.get(key)
            if cid is None:
                stale += 1
            else:
                expected_ids.append(cid)

        if not expected_ids:
            rows.append({"query": query, "error": "ground truth not found in collection"})
            continue

        search = hybrid_search.search(query=query, top_k=args.top_k)
        results = search.results if hasattr(search, "results") else search
        metrics = evaluator.evaluate(query, results, ground_truth=expected_ids)
        rows.append({"query": query, "metrics": metrics, "expected": len(expected_ids)})

    def _avg(name: str) -> float:
        values = [r["metrics"][name] for r in rows if "metrics" in r]
        return sum(values) / len(values) if values else 0.0

    aggregate = {"hit_rate": _avg("hit_rate"), "mrr": _avg("mrr"), "ndcg": _avg("ndcg")}
    thresholds = {
        "hit_rate": args.min_hit_rate,
        "mrr": args.min_mrr,
        "ndcg": args.min_ndcg,
    }
    failed = {
        name: (aggregate[name], thresholds[name])
        for name in aggregate
        if aggregate[name] < thresholds[name]
    }

    report = {
        "collection": args.collection,
        "test_set": args.test_set,
        "top_k": args.top_k,
        "cases": len(rows),
        "ground_truth_errors": sum(1 for r in rows if "error" in r),
        "stale_labels": stale,
        "aggregate": {k: round(v, 4) for k, v in aggregate.items()},
        "thresholds": thresholds,
        "gate": "PASS" if not failed and not rows_errors(rows) else "FAIL",
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("=" * 56)
        print("  QUALITY GATE —", report["gate"])
        print("=" * 56)
        for name in ("hit_rate", "mrr", "ndcg"):
            mark = "✅" if aggregate[name] >= thresholds[name] else "❌"
            print(f"  {mark} {name:<10} {aggregate[name]:.4f}  (≥ {thresholds[name]})")
        if stale:
            print(f"  ⚠️  {stale} ground-truth locations not found in collection")
        for r in rows:
            if "error" in r:
                print(f"  ⚠️  {r['query'][:40]}: {r['error']}")

    if report["gate"] == "FAIL":
        print("[FAIL] quality gate failed — merge blocked", file=sys.stderr)
        Path("reports/quality_gate_fail.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 1
    return 0


def rows_errors(rows: List[Dict[str, Any]]) -> int:
    return sum(1 for r in rows if "error" in r)


if __name__ == "__main__":
    sys.exit(main())
