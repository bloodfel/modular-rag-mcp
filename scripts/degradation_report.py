#!/usr/bin/env python
"""Degradation report from logs/traces.jsonl.

Counts how often the pipeline silently fell back because a dependency failed.
Without this the fallbacks are invisible: a query served by BM25 alone looks
identical to a healthy one unless you read every trace.

Usage:
    .venv/bin/python scripts/degradation_report.py
    .venv/bin/python scripts/degradation_report.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observability.dashboard.services.trace_service import TraceService


def _stages(trace: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    return [s for s in trace.get("stages", []) if s.get("stage") == name]


def analyse(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count degradation events and empty-result queries across *traces*."""
    query_traces = [t for t in traces if t.get("trace_type") == "query"]

    retrieval_fallback = 0
    failed_paths: Counter = Counter()
    rerank_fallback = 0
    rerank_reasons: Counter = Counter()
    empty_results = 0

    for trace in query_traces:
        for stage in _stages(trace, "retrieval_fallback"):
            retrieval_fallback += 1
            data = stage.get("data") or {}
            if data.get("dense_failed"):
                failed_paths["dense"] += 1
            if data.get("sparse_failed"):
                failed_paths["sparse"] += 1

        for stage in _stages(trace, "rerank"):
            data = stage.get("data") or {}
            if data.get("fallback"):
                rerank_fallback += 1
                rerank_reasons[str(data.get("fallback_reason") or "unknown")[:80]] += 1

        fusion = _stages(trace, "fusion")
        if fusion and not (fusion[-1].get("data") or {}).get("result_count"):
            empty_results += 1

    total = len(query_traces)
    return {
        "traces_total": len(traces),
        "query_traces": total,
        "retrieval_fallback": retrieval_fallback,
        "retrieval_fallback_rate": retrieval_fallback / total if total else 0.0,
        "failed_paths": dict(failed_paths),
        "rerank_fallback": rerank_fallback,
        "rerank_fallback_rate": rerank_fallback / total if total else 0.0,
        "rerank_reasons": dict(rerank_reasons.most_common(5)),
        "empty_result_queries": empty_results,
    }


def _print(report: Dict[str, Any]) -> None:
    total = report["query_traces"]
    print("=" * 62)
    print("  DEGRADATION REPORT")
    print("=" * 62)
    print(f"  Traces scanned: {report['traces_total']}  (query: {total})")
    print()

    def line(label: str, count: int, rate: float) -> None:
        share = f"{rate:6.1%}" if total else "   n/a"
        print(f"  {label:<26} {count:>4} 次   {share}")

    line("检索降级（一路挂掉）", report["retrieval_fallback"], report["retrieval_fallback_rate"])
    for path, count in report["failed_paths"].items():
        print(f"      └─ {path} 路径挂掉: {count}")
    line("重排降级（回退融合序）", report["rerank_fallback"], report["rerank_fallback_rate"])
    for reason, count in report["rerank_reasons"].items():
        print(f"      └─ {reason}: {count}")
    print(f"  {'0 结果查询':<26} {report['empty_result_queries']:>4} 次")
    print()

    if total and not report["retrieval_fallback"] and not report["rerank_fallback"]:
        print("  结论：窗口内没有发生降级（不是没统计，是真的没挂）。")
    elif not total:
        print("  结论：窗口内没有查询 trace，无法判断。")
    print("=" * 62)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report degradation events from traces.jsonl.")
    parser.add_argument("--json", action="store_true", help="Output the report as JSON")
    parser.add_argument("--limit", type=int, default=1000, help="Max traces to scan")
    args = parser.parse_args()

    traces = TraceService().list_traces(limit=args.limit)
    report = analyse(traces)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
