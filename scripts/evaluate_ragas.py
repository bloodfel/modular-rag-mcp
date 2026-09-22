#!/usr/bin/env python
"""Run Ragas LLM-as-Judge metrics over the golden set, report scores to Langfuse.

Per golden case this script:
  1. retrieves the top-k chunks with the project's hybrid search,
  2. generates an answer from that context with the configured LLM,
  3. scores the run with Ragas — faithfulness, answer_relevancy,
     context_precision, context_recall,
  4. writes each score back onto that query's Langfuse trace (``ragas.*``),
     so the numbers appear next to the trace that produced them.

Usage:
    # Full golden set against the deterministic corpus
    .venv/bin/python scripts/evaluate_ragas.py --collection project_docs_ci

    # Quick smoke over the first three cases, no score write-back
    .venv/bin/python scripts/evaluate_ragas.py --limit 3 --no-langfuse

    # Machine-readable output
    .venv/bin/python scripts/evaluate_ragas.py --json

Exit codes:
    0 - success (scores may still be partial if single cases failed)
    1 - the run failed entirely
    2 - configuration error
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

ANSWER_PROMPT = """你是知识库问答助手。只依据下面的「检索到的资料」回答用户问题，不要引入资料之外的信息。

要求：
1. 直接回答问题，简洁准确；
2. 资料不足以回答时，明确说明"资料中没有足够信息"；
3. 不要输出引用编号、Markdown 标题等格式装饰。

检索到的资料：
{context}

用户问题：{query}

回答："""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ragas metrics over the golden set and report to Langfuse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set.json",
        help="Path to the golden set JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--collection",
        default="project_docs_ci",
        help="Collection to search (default: %(default)s)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Chunks per query (default: 5)")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N cases (0 = all)")
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Skip score write-back (traces are still reported by TraceCollector)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    parser.add_argument(
        "--outdir", default="reports", help="Where to write the report (default: %(default)s)"
    )
    return parser.parse_args()


def _load_cases(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("test_cases", [])


def _context_text(results: List[Any], top_k: int) -> str:
    blocks = []
    for rank, item in enumerate(results[:top_k], start=1):
        text = getattr(item, "text", None) or (
            item.get("text") if isinstance(item, dict) else str(item)
        )
        source = (getattr(item, "metadata", None) or {}).get("source_path", "")
        blocks.append(f"[{rank}] {Path(source).name}\n{text}")
    return "\n\n".join(blocks)


def _generate_answer(llm: Any, query: str, results: List[Any], top_k: int, trace: Any, label: str) -> str:
    from src.libs.llm.base_llm import Message

    messages = [
        Message(
            role="user",
            content=ANSWER_PROMPT.format(context=_context_text(results, top_k), query=query),
        )
    ]
    response = llm.chat(messages, trace=trace, purpose="answer_generation", label=label)
    return (getattr(response, "content", None) or "").strip()


def main() -> int:
    args = parse_args()

    try:
        from src.core.settings import load_settings
        from src.core.trace import TraceCollector, TraceContext
        from src.core.query_engine.bootstrap import build_query_components
        from src.libs.llm.llm_factory import LLMFactory
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        settings = load_settings()
        hybrid_search, _ = build_query_components(settings, args.collection)
        llm = LLMFactory.create(settings)
        evaluator = RagasEvaluator(settings=settings, metrics=METRICS)
    except Exception as exc:
        print(f"[FAIL] Configuration error: {exc}", file=sys.stderr)
        return 2

    cases = _load_cases(args.test_set)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("[FAIL] Golden set has no test cases.", file=sys.stderr)
        return 2

    from src.observability.langfuse_sink import send_scores

    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        query = case["query"]
        trace = TraceContext(trace_type="query")
        trace.metadata["query"] = query[:200]
        trace.metadata["collection"] = args.collection
        trace.metadata["eval"] = "ragas"

        row: Dict[str, Any] = {"query": query, "scores": {}, "error": None}
        try:
            search_result = hybrid_search.search(query=query, top_k=args.top_k, trace=trace)
            results = (
                search_result.results
                if hasattr(search_result, "results")
                else search_result
            )
            answer = _generate_answer(
                llm, query, results, args.top_k, trace, label=f"case{index}"
            )
            scores = evaluator.evaluate(
                query,
                results,
                generated_answer=answer,
                ground_truth=case.get("reference_answer"),
            )
            row["scores"] = scores
            row["answer"] = answer
            row["retrieved"] = len(results)
        except Exception as exc:  # keep going: one bad case must not kill the batch
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            trace.finish()
            TraceCollector().collect(trace)
            row["trace_id"] = trace.trace_id

        if row["scores"] and not args.no_langfuse:
            try:
                row["scores_written"] = send_scores(trace.trace_id, row["scores"])
            except Exception as exc:  # noqa: BLE001 - write-back is best effort
                row["scores_written"] = 0
                row["error"] = (row["error"] or "") + f" [score write: {exc}]"

        status = "ok" if row["scores"] else "FAIL"
        summary = " ".join(f"{k}={v:.2f}" for k, v in row["scores"].items())
        print(f"[{index}/{len(cases)}] {status} {query[:40]} {summary} {row['error'] or ''}")
        rows.append(row)

    aggregates = {
        metric: mean([r["scores"][metric] for r in rows if metric in r["scores"]])
        for metric in METRICS
        if any(metric in r["scores"] for r in rows)
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": args.collection,
        "test_set": args.test_set,
        "top_k": args.top_k,
        "cases": len(rows),
        "failed": sum(1 for r in rows if not r["scores"]),
        "aggregate": {k: round(v, 4) for k, v in aggregates.items()},
        "results": rows,
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("\n" + "=" * 60)
        print("  RAGAS EVALUATION")
        print("=" * 60)
        for metric, value in aggregates.items():
            print(f"  {metric:<20} {value:.4f}")
        print(f"  cases: {len(rows)} ({report['failed']} failed)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    json_path = outdir / f"ragas_eval_{stamp}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] report: {json_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
