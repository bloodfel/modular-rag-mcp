#!/usr/bin/env python
"""Show retrieval candidates for every golden-set query, for ground-truth labelling.

The golden set's ``expected_chunk_ids`` must be filled by *reading what each
candidate chunk actually says*, not by copying whatever the retriever ranked
first — copying the ranking would make every metric self-fulfilling.

This script only shows the candidates. Writing the labels is a judgement call.

Usage:
    .venv/bin/python scripts/label_golden_set.py
    .venv/bin/python scripts/label_golden_set.py --collection default --top-k 10
    .venv/bin/python scripts/label_golden_set.py --json > candidates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_GOLDEN = "tests/fixtures/golden_test_set.json"
PREVIEW_CHARS = 300


def _load_cases(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("test_cases", [])


def _candidates(search: Any, query: str, top_k: int) -> List[Dict[str, Any]]:
    results = search.search(query=query, top_k=top_k)
    results = results if isinstance(results, list) else results.results
    out = []
    for rank, r in enumerate(results, start=1):
        text = (r.text or "").strip().replace("\n", " ")
        out.append(
            {
                "rank": rank,
                "chunk_id": r.chunk_id,
                "score": round(float(r.score), 4),
                "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                "chunk_index": r.metadata.get("chunk_index"),
                "preview": text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else ""),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Show golden-set retrieval candidates.")
    parser.add_argument("--test-set", default=DEFAULT_GOLDEN)
    parser.add_argument("--collection", default="default")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cases = _load_cases(args.test_set)
    if not cases:
        print(f"[skip] no test cases in {args.test_set}", file=sys.stderr)
        return 1

    from src.core.query_engine.bootstrap import build_query_components
    from src.core.settings import load_settings

    settings = load_settings()
    search, _ = build_query_components(settings, args.collection)

    report = []
    for case in cases:
        query = case["query"]
        cands = _candidates(search, query, args.top_k)
        report.append({"query": query, "candidates": cands})

        if args.json:
            continue

        print("=" * 78)
        print(f"QUERY: {query}")
        print(f"  现有 expected_chunk_ids: {case.get('expected_chunk_ids') or '(空)'}")
        print("=" * 78)
        for c in cands:
            print(f"  #{c['rank']:<2} score={c['score']:<8} idx={c['chunk_index']}  {c['chunk_id']}")
            print(f"      source: {c['source']}")
            print(f"      text  : {c['preview']}")
            print()

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
