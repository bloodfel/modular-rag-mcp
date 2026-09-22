#!/usr/bin/env python
"""Draft a Ragas golden set by asking the LLM for one QA pair per corpus chunk.

Each pair keeps its source chunk (id, file, index) as provenance, so a
reviewer can check every question against the text that supposedly answers
it. The drafting is machine work; verification is a human judgement call —
reject any pair whose chunk does not actually support the reference answer.

Usage:
    # 60 pairs drawn from the deterministic corpus
    .venv/bin/python scripts/generate_golden_set.py --num 60

    # A different slice (same seed = same sample)
    .venv/bin/python scripts/generate_golden_set.py --num 60 --seed 7

Exit codes:
    0 - success (individual draft failures are reported, not fatal)
    1 - the run failed entirely
    2 - configuration error
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

DRAFT_PROMPT = """你是评测数据标注员。根据下面这段资料，写两样东西：

1. 一个只能依靠该资料回答的问题（不要问需要资料之外知识的问题，也不要问"这段资料讲了什么"这类元问题）；
2. 一个简洁的参考答案（只依据资料，1-3 句，不要展开）。

用资料本身的语言书写。只输出 JSON，不要输出其它任何内容：
{{"question": "...", "reference_answer": "..."}}

资料（来源：{source}）：
{text}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draft a Ragas golden set from corpus chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--collection", default="project_docs_ci", help="Source collection")
    parser.add_argument("--num", type=int, default=60, help="How many pairs to draft")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--min-chars", type=int, default=200, help="Skip chunks shorter than this")
    parser.add_argument(
        "--max-source-share",
        type=float,
        default=0.6,
        help="At most this share of pairs may come from one source file",
    )
    parser.add_argument(
        "--out",
        default="tests/fixtures/golden_ragas_set.json",
        help="Where to write the draft set (default: %(default)s)",
    )
    return parser.parse_args()


def _sample_chunks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from src.core.settings import load_settings
    from src.libs.vector_store.chroma_store import ChromaStore

    store = ChromaStore(settings=load_settings(), collection_name=args.collection)
    data = store.client.get_collection(args.collection).get(include=["documents", "metadatas"])

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        if len(text) < args.min_chars:
            continue
        source = str(meta.get("source_path", ""))
        by_source.setdefault(source, []).append(
            {"id": cid, "text": text, "source": source, "chunk_index": meta.get("chunk_index")}
        )

    rng = random.Random(args.seed)
    for chunks in by_source.values():
        rng.shuffle(chunks)

    cap = max(1, int(args.num * args.max_source_share))
    picked: List[Dict[str, Any]] = []
    # Round-robin across sources so no single file dominates the set.
    sources = sorted(by_source)
    offset = 0
    while len(picked) < args.num:
        added = False
        for source in sources:
            if len(picked) >= args.num:
                break
            chunks = by_source[source]
            taken = sum(1 for p in picked if p["source"] == source)
            if offset < len(chunks) and taken < cap:
                picked.append(chunks[offset])
                added = True
        if not added:
            break
        offset += 1
    return picked


def _draft_pair(llm: Any, chunk: Dict[str, Any]) -> Dict[str, Any]:
    from src.libs.llm.base_llm import Message

    prompt = DRAFT_PROMPT.format(source=Path(chunk["source"]).name, text=chunk["text"])
    response = llm.chat(
        [Message(role="user", content=prompt)],
        purpose="golden_set_draft",
        label=chunk["id"],
    )
    raw = (getattr(response, "content", None) or "").strip()
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"no JSON in draft response: {raw[:120]}")
    parsed = json.loads(match.group(0))
    question = str(parsed.get("question", "")).strip()
    reference = str(parsed.get("reference_answer", "")).strip()
    if not question or not reference:
        raise ValueError("draft is missing question or reference_answer")
    return {"question": question, "reference_answer": reference}


def main() -> int:
    args = parse_args()

    try:
        from src.core.settings import load_settings
        from src.libs.llm.llm_factory import LLMFactory

        llm = LLMFactory.create(load_settings())
        chunks = _sample_chunks(args)
    except Exception as exc:
        print(f"[FAIL] Configuration error: {exc}", file=sys.stderr)
        return 2

    if not chunks:
        print("[FAIL] No chunks selected — check collection and --min-chars.", file=sys.stderr)
        return 2

    cases: List[Dict[str, Any]] = []
    failures = 0
    for index, chunk in enumerate(chunks, start=1):
        try:
            pair = _draft_pair(llm, chunk)
            cases.append(
                {
                    "query": pair["question"],
                    "reference_answer": pair["reference_answer"],
                    "expected_chunk_ids": [],
                    "expected_sources": [Path(chunk["source"]).name],
                    "provenance": {
                        "chunk_id": chunk["id"],
                        "source_path": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                    },
                }
            )
        except Exception as exc:
            failures += 1
            print(f"[{index}/{len(chunks)}] draft failed ({chunk['id']}): {exc}")
        else:
            print(f"[{index}/{len(chunks)}] {cases[-1]['query'][:50]}")

    payload = {
        "description": (
            "Ragas golden set drafted by an LLM from corpus chunks, one QA pair per chunk. "
            "provenance.chunk_id points at the chunk the pair was drafted from — verify each "
            "pair against that text before trusting the metrics."
        ),
        "version": "1.0-draft",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": args.collection,
        "seed": args.seed,
        "test_cases": cases,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ok] drafted {len(cases)} pairs ({failures} failed) -> {out_path}")
    return 0 if cases else 1


if __name__ == "__main__":
    sys.exit(main())
