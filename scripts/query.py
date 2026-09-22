#!/usr/bin/env python
"""Query script for the Modular RAG MCP Server.

This script provides a command-line interface for querying the knowledge hub
using HybridSearch (Dense + Sparse + RRF) with optional reranking.

Usage:
    # Run a single query
    python scripts/query.py --query "Azure OpenAI 配置步骤" --collection technical_docs

    # Verbose mode (show dense/sparse/fusion/rerank results)
    python scripts/query.py --query "RRF 是什么" --verbose

    # Disable reranking
    python scripts/query.py --query "RRF 是什么" --no-rerank

Exit codes:
    0 - Success
    1 - Query failure
    2 - Configuration error
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project root is in path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import load_settings
from src.core.trace import TraceContext, TraceCollector
from src.observability.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query documents from the Modular RAG knowledge hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Query string."
    )

    parser.add_argument(
        "--collection", "-c",
        default="default",
        help="Collection name (default: 'default')"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Max number of results (default: 10)"
    )

    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "settings.yaml"),
        help="Path to configuration file (default: config/settings.yaml)"
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking even if enabled in settings"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate results (dense/sparse/fusion/rerank)"
    )

    return parser.parse_args()


def _format_filters(filters: Dict[str, Any]) -> str:
    if not filters:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in filters.items())


def _print_results(results: List[Any], top_k: int, title: str = "RESULTS") -> None:
    print("\n" + "=" * 60)
    print(f"{title} (top_k={top_k}, returned={len(results)})")
    print("=" * 60)

    for idx, result in enumerate(results, start=1):
        metadata = result.metadata or {}
        source_path = metadata.get("source_path", "")
        chunk_index = metadata.get("chunk_index", "")
        page_num = metadata.get("page_num", "")
        snippet = (result.text or "").replace("\n", " ")[:200]

        print(f"#{idx:02d}  score={result.score:.4f}  id={result.chunk_id}")
        print(f"     source_path={source_path}")
        if chunk_index != "":
            print(f"     chunk_index={chunk_index}")
        if page_num != "":
            print(f"     page_num={page_num}")
        print(f"     text={snippet}...")

    print("=" * 60)


def _build_components(settings, collection: str):
    """Delegate to the shared query-component wiring (bootstrap)."""
    from src.core.query_engine.bootstrap import build_query_components

    return build_query_components(settings, collection)


def _run_query(
    hybrid_search,
    reranker,
    query: str,
    top_k: Optional[int],
    use_rerank: bool,
    verbose: bool,
    collection: str = "default",
) -> int:
    trace = TraceContext(trace_type="query")
    trace.metadata["query"] = query[:200]
    trace.metadata["top_k"] = top_k
    trace.metadata["collection"] = collection

    try:
        hybrid_result = hybrid_search.search(
            query=query,
            top_k=top_k,
            filters=None,
            trace=trace,
            return_details=verbose,
        )
    except Exception as e:
        print(f"[FAIL] Hybrid search failed: {e}")
        TraceCollector().collect(trace)
        return 1

    if verbose:
        results = hybrid_result.results
        if hybrid_result.used_fallback:
            print(
                f"[WARN] HybridSearch fallback used. "
                f"dense_error={hybrid_result.dense_error}, "
                f"sparse_error={hybrid_result.sparse_error}"
            )
        if hybrid_result.processed_query:
            print(
                f"[INFO] ProcessedQuery keywords={hybrid_result.processed_query.keywords} "
                f"filters={_format_filters(hybrid_result.processed_query.filters)}"
            )
        _print_results(hybrid_result.dense_results or [], top_k=top_k, title="DENSE RESULTS")
        _print_results(hybrid_result.sparse_results or [], top_k=top_k, title="SPARSE RESULTS")
        _print_results(hybrid_result.results, top_k=top_k, title="FUSION RESULTS")
    else:
        results = hybrid_result

    effective_top_k = top_k if top_k is not None else len(results)

    if not results:
        print("[INFO] 未找到相关文档，请先运行 ingest.py 摄取数据。")
        return 0

    # Optional reranking
    if use_rerank and reranker.is_enabled:
        try:
            rerank_result = reranker.rerank(query=query, results=results, top_k=top_k, trace=trace)
            results = rerank_result.results
            if verbose and rerank_result.used_fallback:
                print(
                    f"[WARN] Rerank fallback used: {rerank_result.fallback_reason} "
                    f"(reranker={rerank_result.reranker_type})"
                )
            if verbose:
                _print_results(results, top_k=top_k, title="RERANK RESULTS")
        except Exception as e:
            print(f"[WARN] Reranking failed: {e}. Using original order.")
    elif verbose and not reranker.is_enabled:
        print("[INFO] Reranking disabled by settings.")

    _print_results(results, top_k=effective_top_k)
    # Local write plus the Langfuse send happen inside collect().
    TraceCollector().collect(trace)
    return 0


def main() -> int:
    args = parse_args()

    # Load configuration
    try:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[FAIL] Configuration file not found: {config_path}")
            return 2
        settings = load_settings(str(config_path))
        print(f"[OK] Configuration loaded from: {config_path}")
    except Exception as e:
        print(f"[FAIL] Failed to load configuration: {e}")
        return 2

    print("[*] Modular RAG Query Script")
    print("=" * 60)
    print(f"Collection: {args.collection}")

    try:
        hybrid_search, reranker = _build_components(settings, args.collection)
    except Exception as e:
        print(f"[FAIL] Failed to initialize query components: {e}")
        logger.exception("Query initialization failed")
        return 2

    use_rerank = not args.no_rerank

    # Single-query mode
    return _run_query(
        hybrid_search=hybrid_search,
        reranker=reranker,
        query=args.query,
        top_k=args.top_k,
        use_rerank=use_rerank,
        verbose=args.verbose,
        collection=args.collection,
    )


if __name__ == "__main__":
    sys.exit(main())
