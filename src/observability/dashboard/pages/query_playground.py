"""Query Playground – run retrieval queries interactively from the dashboard.

Debug/try-out entry point (the production entry points are the MCP tools
and the CLI script).  Every playground query goes through the same shared
component wiring (``build_query_components``) and writes a query trace, so
runs appear on the Query Traces page and can be exported to Langfuse.

UI controls mirror the CLI flags of ``scripts/query.py``:
    query text   -> --query
    collection   -> --collection
    top-k        -> --top-k
    rerank toggle-> --no-rerank (inverted)
    mid-process   -> --verbose (always available as an expander)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import streamlit as st

from src.core.settings import load_settings
from src.core.query_engine.bootstrap import build_query_components
from src.core.trace import TraceContext, TraceCollector
from src.observability.dashboard.services.data_service import DataService
from src.observability.langfuse_sink import langfuse_configured


def _list_collections() -> List[str]:
    """Return available collection names (never raises)."""
    try:
        return DataService().list_collections()
    except Exception:
        return ["default"]


def _result_rows(results: List[Any]) -> List[Dict[str, Any]]:
    """Convert RetrievalResult objects into display rows."""
    rows = []
    for idx, r in enumerate(results, start=1):
        meta = r.metadata or {}
        rows.append(
            {
                "rank": idx,
                "score": round(float(r.score), 4),
                "source": meta.get("source_path", ""),
                "chunk_index": meta.get("chunk_index", ""),
                "chunk_id": r.chunk_id,
                "text": (r.text or "").replace("\n", " ")[:160],
            }
        )
    return rows


def _show_stage_table(trace: TraceContext) -> None:
    stages = trace.stages or []
    if not stages:
        return
    rows = []
    for i, s in enumerate(stages):
        end = stages[i + 1]["timestamp"] if i + 1 < len(stages) else s.get("timestamp")
        rows.append(
            {
                "stage": s.get("stage", ""),
                "timestamp": s.get("timestamp", ""),
                "details": json.dumps(s.get("details", {}), ensure_ascii=False)[:120],
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _run_query(
    query: str,
    collection: str,
    top_k: int,
    use_rerank: bool,
    settings: Any,
) -> None:
    """Execute one query and render results + mid-process breakdown."""
    hybrid_search, reranker = build_query_components(settings, collection)

    trace = TraceContext(trace_type="query")
    trace.metadata["query"] = query[:200]
    trace.metadata["top_k"] = top_k
    trace.metadata["collection"] = collection

    with st.spinner("检索中…（dense + BM25 并行召回 → RRF 融合 → 重排）"):
        try:
            hybrid_result = hybrid_search.search(
                query=query,
                top_k=top_k,
                filters=None,
                trace=trace,
                return_details=True,
            )
        except Exception as e:
            st.error(f"检索失败：{e}")
            TraceCollector().collect(trace)
            return

    results = hybrid_result.results
    used_fallback = hybrid_result.used_fallback
    fallback_reason = (
        f"dense_error={hybrid_result.dense_error}, sparse_error={hybrid_result.sparse_error}"
    )

    rerank_fallback = None
    if use_rerank and reranker.is_enabled:
        try:
            rerank_result = reranker.rerank(
                query=query, results=results, top_k=top_k, trace=trace
            )
            results = rerank_result.results
            if rerank_result.used_fallback:
                rerank_fallback = (
                    f"{rerank_result.fallback_reason} (reranker={rerank_result.reranker_type})"
                )
        except Exception as e:
            rerank_fallback = str(e)
    elif not reranker.is_enabled:
        st.info("当前 settings.rerank.enabled=false，结果为 RRF 融合排序（未重排）。")

    # Local write plus the Langfuse send happen inside collect(); the result
    # is reported below rather than toggled per run.
    TraceCollector().collect(trace)

    if langfuse_configured():
        st.caption("✈️ 本条 trace 已上报 Langfuse（若发送失败会记入日志，查询结果不受影响）")

    # ---- results ----
    if not results:
        st.warning("未找到相关文档。先在 Ingestion Manager 上传文件，或检查集合选择。")
        return

    st.subheader(f"检索结果（{len(results)} 条）")
    st.dataframe(_result_rows(results), use_container_width=True, hide_index=True)

    for idx, r in enumerate(results, start=1):
        meta = r.metadata or {}
        label = f"#{idx} · score={r.score:.4f} · {meta.get('source_path', '')}"
        with st.expander(label):
            st.markdown(r.text or "（空）")
            meta_clean = {k: v for k, v in meta.items() if not k.startswith("_")}
            st.caption(json.dumps(meta_clean, ensure_ascii=False, default=str))

    # ---- mid-process (--verbose equivalent) ----
    with st.expander("中间过程（dense / sparse / fusion、降级与重排状态）"):
        if used_fallback:
            st.warning(f"HybridSearch 触发降级：{fallback_reason}")
        if rerank_fallback:
            st.warning(f"重排失败已回退原序：{rerank_fallback}")
        if hybrid_result.processed_query:
            st.caption(
                f"关键词提取：{hybrid_result.processed_query.keywords or '（无）'} · "
                f"filters={hybrid_result.processed_query.filters or '（无）'}"
            )
        st.markdown("**Dense 召回**")
        st.dataframe(_result_rows(hybrid_result.dense_results or []), hide_index=True)
        st.markdown("**Sparse (BM25) 召回**")
        st.dataframe(_result_rows(hybrid_result.sparse_results or []), hide_index=True)
        st.markdown("**RRF 融合（重排前）**")
        st.dataframe(_result_rows(hybrid_result.results), hide_index=True)
        st.markdown("**阶段时间线（本条 trace）**")
        _show_stage_table(trace)

    st.caption(
        f"本次查询已写入 trace（`{trace.trace_id[:12]}…`）——"
        "每条 trace 自动写入 logs/traces.jsonl 并上报 Langfuse（未配置凭证时只写本地）。"
    )


def render() -> None:
    st.header("🧪 Query Playground")
    st.caption(
        "调试试用入口：与 CLI / MCP 共用同一套检索组件，"
        "每条查询自动写入 trace。生产接入请用 MCP（stdio / HTTP）。"
    )

    settings = load_settings()
    collections = _list_collections()
    default_idx = collections.index("default") if "default" in collections else 0

    col_q, col_c = st.columns([3, 1])
    with col_q:
        query = st.text_input(
            "Query（对应 --query）", value="", key="pg_query",
            placeholder="例如：远程接入指南里怎么上传文档？"
        )
    with col_c:
        collection = st.selectbox(
            "Collection（对应 --collection）", collections or ["default"],
            index=default_idx, key="pg_collection",
        )

    col_k, col_r = st.columns([1, 3])
    with col_k:
        top_k = st.slider("Top-K（对应 --top-k）", min_value=1, max_value=20, value=5, key="pg_top_k")
    with col_r:
        use_rerank = st.toggle(
            "启用重排（关闭 = --no-rerank，直接用 RRF 融合序）",
            value=bool(getattr(settings.rerank, "enabled", False)),
            key="pg_rerank",
        )
        if getattr(settings.rerank, "provider", "") == "llm" and use_rerank:
            st.warning("当前重排后端是聊天模型（llm），单次查询可能需要 ~40s。")

    if not langfuse_configured():
        st.info(
            "未配置 Langfuse（.env 里缺 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY），"
            "trace 只写入 logs/traces.jsonl，Streamlit 页面照常可看。"
        )

    if st.button("🔍 查询", type="primary", disabled=not query.strip(), key="pg_run"):
        _run_query(query.strip(), collection, top_k, use_rerank, settings)
