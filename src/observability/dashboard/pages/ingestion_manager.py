"""Ingestion Manager page – upload files, trigger ingestion, delete documents.

Layout:
1. File uploader + collection selector
2. Ingest button → progress bar (using on_progress callback)
3. Document list with delete buttons
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def _run_ingestion(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
    force: bool = False,
) -> None:
    """Save the uploaded file to a temp location and run the pipeline."""
    from src.core.settings import load_settings
    from src.core.trace import TraceContext, TraceCollector
    from src.ingestion.pipeline import IngestionPipeline

    settings = load_settings()

    # Write uploaded file to a temp location
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    _STAGE_LABELS = {
        "integrity": "🔍 Checking file integrity…",
        "load": "📄 Loading document…",
        "split": "✂️ Chunking document…",
        "transform": "🔄 Transforming chunks (LLM refine + enrich)…",
        "embed": "🔢 Encoding vectors…",
        "upsert": "💾 Storing to database…",
    }

    def on_progress(stage: str, current: int, total: int) -> None:
        frac = (current - 1) / total  # stage just started, show partial progress
        label = _STAGE_LABELS.get(stage, stage)
        progress_bar.progress(frac, text=f"[{current}/{total}] {label}")
        status_text.caption(label)

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["source_path"] = uploaded_file.name
    trace.metadata["collection"] = collection
    trace.metadata["source"] = "dashboard"

    try:
        pipeline = IngestionPipeline(settings, collection=collection, force=force)
        pipeline.run(
            file_path=tmp_path,
            trace=trace,
            on_progress=on_progress,
        )
        progress_bar.progress(1.0, text="✅ Complete")
        if force:
            status_text.success(
                f"Successfully re-ingested **{uploaded_file.name}** into **{collection}**."
            )
        else:
            status_text.success(f"Successfully ingested **{uploaded_file.name}** into collection **{collection}**.")
    except Exception as exc:
        status_text.error(f"Ingestion failed: {exc}")
    finally:
        TraceCollector().collect(trace)
        # Clean up temp file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def render() -> None:
    """Render the Ingestion Manager page."""
    st.header("📥 Ingestion Manager")

    svc = None
    existing_collections: List[str] = ["default"]
    try:
        svc = DataService()
        existing_collections = svc.list_collections() or ["default"]
    except Exception as exc:
        st.error(f"Failed to initialise DataService: {exc}")
        return

    # ── Upload section ─────────────────────────────────────────────
    st.subheader("📤 Upload & Ingest")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader(
            "Select a file to ingest",
            type=["pdf", "txt", "md", "docx"],
            key="ingest_uploader",
        )
    with col2:
        create_label = "➕ 新建集合…"
        options = existing_collections + [create_label]
        default_idx = options.index("default") if "default" in options else 0
        choice = st.selectbox("Collection", options, index=default_idx, key="ingest_collection_choice")
        if choice == create_label:
            collection = st.text_input("新集合名称", value="", key="ingest_collection_new")
        else:
            collection = choice

    if uploaded is not None:
        force = st.checkbox(
            "强制重新摄取（force）—— 文件内容没变时，重复上传会被跳过；勾选此项强制重跑",
            value=False,
            key="ingest_force",
        )
        if st.button("🚀 Start Ingestion", key="btn_ingest"):
            progress_bar = st.progress(0, text="Preparing…")
            status_text = st.empty()
            _run_ingestion(
                uploaded,
                collection.strip() or "default",
                progress_bar,
                status_text,
                force=force,
            )

    st.divider()

    # ── Collection management section ─────────────────────────────
    st.subheader("📁 Collections")

    col_new, col_del = st.columns(2)
    with col_new:
        new_name = st.text_input("新建空集合", value="", key="new_collection_name")
        if st.button("➕ 创建", key="btn_create_collection", disabled=not new_name.strip()):
            try:
                svc.create_collection(new_name)
                st.success(f"Collection '{new_name.strip()}' created.")
                st.rerun()
            except Exception as exc:
                st.error(f"Create failed: {exc}")

    with col_del:
        deletable = [c for c in existing_collections if c != "default"]
        if not deletable:
            st.caption("没有可删除的集合（'default' 受保护）。")
        else:
            del_name = st.selectbox("删除集合", deletable, key="delete_collection_choice")
            confirm = st.checkbox(
                f"我确认删除 '{del_name}' 及其全部文档（不可恢复）", key="delete_collection_confirm"
            )
            if st.button("🗑️ 删除", key="btn_delete_collection", disabled=not confirm):
                try:
                    svc.delete_collection(del_name)
                    st.success(f"Collection '{del_name}' deleted.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")

    st.divider()

    # ── Document management section ────────────────────────────────
    st.subheader("🗑️ Manage Documents")

    try:
        docs = svc.list_documents()
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    if not docs:
        st.info(
            "**No documents ingested yet.** "
            "Upload a PDF, TXT, MD, or DOCX file above and click \"Start Ingestion\" to begin."
        )
        return

    for idx, doc in enumerate(docs):
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"**{doc['source_path']}** — "
                f"collection: `{doc.get('collection', '—')}` | "
                f"chunks: {doc['chunk_count']} | "
                f"images: {doc['image_count']}"
            )
        with col_btn:
            if st.button("🗑️ Delete", key=f"del_{idx}"):
                try:
                    result = svc.delete_document(
                        source_path=doc["source_path"],
                        collection=doc.get("collection", "default"),
                        source_hash=doc.get("source_hash"),
                    )
                    if result.success:
                        st.success(
                            f"Deleted: {result.chunks_deleted} chunks, "
                            f"{result.images_deleted} images removed."
                        )
                        st.rerun()
                    else:
                        st.warning(f"Partial delete. Errors: {result.errors}")
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
