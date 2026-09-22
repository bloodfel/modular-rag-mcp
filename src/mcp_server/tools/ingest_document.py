"""MCP Tool: ingest_document

Remote ingestion entry point: lets an MCP client add a text/markdown
document to the knowledge base over the same connection it queries with
(no local CLI / dashboard access required).  Typical remote flow:

    1. ingest_document(name="handbook", content="# ...", collection="docs")
    2. query_knowledge_hub(query="...", collection="docs")

Text documents bypass the PDF loader and LLM transform stages; they go
through the same chunk -> embed -> upsert path as file ingestion (chunk
sizes from settings.yaml).  Re-ingesting a document under the same name
replaces its previous chunks in both Chroma and the BM25 index.

PDF / image-heavy documents should still be ingested server-side via
``scripts/ingest.py`` or the dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, Optional, TYPE_CHECKING

from mcp import types

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)

TOOL_NAME = "ingest_document"
TOOL_DESCRIPTION = """Add a text/markdown document to the knowledge base.

The document is chunked, embedded and indexed (vector + BM25) so it is
immediately searchable via query_knowledge_hub. Re-ingesting the same
name replaces the previous version.

Use this for text/markdown content. PDFs should be ingested server-side
(scripts/ingest.py)."""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Document name used as its identifier and title. "
            "Re-ingesting the same name replaces the old version.",
        },
        "content": {
            "type": "string",
            "description": "Full text or markdown content of the document.",
        },
        "collection": {
            "type": "string",
            "description": "Target collection. Defaults to 'default'.",
            "default": "default",
        },
    },
    "required": ["name", "content"],
}

# Chars that would corrupt chunk-id prefixes / source_path values.
_UNSAFE_ID_CHARS = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._-]+")


def sanitize_doc_name(name: str) -> str:
    """Reduce a document name to a safe identifier fragment."""
    cleaned = _UNSAFE_ID_CHARS.sub("-", name.strip())
    return cleaned.strip("-") or "untitled"


class IngestDocumentTool:
    """MCP Tool for remote text-document ingestion."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            from src.core.settings import load_settings
            self._settings = load_settings()
        return self._settings

    def ingest(
        self,
        name: str,
        content: str,
        collection: str = "default",
    ) -> str:
        """Chunk, embed and index one text document (blocking; run in a thread).

        Args:
            name: Document identifier / title.
            content: Text or markdown body.
            collection: Target collection name.

        Returns:
            Human-readable summary of what was indexed.
        """
        from src.core.settings import resolve_path
        from src.core.types import Document
        from src.ingestion.chunking.document_chunker import DocumentChunker
        from src.ingestion.embedding.dense_encoder import DenseEncoder
        from src.ingestion.embedding.sparse_encoder import SparseEncoder
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.ingestion.storage.vector_upserter import VectorUpserter
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.chroma_store import ChromaStore

        collection = collection.strip() or "default"
        safe_name = sanitize_doc_name(name)
        doc_id = f"mcp-{collection}-{safe_name}"
        source_path = f"mcp://{collection}/{safe_name}"

        document = Document(
            id=doc_id,
            text=content,
            metadata={
                "source_path": source_path,
                "doc_type": "text",
                "title": name.strip(),
            },
        )

        # Drop previous chunks of an earlier version with the same name:
        # chunk ids embed a content hash, so without this an updated
        # re-ingest would leave stale chunks searchable alongside new ones.
        stale_deleted = 0
        try:
            store = ChromaStore(self.settings, collection_name=collection)
            stale_deleted = store.delete_by_metadata({"source_path": source_path})
        except Exception as e:
            logger.warning("Could not clean previous version of '%s': %s", doc_id, e)

        chunks = DocumentChunker(self.settings).split_document(document)
        if not chunks:
            return f"Document '{name}' produced no chunks (empty content?)."

        embedding = EmbeddingFactory.create(self.settings)
        dense_encoder = DenseEncoder(embedding)
        upserter = VectorUpserter(self.settings, collection_name=collection)
        vector_ids = upserter.upsert(chunks, dense_encoder.encode(chunks))

        stats = SparseEncoder().encode(chunks)
        for stat, vid in zip(stats, vector_ids):
            stat["chunk_id"] = vid
        # Per-collection index dir — matches what every query entry point
        # (CLI / MCP / dashboard) reads via build_query_components.
        indexer = BM25Indexer(
            index_dir=str(resolve_path(f"data/db/bm25/{collection}"))
        )
        indexer.add_documents(stats, collection=collection, doc_id=doc_id)

        logger.info(
            "ingest_document '%s' -> %s: %d chunks (%d stale removed)",
            doc_id,
            collection,
            len(chunks),
            stale_deleted,
        )
        return (
            f"Ingested '{name}' into collection '{collection}': "
            f"{len(chunks)} chunk(s) indexed"
            + (f", {stale_deleted} stale chunk(s) from a previous version replaced" if stale_deleted else "")
            + ". It is now searchable via query_knowledge_hub."
        )

    async def execute(
        self,
        name: str,
        content: str,
        collection: str = "default",
    ) -> types.CallToolResult:
        """Execute the ingest_document tool."""
        if not name or not name.strip():
            return self._error("Parameter 'name' must be a non-empty string.")
        if not content or not content.strip():
            return self._error("Parameter 'content' must be a non-empty string.")

        try:
            summary = await asyncio.to_thread(
                self.ingest, name, content, collection
            )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=summary)],
                isError=False,
            )
        except Exception as e:
            logger.exception("Error executing ingest_document")
            return self._error(f"Ingestion failed: {e}")

    @staticmethod
    def _error(message: str) -> types.CallToolResult:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=message)],
            isError=True,
        )


def register_tool(protocol_handler: Any) -> None:
    """Register the ingest_document tool with the protocol handler."""
    tool = IngestDocumentTool()

    async def handler(
        name: str,
        content: str,
        collection: str = "default",
    ) -> types.CallToolResult:
        return await tool.execute(name=name, content=content, collection=collection)

    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )
    logger.info(f"Registered MCP tool: {TOOL_NAME}")
