"""Unit tests for TextLoader (Markdown / txt ingestion).

TextLoader exists because the repo's own Markdown docs are the golden-set
corpus and PdfLoader rejects anything that is not a PDF — the dashboard
uploader already advertised md/txt support.
"""

from __future__ import annotations

import pytest

from src.core.types import Document
from src.libs.loader.text_loader import TextLoader


class TestTextLoader:
    def test_loads_markdown_as_is(self, tmp_path):
        """Markdown is the pipeline's native format: text must pass through untouched."""
        src = tmp_path / "guide.md"
        content = "# 标题\n\n正文一段。\n\n## 小节\n\n- 列表项\n"
        src.write_text(content, encoding="utf-8")

        doc = TextLoader().load(src)

        assert isinstance(doc, Document)
        assert doc.text == content
        assert doc.metadata["source_path"] == str(src.resolve())
        assert doc.metadata["doc_type"] == "md"
        assert doc.id.startswith("doc_")

    def test_loads_txt(self, tmp_path):
        src = tmp_path / "notes.txt"
        src.write_text("plain text", encoding="utf-8")

        doc = TextLoader().load(src)

        assert doc.text == "plain text"
        assert doc.metadata["doc_type"] == "txt"

    def test_doc_id_is_content_hash(self, tmp_path):
        """Same bytes must yield the same id — ingestion history depends on it."""
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("same content", encoding="utf-8")
        b.write_text("same content", encoding="utf-8")

        assert TextLoader().load(a).id == TextLoader().load(b).id

    def test_rejects_pdf(self, tmp_path):
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4 fake")

        with pytest.raises(ValueError, match="not a text document"):
            TextLoader().load(src)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TextLoader().load(tmp_path / "nope.md")

    def test_undecodable_bytes_do_not_fail_the_load(self, tmp_path):
        """A stray non-UTF-8 byte must not block ingestion of the whole file."""
        src = tmp_path / "mixed.md"
        src.write_bytes("正常内容".encode("utf-8") + b"\xff\xfe" + b"tail")

        doc = TextLoader().load(src)

        assert "正常内容" in doc.text
        assert "tail" in doc.text


class TestPipelineLoaderDispatch:
    def test_md_suffix_routes_to_text_loader(self):
        """The pipeline's dispatch rule: text suffixes skip the PDF loader."""
        assert ".md" in TextLoader.SUPPORTED_SUFFIXES
        assert ".markdown" in TextLoader.SUPPORTED_SUFFIXES
        assert ".txt" in TextLoader.SUPPORTED_SUFFIXES
        assert ".pdf" not in TextLoader.SUPPORTED_SUFFIXES
