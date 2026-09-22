"""Loader for plain-text documents (Markdown, txt).

The pipeline needs this because the repo's own documentation — the golden-set
corpus — is Markdown, and PdfLoader rejects anything that is not a PDF.
Text formats carry no embedded images, so there is nothing to extract and the
file content is used as-is (Markdown is already the pipeline's native format).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader


class TextLoader(BaseLoader):
    """Load Markdown / txt files as UTF-8 text.

    Configuration: none — a text file is its own content.

    Graceful Degradation:
        Undecodable bytes are replaced (``errors="replace"``) instead of
        failing the load, so a stray non-UTF-8 character cannot block
        ingestion of an otherwise fine document.
    """

    SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}

    def load(self, file_path: str | Path) -> Document:
        """Load a text file.

        Args:
            file_path: Path to a .md / .markdown / .txt file.

        Returns:
            Document whose text is the file content.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the suffix is not a supported text format.
        """
        path = self._validate_file(file_path)
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(
                f"File is not a text document: {path} "
                f"(supported: {sorted(self.SUPPORTED_SUFFIXES)})"
            )

        raw = path.read_bytes()
        doc_hash = hashlib.sha256(raw).hexdigest()

        return Document(
            id=f"doc_{doc_hash[:16]}",
            text=raw.decode("utf-8", errors="replace"),
            metadata={
                "source_path": str(path),
                "doc_type": suffix.lstrip("."),
                "doc_hash": doc_hash,
            },
        )
