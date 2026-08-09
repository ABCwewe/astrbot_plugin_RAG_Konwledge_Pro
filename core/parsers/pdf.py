"""PDF parser using PyMuPDF (AGENTS.md §8).

Page numbers are preserved: each non-empty page becomes a ``(page_no, text)``
entry in :attr:`ParsedDocument.pages` so the indexer can tag every chunk with
its source page.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from ..exceptions import ParserError
from ..models import Document, ParsedDocument
from .base import DocumentParser


class PDFParser(DocumentParser):
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    async def parse(self, path: Path) -> ParsedDocument:
        try:
            pages = await asyncio.to_thread(_extract_pages, path)
        except ParserError:
            raise
        except Exception as exc:  # fitz raises many exception types
            raise ParserError(f"解析 PDF 失败 {path}: {exc}") from exc
        if not pages:
            raise ParserError(f"PDF 无可用文本: {path}")
        return ParsedDocument(
            document=_document_for(path),
            pages=pages,
            metadata={"parser": "pdf", "filename": path.name, "page_count": len(pages)},
        )


def _extract_pages(path: Path) -> list[tuple[int, str]]:
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    try:
        out: list[tuple[int, str]] = []
        for page_no in range(doc.page_count):
            text = doc.load_page(page_no).get_text("text")
            if text and text.strip():
                out.append((page_no + 1, text))
        return out
    finally:
        doc.close()


def _document_for(path: Path) -> Document:
    data = path.read_bytes()
    return Document(
        id="",
        source=path.name,
        filename=path.name,
        content_hash=hashlib.sha256(data).hexdigest(),
        metadata={},
    )
