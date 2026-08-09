"""Plain-text / Markdown parser."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ..exceptions import ParserError
from ..models import Document, ParsedDocument
from .base import DocumentParser

_SUPPORTED = {".txt", ".md", ".markdown", ".text"}


class TextParser(DocumentParser):
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in _SUPPORTED

    async def parse(self, path: Path) -> ParsedDocument:
        try:
            text = await asyncio.to_thread(
                _read_text, path
            )
        except OSError as exc:
            raise ParserError(f"读取文本文件失败 {path}: {exc}") from exc
        if not text.strip():
            raise ParserError(f"文本文件为空: {path}")
        return ParsedDocument(
            document=_document_for(path),
            text=text,
            metadata={"parser": "text", "filename": path.name},
        )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _document_for(path: Path) -> Document:
    import hashlib

    data = path.read_bytes()
    return Document(
        id="",
        source=path.name,
        filename=path.name,
        content_hash=hashlib.sha256(data).hexdigest(),
        metadata={},
    )
