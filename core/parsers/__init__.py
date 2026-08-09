"""Parser registry — resolves the right parser for a file by extension."""

from __future__ import annotations

from pathlib import Path

from ..exceptions import ParserError
from .base import DocumentParser
from .image import ImageParser
from .pdf import PDFParser
from .text import TextParser

__all__ = ["DocumentParser", "ImageParser", "PDFParser", "ParserRegistry", "TextParser"]


class ParserRegistry:
    def __init__(self, parsers: list[DocumentParser] | None = None) -> None:
        self._parsers = parsers or [TextParser(), PDFParser(), ImageParser()]

    def get_parser(self, path: str | Path) -> DocumentParser:
        path = Path(path)
        for parser in self._parsers:
            if parser.supports(path):
                return parser
        raise ParserError(f"不支持的文件类型: {path.suffix or path.name}")
