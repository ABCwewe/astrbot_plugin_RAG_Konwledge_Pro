"""Image parser — each image becomes an ``image`` chunk (AGENTS.md §9).

No OCR is performed in v1; retrieval relies on the multimodal embedding of
the raw image bytes.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from ..exceptions import ParserError
from ..models import Document, ParsedDocument
from .base import DocumentParser

_SUPPORTED = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


class ImageParser(DocumentParser):
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in _SUPPORTED

    async def parse(self, path: Path) -> ParsedDocument:
        try:
            await asyncio.to_thread(_validate_image, path)
        except ParserError:
            raise
        except Exception as exc:
            raise ParserError(f"读取图片失败 {path}: {exc}") from exc
        data = path.read_bytes()
        return ParsedDocument(
            document=Document(
                id="",
                source=path.name,
                filename=path.name,
                content_hash=hashlib.sha256(data).hexdigest(),
                metadata={},
            ),
            image_paths=[str(path)],
            metadata={"parser": "image", "filename": path.name},
        )


def _validate_image(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as img:
        img.verify()
