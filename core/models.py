"""Business data models for the RAG pipeline.

Qdrant points are never exposed directly to business code; every layer above
the storage adapter consumes these dataclasses instead (see AGENTS.md §6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Document:
    """A single source file managed by a knowledge base."""

    id: str
    source: str
    filename: str
    content_hash: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit produced from a :class:`Document`."""

    id: str
    document_id: str
    type: Literal["text", "image"]
    content: str | None
    image_path: str | None
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A unified retrieval hit, regardless of whether it came from text or
    image vectors."""

    chunk_id: str
    document_id: str
    content: str | None
    image_path: str | None
    vector_score: float
    rerank_score: float | None
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Output of a :class:`DocumentParser`.

    ``text`` holds the full plain text for text-style documents;
    ``pages`` is an optional list of ``(page_number, page_text)`` pairs used by
    the PDF parser so page metadata survives into chunks; ``image_paths`` is
    used by image parsers (each path becomes one image chunk).
    """

    document: Document
    text: str | None = None
    pages: list[tuple[int, str]] | None = None
    image_paths: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
