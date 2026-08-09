"""RAG plugin exception hierarchy.

All plugin errors derive from :class:`RAGError` so callers can catch a single
base type, while distinct subclasses allow targeted handling (e.g. a failed
rebuild must never take down an already READY index).
"""

from __future__ import annotations


class RAGError(Exception):
    """Base class for all RAG plugin errors."""


class ConfigurationError(RAGError):
    """Invalid or missing plugin configuration."""


class EmbeddingAPIError(RAGError):
    """A remote embedding API call failed."""


class RerankerAPIError(RAGError):
    """A remote reranker API call failed."""


class QdrantError(RAGError):
    """A Qdrant operation failed."""


class ParserError(RAGError):
    """A document could not be parsed."""


class IndexBuildError(RAGError):
    """Index construction failed (rebuild/verify/switch)."""


class IndexBusyError(IndexBuildError):
    """A rebuild is already in progress for this knowledge base."""


class IndexNotFoundError(RAGError):
    """No usable (READY) index exists for the requested knowledge base."""
