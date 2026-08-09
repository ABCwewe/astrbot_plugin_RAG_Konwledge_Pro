"""Vector store abstraction (AGENTS.md §5.1).

Business code (indexer, retriever, engine) talks only to
:class:`VectorStore`; Qdrant specifics are confined to
:class:`~core.storage.qdrant_store.QdrantStore`. Swapping the backend must
never touch the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class VectorPoint:
    """Business-level point handed to the store for upsert."""

    id: str
    vectors: dict[str, list[float]]
    payload: dict = field(default_factory=dict)


@dataclass
class ScoredPoint:
    """A raw search hit; payload must be enough to rebuild a SearchResult."""

    point_id: str
    score: float
    payload: dict = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    async def create_collection(
        self, name: str, vectors: dict[str, int]
    ) -> None:
        """Create a collection with one named vector per (name -> dimension)."""

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        ...

    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        ...

    @abstractmethod
    async def list_collections(self) -> list[str]:
        ...

    @abstractmethod
    async def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        ...

    @abstractmethod
    async def delete_by_document(self, collection: str, document_id: str) -> None:
        """Delete every point belonging to a document."""

    @abstractmethod
    async def delete_points(self, collection: str, point_ids: list[str]) -> None:
        ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        vector_name: str,
        query_vector: list[float],
        limit: int,
        *,
        query_filter: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[ScoredPoint]:
        """Nearest-neighbour search against one named vector.

        ``score_threshold`` drops hits below the given similarity score
        (interpretation depends on the distance metric).
        """

    @abstractmethod
    async def count(self, collection: str) -> int:
        ...

    @abstractmethod
    def scroll(self, collection: str) -> AsyncIterator[ScoredPoint]:
        """Yield every point's payload (used for index verification)."""

    @abstractmethod
    async def close(self) -> None:
        ...
