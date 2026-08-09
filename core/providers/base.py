"""Provider abstractions (AGENTS.md §10-§12).

Business code only ever talks to :class:`EmbeddingProvider` and
:class:`RerankerProvider`; concrete HTTP implementations live in sibling
modules and can be swapped without touching the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RerankResult:
    index: int
    score: float
    document: str


class EmbeddingProvider(ABC):
    """Abstract embedding service (text and/or image)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    def supports_text(self) -> bool:
        return True

    @property
    def supports_image(self) -> bool:
        return False

    @abstractmethod
    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Optional keyword args (e.g. an input-type
        hint for asymmetric models) are accepted by concrete providers."""

    async def embed_image(self, images: list[bytes]) -> list[list[float]]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support image embedding"
        )


class RerankerProvider(ABC):
    """Abstract reranking service."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        """Return up to ``top_n`` results ordered by descending score."""
