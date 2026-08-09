"""Shared fakes: in-memory vector store, deterministic embedding, fake
reranker. No real Qdrant / HTTP services are ever touched by unit tests.
"""

from __future__ import annotations

import math

from core.providers import EmbeddingProvider, RerankResult, RerankerProvider
from core.storage.base import ScoredPoint, VectorPoint, VectorStore


class FakeVectorStore(VectorStore):
    """In-memory store keyed by collection -> {vectors config, points}."""

    def __init__(self) -> None:
        self.collections: dict[str, dict] = {}
        self.deleted_collections: list[str] = []

    async def create_collection(self, name, vectors) -> None:
        self.collections[name] = {"vectors": dict(vectors), "points": {}}

    async def collection_exists(self, name) -> bool:
        return name in self.collections

    async def delete_collection(self, name) -> None:
        self.collections.pop(name, None)
        self.deleted_collections.append(name)

    async def list_collections(self) -> list[str]:
        return list(self.collections)

    async def upsert(self, collection, points) -> None:
        coll = self.collections.setdefault(collection, {"vectors": {}, "points": {}})
        for p in points:
            coll["points"][p.id] = p

    async def delete_by_document(self, collection, document_id) -> None:
        coll = self.collections.get(collection)
        if not coll:
            return
        for pid in [
            pid
            for pid, p in coll["points"].items()
            if p.payload.get("document_id") == document_id
        ]:
            del coll["points"][pid]

    async def delete_points(self, collection, point_ids) -> None:
        coll = self.collections.get(collection)
        if not coll:
            return
        for pid in point_ids:
            coll["points"].pop(pid, None)

    async def search(self, collection, vector_name, query_vector, limit, *, query_filter=None, score_threshold=None) -> list[ScoredPoint]:
        coll = self.collections.get(collection)
        if not coll:
            return []
        scored = []
        for pid, p in coll["points"].items():
            if vector_name not in p.vectors:
                continue
            score = _cosine(query_vector, p.vectors[vector_name])
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append(ScoredPoint(pid, score, p.payload))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:limit]

    async def count(self, collection) -> int:
        coll = self.collections.get(collection)
        return len(coll["points"]) if coll else 0

    async def scroll(self, collection):
        coll = self.collections.get(collection)
        for p in (coll["points"].values() if coll else []):
            yield ScoredPoint(p.id, 0.0, p.payload)

    async def close(self) -> None:
        pass


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class FakeEmbedding(EmbeddingProvider):
    """Deterministic char-based embedding: similar texts get similar vectors."""

    def __init__(self, model: str = "fake-embed", dim: int = 8, supports_image: bool = False) -> None:
        self._model = model
        self._dim = dim
        self._supports_image = supports_image
        self.text_calls: list[list[str]] = []
        self.image_calls: list[list[bytes]] = []

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def supports_image(self) -> bool:
        return self._supports_image

    async def embed_text(self, texts, *, input_type=None):
        self.text_calls.append(list(texts))
        return [self._vec(t) for t in texts]

    async def embed_image(self, images):
        self.image_calls.append(list(images))
        return [self._vec(b) for b in images]

    def _vec(self, data) -> list[float]:
        s = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
        padded = (s[: self._dim] + " " * self._dim)[: self._dim]
        return [ord(c) % 32 / 32 for c in padded]


class FakeReranker(RerankerProvider):
    """Scores documents by character overlap with the query."""

    def __init__(self, model: str = "fake-rerank") -> None:
        self._model = model
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def model_name(self) -> str:
        return self._model

    async def rerank(self, query, documents, top_n):
        self.calls.append((query, list(documents)))
        scored = sorted(
            range(len(documents)),
            key=lambda i: _overlap(query, documents[i]),
            reverse=True,
        )
        return [
            RerankResult(index=i, score=_overlap(query, documents[i]), document=documents[i])
            for i in scored[:top_n]
        ]


def _overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    common = sum(1 for ca in set(a) if ca in b)
    return common / len(set(a))
