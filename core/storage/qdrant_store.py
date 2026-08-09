"""Qdrant-backed vector store.

Uses the async client and the modern ``query_points`` API (the legacy
``search``/``search_batch``/``NamedVector`` APIs were removed in
qdrant-client 1.19 — named vectors are addressed through the ``using``
parameter). Upserts are chunked (64/request) per upstream guidance; deletes
use filter selectors so a whole document can be removed in one call.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from ..exceptions import QdrantError
from .base import ScoredPoint, VectorPoint, VectorStore

logger = logging.getLogger("rag.storage.qdrant")

_UPSERT_BATCH = 64
_DELETE_BATCH = 100
_SCROLL_BATCH = 100


class QdrantStore(VectorStore):
    def __init__(self, url: str, api_key: str | None = None, timeout: float = 60.0) -> None:
        try:
            self._client = AsyncQdrantClient(
                url=url, api_key=api_key or None, timeout=timeout
            )
        except Exception as exc:  # pragma: no cover - construction only
            raise QdrantError(f"无法连接 Qdrant ({url}): {exc}") from exc
        self._url = url

    # -- collection management -------------------------------------------

    async def create_collection(self, name: str, vectors: dict[str, int]) -> None:
        cfg = {
            vec_name: VectorParams(size=dim, distance=Distance.COSINE)
            for vec_name, dim in vectors.items()
        }
        try:
            await self._client.create_collection(collection_name=name, vectors_config=cfg)
        except Exception as exc:
            raise QdrantError(f"创建 collection {name} 失败: {exc}") from exc

    async def collection_exists(self, name: str) -> bool:
        try:
            return await self._client.collection_exists(collection_name=name)
        except Exception as exc:
            raise QdrantError(f"检查 collection {name} 失败: {exc}") from exc

    async def collection_has_vector(self, name: str, vector_name: str) -> bool:
        try:
            info = await self._client.get_collection(collection_name=name)
        except Exception as exc:
            raise QdrantError(f"获取 collection {name} 配置失败: {exc}") from exc
        vectors = getattr(getattr(info.config, "params", None), "vectors", None)
        if isinstance(vectors, dict):
            return vector_name in vectors
        return False

    async def delete_collection(self, name: str) -> None:
        try:
            await self._client.delete_collection(collection_name=name)
        except Exception as exc:
            raise QdrantError(f"删除 collection {name} 失败: {exc}") from exc

    async def list_collections(self) -> list[str]:
        try:
            resp = await self._client.get_collections()
            return [c.name for c in resp.collections]
        except Exception as exc:
            raise QdrantError(f"获取 collection 列表失败: {exc}") from exc

    # -- points -----------------------------------------------------------

    async def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        if not points:
            return
        structs = [
            PointStruct(id=p.id, vector=p.vectors, payload=p.payload) for p in points
        ]
        for start in range(0, len(structs), _UPSERT_BATCH):
            batch = structs[start : start + _UPSERT_BATCH]
            try:
                await self._client.upsert(collection_name=collection, points=batch, wait=True)
            except Exception as exc:
                raise QdrantError(f"upsert {len(batch)} 个点失败: {exc}") from exc

    async def delete_by_document(self, collection: str, document_id: str) -> None:
        selector = FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )
        )
        try:
            await self._client.delete(collection_name=collection, points_selector=selector, wait=True)
        except Exception as exc:
            raise QdrantError(f"删除文档 {document_id} 的点失败: {exc}") from exc

    async def delete_points(self, collection: str, point_ids: list[str]) -> None:
        if not point_ids:
            return
        for start in range(0, len(point_ids), _DELETE_BATCH):
            batch = point_ids[start : start + _DELETE_BATCH]
            try:
                await self._client.delete(
                    collection_name=collection,
                    points_selector=PointIdsList(points=batch),
                    wait=True,
                )
            except Exception as exc:
                raise QdrantError(f"删除 {len(batch)} 个点失败: {exc}") from exc

    # -- search -----------------------------------------------------------

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
        try:
            resp = await self._client.query_points(
                collection_name=collection,
                query=query_vector,
                using=vector_name,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold,
            )
        except Exception as exc:
            raise QdrantError(f"检索 {collection} 失败: {exc}") from exc
        return [
            ScoredPoint(point_id=str(p.id), score=p.score, payload=p.payload or {})
            for p in resp.points
        ]

    async def count(
        self, collection: str, *, count_filter: dict | None = None
    ) -> int:
        qdrant_filter = None
        if count_filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in count_filter.items()
                ]
            )
        try:
            resp = await self._client.count(
                collection_name=collection,
                count_filter=qdrant_filter,
                exact=True,
            )
            return resp.count
        except Exception as exc:
            raise QdrantError(f"统计 {collection} 点数失败: {exc}") from exc

    async def scroll(self, collection: str) -> AsyncIterator[ScoredPoint]:
        offset = None
        while True:
            try:
                records, offset = await self._client.scroll(
                    collection_name=collection,
                    offset=offset,
                    limit=_SCROLL_BATCH,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                raise QdrantError(f"遍历 {collection} 失败: {exc}") from exc
            for r in records:
                yield ScoredPoint(point_id=str(r.id), score=0.0, payload=r.payload or {})
            if offset is None:
                break

    async def close(self) -> None:
        await self._client.close()
