"""Retriever — Top-K vector search → rerank → Top-N (AGENTS.md §24-§26).

Supports both single-collection and multi-collection retrieval: for multiple
knowledge bases the query embedding is computed once, each collection
contributes its Top-K hits, the pools are merged and de-duplicated by vector
score, and the combined candidates are reranked in ONE call before Top-N is
returned (aggregation by match score).

Text queries search the ``text`` named vector by default. When image search
is enabled, the same query embedding also searches the ``image`` named vector
and the hits are merged before reranking. Image hits are given a text
representation (source / page / filename) so an ordinary text reranker can
score them (AGENTS.md §26).
"""

from __future__ import annotations

from pathlib import Path

from ..config import RAGConfig
from ..models import SearchResult
from ..providers import EmbeddingProvider, RerankerProvider
from ..storage.base import ScoredPoint, VectorStore


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedding: EmbeddingProvider,
        reranker: RerankerProvider | None,
        config: RAGConfig,
        image_embedding: EmbeddingProvider | None = None,
    ) -> None:
        self._store = store
        self._embedding = embedding
        self._image_embedding = image_embedding
        self._reranker = reranker
        self._config = config

    async def retrieve(
        self,
        collection: str,
        query: str,
        *,
        top_k: int | None = None,
        top_n: int | None = None,
        include_images: bool | None = None,
    ) -> list[SearchResult]:
        return await self.retrieve_from_collections(
            [collection],
            query,
            top_k=top_k,
            top_n=top_n,
            include_images=include_images,
        )

    async def retrieve_by_image(
        self,
        collection: str,
        image_bytes: bytes,
        *,
        top_n: int | None = None,
    ) -> list[SearchResult]:
        """Image-as-query retrieval: embed the image and search the ``image``
        named vector. Used for automatic image search on incoming messages.
        Hits below ``config.image.min_score`` are dropped (noise guard).
        """
        if self._image_embedding is None:
            return []
        query_vector = (await self._image_embedding.embed_image([image_bytes]))[0]
        hits = await self._store.search(
            collection,
            "image",
            query_vector,
            top_n or self._config.top_n,
            score_threshold=self._config.image.min_score or None,
        )
        return [self._to_result(hit) for hit in hits]

    async def retrieve_from_collections(
        self,
        collections: list[str],
        query: str,
        *,
        top_k: int | None = None,
        top_n: int | None = None,
        include_images: bool | None = None,
    ) -> list[SearchResult]:
        """Retrieve across several collections; results are aggregated by
        match score (vector Top-K per KB → merged → one rerank → Top-N).

        Every collection contributes up to ``top_k`` candidates to the rerank
        pool (per-KB quota, so no KB is starved out by vector score); the
        merged pool is capped by ``rerank_pool_size``.
        """
        if not collections or not query or not query.strip():
            return []
        top_k = top_k or self._config.top_k
        top_n = top_n or self._config.top_n
        pool_cap = self._config.rerank_pool_size
        if include_images is None:
            include_images = bool(
                self._config.image.enabled and self._config.image.search_always
            )

        query_vector = (await self._embedding.embed_text([query], input_type="query"))[0]

        merged: list[SearchResult] = []
        seen: set[str] = set()
        for collection in collections:
            per_kb: list[SearchResult] = []
            text_hits = await self._store.search(collection, "text", query_vector, top_k)
            per_kb.extend(self._to_result(hit) for hit in text_hits)
            if include_images and self._image_embedding is not None:
                image_hits = await self._store.search(collection, "image", query_vector, top_k)
                per_kb.extend(self._to_result(hit) for hit in image_hits)
            for result in _dedupe_and_sort(per_kb, top_k):
                if result.chunk_id in seen:
                    continue
                seen.add(result.chunk_id)
                merged.append(result)
            if len(merged) >= pool_cap:
                break
        results = merged[:pool_cap]
        if not results:
            return []

        if self._reranker is not None:
            documents = [self._rerank_text(r) for r in results]
            ranked = await self._reranker.rerank(query, documents, top_n)
            out: list[SearchResult] = []
            for item in ranked:
                if 0 <= item.index < len(results):
                    result = results[item.index]
                    result.rerank_score = item.score
                    out.append(result)
            return out[:top_n]

        return results[:top_n]

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _to_result(hit: ScoredPoint) -> SearchResult:
        payload = hit.payload or {}
        return SearchResult(
            chunk_id=payload.get("chunk_id") or hit.point_id,
            document_id=payload.get("document_id", ""),
            content=payload.get("content"),
            image_path=payload.get("image_path"),
            vector_score=hit.score,
            rerank_score=None,
            metadata={
                "source": payload.get("source"),
                "page": payload.get("page"),
                "chunk_index": payload.get("chunk_index"),
                "type": payload.get("type"),
                "kb_id": payload.get("kb_id"),
            },
        )

    @staticmethod
    def _rerank_text(result: SearchResult) -> str:
        """Textual representation of a hit for the reranker (AGENTS.md §26)."""
        if result.content:
            return result.content
        parts: list[str] = []
        source = result.metadata.get("source")
        if source:
            parts.append(source)
        page = result.metadata.get("page")
        if page is not None:
            parts.append(f"第{page}页")
        if result.image_path:
            parts.append(f"图片: {Path(result.image_path).name}")
        return " ".join(parts) or "（无文本内容）"


def _dedupe_and_sort(results: list[SearchResult], limit: int) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in sorted(results, key=lambda r: r.vector_score, reverse=True):
        if r.chunk_id in seen:
            continue
        seen.add(r.chunk_id)
        out.append(r)
        if len(out) >= limit:
            break
    return out
