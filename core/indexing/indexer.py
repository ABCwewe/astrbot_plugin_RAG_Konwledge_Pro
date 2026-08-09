"""Document indexer (AGENTS.md §20).

Drives the pipeline: content-hash comparison → parse → chunk → (cached)
embed → upsert. Handles new documents, changed documents (delete old chunks
then re-index) and deletions (drop all chunks of a missing source). The
embedding cache makes rebuilds and re-imports cheap (AGENTS.md §22-§23).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..chunking import TextChunker
from ..models import Chunk
from ..parsers import ParserRegistry
from ..providers import EmbeddingProvider
from ..providers.cache import EmbeddingCache
from ..storage.base import VectorPoint, VectorStore
from .manifest import DocumentRecord, document_id_for

logger = logging.getLogger("rag.indexer")

ProgressFn = Callable[[int, int, int], None]
"""Callback receiving ``(documents_processed, total_documents, chunks_processed)``."""

_CHUNK_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


def chunk_id_for(document_id: str, suffix: str) -> str:
    """Deterministic UUID chunk id.

    Qdrant point IDs must be unsigned integers or UUIDs, so the document id
    + index is hashed into a stable UUIDv5 (AGENTS.md §43: no random UUIDs —
    incremental updates must be able to reproduce the same ids).
    """
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{document_id}#{suffix}"))


@dataclass
class IndexStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    chunks_written: int = 0
    errors: list[str] = field(default_factory=list)


class Indexer:
    def __init__(
        self,
        store: VectorStore,
        embedding: EmbeddingProvider,
        chunker: TextChunker,
        cache: EmbeddingCache,
        parsers: ParserRegistry | None = None,
        image_embedding: EmbeddingProvider | None = None,
    ) -> None:
        self._store = store
        self._embedding = embedding
        self._image_embedding = image_embedding
        self._chunker = chunker
        self._cache = cache
        self._parsers = parsers or ParserRegistry()

    # -- public API -------------------------------------------------------

    async def sync_documents(
        self,
        collection: str,
        kb_id: str,
        docs_root: Path,
        doc_paths: list[Path],
        documents: dict[str, DocumentRecord],
        *,
        progress: ProgressFn | None = None,
    ) -> IndexStats:
        """Add/modify/delete documents to match the current file set.

        ``documents`` is mutated in place (records added/updated/removed).
        """
        stats = IndexStats()
        seen_sources: set[str] = set()
        total = len(doc_paths)
        chunks_done = 0
        for i, path in enumerate(doc_paths):
            source = _source_identity(docs_root, path)
            seen_sources.add(source)
            try:
                chunks_written, kind = await self._sync_one(
                    collection, kb_id, path, source, documents
                )
                chunks_done += chunks_written
                if kind == "added":
                    stats.added += 1
                elif kind == "updated":
                    stats.updated += 1
                else:
                    stats.unchanged += 1
            except Exception as exc:
                logger.exception("[INDEX] 文档索引失败: %s", path)
                stats.errors.append(f"{source}: {exc}")
            if progress:
                progress(i + 1, total, chunks_done)

        # Remove documents whose source file no longer exists.
        for doc_id, record in list(documents.items()):
            if record.source not in seen_sources:
                try:
                    await self._store.delete_by_document(collection, doc_id)
                except Exception as exc:
                    stats.errors.append(f"{record.source} (delete): {exc}")
                    continue
                del documents[doc_id]
                stats.deleted += 1

        # Final chunk tally for reporting.
        stats.chunks_written = sum(
            rec.metadata.get("chunks", 0) for rec in documents.values()
        )
        return stats

    # -- internals --------------------------------------------------------

    async def _sync_one(
        self,
        collection: str,
        kb_id: str,
        path: Path,
        source: str,
        documents: dict[str, DocumentRecord],
    ) -> tuple[int, str]:
        """Index one file.

        Returns ``(chunks_written, kind)`` where kind is ``added``, ``updated``
        or ``unchanged``.
        """
        try:
            content_hash = await _sha256_file(path)
        except OSError as exc:
            raise ValueError(f"读取文件失败: {exc}") from exc

        doc_id = document_id_for(kb_id, source)
        existing = documents.get(doc_id)

        if existing and existing.content_hash == content_hash:
            logger.debug("[INDEX] 未变化，跳过: %s", source)
            return 0, "unchanged"
        if existing:
            logger.info("[INDEX] 内容变化，删除旧块后重建: %s", source)
            await self._store.delete_by_document(collection, doc_id)
            kind = "updated"
        else:
            logger.info("[INDEX] 新增文档: %s", source)
            kind = "added"

        parsed = await self._parse(path)
        chunks = self._build_chunks(parsed, doc_id, source)
        if not chunks:
            raise ValueError("文档解析后没有可索引的块")

        points = await self._build_points(chunks, kb_id)
        await self._store.upsert(collection, points)
        vector_names = sorted({name for p in points for name in p.vectors})
        logger.info(
            "[INDEX] 写入 %d 个块: %s (%s)",
            len(chunks),
            source,
            ", ".join(vector_names),
        )

        documents[doc_id] = DocumentRecord(
            document_id=doc_id,
            source=source,
            filename=path.name,
            content_hash=content_hash,
            indexed_at=_now(),
            metadata={"chunks": len(chunks)},
        )
        return len(chunks), kind

    async def _parse(self, path: Path):
        parser = self._parsers.get_parser(path)
        return await parser.parse(path)

    def _build_chunks(self, parsed, doc_id: str, source: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        if parsed.pages:
            idx = 0
            for page_no, text in parsed.pages:
                for seg in self._chunker.split(text):
                    chunks.append(
                        Chunk(
                            id=chunk_id_for(doc_id, str(idx)),
                            document_id=doc_id,
                            type="text",
                            content=seg,
                            image_path=None,
                            chunk_index=idx,
                            metadata={"source": source, "page": page_no},
                        )
                    )
                    idx += 1
        elif parsed.text is not None:
            for idx, seg in enumerate(self._chunker.split(parsed.text)):
                chunks.append(
                    Chunk(
                        id=chunk_id_for(doc_id, str(idx)),
                        document_id=doc_id,
                        type="text",
                        content=seg,
                        image_path=None,
                        chunk_index=idx,
                        metadata={"source": source},
                    )
                )
        img_idx = 0
        for image_path in parsed.image_paths:
            chunks.append(
                Chunk(
                    id=chunk_id_for(doc_id, f"img{img_idx}"),
                    document_id=doc_id,
                    type="image",
                    content=None,
                    image_path=image_path,
                    chunk_index=img_idx,
                    metadata={"source": source, "filename": Path(image_path).name},
                )
            )
            img_idx += 1
        return chunks

    async def _build_points(self, chunks: list[Chunk], kb_id: str) -> list[VectorPoint]:
        text_chunks = [c for c in chunks if c.type == "text"]
        image_chunks = [c for c in chunks if c.type == "image"]

        text_vecs = await self._embed_texts(text_chunks) if text_chunks else []
        image_vecs = []
        if image_chunks:
            if self._image_embedding is None:
                raise ValueError("存在图片块但未配置图片 Embedding")
            image_vecs = await self._embed_images(image_chunks)

        points: list[VectorPoint] = []
        for chunk, vec in zip(text_chunks, text_vecs):
            points.append(
                VectorPoint(
                    id=chunk.id,
                    vectors={"text": vec},
                    payload={
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "kb_id": kb_id,
                        "type": "text",
                        "content": chunk.content,
                        "source": chunk.metadata.get("source"),
                        "page": chunk.metadata.get("page"),
                        "chunk_index": chunk.chunk_index,
                    },
                )
            )
        for chunk, vec in zip(image_chunks, image_vecs):
            points.append(
                VectorPoint(
                    id=chunk.id,
                    vectors={"image": vec},
                    payload={
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "kb_id": kb_id,
                        "type": "image",
                        "image_path": chunk.image_path,
                        "source": chunk.metadata.get("source"),
                        "page": chunk.metadata.get("page"),
                        "chunk_index": chunk.chunk_index,
                    },
                )
            )
        return points

    async def _embed_texts(self, chunks: list[Chunk]) -> list[list[float]]:
        model = self._embedding.model_name
        hashes = [EmbeddingCache.content_hash_of_text(c.content or "") for c in chunks]
        hits = await self._cache.get_many("text", model, hashes)
        missing_hashes = [h for h, v in zip(hashes, hits) if v is None]
        missing_chunks = [c for c, v in zip(chunks, hits) if v is None]
        vecs: list[list[float]] = []
        if missing_chunks:
            vecs = await self._embedding.embed_text(
                [c.content or "" for c in missing_chunks], input_type="passage"
            )
            await self._cache.set_many("text", model, missing_hashes, vecs)
        by_hash = dict(zip(missing_hashes, vecs))
        return [v if v is not None else by_hash[h] for h, v in zip(hashes, hits)]

    async def _embed_images(self, chunks: list[Chunk]) -> list[list[float]]:
        model = self._image_embedding.model_name
        payloads: list[bytes] = []
        for c in chunks:
            payloads.append(await _read_bytes(c.image_path))
        hashes = [EmbeddingCache.content_hash_of_bytes(p) for p in payloads]
        hits = await self._cache.get_many("image", model, hashes)
        missing_idx = [i for i, v in enumerate(hits) if v is None]
        vecs: list[list[float]] = []
        if missing_idx:
            vecs = await self._image_embedding.embed_image(
                [payloads[i] for i in missing_idx]
            )
            await self._cache.set_many(
                "image", model, [hashes[i] for i in missing_idx], vecs
            )
        by_hash = dict(
            zip([hashes[i] for i in missing_idx], vecs)
        )
        return [v if v is not None else by_hash[h] for h, v in zip(hashes, hits)]


async def _sha256_file(path: Path) -> str:
    import hashlib

    data = await _read_bytes(path)
    return hashlib.sha256(data).hexdigest()


async def _read_bytes(path: str | Path) -> bytes:
    import asyncio

    return await asyncio.to_thread(Path(path).read_bytes)


def _source_identity(docs_root: Path, path: Path) -> str:
    try:
        return path.relative_to(docs_root).as_posix()
    except ValueError:
        return path.name


def _now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
