"""RAGEngine — top-level facade over the whole pipeline.

Constructed from a :class:`~core.config.RAGConfig` plus an on-disk data root.
The engine owns a knowledge base's documents directory, the index manager
(rebuild/incremental/rollback), the retriever, and the embedding cache, and
exposes the small API that the AstrBot adapter and CLI/WebUI consume.

Business code never touches Qdrant or the HTTP providers directly.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from .chunking import TextChunker
from .config import RAGConfig
from .exceptions import ParserError
from .indexing import BuildProgress, IndexManager
from .models import SearchResult
from .providers import EmbeddingProvider, RerankerProvider
from .providers.cache import EmbeddingCache
from .providers.openai_compatible import OpenAICompatibleEmbedding
from .providers.reranker import SiliconFlowReranker
from .retrieval import Retriever
from .storage.base import VectorStore
from .storage.qdrant_store import QdrantStore

logger = logging.getLogger("rag.engine")


class RAGEngine:
    def __init__(
        self,
        config: RAGConfig,
        data_root: str | Path,
        *,
        store: VectorStore | None = None,
        embedding: EmbeddingProvider | None = None,
        image_embedding: EmbeddingProvider | None = None,
        reranker: RerankerProvider | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._data_root = Path(data_root)
        self._data_root.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_kbs()

        self._store = store or QdrantStore(
            config.qdrant.url, config.qdrant.api_key, config.qdrant.timeout
        )
        self._embedding = embedding or OpenAICompatibleEmbedding(
            config.embedding.api_base,
            config.embedding.api_key,
            config.embedding.model,
            config.embedding.dimension,
            batch_size=config.embedding.batch_size,
            concurrency=config.embedding.concurrency,
            timeout=config.embedding.timeout,
            extra_params=config.embedding.extra_params,
        )
        if config.image.enabled:
            self._image_embedding = image_embedding or OpenAICompatibleEmbedding(
                config.image.api_base,
                config.image.api_key,
                config.image.model,
                int(config.image.dimension),
                batch_size=config.image.batch_size,
                concurrency=config.image.concurrency,
                timeout=config.image.timeout,
                extra_params=config.image.extra_params,
            )
        else:
            self._image_embedding = image_embedding
        self._reranker = (
            reranker
            if reranker is not None
            else (
                SiliconFlowReranker(
                    config.rerank.api_base,
                    config.rerank.api_key,
                    config.rerank.model,
                    concurrency=config.rerank.concurrency,
                    timeout=config.rerank.timeout,
                )
                if config.rerank.enabled
                else None
            )
        )

        self._chunker = TextChunker(
            config.chunking.separator,
            config.chunking.chunk_size,
            config.chunking.chunk_overlap,
        )
        self._cache = EmbeddingCache(self._data_root / "cache.db")
        self._manager = IndexManager(
            self._store,
            self._embedding,
            self._chunker,
            self._cache,
            self._data_root,
            image_embedding=self._image_embedding,
        )
        self._retriever = Retriever(
            self._store,
            self._embedding,
            self._reranker,
            config,
            image_embedding=self._image_embedding,
        )

    # -- knowledge base management ----------------------------------------

    def kb_root(self, kb_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in kb_id)
        return self._data_root / "kbs" / safe

    def _migrate_legacy_kbs(self) -> None:
        """Move KB roots from the old layout (``data_root/<kb_id>``) into the
        parent folder ``data_root/kbs/`` so engine-level directories (tmp/,
        cache.db) never mix with knowledge bases."""
        kbs_dir = self._data_root / "kbs"
        for child in self._data_root.iterdir():
            if not child.is_dir() or child.name in ("kbs", "tmp"):
                continue
            looks_like_kb = (
                (child / "documents").is_dir()
                or (child / "active.json").exists()
                or any(child.glob("manifest_v*.json"))
            )
            if not looks_like_kb:
                continue
            dest = kbs_dir / child.name
            if dest.exists():
                logger.warning("[RAG] 迁移目标已存在，跳过: %s", dest)
                continue
            kbs_dir.mkdir(parents=True, exist_ok=True)
            child.rename(dest)
            logger.info("[RAG] 迁移知识库目录: %s -> %s", child, dest)

    def docs_dir(self, kb_id: str) -> Path:
        return self.kb_root(kb_id) / "documents"

    async def ingest(self, kb_id: str, paths: list[str | Path]) -> dict:
        """Copy files into the KB document directory, then sync the index.

        Files already inside the KB document directory are not re-copied
        (copying a file onto itself fails with a lock error on Windows).
        """
        docs = self.docs_dir(kb_id)
        docs.mkdir(parents=True, exist_ok=True)
        for raw in paths:
            src = Path(raw)
            if not src.is_file():
                raise ParserError(f"文件不存在: {src}")
            dest = docs / src.name
            if src.resolve() == dest.resolve():
                logger.info("[RAG] 文件已在知识库目录，跳过复制: %s", src.name)
            else:
                await asyncio.to_thread(shutil.copy2, src, dest)
                logger.info("[RAG] 已导入文档: %s -> %s", src.name, dest)
        return await self._manager.ensure_index(kb_id, self.config)

    async def remove_document(self, kb_id: str, filename: str) -> dict:
        """Delete one document file from the KB and sync the index."""
        dest = self.docs_dir(kb_id) / filename
        if not dest.exists():
            raise ParserError(f"文档不存在: {filename}")
        await asyncio.to_thread(dest.unlink)
        return await self._manager.sync(kb_id, self.config)

    async def list_documents(self, kb_id: str) -> list[dict]:
        """List documents of a knowledge base, joined with index metadata."""
        from .indexing.manifest import ManifestStore, document_id_for

        docs_dir = self.docs_dir(kb_id)
        if not docs_dir.exists():
            return []
        store = ManifestStore(self.kb_root(kb_id))
        records = await store.load_documents()
        out: list[dict] = []
        for path in sorted(docs_dir.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            source = path.relative_to(docs_dir).as_posix()
            record = records.get(document_id_for(kb_id, source))
            out.append(
                {
                    "filename": path.name,
                    "source": source,
                    "size": path.stat().st_size,
                    "indexed": record is not None,
                    "chunks": (record.metadata.get("chunks", 0) if record else 0),
                    "indexed_at": record.indexed_at if record else None,
                }
            )
        return out

    async def rebuild(self, kb_id: str) -> dict:
        """Force a full rebuild (new version, atomic switch on success)."""
        return await self._manager.rebuild(kb_id, self.config)

    async def sync(self, kb_id: str) -> dict:
        """Incremental sync of the KB document directory."""
        return await self._manager.sync(kb_id, self.config)

    async def delete_kb(self, kb_id: str) -> None:
        await self._manager.delete_kb(kb_id)

    async def list_kbs(self) -> list[str]:
        kbs_dir = self._data_root / "kbs"
        if not kbs_dir.exists():
            return []
        return sorted(p.name for p in kbs_dir.iterdir() if p.is_dir())

    async def status(self, kb_id: str) -> dict:
        return await self._manager.status(kb_id, self.config)

    def get_build_progress(self, kb_id: str) -> BuildProgress | None:
        return self._manager.get_progress(kb_id)

    # -- retrieval --------------------------------------------------------

    async def search(
        self,
        kb_id: str,
        query: str,
        *,
        top_k: int | None = None,
        top_n: int | None = None,
        include_images: bool | None = None,
    ) -> list[SearchResult]:
        collection = await self._manager.active_collection(kb_id)
        if collection is None:
            from .exceptions import IndexNotFoundError

            raise IndexNotFoundError(
                f"知识库 {kb_id} 尚未构建索引，请先导入文档或执行 /rag rebuild"
            )
        return await self._retriever.retrieve(
            collection,
            query,
            top_k=top_k,
            top_n=top_n,
            include_images=include_images,
        )

    async def search_multi(
        self,
        kb_ids: list[str],
        query: str,
        *,
        top_k: int | None = None,
        top_n: int | None = None,
        include_images: bool | None = None,
    ) -> list[SearchResult]:
        """Search several knowledge bases at once; results are aggregated by
        match score (vector Top-K per KB → merged → one rerank → Top-N).

        KBs without a READY index are skipped; if none is usable an
        :class:`~core.exceptions.IndexNotFoundError` is raised.
        """
        from .exceptions import IndexNotFoundError

        kb_ids = [kb for kb in kb_ids if kb and kb.strip()]
        if not kb_ids:
            return []
        collections: list[str] = []
        skipped: list[str] = []
        for kb_id in kb_ids:
            collection = await self._manager.active_collection(kb_id)
            if collection is not None:
                collections.append(collection)
            else:
                skipped.append(kb_id)
        if not collections:
            raise IndexNotFoundError(
                f"所选知识库均未构建索引: {', '.join(kb_ids)}"
            )
        if skipped:
            logger.warning("[RAG] 以下知识库无可用索引，已跳过: %s", ", ".join(skipped))
        return await self._retriever.retrieve_from_collections(
            collections,
            query,
            top_k=top_k,
            top_n=top_n,
            include_images=include_images,
        )

    # -- knowledge base management ----------------------------------------

    async def create_kb(self, kb_id: str) -> dict:
        """Create an empty knowledge base (empty READY index)."""
        return await self._manager.ensure_index(kb_id, self.config)

    async def drop_index(self, kb_id: str) -> None:
        """Delete the KB's index (all versions/collections + manifests) but
        keep the knowledge base itself and its document files. The index is
        rebuilt on the next ingest/rebuild."""
        await self._manager.delete_index(kb_id)

    # -- context formatting -----------------------------------------------

    @staticmethod
    def format_context(results: list[SearchResult]) -> str:
        """Render SearchResults as RAG context (AGENTS.md §45).

        Format is decoupled from Qdrant payloads:
        ``[Source: <file>, Page: N]\\n\\n<content>``
        """
        blocks: list[str] = []
        for r in results:
            header = "[Source: "
            source = r.metadata.get("source") or r.document_id
            header += str(source)
            page = r.metadata.get("page")
            if page is not None:
                header += f", Page: {page}"
            header += "]"
            if r.content:
                body = r.content
            elif r.image_path:
                body = f"[Image: {Path(r.image_path).name}]"
            else:
                body = "（无内容）"
            blocks.append(f"{header}\n\n{body}")
        return "\n\n".join(blocks)

    # -- lifecycle --------------------------------------------------------

    async def close(self) -> None:
        for client in (self._embedding, self._image_embedding, self._reranker):
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # pragma: no cover - teardown
                    logger.exception("[RAG] 关闭客户端失败")
        self._cache.close()
        await self._store.close()
