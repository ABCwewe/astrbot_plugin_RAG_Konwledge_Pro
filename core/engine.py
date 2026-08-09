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
import os
import shutil
import uuid
from pathlib import Path

from .chunking import TextChunker
from .config import RAGConfig
from .exceptions import IndexBusyError, IndexNotFoundError, ParserError
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
            ingest_concurrency=config.ingest_concurrency,
        )
        self._retriever = Retriever(
            self._store,
            self._embedding,
            self._reranker,
            config,
            image_embedding=self._image_embedding,
        )
        # 上传队列：收件箱原子落盘 → 后台工人按波次合并索引
        self._inflight_parts: set[str] = set()
        self._worker_task = asyncio.create_task(
            self._upload_worker(), name="rag-upload-worker"
        )

    # -- knowledge base management ----------------------------------------

    def kb_root(self, kb_id: str) -> Path:
        from .exceptions import ConfigurationError

        if not kb_id or not kb_id.strip():
            raise ConfigurationError("知识库 ID 不能为空")
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

    def inbox_dir(self, kb_id: str) -> Path:
        """上传收件箱：文件先原子落盘到这里，由后台工人波次移入 documents。"""
        return self.kb_root(kb_id) / "inbox"

    async def receive_upload(
        self, kb_id: str, filename: str, saver
    ) -> str:
        """原子接收一个上传文件（AGENTS.md 数据一致性）。

        写入 ``inbox/.upload-<uuid>.part``，完整成功后才 ``os.replace``
        发布为正式文件名；任何中断/异常都会清理临时文件，知识库内
        不会出现半截文件。``saver(tmp_path)`` 为异步写入回调。
        """
        from .exceptions import ConfigurationError, ParserError

        if not kb_id or not kb_id.strip():
            raise ConfigurationError("知识库 ID 不能为空")
        name = Path(filename).name
        if not name or name.startswith("."):
            raise ParserError(f"非法的文件名: {filename}")
        inbox = self.inbox_dir(kb_id)
        inbox.mkdir(parents=True, exist_ok=True)
        tmp = inbox / f".upload-{uuid.uuid4().hex}.part"
        self._inflight_parts.add(str(tmp))
        try:
            await saver(tmp)
            os.replace(tmp, inbox / name)
        except BaseException:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        finally:
            self._inflight_parts.discard(str(tmp))
        return str(inbox / name)

    # -- 上传队列后台工人 --------------------------------------------------

    async def _upload_worker(self) -> None:
        """后台索引工人：轮询各知识库收件箱，按波次合并索引。

        - 波次 = 一次收集当前全部已发布文件 → 移入 documents → 一次
          ensure_index（受 ingest_concurrency 限流）
        - 上传与索引完全解耦：前端可先传完，后端按动态队列持续消费
        """
        while True:
            try:
                await self._drain_uploads_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[RAG] 上传队列处理异常")
            await asyncio.sleep(0.5)

    async def _drain_uploads_once(self) -> None:
        kbs_dir = self._data_root / "kbs"
        if not kbs_dir.exists():
            return
        waves: list[asyncio.Task] = []
        for child in kbs_dir.iterdir():
            if not child.is_dir():
                continue
            inbox = child / "inbox"
            if not inbox.is_dir():
                continue
            # 清理残留临时文件（非进行中的 .part）
            for part in list(inbox.iterdir()):
                if (
                    part.is_file()
                    and part.name.startswith(".")
                    and str(part) not in self._inflight_parts
                ):
                    try:
                        part.unlink()
                    except OSError:
                        pass
            published = [
                p
                for p in inbox.iterdir()
                if p.is_file() and not p.name.startswith(".")
            ]
            if not published:
                continue
            waves.append(
                asyncio.create_task(
                    self._index_wave(child.name, published), name="rag-wave"
                )
            )
        if waves:
            await asyncio.gather(*waves, return_exceptions=True)

    async def _index_wave(self, kb_id: str, inbox_files: list[Path]) -> None:
        docs = self.docs_dir(kb_id)
        docs.mkdir(parents=True, exist_ok=True)
        moved: list[Path] = []
        for src in inbox_files:
            try:
                os.replace(src, docs / src.name)
                moved.append(src)
            except OSError as exc:
                logger.warning("[RAG] 移动上传文件失败: %s (%s)", src, exc)
        if moved:
            logger.info("[RAG] 上传队列波次: kb=%s 文件=%d", kb_id, len(moved))
        try:
            await self._manager.ensure_index(kb_id, self.config)
        except IndexBusyError:
            # 库忙（重建/删除进行中）：文件放回收件箱，下一轮重试
            if self.kb_root(kb_id).exists():
                inbox = self.inbox_dir(kb_id)
                inbox.mkdir(parents=True, exist_ok=True)
                for src in moved:
                    try:
                        os.replace(docs / src.name, src)
                    except OSError:
                        pass
                logger.info("[RAG] 波次重试放回收件箱 (kb=%s)", kb_id)
            else:
                logger.info("[RAG] 波次目标知识库已删除，丢弃文件 (kb=%s)", kb_id)
        except IndexNotFoundError:
            # 知识库已删除/无索引：丢弃本次波次文件，不复活旧数据
            logger.info("[RAG] 波次目标知识库已删除，丢弃文件 (kb=%s)", kb_id)

    def get_upload_stats(self) -> dict:
        """动态队列状态：收件箱待索引数 + 索引限流统计。"""
        pending = 0
        by_kb: dict[str, int] = {}
        kbs_dir = self._data_root / "kbs"
        if kbs_dir.exists():
            for child in kbs_dir.iterdir():
                if not child.is_dir():
                    continue
                inbox = child / "inbox"
                if not inbox.is_dir():
                    continue
                n = sum(
                    1
                    for f in inbox.iterdir()
                    if f.is_file() and not f.name.startswith(".")
                )
                if n:
                    by_kb[child.name] = n
                    pending += n
        return {
            **self._manager.get_index_stats(),
            "inbox_pending": pending,
            "inbox_by_kb": by_kb,
        }

    def get_index_op_counts(self) -> dict:
        """索引操作计数（rebuild/sync），测试与观测用。"""
        return self._manager.get_op_counts()

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

    def get_index_stats(self) -> dict:
        """Live index-queue stats (running / queued / completed / max)."""
        return self._manager.get_index_stats()

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

    async def search_image_by_path(
        self,
        kb_id: str,
        image_path: str | Path,
        *,
        top_n: int | None = None,
    ) -> list[SearchResult]:
        """Image-as-query retrieval: embed a local image file and search the
        ``image`` named vector of one KB (no rerank — a text reranker cannot
        judge images).
        """
        from .exceptions import ConfigurationError, IndexNotFoundError

        if not self.config.image.enabled or self._image_embedding is None:
            raise ConfigurationError("图片检索未启用（image.enabled=false）")
        collection = await self._manager.active_collection(kb_id)
        if collection is None:
            raise IndexNotFoundError(
                f"知识库 {kb_id} 尚未构建索引，请先导入文档或执行 /rag rebuild"
            )
        data = await asyncio.to_thread(Path(image_path).read_bytes)
        return await self._retriever.retrieve_by_image(
            collection, data, top_n=top_n
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
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass
        for client in (self._embedding, self._image_embedding, self._reranker):
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # pragma: no cover - teardown
                    logger.exception("[RAG] 关闭客户端失败")
        self._cache.close()
        await self._store.close()
