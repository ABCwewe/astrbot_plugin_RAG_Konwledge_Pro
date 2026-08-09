"""Index manager — versioning, rebuild safety, rollback (AGENTS.md §18, §41).

Rebuild flow:
    active v2 → create v3 (CREATING) → build (BUILDING) → verify → READY
              → switch active = v3 → delete v2

Any failure before READY marks v3 FAILED and deletes its collection; the old
active version stays untouched and usable. A rebuild is serialized per
knowledge base (at most one running task, AGENTS.md §27).

Versioned collection names come from
:func:`~core.indexing.manifest.collection_name_for`.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from ..chunking import TextChunker
from ..config import RAGConfig
from ..exceptions import (
    ConfigurationError,
    IndexBuildError,
    IndexBusyError,
    IndexNotFoundError,
    QdrantError,
)
from ..parsers import ParserRegistry
from ..providers import EmbeddingProvider
from ..providers.cache import EmbeddingCache
from ..storage.base import VectorStore
from .indexer import IndexStats, Indexer
from .manifest import (
    STATUS_BUILDING,
    STATUS_CREATING,
    STATUS_FAILED,
    STATUS_READY,
    ActiveState,
    DocumentRecord,
    IndexManifest,
    ManifestStore,
    collection_name_for,
)

logger = logging.getLogger("rag.index.manager")


class _IndexLimiter:
    """Global concurrency limiter for index operations (rebuild/sync).

    Reports live queue stats (running / queued / completed) so the WebUI can
    mirror indexing progress while uploads stream in.
    """

    def __init__(self, max_concurrent: int) -> None:
        self._max = max(1, max_concurrent)
        self._running = 0
        self._queued = 0
        self._completed = 0
        self._cond = asyncio.Condition()

    async def __aenter__(self) -> "_IndexLimiter":
        async with self._cond:
            if self._running >= self._max:
                self._queued += 1
                try:
                    await self._cond.wait_for(lambda: self._running < self._max)
                finally:
                    self._queued -= 1
            self._running += 1
        return self

    async def __aexit__(self, *exc) -> None:
        async with self._cond:
            self._running -= 1
            self._completed += 1
            self._cond.notify_all()

    def stats(self) -> dict:
        return {
            "running": self._running,
            "queued": self._queued,
            "completed": self._completed,
            "max_concurrent": self._max,
        }


@dataclass
class BuildProgress:
    kb_id: str
    status: str = "idle"
    current_version: int | None = None
    target_version: int | None = None
    total_documents: int = 0
    processed_documents: int = 0
    processed_chunks: int = 0
    error: str | None = None
    started_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "kb_id": self.kb_id,
            "status": self.status,
            "current_version": self.current_version,
            "target_version": self.target_version,
            "total_documents": self.total_documents,
            "processed_documents": self.processed_documents,
            "processed_chunks": self.processed_chunks,
            "error": self.error,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


class IndexManager:
    def __init__(
        self,
        store: VectorStore,
        embedding: EmbeddingProvider,
        chunker: TextChunker,
        cache: EmbeddingCache,
        data_root: str | Path,
        *,
        image_embedding: EmbeddingProvider | None = None,
        parsers: ParserRegistry | None = None,
        ingest_concurrency: int = 2,
    ) -> None:
        self._store = store
        self._embedding = embedding
        self._image_embedding = image_embedding
        self._chunker = chunker
        self._cache = cache
        self._parsers = parsers or ParserRegistry()
        self._data_root = Path(data_root)
        self._locks: dict[str, asyncio.Lock] = {}
        self._progress: dict[str, BuildProgress] = {}
        self._limiter = _IndexLimiter(ingest_concurrency)
        self._op_counts = {"rebuild": 0, "sync": 0}

    def get_index_stats(self) -> dict:
        """Live index-queue stats: running / queued / completed / max."""
        return self._limiter.stats()

    def get_op_counts(self) -> dict:
        """Index operation counters (rebuild / sync waves)."""
        return dict(self._op_counts)

    # -- paths ------------------------------------------------------------

    def _kb_root(self, kb_id: str) -> Path:
        if not kb_id or not kb_id.strip():
            raise ConfigurationError("知识库 ID 不能为空")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in kb_id)
        # 所有知识库统一放在 <data_root>/kbs/ 下，与 tmp/、cache.db 等
        # 引擎级目录同级隔离。
        return self._data_root / "kbs" / safe

    def _docs_dir(self, kb_id: str) -> Path:
        return self._kb_root(kb_id) / "documents"

    def _manifest_store(self, kb_id: str) -> ManifestStore:
        return ManifestStore(self._kb_root(kb_id))

    def _make_indexer(self) -> Indexer:
        return Indexer(
            self._store,
            self._embedding,
            self._chunker,
            self._cache,
            parsers=self._parsers,
            image_embedding=self._image_embedding,
        )

    # -- progress ---------------------------------------------------------

    def get_progress(self, kb_id: str) -> BuildProgress | None:
        return self._progress.get(kb_id)

    def _progress_for(self, kb_id: str) -> BuildProgress:
        return self._progress.setdefault(kb_id, BuildProgress(kb_id=kb_id))

    # -- public API -------------------------------------------------------

    async def ensure_index(self, kb_id: str, config: RAGConfig) -> dict:
        """Guarantee a READY index for ``kb_id`` under ``config``.

        - no active index or config hash changed → rebuild
        - otherwise → incremental sync against the active collection
        """
        config.validate()
        store = self._manifest_store(kb_id)
        active = await store.load_active()
        if active is None or active.config_hash != config.config_hash():
            logger.info(
                "[INDEX] 配置变更或无活动索引，触发重建 (kb=%s)", kb_id
            )
            return await self.rebuild(kb_id, config)
        return await self.sync(kb_id, config)

    async def rebuild(self, kb_id: str, config: RAGConfig) -> dict:
        """Full rebuild into a new versioned collection (serialized per kb)."""
        config.validate()
        lock = self._locks.setdefault(kb_id, asyncio.Lock())
        if lock.locked():
            raise IndexBusyError(f"知识库 {kb_id} 正在重建中")
        async with lock:
            return await self._rebuild_locked(kb_id, config)

    async def sync(self, kb_id: str, config: RAGConfig) -> dict:
        """Incremental add/update/delete against the active READY index.

        Holds the per-KB lock so delete_kb/delete_index can never race an
        in-flight sync (stale registry resurrection after a delete).
        """
        self._op_counts["sync"] += 1
        lock = self._locks.setdefault(kb_id, asyncio.Lock())
        async with lock:
            return await self._sync_locked(kb_id, config)

    async def _sync_locked(self, kb_id: str, config: RAGConfig) -> dict:
        store = self._manifest_store(kb_id)
        active = await store.load_active()
        manifest = (
            await store.load_manifest(active.active_version)
            if active and active.active_version is not None
            else None
        )
        if (
            not active
            or manifest is None
            or manifest.status != STATUS_READY
            or not manifest.collection_name
        ):
            raise IndexNotFoundError(f"知识库 {kb_id} 没有可用的 READY 索引")

        async with self._limiter:
            doc_paths = self._list_docs(kb_id)
            documents = await store.load_documents()
            indexer = self._make_indexer()
            stats = await indexer.sync_documents(
                manifest.collection_name,
                kb_id,
                self._docs_dir(kb_id),
                doc_paths,
                documents,
            )
            await store.save_documents(documents)
            manifest.document_count = len(documents)
            manifest.chunk_count = stats.chunks_written
            await store.save_manifest(manifest)
        logger.info(
            "[INDEX] 增量同步完成 (kb=%s): added=%d updated=%d unchanged=%d deleted=%d",
            kb_id,
            stats.added,
            stats.updated,
            stats.unchanged,
            stats.deleted,
        )
        return {
            "action": "synced",
            "added": stats.added,
            "updated": stats.updated,
            "unchanged": stats.unchanged,
            "deleted": stats.deleted,
            "chunks": stats.chunks_written,
        }

    async def active_collection(self, kb_id: str) -> str | None:
        """Name of the READY active collection, or ``None`` if unavailable."""
        store = self._manifest_store(kb_id)
        active = await store.load_active()
        if not active or active.active_version is None:
            return None
        manifest = await store.load_manifest(active.active_version)
        if not manifest or manifest.status != STATUS_READY:
            return None
        return manifest.collection_name

    async def delete_kb(self, kb_id: str) -> None:
        """Drop every collection belonging to the kb and its local data.

        Holds the per-KB lock so in-flight index operations for this KB finish
        before deletion (no resurrection / stale registry).
        """
        lock = self._locks.setdefault(kb_id, asyncio.Lock())
        async with lock:
            store = self._manifest_store(kb_id)
            manifests = await store.list_manifests()
            for manifest in manifests.values():
                try:
                    await self._store.delete_collection(manifest.collection_name)
                except QdrantError as exc:
                    logger.warning("[QDRANT] 删除 collection 失败: %s", exc)
            root = self._kb_root(kb_id)
            if root.exists():
                await asyncio.to_thread(_rmtree_retry, root)
            self._progress.pop(kb_id, None)

    async def delete_index(self, kb_id: str) -> None:
        """Delete only the KB's index (collections + manifests + registry),
        keeping the KB root and its document files."""
        lock = self._locks.setdefault(kb_id, asyncio.Lock())
        async with lock:
            store = self._manifest_store(kb_id)
            manifests = await store.list_manifests()
            for manifest in manifests.values():
                try:
                    await self._store.delete_collection(manifest.collection_name)
                except QdrantError as exc:
                    logger.warning("[QDRANT] 删除 collection 失败: %s", exc)
                await store.delete_manifest(manifest.version)
            for path in (store.active_path, store.documents_path):
                if path.exists():
                    await asyncio.to_thread(path.unlink)
            self._progress.pop(kb_id, None)
            logger.info("[INDEX] 已删除知识库 %s 的索引（文档保留）", kb_id)

    async def status(self, kb_id: str, config: RAGConfig | None = None) -> dict:
        """Human/UI-readable snapshot of a knowledge base."""
        store = self._manifest_store(kb_id)
        active = await store.load_active()
        manifests = await store.list_manifests()
        current_config_hash = config.config_hash() if config else None
        return {
            "kb_id": kb_id,
            "active_version": active.active_version if active else None,
            "config_changed": bool(
                active and current_config_hash and active.config_hash != current_config_hash
            ),
            "active_config_hash": active.config_hash if active else None,
            "current_config_hash": current_config_hash,
            "manifest_count": len(manifests),
            "versions": {
                v: {
                    "status": m.status,
                    "document_count": m.document_count,
                    "chunk_count": m.chunk_count,
                    "collection": m.collection_name,
                    "error": m.error,
                }
                for v, m in sorted(manifests.items())
            },
            "progress": (
                self._progress.get(kb_id).to_dict() if kb_id in self._progress else None
            ),
        }

    # -- internals --------------------------------------------------------

    async def _rebuild_locked(self, kb_id: str, config: RAGConfig) -> dict:
        self._op_counts["rebuild"] += 1
        store = self._manifest_store(kb_id)
        manifests = await store.list_manifests()
        active = await store.load_active()

        next_version = (max(manifests) + 1) if manifests else 1
        collection = collection_name_for(kb_id, next_version)
        manifest = IndexManifest.from_config(
            kb_id, next_version, config, status=STATUS_CREATING
        )
        await store.save_manifest(manifest)

        progress = self._progress_for(kb_id)
        progress.kb_id = kb_id
        progress.status = STATUS_CREATING
        progress.current_version = active.active_version if active else None
        progress.target_version = next_version
        progress.error = None
        progress.started_at = _now()
        _touch(progress)

        async with self._limiter:
            try:
                vectors = {"text": config.embedding.dimension}
                if config.image.enabled:
                    vectors["image"] = int(config.image.dimension or 0)
                # A collection with the target name can never be referenced by a
                # READY manifest (next_version > every manifest version), so if it
                # exists it is an orphan from an aborted build — drop it first.
                if await self._store.collection_exists(collection):
                    logger.warning("[INDEX] 清理遗留 collection: %s", collection)
                    await self._store.delete_collection(collection)
                await self._store.create_collection(collection, vectors)
                logger.info("[INDEX] 已创建新 collection: %s", collection)

                manifest.status = STATUS_BUILDING
                await store.save_manifest(manifest)
                progress.status = STATUS_BUILDING
                _touch(progress)

                doc_paths = self._list_docs(kb_id)
                progress.total_documents = len(doc_paths)
                # Fresh registry: a rebuild must re-index everything, even files
                # whose content hash is unchanged (chunking/embedding may differ).
                documents: dict[str, DocumentRecord] = {}
                indexer = self._make_indexer()
                stats = await indexer.sync_documents(
                    collection,
                    kb_id,
                    self._docs_dir(kb_id),
                    doc_paths,
                    documents,
                    progress=lambda done, total, chunks: (
                        self._tick_progress(kb_id, done, total, chunks)
                    ),
                )
                if stats.errors:
                    raise IndexBuildError(
                        "部分文档索引失败: " + "; ".join(stats.errors[:5])
                    )

                await self._verify(collection, documents, stats)

                manifest.document_count = len(documents)
                manifest.chunk_count = stats.chunks_written
                manifest.status = STATUS_READY
                manifest.error = None
                await store.save_manifest(manifest)
                await store.save_documents(documents)

                await store.save_active(
                    ActiveState(
                        kb_id=kb_id,
                        active_version=next_version,
                        config_hash=config.config_hash(),
                    )
                )
                progress.status = STATUS_READY
                progress.processed_chunks = stats.chunks_written
                _touch(progress)
                logger.info(
                    "[INDEX] 索引构建完成，切换到 %s (kb=%s)", collection, kb_id
                )

                # Best-effort cleanup of superseded versions.
                for version, old in manifests.items():
                    if version == next_version:
                        continue
                    try:
                        await self._store.delete_collection(old.collection_name)
                        logger.info(
                            "[INDEX] 已删除旧 collection: %s", old.collection_name
                        )
                    except QdrantError as exc:
                        logger.warning("[QDRANT] 删除旧 collection 失败: %s", exc)
                    await store.delete_manifest(version)

                return {
                    "action": "rebuilt",
                    "version": next_version,
                    "collection": collection,
                    "documents": len(documents),
                    "chunks": stats.chunks_written,
                }
            except Exception as exc:
                logger.exception("[INDEX] 重建失败，回滚 (kb=%s)", kb_id)
                manifest.status = STATUS_FAILED
                manifest.error = str(exc)
                await store.save_manifest(manifest)
                progress.status = STATUS_FAILED
                progress.error = str(exc)
                _touch(progress)
                try:
                    await self._store.delete_collection(collection)
                except QdrantError:
                    logger.warning("[QDRANT] 回滚删除失败: %s", collection)
                if isinstance(exc, (IndexBuildError, ConfigurationError)):
                    raise
                raise IndexBuildError(f"索引重建失败: {exc}") from exc

    async def _verify(
        self, collection: str, documents: dict[str, DocumentRecord], stats: IndexStats
    ) -> None:
        """Consistency checks before switching active (AGENTS.md §42)."""
        if not await self._store.collection_exists(collection):
            raise IndexBuildError("新 collection 不存在，拒绝切换")
        count = await self._store.count(collection)
        expected = stats.chunks_written
        if expected > 0 and count != expected:
            raise IndexBuildError(
                f"向量点数量不一致: 期望 {expected}, 实际 {count}，拒绝切换"
            )
        if count == 0 and documents:
            raise IndexBuildError("知识库非空但向量点数为 0，拒绝切换")

    def _tick_progress(self, kb_id: str, done: int, total: int, chunks: int) -> None:
        progress = self._progress_for(kb_id)
        progress.processed_documents = done
        progress.total_documents = total
        progress.processed_chunks = chunks
        _touch(progress)

    def _list_docs(self, kb_id: str) -> list[Path]:
        docs_dir = self._docs_dir(kb_id)
        if not docs_dir.exists():
            return []
        return sorted(
            p
            for p in docs_dir.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        )


def _rmtree_retry(root: Path, attempts: int = 5, delay: float = 0.2) -> None:
    """Delete a directory tree robustly (Windows: locked files can make
    rmtree fail mid-way with ignore_errors silently leaving data behind —
    which would keep the stale document registry and cause re-uploads to be
    skipped as "unchanged"). Retries and verifies; raises on final failure."""
    for attempt in range(attempts):
        try:
            shutil.rmtree(root)
        except OSError as exc:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            continue
        if not root.exists():
            return
        time.sleep(delay)
    raise OSError(f"无法完全删除目录: {root}")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _touch(progress: BuildProgress) -> None:
    progress.updated_at = _now()
