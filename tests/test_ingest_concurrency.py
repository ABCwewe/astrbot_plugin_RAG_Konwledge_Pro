"""Tests for the index concurrency limiter (ingest_concurrency)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.chunking import TextChunker
from core.config import RAGConfig
from core.engine import RAGEngine
from core.indexing.manager import IndexManager
from core.providers.cache import EmbeddingCache

from tests.fakes import FakeEmbedding, FakeVectorStore


def _config(concurrency: int = 1) -> RAGConfig:
    return RAGConfig.from_dict(
        {
            "embedding": {"api_base": "x", "api_key": "k", "model": "m", "dimension": 8},
            "rerank": {"enabled": False},
            "ingest_concurrency": concurrency,
        }
    )


class CountingStore(FakeVectorStore):
    """Tracks concurrent create_collection calls (inside the limiter scope)."""

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0

    async def create_collection(self, name, vectors) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.05)  # widen the overlap window
            await super().create_collection(name, vectors)
        finally:
            self.active -= 1


def _write(root: Path, name: str, text: str) -> None:
    docs = root / "kbs" / "kb" / "documents"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / name).write_text(text, encoding="utf-8")


async def test_concurrency_limit_serializes_index_ops(tmp_path):
    store = CountingStore()
    manager = IndexManager(
        store,
        FakeEmbedding(dim=8),
        TextChunker(separator="\n\n", chunk_size=100, chunk_overlap=10),
        EmbeddingCache(tmp_path / "cache.db"),
        tmp_path / "rag",
        ingest_concurrency=1,
    )
    _write(tmp_path / "rag" / "kbs" / "kb1", "a.md", "内容A")
    _write(tmp_path / "rag" / "kbs" / "kb2", "b.md", "内容B")
    _write(tmp_path / "rag" / "kbs" / "kb3", "c.md", "内容C")

    results = await asyncio.gather(
        manager.rebuild("kb1", _config()),
        manager.rebuild("kb2", _config()),
        manager.rebuild("kb3", _config()),
    )
    assert [r["action"] for r in results] == ["rebuilt"] * 3
    # With concurrency=1 the three rebuilds must never overlap.
    assert store.max_active == 1
    stats = manager.get_index_stats()
    assert stats["completed"] == 3
    assert stats["running"] == 0
    assert stats["queued"] == 0
    assert stats["max_concurrent"] == 1


async def test_concurrency_limit_allows_parallel_up_to_max(tmp_path):
    store = CountingStore()
    manager = IndexManager(
        store,
        FakeEmbedding(dim=8),
        TextChunker(separator="\n\n", chunk_size=100, chunk_overlap=10),
        EmbeddingCache(tmp_path / "cache.db"),
        tmp_path / "rag",
        ingest_concurrency=2,
    )
    _write(tmp_path / "rag" / "kbs" / "kb1", "a.md", "A")
    _write(tmp_path / "rag" / "kbs" / "kb2", "b.md", "B")
    _write(tmp_path / "rag" / "kbs" / "kb3", "c.md", "C")

    await asyncio.gather(
        manager.rebuild("kb1", _config(2)),
        manager.rebuild("kb2", _config(2)),
        manager.rebuild("kb3", _config(2)),
    )
    # Two may run at once, never three.
    assert store.max_active == 2
    assert manager.get_index_stats()["completed"] == 3


async def test_config_parse_and_validation():
    cfg = _config(4)
    assert cfg.ingest_concurrency == 4
    # invalid → ConfigurationError
    from core.exceptions import ConfigurationError

    try:
        _config(0).validate()
        assert False, "expected ConfigurationError"
    except ConfigurationError:
        pass


async def test_engine_exposes_index_stats(tmp_path):
    engine = RAGEngine(
        _config(2),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=FakeEmbedding(dim=8),
        image_embedding=None,
        reranker=None,
    )
    stats = engine.get_index_stats()
    assert stats["max_concurrent"] == 2
    assert stats["completed"] == 0
    await engine.close()


class _SlowEmbedding(FakeEmbedding):
    async def embed_text(self, texts, *, input_type=None):
        await asyncio.sleep(0.15)
        return await super().embed_text(texts, input_type=input_type)


async def test_incremental_sync_reports_doc_progress(tmp_path):
    """增量同步必须逐文档上报进度（status=SYNCING → 完成回 idle）。"""
    engine = RAGEngine(
        _config(2),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=_SlowEmbedding(dim=8),
        image_embedding=None,
        reranker=None,
    )
    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.md").write_text("初始内容", encoding="utf-8")
    await engine.ingest("kb", [docs / "a.md"])

    # 追加 3 个文档，触发一次增量同步
    for i in range(3):
        (tmp_path / "rag" / "kbs" / "kb" / "documents" / f"new{i}.md").write_text(
            f"新内容{i}", encoding="utf-8"
        )
    task = asyncio.create_task(engine._manager.sync("kb", _config(2)))

    observed = None
    for _ in range(100):  # ≤5s
        p = engine._manager.get_progress("kb")
        if p and p.status == "SYNCING" and p.processed_documents > 0:
            observed = p
            break
        await asyncio.sleep(0.05)
    assert observed is not None, "未观察到增量同步的逐文档进度"
    assert observed.processed_documents >= 1
    assert observed.total_documents >= 1

    await task
    assert engine._manager.get_progress("kb").status == "idle"
    assert len(await engine.list_documents("kb")) == 4
    await engine.close()
