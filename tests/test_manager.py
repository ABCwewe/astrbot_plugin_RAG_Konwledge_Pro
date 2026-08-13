"""IndexManager tests — the rebuild/rollback/version-switch safety core
(AGENTS.md §18, §41-§42)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.chunking import TextChunker
from core.config import RAGConfig
from core.exceptions import IndexBusyError, IndexBuildError, IndexNotFoundError
from core.indexing.manager import IndexManager
from core.providers.cache import EmbeddingCache

from tests.fakes import FakeEmbedding, FakeVectorStore


def _config(overrides: dict | None = None) -> RAGConfig:
    data = {
        "embedding": {
            "api_base": "http://localhost:1",
            "api_key": "k",
            "model": "m",
            "dimension": 8,
        },
        "rerank": {"enabled": False},
        "chunking": {"separator": "\n\n", "chunk_size": 100, "chunk_overlap": 10},
        **(overrides or {}),
    }
    return RAGConfig.from_dict(data)


def _manager(tmp_path, store=None, embedding=None, prefix: str = "") -> tuple[IndexManager, FakeVectorStore, FakeEmbedding]:
    store = store or FakeVectorStore()
    embedding = embedding or FakeEmbedding(dim=8)
    cache = EmbeddingCache(tmp_path / "cache.db")
    manager = IndexManager(
        store,
        embedding,
        TextChunker(separator="\n\n", chunk_size=100, chunk_overlap=10),
        cache,
        tmp_path / "rag",
        collection_prefix=prefix,
    )
    return manager, store, embedding


def _write(root: Path, name: str, text: str) -> Path:
    docs = root / "documents"
    docs.mkdir(parents=True, exist_ok=True)
    path = docs / name
    path.write_text(text, encoding="utf-8")
    return path


async def test_initial_rebuild_creates_ready_v1(tmp_path):
    manager, store, _ = _manager(tmp_path)
    _write(tmp_path / "rag" / "kbs" / "kb", "a.md", "第一个文档内容")

    result = await manager.rebuild("kb", _config())
    assert result["version"] == 1
    assert result["documents"] == 1
    assert await manager.active_collection("kb") == "astrbot_rag_kb_v1"

    status = await manager.status("kb")
    assert status["active_version"] == 1
    assert status["versions"][1]["status"] == "READY"


async def test_config_change_triggers_rebuild_and_old_removed(tmp_path):
    manager, store, _ = _manager(tmp_path)
    _write(tmp_path / "rag" / "kbs" / "kb", "a.md", "内容")
    await manager.rebuild("kb", _config())
    assert store.collections.get("astrbot_rag_kb_v1") is not None

    result = await manager.ensure_index("kb", _config({"chunking": {"separator": "\n", "chunk_size": 100, "chunk_overlap": 10}}))
    assert result["action"] == "rebuilt"
    assert result["version"] == 2
    assert "astrbot_rag_kb_v1" not in store.collections  # old deleted
    assert "astrbot_rag_kb_v2" in store.collections


async def test_unchanged_config_performs_incremental_sync(tmp_path):
    manager, store, _ = _manager(tmp_path)
    _write(tmp_path / "rag" / "kbs" / "kb", "a.md", "内容A")
    await manager.rebuild("kb", _config())

    result = await manager.ensure_index("kb", _config())
    assert result["action"] == "synced"
    assert result["added"] == 0
    assert await manager.active_collection("kb") == "astrbot_rag_kb_v1"


async def test_rebuild_failure_rolls_back_and_keeps_old_active(tmp_path):
    manager, store, _ = _manager(tmp_path)
    _write(tmp_path / "rag" / "kbs" / "kb", "a.md", "内容A")
    await manager.rebuild("kb", _config())

    # Next rebuild will fail: embedding provider raises.
    class BoomEmbedding(FakeEmbedding):
        async def embed_text(self, texts, *, input_type=None):
            raise RuntimeError("api down")

    boom_manager, store2, _ = _manager(tmp_path, store=store, embedding=BoomEmbedding())
    _write(tmp_path / "rag" / "kbs" / "kb", "b.md", "会触发重建的新文档")

    with pytest.raises(IndexBuildError):
        await boom_manager.rebuild("kb", _config({"chunking": {"separator": "\n\n", "chunk_size": 50, "chunk_overlap": 5}}))

    # Old index intact and still active; failed v2 cleaned up.
    assert await boom_manager.active_collection("kb") == "astrbot_rag_kb_v1"
    assert "astrbot_rag_kb_v2" not in store.collections
    status = await boom_manager.status("kb")
    assert status["versions"][2]["status"] == "FAILED"
    assert status["versions"][2]["error"]


async def test_incremental_sync_adds_and_removes(tmp_path):
    manager, store, _ = _manager(tmp_path)
    _write(tmp_path / "rag" / "kbs" / "kb", "a.md", "内容A")
    await manager.rebuild("kb", _config())

    _write(tmp_path / "rag" / "kbs" / "kb", "b.md", "内容B")
    result = await manager.sync("kb", _config())
    assert result["added"] == 1
    assert await store.count("astrbot_rag_kb_v1") == 2

    (tmp_path / "rag" / "kbs" / "kb" / "documents" / "a.md").unlink()
    result = await manager.sync("kb", _config())
    assert result["deleted"] == 1
    assert await store.count("astrbot_rag_kb_v1") == 1


async def test_sync_without_ready_index_raises(tmp_path):
    manager, _, _ = _manager(tmp_path)
    with pytest.raises(IndexNotFoundError):
        await manager.sync("nokb", _config())


async def test_concurrent_rebuild_is_rejected(tmp_path):
    manager, store, embedding = _manager(tmp_path)
    _write(tmp_path / "rag" / "kbs" / "kb", "a.md", "内容A")

    original_embed = embedding.embed_text

    async def slow_embed(texts, *, input_type=None):
        await asyncio.sleep(0.2)
        return await original_embed(texts, input_type=input_type)

    embedding.embed_text = slow_embed  # type: ignore[method-assign]

    task = asyncio.create_task(manager.rebuild("kb", _config()))
    await asyncio.sleep(0.05)
    with pytest.raises(IndexBusyError):
        await manager.rebuild("kb", _config())
    await task


async def test_delete_kb_removes_collections_and_data(tmp_path):
    manager, store, _ = _manager(tmp_path)
    _write(tmp_path / "rag" / "kbs" / "kb", "a.md", "内容A")
    await manager.rebuild("kb", _config())

    await manager.delete_kb("kb")
    assert not store.collections
    assert not (tmp_path / "rag" / "kbs" / "kb").exists()


# ---------- collection 命名空间隔离（多 client 共享 Qdrant） ----------


async def test_custom_prefix_namespaces_collections(tmp_path):
    manager, store, _ = _manager(tmp_path, prefix="tenantA")
    _write(tmp_path / "rag" / "kbs" / "kb", "a.md", "内容")
    # 配置与构造器前缀保持一致（生产里由 adapter 统一解析后注入）
    await manager.rebuild("kb", _config({"qdrant": {"collection_prefix": "tenantA"}}))

    assert await manager.active_collection("kb") == "tenantA_kb_v1"
    assert "tenantA_kb_v1" in store.collections
    assert "astrbot_rag_kb_v1" not in store.collections


def test_owns_collection_guard(tmp_path):
    manager, _, _ = _manager(tmp_path, prefix="tenantA")
    assert manager._owns_collection("tenantA_kb_v1")           # 本命名空间
    assert manager._owns_collection("astrbot_rag_kb_v1")       # 遗留前缀兼容
    assert not manager._owns_collection("other_client_kb_v1")  # 外来集合
    assert not manager._owns_collection("tenantA")             # 缺 "_" 分隔
    assert not manager._owns_collection("")


async def test_delete_index_skips_foreign_collections(tmp_path):
    from core.indexing.manifest import STATUS_READY, IndexManifest, ManifestStore

    manager, store, _ = _manager(tmp_path, prefix="tenantA")
    kb_root = tmp_path / "rag" / "kbs" / "kb"
    kb_root.mkdir(parents=True, exist_ok=True)
    mstore = ManifestStore(kb_root)

    foreign = "other_client_kb_v1"
    own = "tenantA_kb_v1"
    await store.create_collection(foreign, {"text": 8})
    await store.create_collection(own, {"text": 8})

    cfg = _config({"qdrant": {"collection_prefix": "tenantA"}})
    bad = IndexManifest.from_config("kb", 1, cfg, status=STATUS_READY)
    bad.collection_name = foreign  # 模拟 manifest 错误引用外来集合
    await mstore.save_manifest(bad)
    good = IndexManifest.from_config("kb", 2, cfg, status=STATUS_READY)
    good.collection_name = own
    await mstore.save_manifest(good)

    await manager.delete_index("kb")

    assert foreign in store.collections                       # 外来集合保留
    assert own not in store.collections                       # 本命名空间集合被删
    assert not list(kb_root.glob("manifest_v*.json"))         # manifest 文件仍清理
