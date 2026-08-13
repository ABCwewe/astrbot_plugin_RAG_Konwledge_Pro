"""IndexManifest / ManifestStore / config_hash tests (AGENTS.md §16-§17)."""

from __future__ import annotations

from core.config import RAGConfig
from core.indexing.manifest import (
    STATUS_READY,
    ActiveState,
    DocumentRecord,
    IndexManifest,
    ManifestStore,
    collection_name_for,
    document_id_for,
)


def _config(overrides: dict | None = None) -> RAGConfig:
    data = {
        "embedding": {
            "api_base": "http://localhost:1",
            "api_key": "k",
            "model": "m",
            "dimension": 8,
        },
        "rerank": {"enabled": False},
        **(overrides or {}),
    }
    return RAGConfig.from_dict(data)


def test_config_hash_changes_with_relevant_settings():
    base = _config().config_hash()
    assert _config({"embedding": {"api_base": "http://localhost:1", "api_key": "k", "model": "other", "dimension": 8}}).config_hash() != base
    assert _config({"chunking": {"separator": "\n", "chunk_size": 800, "chunk_overlap": 100}}).config_hash() != base
    assert _config({"chunking": {"separator": "\n\n", "chunk_size": 900, "chunk_overlap": 100}}).config_hash() != base


def test_config_hash_ignores_retrieval_only_settings():
    a = _config({"top_k": 30, "top_n": 6}).config_hash()
    b = _config({"top_k": 50, "top_n": 3}).config_hash()
    assert a == b


def test_config_hash_enables_image_changes():
    base = _config().config_hash()
    with_img = _config(
        {
            "image": {
                "enabled": True,
                "api_base": "http://localhost:1",
                "api_key": "k",
                "model": "m-img",
                "dimension": 8,
            }
        }
    ).config_hash()
    assert base != with_img


def test_collection_and_document_ids_are_stable():
    # 空/遗留前缀回退到 astrbot_rag（向后兼容）
    assert collection_name_for("", "default", 1) == "astrbot_rag_default_v1"
    assert collection_name_for("astrbot_rag", "default", 3) == "astrbot_rag_default_v3"
    assert collection_name_for("", "my kb/1", 2) == "astrbot_rag_my_kb_1_v2"
    # 自定义命名空间隔离（多 client 共享 Qdrant 时每个 client 唯一）
    assert collection_name_for("tenantA", "default", 1) == "tenantA_default_v1"
    assert collection_name_for("a b", "default", 1) == "a_b_default_v1"
    assert collection_name_for("", "default", 1) != collection_name_for("tenantA", "default", 1)
    # 文档 ID 稳定
    assert document_id_for("default", "a.md") == document_id_for("default", "a.md")
    assert document_id_for("default", "a.md") != document_id_for("other", "a.md")
    assert document_id_for("default", "a.md") != document_id_for("default", "b.md")


def test_manifest_round_trip(tmp_path):
    store = ManifestStore(tmp_path)
    manifest = IndexManifest.from_config("default", 2, _config(), status=STATUS_READY)
    manifest.document_count = 5
    manifest.chunk_count = 42
    # async store writes to disk
    import asyncio

    async def run():
        await store.save_manifest(manifest)
        loaded = await store.load_manifest(2)
        assert loaded is not None
        assert loaded.status == STATUS_READY
        assert loaded.kb_id == "default"
        assert loaded.version == 2
        assert loaded.collection_name == "astrbot_rag_default_v2"
        assert loaded.document_count == 5
        assert loaded.chunk_count == 42
        assert loaded.config_hash == _config().config_hash()

    asyncio.run(run())


def test_active_and_documents_persistence(tmp_path):
    import asyncio

    async def run():
        store = ManifestStore(tmp_path)
        assert await store.load_active() is None
        await store.save_active(ActiveState(kb_id="default", active_version=2, config_hash="h"))
        state = await store.load_active()
        assert state.active_version == 2
        assert state.config_hash == "h"

        records = {"id1": DocumentRecord(document_id="id1", source="a.md", filename="a.md", content_hash="c1")}
        await store.save_documents(records)
        loaded = await store.load_documents()
        assert loaded["id1"].source == "a.md"
        assert loaded["id1"].content_hash == "c1"

    asyncio.run(run())


def test_manifest_from_config_has_expected_fields():
    manifest = IndexManifest.from_config("default", 1, _config())
    assert manifest.embedding_model == "m"
    assert manifest.embedding_dimension == 8
    assert manifest.embedding_provider == "openai_compatible"
    assert manifest.image_embedding_model is None
    assert manifest.chunk_size == 800
    assert manifest.chunk_overlap == 100
    assert manifest.chunk_separator == "\n\n"
