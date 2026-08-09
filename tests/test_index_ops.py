"""Tests for drop_index (keep KB + docs) and default search KBs."""

from __future__ import annotations

from pathlib import Path

from core.config import RAGConfig
from core.engine import RAGEngine
from core.exceptions import IndexNotFoundError

from adapter.config_utils import coerce_value, normalize_kb_ids
from tests.fakes import FakeEmbedding, FakeVectorStore


def _config() -> RAGConfig:
    return RAGConfig.from_dict(
        {
            "embedding": {
                "api_base": "http://localhost:1",
                "api_key": "k",
                "model": "m",
                "dimension": 8,
            },
            "rerank": {"enabled": False},
        }
    )


async def test_drop_index_keeps_kb_and_documents(tmp_path):
    engine = RAGEngine(
        _config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=FakeEmbedding(),
        image_embedding=None,
        reranker=None,
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("文档内容一", encoding="utf-8")
    await engine.ingest("kb", [docs / "a.md"])
    st = await engine.status("kb")
    assert st["active_version"] == 1

    await engine.drop_index("kb")
    st = await engine.status("kb")
    assert st["active_version"] is None
    assert st["versions"] == {}
    # KB root + documents survive, registry cleared → docs show as unindexed
    assert (tmp_path / "rag" / "kbs" / "kb" / "documents" / "a.md").exists()
    docs_list = await engine.list_documents("kb")
    assert len(docs_list) == 1
    assert docs_list[0]["indexed"] is False

    # re-ingest rebuilds the index from the kept documents
    result = await engine.ingest("kb", [tmp_path / "rag" / "kbs" / "kb" / "documents" / "a.md"])
    assert result["action"] == "rebuilt"
    assert (await engine.status("kb"))["active_version"] == 1
    await engine.close()


async def test_search_without_index_raises(tmp_path):
    engine = RAGEngine(
        _config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=FakeEmbedding(),
        image_embedding=None,
        reranker=None,
    )
    try:
        await engine.search("nokb", "问题")
        assert False, "expected IndexNotFoundError"
    except IndexNotFoundError:
        pass
    await engine.close()


def test_normalize_kb_ids():
    assert normalize_kb_ids(["kb1", "kb2", "", "kb1", "kb2"]) == ["kb1", "kb2"]
    assert normalize_kb_ids([]) == []
    assert normalize_kb_ids(None) == []


def test_coerce_list_type():
    schema = {"default_kb_ids": {"type": "list"}}
    assert coerce_value("default_kb_ids", ["a", "b"], schema) == ["a", "b"]
    assert coerce_value("default_kb_ids", "a,b, c", schema) == ["a", "b", "c"]
    assert coerce_value("default_kb_ids", ("a",), schema) == ["a"]


async def test_adapter_defaults_config_backed(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    from adapter.astrbot import AstrBotRAGAdapter

    engine = RAGEngine(
        _config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=FakeEmbedding(dim=8),
        image_embedding=None,
        reranker=None,
    )
    adapter = AstrBotRAGAdapter(None, {"default_kb_id": "default"})
    adapter._engine = engine

    # 初始为空
    assert adapter.get_default_search_kbs() == []
    # 新建自动加入聚合集合
    await adapter.create_kb("fresh")
    assert adapter.get_default_search_kbs() == ["fresh"]
    # 手动设置去重
    adapter.set_default_search_kbs(["a", "a", "b"])
    assert adapter.get_default_search_kbs() == ["a", "b"]
    # 删除自动移出
    await adapter.delete_kb("a")
    assert adapter.get_default_search_kbs() == ["b"]
    await engine.close()


async def test_llm_search_uses_aggregation_set(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    from adapter.astrbot import AstrBotRAGAdapter

    engine = RAGEngine(
        _config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=FakeEmbedding(dim=8),
        image_embedding=None,
        reranker=None,
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("今州岁主的传说内容", encoding="utf-8")
    (docs / "b.md").write_text("黑海岸守岸人的故事", encoding="utf-8")
    await engine.ingest("kb1", [docs / "a.md"])
    await engine.ingest("kb2", [docs / "b.md"])

    adapter = AstrBotRAGAdapter(None, {"default_kb_id": "kb1"})
    adapter._engine = engine
    adapter.set_default_search_kbs(["kb1", "kb2"])
    # 聚合集合命中（跨库）→ 内容来自集合
    ctx = await adapter.llm_search("今州岁主")
    assert ctx and "今州" in ctx
    # 清空集合 → 退回兜底 default_kb（kb1）
    adapter.set_default_search_kbs([])
    ctx2 = await adapter.llm_search("今州岁主")
    assert ctx2 and "今州" in ctx2
    await engine.close()
