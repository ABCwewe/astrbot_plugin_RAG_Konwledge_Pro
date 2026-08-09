"""Tests for drop_index (keep KB + docs) and default search KBs."""

from __future__ import annotations

from pathlib import Path

from core.config import RAGConfig
from core.engine import RAGEngine
from core.exceptions import IndexNotFoundError

from adapter.config_utils import load_search_defaults, save_search_defaults
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
    assert (tmp_path / "rag" / "kb" / "documents" / "a.md").exists()
    docs_list = await engine.list_documents("kb")
    assert len(docs_list) == 1
    assert docs_list[0]["indexed"] is False

    # re-ingest rebuilds the index from the kept documents
    result = await engine.ingest("kb", [tmp_path / "rag" / "kb" / "documents" / "a.md"])
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


def test_search_defaults_persistence(tmp_path):
    path = tmp_path / "search_defaults.json"
    assert load_search_defaults(path) == []
    saved = save_search_defaults(path, ["kb1", "kb2", "", "kb1", "kb2"])
    assert saved == ["kb1", "kb2"]
    assert load_search_defaults(path) == ["kb1", "kb2"]
    # survives across instances (file-based)
    assert load_search_defaults(Path(path)) == ["kb1", "kb2"]
    # corrupt file → empty
    path.write_text("{broken", encoding="utf-8")
    assert load_search_defaults(path) == []
