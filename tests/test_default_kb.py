"""Tests for disabling the auto "default" knowledge base.

- empty kb_id is rejected at the engine layer (path guard)
- empty default_kb_id disables the LLM tool / auto image search silently
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import RAGConfig
from core.engine import RAGEngine
from core.exceptions import ConfigurationError, IndexNotFoundError

from tests.fakes import FakeEmbedding, FakeVectorStore


def _config() -> RAGConfig:
    return RAGConfig.from_dict(
        {
            "embedding": {"api_base": "x", "api_key": "k", "model": "m", "dimension": 8},
            "rerank": {"enabled": False},
        }
    )


def _engine(tmp_path) -> RAGEngine:
    return RAGEngine(
        _config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=FakeEmbedding(dim=8),
        image_embedding=None,
        reranker=None,
    )


async def test_empty_kb_id_rejected_everywhere(tmp_path):
    engine = _engine(tmp_path)
    try:
        await engine.search("", "q")
        assert False, "expected ConfigurationError"
    except ConfigurationError:
        pass
    try:
        await engine.ingest("  ", [Path(__file__)])
        assert False, "expected ConfigurationError"
    except ConfigurationError:
        pass
    try:
        await engine.status("")
        assert False, "expected ConfigurationError"
    except ConfigurationError:
        pass
    # list_kbs needs no kb_id and stays safe
    assert await engine.list_kbs() == []
    await engine.close()


async def test_no_default_creates_nothing(tmp_path):
    engine = _engine(tmp_path)
    # nothing at startup
    assert await engine.list_kbs() == []
    assert not (tmp_path / "rag" / "kbs" / "default").exists()
    assert await engine.list_kbs() == []
    await engine.close()


async def test_empty_default_kb_disables_llm_tool(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    from adapter.astrbot import AstrBotRAGAdapter

    adapter = AstrBotRAGAdapter(None, {"default_kb_id": ""})
    # engine-less adapter: default_kb empty short-circuits before _require_engine
    assert adapter.default_kb == ""
    assert await adapter.llm_search("任何问题") is None
    assert await adapter.auto_image_search_context(object()) is None

    # with engine attached, still skipped (no index created)
    engine = _engine(tmp_path)
    adapter._engine = engine
    assert await adapter.llm_search("任何问题") is None
    assert await engine.list_kbs() == []
    await engine.close()
