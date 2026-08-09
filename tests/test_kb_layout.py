"""KB layout tests: all KB roots live under <data_root>/kbs/, engine-level
directories (tmp/) never appear as knowledge bases; legacy layout migrates."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from core.config import RAGConfig
from core.engine import RAGEngine

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


async def test_tmp_is_not_listed_as_kb(tmp_path):
    engine = RAGEngine(
        _config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=FakeEmbedding(),
        image_embedding=None,
        reranker=None,
    )
    # engine-level dirs the old code would have listed as KBs
    (tmp_path / "rag" / "tmp").mkdir(parents=True, exist_ok=True)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("内容", encoding="utf-8")
    await engine.ingest("kb1", [docs / "a.md"])
    await engine.ingest("kb2", [docs / "a.md"])

    kbs = await engine.list_kbs()
    assert kbs == ["kb1", "kb2"]
    assert "tmp" not in kbs
    assert (tmp_path / "rag" / "kbs" / "kb1").is_dir()
    assert (tmp_path / "rag" / "kbs" / "kb2").is_dir()
    assert not (tmp_path / "rag" / "kb1").exists()  # not at engine root
    await engine.close()


async def test_legacy_kb_roots_migrate_into_kbs(tmp_path):
    # simulate old layout: data_root/<kb_id> with manifest + documents
    legacy = tmp_path / "rag" / "oldkb"
    (legacy / "documents").mkdir(parents=True)
    (legacy / "documents" / "a.md").write_text("旧文档", encoding="utf-8")
    (legacy / "active.json").write_text("{}", encoding="utf-8")
    (legacy / "manifest_v1.json").write_text("{}", encoding="utf-8")

    engine = RAGEngine(
        _config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=FakeEmbedding(),
        image_embedding=None,
        reranker=None,
    )
    assert not (tmp_path / "rag" / "oldkb").exists()
    assert (tmp_path / "rag" / "kbs" / "oldkb" / "documents" / "a.md").exists()
    assert await engine.list_kbs() == ["oldkb"]
    await engine.close()


async def test_legacy_migration_skips_non_kb_dirs(tmp_path):
    (tmp_path / "rag" / "misc").mkdir(parents=True)
    (tmp_path / "rag" / "misc" / "notes.txt").write_text("无关目录", encoding="utf-8")
    engine = RAGEngine(
        _config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=FakeEmbedding(),
        image_embedding=None,
        reranker=None,
    )
    # non-KB dir stays in place, not listed as KB
    assert (tmp_path / "rag" / "misc").exists()
    assert await engine.list_kbs() == []
    await engine.close()
