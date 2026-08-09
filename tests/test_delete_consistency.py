"""Delete-consistency tests.

Regression for: deleting a KB then re-uploading the same documents to a
recreated same-name KB must NOT skip indexing. Root causes addressed:

- rmtree(ignore_errors=True) could silently leave the stale documents.json on
  Windows (locked files) → re-upload saw matching hashes and skipped.
- delete_kb/delete_index now hold the per-KB lock (no in-flight sync races)
  and use a retrying/verifying rmtree.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from core.config import RAGConfig
from core.engine import RAGEngine

from tests.fakes import FakeEmbedding, FakeVectorStore
from tests.test_upload_queue import _saver, _wait_until, _docs


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


async def test_delete_recreate_reupload_is_not_skipped(tmp_path):
    engine = _engine(tmp_path)
    try:
        await engine.receive_upload("kb", "a.md", _saver("相同内容"))

        async def _indexed() -> bool:
            docs = await _docs(engine)
            return len(docs) == 1 and docs[0]["indexed"]

        await _wait_until(_indexed)
        counts_after_first = engine.get_index_op_counts()

        # 删除知识库（含文档和索引）
        await engine.delete_kb("kb")
        assert not (tmp_path / "rag" / "kbs" / "kb").exists()
        assert await engine.list_kbs() == []

        # 重建同名库 + 重传相同文档：必须重新索引（added），不能跳过
        await engine.create_kb("kb")
        await engine.receive_upload("kb", "a.md", _saver("相同内容"))
        await _wait_until(_indexed)

        docs = await _docs(engine)
        assert len(docs) == 1 and docs[0]["indexed"]
        assert engine.get_index_op_counts()["rebuild"] > counts_after_first["rebuild"]
    finally:
        await engine.close()


async def test_delete_rmtree_retries_on_locked_files(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    await engine.receive_upload("kb", "a.md", _saver("x"))

    async def _indexed() -> bool:
        docs = await _docs(engine)
        return len(docs) == 1 and docs[0]["indexed"]

    await _wait_until(_indexed)

    # 模拟 Windows 文件占用：第一次 rmtree 抛 OSError，第二次成功
    real_rmtree = shutil.rmtree
    calls = {"n": 0}

    def flaky_rmtree(path, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("文件被占用")
        return real_rmtree(path, *a, **kw)

    monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)
    await engine.delete_kb("kb")
    assert calls["n"] >= 2
    assert not (tmp_path / "rag" / "kbs" / "kb").exists()
    await engine.close()


class SlowEmbedding(FakeEmbedding):
    async def embed_text(self, texts, *, input_type=None):
        await asyncio.sleep(0.3)
        return await super().embed_text(texts, input_type=input_type)


async def test_delete_waits_for_inflight_sync(tmp_path):
    config = _config()
    engine = RAGEngine(
        config,
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=SlowEmbedding(dim=8),
        image_embedding=None,
        reranker=None,
    )
    await engine.receive_upload("kb", "a.md", _saver("内容"))

    async def _indexed() -> bool:
        docs = await _docs(engine)
        return len(docs) == 1 and docs[0]["indexed"]

    await _wait_until(_indexed)

    # 触发一次慢同步（持有 kb 锁），同时发起删除：删除必须等待，不能竞态复活
    sync_task = asyncio.create_task(engine._manager.sync("kb", config))
    delete_task = asyncio.create_task(engine.delete_kb("kb"))
    await asyncio.gather(sync_task, delete_task)

    assert not (tmp_path / "rag" / "kbs" / "kb").exists()
    assert await engine.list_kbs() == []
    await engine.close()
