"""Tests for the async upload queue: atomic inbox receive, background worker,
wave coalescing, .part cleanup, restart resume, queue stats."""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.config import RAGConfig
from core.engine import RAGEngine
from core.exceptions import ConfigurationError, ParserError

from tests.fakes import FakeEmbedding, FakeVectorStore


def _config(concurrency: int = 2) -> RAGConfig:
    return RAGConfig.from_dict(
        {
            "embedding": {"api_base": "x", "api_key": "k", "model": "m", "dimension": 8},
            "rerank": {"enabled": False},
            "ingest_concurrency": concurrency,
        }
    )


def _engine(tmp_path, concurrency: int = 2) -> RAGEngine:
    return RAGEngine(
        _config(concurrency),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=FakeEmbedding(dim=8),
        image_embedding=None,
        reranker=None,
    )


def _saver(text: str):
    async def _save(tmp) -> None:
        await asyncio.to_thread(Path(tmp).write_text, text, encoding="utf-8")

    return _save


def _failing_saver():
    async def _save(tmp) -> None:
        raise ConnectionError("中断")

    return _save


async def _wait_until(cond, timeout: float = 12.0) -> None:
    async def _poll() -> None:
        while True:
            result = cond()
            if asyncio.iscoroutine(result):
                result = await result
            if result:
                return
            await asyncio.sleep(0.1)

    await asyncio.wait_for(_poll(), timeout)


async def _stop_worker(engine: RAGEngine) -> None:
    """Cancel the background worker for deterministic manual drain."""
    engine._worker_task.cancel()
    try:
        await engine._worker_task
    except (asyncio.CancelledError, Exception):
        pass


# ---------- 原子接收 ----------


async def test_receive_upload_atomic_success(tmp_path):
    engine = _engine(tmp_path)
    try:
        path = await engine.receive_upload("kb", "a.md", _saver("内容A"))
        inbox = engine.inbox_dir("kb")
        assert Path(path).name == "a.md"
        assert (inbox / "a.md").is_file()
        assert not list(inbox.glob("*.part"))  # 无残留临时文件
    finally:
        await engine.close()


async def test_receive_upload_interrupted_leaves_nothing(tmp_path):
    engine = _engine(tmp_path)
    try:
        try:
            await engine.receive_upload("kb", "a.md", _failing_saver())
            assert False, "expected ConnectionError"
        except ConnectionError:
            pass
        inbox = engine.inbox_dir("kb")
        assert not (inbox / "a.md").exists()  # 知识库内不存在
        assert list(inbox.iterdir()) == []     # 也无 .part 残留
    finally:
        await engine.close()


async def test_receive_upload_rejects_bad_names(tmp_path):
    engine = _engine(tmp_path)
    try:
        try:
            await engine.receive_upload("", "a.md", _saver("x"))
            assert False
        except ConfigurationError:
            pass
        try:
            await engine.receive_upload("kb", ".hidden", _saver("x"))
            assert False
        except ParserError:
            pass
        try:
            await engine.receive_upload("kb", "", _saver("x"))
            assert False
        except ParserError:
            pass
    finally:
        await engine.close()


# ---------- 波次合并（确定性：手动驱动 drain） ----------


async def test_wave_coalesces_files_into_single_index_op(tmp_path):
    engine = _engine(tmp_path)
    await _stop_worker(engine)
    try:
        for i in range(3):
            await engine.receive_upload("kb", f"a{i}.md", _saver(f"内容{i}"))
        assert engine.get_upload_stats()["inbox_pending"] == 3

        await engine._drain_uploads_once()

        docs = await engine.list_documents("kb")
        assert len(docs) == 3
        assert all(d["indexed"] for d in docs)
        counts = engine.get_index_op_counts()
        assert counts == {"rebuild": 1, "sync": 0}  # 3 个文件 = 1 次波次（重建）

        # 第二批两个文件：下一次 drain = 1 次增量同步
        await engine.receive_upload("kb", "b1.md", _saver("B1"))
        await engine.receive_upload("kb", "b2.md", _saver("B2"))
        await engine._drain_uploads_once()
        counts = engine.get_index_op_counts()
        assert counts["rebuild"] == 1
        assert counts["sync"] == 1  # 合并为一次增量同步
        assert len(await engine.list_documents("kb")) == 5
    finally:
        await engine.close()


# ---------- 后台工人自动消费 ----------


async def test_worker_drains_inbox_automatically(tmp_path):
    engine = _engine(tmp_path)
    try:
        await asyncio.gather(
            *(engine.receive_upload("kb", f"f{i}.md", _saver(f"内容{i}")) for i in range(3))
        )

        async def _all_indexed() -> bool:
            docs = await _docs(engine)
            return len(docs) == 3 and all(d["indexed"] for d in docs)

        await _wait_until(_all_indexed)
        counts = engine.get_index_op_counts()
        assert counts["rebuild"] == 1
        assert counts["sync"] in (0, 1)  # 波次可能拆成两波，但每波合并
        assert engine.get_upload_stats()["inbox_pending"] == 0
    finally:
        await engine.close()


async def _docs(engine: RAGEngine):
    return await engine.list_documents("kb")


# ---------- .part 忽略与清理 ----------


async def test_worker_ignores_and_purges_part_files(tmp_path):
    engine = _engine(tmp_path)
    await _stop_worker(engine)
    try:
        await engine.receive_upload("kb", "real.md", _saver("真文件"))
        inbox = engine.inbox_dir("kb")
        (inbox / ".upload-dead.part").write_text("半截", encoding="utf-8")
        await engine._drain_uploads_once()

        docs = await engine.list_documents("kb")
        assert len(docs) == 1 and docs[0]["filename"] == "real.md"
        assert not (inbox / ".upload-dead.part").exists()  # 残留被清理
    finally:
        await engine.close()


# ---------- 重启恢复 ----------


async def test_restart_resumes_inbox(tmp_path):
    engine_a = _engine(tmp_path)
    await engine_a.receive_upload("kb", "pending.md", _saver("未索引"))
    await engine_a.close()  # 不 drain，文件留在 inbox（模拟中断）

    engine_b = _engine(tmp_path)
    try:
        async def _resumed() -> bool:
            docs = await _docs(engine_b)
            return len(docs) == 1 and docs[0]["indexed"]

        await _wait_until(_resumed)
        assert engine_b.get_upload_stats()["inbox_pending"] == 0
    finally:
        await engine_b.close()


# ---------- 队列统计 ----------


async def test_upload_stats_reflect_queue(tmp_path):
    engine = _engine(tmp_path)
    await _stop_worker(engine)
    try:
        await engine.receive_upload("kb", "x1.md", _saver("1"))
        await engine.receive_upload("kb", "x2.md", _saver("2"))
        stats = engine.get_upload_stats()
        assert stats["inbox_pending"] == 2
        assert stats["inbox_by_kb"] == {"kb": 2}
        await engine._drain_uploads_once()
        assert engine.get_upload_stats()["inbox_pending"] == 0
    finally:
        await engine.close()
