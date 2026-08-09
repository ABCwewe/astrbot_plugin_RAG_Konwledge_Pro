"""Multi-collection retrieval + engine multi-KB tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.chunking import TextChunker
from core.config import RAGConfig
from core.engine import RAGEngine
from core.exceptions import IndexNotFoundError
from core.indexing.manager import IndexManager
from core.providers.cache import EmbeddingCache
from core.retrieval import Retriever
from core.storage.base import VectorPoint

from tests.fakes import FakeEmbedding, FakeReranker, FakeVectorStore


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
            "top_k": 10,
            "top_n": 3,
        }
    )


async def _seed(store: FakeVectorStore) -> None:
    """Two collections with overlapping query relevance."""
    await store.create_collection("kb1", {"text": 8})
    await store.create_collection("kb2", {"text": 8})
    emb = FakeEmbedding()
    await store.upsert(
        "kb1",
        [
            VectorPoint(
                id="a1",
                vectors={"text": emb._vec("今州岁主角的传说")},
                payload={
                    "chunk_id": "a1",
                    "document_id": "d1",
                    "kb_id": "kb1",
                    "type": "text",
                    "content": "今州岁主角的传说内容",
                    "source": "a.md",
                },
            ),
            VectorPoint(
                id="a2",
                vectors={"text": emb._vec("黑海岸守岸人")},
                payload={
                    "chunk_id": "a2",
                    "document_id": "d2",
                    "kb_id": "kb1",
                    "type": "text",
                    "content": "黑海岸守岸人内容",
                    "source": "b.md",
                },
            ),
        ],
    )
    await store.upsert(
        "kb2",
        [
            VectorPoint(
                id="b1",
                vectors={"text": emb._vec("岁主与今州令尹")},
                payload={
                    "chunk_id": "b1",
                    "document_id": "d3",
                    "kb_id": "kb2",
                    "type": "text",
                    "content": "岁主与今州令尹的往事",
                    "source": "c.md",
                },
            ),
            VectorPoint(
                id="b2",
                vectors={"text": emb._vec("七丘角斗士露帕")},
                payload={
                    "chunk_id": "b2",
                    "document_id": "d4",
                    "kb_id": "kb2",
                    "type": "text",
                    "content": "七丘角斗士露帕的故事",
                    "source": "d.md",
                },
            ),
        ],
    )


async def test_multi_collection_merges_and_reranks_once():
    store = FakeVectorStore()
    await _seed(store)
    embedding = FakeEmbedding()
    reranker = FakeReranker()
    retriever = Retriever(store, embedding, reranker, _config())

    results = await retriever.retrieve_from_collections(
        ["kb1", "kb2"], "今州的岁主", top_k=10, top_n=3
    )
    # Both KBs contribute; reranker ran once over the merged pool.
    assert len(reranker.calls) == 1
    assert len(reranker.calls[0][1]) == 4  # merged candidate pool
    assert len(results) <= 3
    assert all(r.rerank_score is not None for r in results)
    kb_ids = {r.metadata.get("kb_id") for r in results}
    assert kb_ids <= {"kb1", "kb2"}
    # top result should be the most relevant (今州/岁主 overlap)
    assert results[0].content in ("今州岁主角的传说内容", "岁主与今州令尹的往事")


async def test_multi_collection_dedupes_same_chunk():
    store = FakeVectorStore()
    await _seed(store)
    retriever = Retriever(store, FakeEmbedding(), None, _config())
    results = await retriever.retrieve_from_collections(
        ["kb1", "kb1"], "今州"  # same collection twice
    )
    chunk_ids = [r.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))


async def test_engine_search_multi_skips_unindexed_kb(tmp_path):
    engine = RAGEngine(
        _config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=FakeEmbedding(),
        image_embedding=None,
        reranker=None,
    )
    # build kb1 via ingest, kb2 stays unindexed
    docs1 = tmp_path / "d1"
    docs1.mkdir()
    (docs1 / "a.md").write_text("今州岁主角的传说内容", encoding="utf-8")
    await engine.ingest("kb1", [docs1 / "a.md"])

    results = await engine.search_multi(["kb1", "missing"], "今州岁主")
    assert results
    assert all(r.metadata.get("kb_id") == "kb1" for r in results)

    with pytest.raises(IndexNotFoundError):
        await engine.search_multi(["missing", "nope"], "今州岁主")
    await engine.close()


async def test_engine_create_kb_and_independent_documents(tmp_path):
    engine = RAGEngine(
        _config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=FakeEmbedding(),
        image_embedding=None,
        reranker=None,
    )
    result = await engine.create_kb("fresh")
    assert result["action"] == "rebuilt"
    assert result["documents"] == 0
    assert await engine.list_documents("fresh") == []

    # ingest into fresh → only fresh has the doc
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text("独立文档内容", encoding="utf-8")
    await engine.ingest("fresh", [docs / "x.md"])
    assert len(await engine.list_documents("fresh")) == 1
    assert await engine.list_documents("other") == []
    await engine.close()
