"""Retriever tests — Top-K → rerank → Top-N, image merge, empty results."""

from __future__ import annotations

from core.config import RAGConfig
from core.retrieval import Retriever

from tests.fakes import FakeEmbedding, FakeReranker, FakeVectorStore


def _config(**overrides) -> RAGConfig:
    data = {
        "embedding": {
            "api_base": "http://localhost:1",
            "api_key": "k",
            "model": "m",
            "dimension": 8,
        },
        "rerank": {"enabled": False},
        "top_k": 10,
        "top_n": 3,
        **overrides,
    }
    return RAGConfig.from_dict(data)


async def _seed(store: FakeVectorStore) -> None:
    from core.storage.base import VectorPoint

    await store.create_collection("c", {"text": 8, "image": 8})
    await store.upsert(
        "c",
        [
            VectorPoint(
                id="t1",
                vectors={"text": FakeEmbedding()._vec("今州岁主角")},
                payload={
                    "chunk_id": "t1",
                    "document_id": "d1",
                    "type": "text",
                    "content": "今州的岁主是角",
                    "source": "剧情总结.md",
                    "page": 1,
                },
            ),
            VectorPoint(
                id="t2",
                vectors={"text": FakeEmbedding()._vec("黑海岸的守岸人")},
                payload={
                    "chunk_id": "t2",
                    "document_id": "d2",
                    "type": "text",
                    "content": "黑海岸的守岸人守护着系统",
                    "source": "剧情总结.md",
                },
            ),
            VectorPoint(
                id="i1",
                vectors={"image": FakeEmbedding()._vec("一张今州城图片")},
                payload={
                    "chunk_id": "i1",
                    "document_id": "d3",
                    "type": "image",
                    "image_path": "/data/x.png",
                    "source": "photos.md",
                },
            ),
        ],
    )


async def test_returns_top_n_with_rerank_scores():
    store = FakeVectorStore()
    await _seed(store)
    embedding = FakeEmbedding()
    reranker = FakeReranker()
    retriever = Retriever(store, embedding, reranker, _config())

    results = await retriever.retrieve("c", "今州的岁主是谁")
    assert results
    assert len(results) <= 3
    assert results[0].chunk_id == "t1"  # query overlaps 今州/岁主/角
    assert all(r.rerank_score is not None for r in results)
    assert all(r.metadata["source"] == "剧情总结.md" for r in results)


async def test_rerank_disabled_keeps_vector_order_and_null_rerank_score():
    store = FakeVectorStore()
    await _seed(store)
    embedding = FakeEmbedding()
    retriever = Retriever(store, embedding, None, _config())
    results = await retriever.retrieve("c", "今州的岁主是谁")
    assert results
    assert results[0].chunk_id == "t1"
    assert all(r.rerank_score is None for r in results)


async def test_image_merge_only_when_requested():
    store = FakeVectorStore()
    await _seed(store)
    embedding = FakeEmbedding(supports_image=True)
    retriever = Retriever(
        store,
        embedding,
        None,
        _config(),
        image_embedding=FakeEmbedding(dim=8, supports_image=True),
    )
    text_only = await retriever.retrieve("c", "今州")
    assert all(r.metadata["type"] == "text" for r in text_only)

    with_images = await retriever.retrieve("c", "今州", include_images=True)
    assert any(r.metadata["type"] == "image" for r in with_images)


async def test_image_results_get_text_representation_for_rerank():
    store = FakeVectorStore()
    await _seed(store)
    embedding = FakeEmbedding(supports_image=True)
    reranker = FakeReranker()
    retriever = Retriever(
        store,
        embedding,
        reranker,
        _config(),
        image_embedding=FakeEmbedding(dim=8, supports_image=True),
    )
    results = await retriever.retrieve("c", "今州城", include_images=True)
    # FakeReranker was given non-empty documents for every hit (incl. image).
    image_docs = [d for (q, d) in reranker.calls for d in d]
    assert image_docs
    assert any("x.png" in d for d in image_docs)  # image text rep carries filename


async def test_empty_query_and_empty_results():
    store = FakeVectorStore()
    await _seed(store)
    retriever = Retriever(store, FakeEmbedding(), FakeReranker(), _config())
    assert await retriever.retrieve("c", "   ") == []

    store2 = FakeVectorStore()
    await store2.create_collection("empty", {"text": 8})
    assert await retriever.retrieve("empty", "没有内容的库") == []
