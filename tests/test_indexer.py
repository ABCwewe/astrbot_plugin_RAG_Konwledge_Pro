"""Indexer tests — incremental add/modify/delete, embedding cache, chunks."""

from __future__ import annotations

from pathlib import Path

from core.chunking import TextChunker
from core.indexing.manifest import DocumentRecord, ManifestStore
from core.indexing.indexer import Indexer
from core.parsers import ParserRegistry
from core.providers.cache import EmbeddingCache

from tests.fakes import FakeEmbedding, FakeVectorStore


def _indexer(
    tmp_path,
    store=None,
    embedding=None,
    image_embedding=None,
    chunk_size=50,
):
    store = store or FakeVectorStore()
    embedding = embedding or FakeEmbedding(dim=8)
    cache = EmbeddingCache(tmp_path / "cache.db")
    indexer = Indexer(
        store,
        embedding,
        TextChunker(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=min(10, chunk_size - 1),
        ),
        cache,
        parsers=ParserRegistry(),
        image_embedding=image_embedding,
    )
    return indexer, store, embedding, cache


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


async def test_new_document_is_added(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = _write(docs, "a.md", "第一章\n\n内容一\n\n内容二")
    indexer, store, embedding, cache = _indexer(tmp_path, chunk_size=5, image_embedding=None)

    records: dict[str, DocumentRecord] = {}
    await indexer.sync_documents("c1", "kb", docs, [path], records)

    assert len(records) == 1
    record = next(iter(records.values()))
    assert record.source == "a.md"
    assert record.filename == "a.md"
    assert record.metadata["chunks"] == 3  # 第一章/内容一/内容二
    assert await store.count("c1") == 3
    # every chunk payload carries document identity + content
    coll = store.collections["c1"]["points"]
    payloads = [p.payload for p in coll.values()]
    assert all(p["document_id"] == record.document_id for p in payloads)
    assert all(p["type"] == "text" for p in payloads)


async def test_unchanged_document_is_skipped(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = _write(docs, "a.md", "同样的内容")
    indexer, store, embedding, cache = _indexer(tmp_path)

    records: dict[str, DocumentRecord] = {}
    await indexer.sync_documents("c1", "kb", docs, [path], records)
    embed_calls = len(embedding.text_calls)
    await indexer.sync_documents("c1", "kb", docs, [path], records)

    assert len(embedding.text_calls) == embed_calls  # no re-embedding
    assert await store.count("c1") == 1


async def test_changed_document_replaces_chunks(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = _write(docs, "a.md", "旧内容")
    indexer, store, embedding, cache = _indexer(tmp_path)

    records: dict[str, DocumentRecord] = {}
    await indexer.sync_documents("c1", "kb", docs, [path], records)
    old_point_ids = set(store.collections["c1"]["points"])

    _write(docs, "a.md", "新内容新内容新内容")
    await indexer.sync_documents("c1", "kb", docs, [path], records)

    new_point_ids = set(store.collections["c1"]["points"])
    # Deterministic chunk ids are reused after delete+recreate; the old point
    # must have been removed and exactly one point with new content remains.
    assert new_point_ids == old_point_ids
    assert len(new_point_ids) == 1
    assert await store.count("c1") == 1
    assert store.collections["c1"]["points"][next(iter(new_point_ids))].payload["content"] == "新内容新内容新内容"


async def test_removed_document_deletes_its_chunks(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    a = _write(docs, "a.md", "A 内容")
    b = _write(docs, "b.md", "B 内容")
    indexer, store, embedding, cache = _indexer(tmp_path)

    records: dict[str, DocumentRecord] = {}
    await indexer.sync_documents("c1", "kb", docs, [a, b], records)
    assert await store.count("c1") == 2

    await indexer.sync_documents("c1", "kb", docs, [b], records)
    assert await store.count("c1") == 1
    assert len(records) == 1
    remaining = next(iter(store.collections["c1"]["points"].values()))
    assert remaining.payload["content"] == "B 内容"


async def test_embedding_cache_hits_avoid_api(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = _write(docs, "a.md", "缓存测试内容")
    indexer, store, embedding, cache = _indexer(tmp_path)

    records: dict[str, DocumentRecord] = {}
    await indexer.sync_documents("c1", "kb", docs, [path], records)
    first_calls = len(embedding.text_calls)

    # Delete the file, then re-add identical content → same chunk texts.
    path.unlink()
    _write(docs, "a.md", "缓存测试内容")
    # New version via a fresh collection to force re-index of identical chunks.
    await indexer.sync_documents("c2", "kb", docs, [path], {})

    assert len(embedding.text_calls) == first_calls  # served from cache
    assert await store.count("c2") == 1


async def test_pdf_pages_carry_page_metadata(tmp_path):
    try:
        import fitz  # noqa: F401
    except ImportError:
        return  # PyMuPDF unavailable → skip
    docs = tmp_path / "docs"
    docs.mkdir()
    import fitz as _fitz

    pdf = docs / "book.pdf"
    doc = _fitz.open()
    for _ in range(2):
        page = doc.new_page()
        page.insert_text((72, 72), "第N页的正文内容用于测试")
    doc.save(str(pdf))
    doc.close()

    indexer, store, embedding, cache = _indexer(tmp_path)
    records: dict[str, DocumentRecord] = {}
    await indexer.sync_documents("c1", "kb", docs, [pdf], records)
    payloads = [p.payload for p in store.collections["c1"]["points"].values()]
    assert payloads
    assert all(p.get("page") in (1, 2) for p in payloads)


async def test_image_document_creates_image_chunk(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    from PIL import Image

    img_path = docs / "photo.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(img_path)

    indexer, store, embedding, cache = _indexer(
        tmp_path,
        embedding=FakeEmbedding(dim=8, supports_image=True),
        image_embedding=FakeEmbedding(dim=8, supports_image=True),
    )
    records: dict[str, DocumentRecord] = {}
    await indexer.sync_documents("c1", "kb", docs, [img_path], records)

    assert await store.count("c1") == 1
    point = next(iter(store.collections["c1"]["points"].values()))
    assert point.payload["type"] == "image"
    assert point.payload["image_path"] == str(img_path)
    assert "image" in point.vectors
