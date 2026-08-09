"""Tests for automatic image-vector search (message image → KB image vectors).

Covers the extraction helper, the retriever's image-as-query path, the engine
wrapper, and the adapter's context builder end-to-end with fakes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import RAGConfig
from core.engine import RAGEngine
from core.exceptions import ConfigurationError
from core.retrieval import Retriever
from core.storage.base import VectorPoint

from adapter.image_utils import find_message_image
from tests.fakes import FakeEmbedding, FakeReranker, FakeVectorStore


class FakeImage:
    def __init__(self, url: str = "", file: str = "") -> None:
        self.url = url
        self.file = file


class FakeReply:
    def __init__(self, chain) -> None:
        self.chain = chain


# ---------- extraction helper ----------


def test_body_image_preferred_over_reply():
    body_img = FakeImage(file="body.png")
    reply_img = FakeImage(url="reply.png")
    assert find_message_image([FakeReply(chain=[reply_img]), body_img]) is body_img


def test_reply_chain_image_found():
    reply_img = FakeImage(file="quoted.png")
    assert find_message_image([FakeReply(chain=[FakeImage(), reply_img])]) is reply_img


def test_no_image():
    assert find_message_image([]) is None
    assert find_message_image([FakeImage(), FakeReply(chain=[])]) is None
    assert find_message_image(None) is None


# ---------- config ----------


def test_config_parses_auto_search():
    cfg = RAGConfig.from_dict(
        {
            "embedding": {"api_base": "x", "api_key": "k", "model": "m", "dimension": 8},
            "image": {"enabled": True, "api_base": "x", "api_key": "k", "model": "m", "dimension": 8},
        }
    )
    assert cfg.image.auto_search is True
    cfg2 = RAGConfig.from_dict(
        {
            "embedding": {"api_base": "x", "api_key": "k", "model": "m", "dimension": 8},
            "image": {"enabled": True, "api_base": "x", "api_key": "k", "model": "m", "dimension": 8, "auto_search": False},
        }
    )
    assert cfg2.image.auto_search is False
    # auto_search is retrieval-side only: must not change the index config hash
    assert cfg.config_hash() == cfg2.config_hash()


# ---------- retriever / engine ----------


def _image_config() -> RAGConfig:
    return RAGConfig.from_dict(
        {
            "embedding": {"api_base": "x", "api_key": "k", "model": "m", "dimension": 8},
            "image": {
                "enabled": True,
                "api_base": "x",
                "api_key": "k",
                "model": "m",
                "dimension": 8,
                "auto_search": True,
                "min_score": 0.9,
            },
            "rerank": {"enabled": False},
            "top_n": 3,
        }
    )


class HashEmbedding(FakeEmbedding):
    """Zero-mean unit vectors from sha256: identical bytes → cosine 1.0,
    different bytes → near-random cosine (~N(0, 1/dim)). Suitable for binary
    image payloads where the char-based FakeEmbedding degenerates."""

    def _vec(self, data):
        import hashlib
        import math

        s = data if isinstance(data, bytes) else data.encode()
        digest = hashlib.sha256(s).digest()
        vals = [(digest[i] / 255) - 0.5 for i in range(self._dim)]
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]


def _png_bytes(seed: str) -> bytes:
    """Deterministic real PNG (Pillow) so the image parser passes validation."""
    import io

    from PIL import Image

    h = sum(ord(c) for c in seed) % 256
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (h, 255 - h, 128)).save(buf, format="PNG")
    return buf.getvalue()


async def _seed_image_store(store: FakeVectorStore, emb: FakeEmbedding) -> None:
    await store.create_collection("kb", {"text": 8, "image": 8})
    await store.upsert(
        "kb",
        [
            VectorPoint(
                id="i1",
                vectors={"image": emb._vec(_png_bytes("角色A"))},
                payload={
                    "chunk_id": "i1",
                    "document_id": "d1",
                    "kb_id": "kb",
                    "type": "image",
                    "image_path": "/kb/角色A.png",
                    "source": "角色A.png",
                },
            ),
            VectorPoint(
                id="i2",
                vectors={"image": emb._vec(_png_bytes("角色B"))},
                payload={
                    "chunk_id": "i2",
                    "document_id": "d2",
                    "kb_id": "kb",
                    "type": "image",
                    "image_path": "/kb/角色B.png",
                    "source": "角色B.png",
                },
            ),
        ],
    )


async def test_retriever_retrieve_by_image():
    store = FakeVectorStore()
    emb = HashEmbedding(dim=32, supports_image=True)
    await _seed_image_store(store, emb)
    retriever = Retriever(store, emb, None, _image_config(), image_embedding=emb)

    results = await retriever.retrieve_by_image("kb", _png_bytes("角色A"))
    assert results
    assert results[0].metadata["source"] == "角色A.png"
    assert all(r.metadata["type"] == "image" for r in results)


async def test_retriever_retrieve_by_image_requires_image_embedding():
    store = FakeVectorStore()
    emb = FakeEmbedding(dim=8, supports_image=False)
    await _seed_image_store(store, emb)
    retriever = Retriever(store, emb, None, _image_config(), image_embedding=None)
    assert await retriever.retrieve_by_image("kb", _png_bytes("x")) == []


async def test_engine_search_image_by_path(tmp_path):
    engine = RAGEngine(
        _image_config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=HashEmbedding(dim=32),
        image_embedding=HashEmbedding(dim=32, supports_image=True),
        reranker=None,
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "角色A.png").write_bytes(_png_bytes("角色A"))
    await engine.ingest("kb", [docs / "角色A.png"])

    probe = tmp_path / "probe.png"
    probe.write_bytes(_png_bytes("角色A"))
    results = await engine.search_image_by_path("kb", probe)
    assert results
    assert results[0].metadata["source"] == "角色A.png"
    await engine.close()


async def test_adapter_search_image_multi_and_single_fallback(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    from adapter.astrbot import AstrBotRAGAdapter

    engine = RAGEngine(
        _image_config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=HashEmbedding(dim=32),
        image_embedding=HashEmbedding(dim=32, supports_image=True),
        reranker=None,
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "角色A.png").write_bytes(_png_bytes("角色A"))
    (docs / "kb2.txt").write_text("纯文本", encoding="utf-8")
    await engine.ingest("kb1", [docs / "角色A.png"])
    await engine.ingest("kb2", [docs / "kb2.txt"])

    adapter = AstrBotRAGAdapter(None, {"default_kb_id": "kb1"})
    adapter._engine = engine

    probe = tmp_path / "probe.png"
    probe.write_bytes(_png_bytes("角色A"))

    # 多库：跳过无图库 kb2
    results = await adapter.search_image(["kb1", "kb2"], str(probe))
    assert results and all(r.metadata.get("kb_id") == "kb1" for r in results)
    # 单库兜底：current_kb = 默认 kb1
    results2 = await adapter.search_image(None, str(probe))
    assert results2 and results2[0].metadata["source"] == "角色A.png"
    await engine.close()


async def test_engine_search_image_by_path_disabled_raises(tmp_path):
    cfg = RAGConfig.from_dict(
        {
            "embedding": {"api_base": "x", "api_key": "k", "model": "m", "dimension": 8},
            "image": {"enabled": False},
            "rerank": {"enabled": False},
        }
    )
    engine = RAGEngine(
        cfg,
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=FakeEmbedding(dim=8),
        image_embedding=None,
        reranker=None,
    )
    try:
        await engine.search_image_by_path("kb", "x.png")
        assert False, "expected ConfigurationError"
    except ConfigurationError:
        pass
    await engine.close()


async def test_search_image_multi_skips_image_less_kbs(tmp_path):
    from core.exceptions import IndexNotFoundError

    engine = RAGEngine(
        _image_config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=HashEmbedding(dim=32),
        image_embedding=HashEmbedding(dim=32, supports_image=True),
        reranker=None,
    )
    # kb1 有图；kb2 只有文本（无 image 向量）
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "角色A.png").write_bytes(_png_bytes("角色A"))
    (docs / "kb2.txt").write_text("纯文本知识库", encoding="utf-8")
    await engine.ingest("kb1", [docs / "角色A.png"])
    await engine.ingest("kb2", [docs / "kb2.txt"])

    probe = tmp_path / "probe.png"
    probe.write_bytes(_png_bytes("角色A"))
    results = await engine.search_image_multi(["kb1", "kb2"], probe)
    # kb2 无图片向量被跳过，结果全部来自 kb1
    assert results
    assert all(r.metadata.get("kb_id") == "kb1" for r in results)

    # 全部无图 → IndexNotFoundError
    try:
        await engine.search_image_multi(["kb2"], probe)
        assert False, "expected IndexNotFoundError"
    except IndexNotFoundError:
        pass
    await engine.close()


# ---------- adapter context builder ----------


class FakeEvent:
    def __init__(self, messages) -> None:
        self._m = messages

    def get_messages(self):
        return self._m


class FileImage(FakeImage):
    """Mimics Image.fromFileSystem: exposes convert_to_file_path."""

    def __init__(self, path: str) -> None:
        super().__init__(file=Path(path).as_uri(), url=str(Path(path).resolve()))
        self._path = str(Path(path).resolve())

    async def convert_to_file_path(self) -> str:
        return self._path


async def test_adapter_auto_image_search_context(tmp_path, monkeypatch):
    # adapter.astrbot 首次导入会触发 astrbot 包初始化（按 cwd 创建 data/），
    # 把 cwd 指向临时目录隔离污染。
    import os

    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    from adapter.astrbot import AstrBotRAGAdapter

    engine = RAGEngine(
        _image_config(),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=HashEmbedding(dim=32),
        image_embedding=HashEmbedding(dim=32, supports_image=True),
        reranker=None,
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "角色A.png").write_bytes(_png_bytes("角色A"))
    await engine.ingest("kb", [docs / "角色A.png"])

    adapter = AstrBotRAGAdapter(None, {})
    adapter._engine = engine
    adapter.default_kb = "kb"

    # image in body → context injected
    probe = tmp_path / "probe.png"
    probe.write_bytes(_png_bytes("角色A"))
    ctx = await adapter.auto_image_search_context(FakeEvent([FileImage(str(probe))]))
    assert ctx and "角色A.png" in ctx

    # no image → None
    assert await adapter.auto_image_search_context(FakeEvent([])) is None

    # image with no match → None
    miss = tmp_path / "miss.png"
    miss.write_bytes(_png_bytes("完全无关内容XYZ"))
    assert await adapter.auto_image_search_context(FakeEvent([FileImage(str(miss))])) is None
    await engine.close()


async def test_adapter_auto_image_search_skips_when_disabled(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    from adapter.astrbot import AstrBotRAGAdapter

    cfg = RAGConfig.from_dict(
        {
            "embedding": {"api_base": "x", "api_key": "k", "model": "m", "dimension": 8},
            "image": {"enabled": True, "api_base": "x", "api_key": "k", "model": "m", "dimension": 8, "auto_search": False},
            "rerank": {"enabled": False},
        }
    )
    engine = RAGEngine(
        cfg,
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=HashEmbedding(dim=32),
        image_embedding=HashEmbedding(dim=32, supports_image=True),
        reranker=None,
    )
    adapter = AstrBotRAGAdapter(None, {})
    adapter._engine = engine
    probe = tmp_path / "p.png"
    probe.write_bytes(_png_bytes("x"))
    assert await adapter.auto_image_search_context(FakeEvent([FileImage(str(probe))])) is None
    await engine.close()
