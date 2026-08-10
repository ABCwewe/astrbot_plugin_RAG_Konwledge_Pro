"""Tests for unified image normalization (index + query, BOX downscale)."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

from PIL import Image

from core.config import RAGConfig
from core.engine import RAGEngine
from core.imaging import normalize_image_bytes

from tests.fakes import FakeEmbedding, FakeVectorStore
from tests.test_auto_image_search import HashEmbedding
from tests.test_upload_queue import _saver, _wait_until, _docs


def _png_bytes(w: int, h: int, seed: str) -> bytes:
    hval = sum(ord(c) for c in seed) % 256
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (hval, 255 - hval, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _bsaver(data: bytes):
    async def _save(tmp) -> None:
        await asyncio.to_thread(Path(tmp).write_bytes, data)

    return _save


def _size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as img:
        return img.size


# ---------- 纯函数 ----------


def test_normalize_disabled_passes_through():
    data = _png_bytes(4000, 3000, "x")
    assert normalize_image_bytes(data, 0) == data
    assert normalize_image_bytes(data, -1) == data


def test_small_image_unchanged():
    data = _png_bytes(512, 512, "small")
    assert normalize_image_bytes(data, 1024) == data  # 无需缩放则不重编码


def test_large_image_scaled_and_deterministic():
    data = _png_bytes(4000, 3000, "big")
    out1 = normalize_image_bytes(data, 1024)
    out2 = normalize_image_bytes(data, 1024)
    w, h = _size(out1)
    assert max(w, h) == 1024
    assert out1 == out2  # 确定性：缓存键稳定
    # 更小上限
    out3 = normalize_image_bytes(data, 256)
    assert max(_size(out3)) == 256


def test_corrupt_bytes_passthrough():
    assert normalize_image_bytes(b"not-an-image", 1024) == b"not-an-image"
    assert normalize_image_bytes(b"", 1024) == b""


def test_config_parse_and_hash():
    base = RAGConfig.from_dict(
        {
            "embedding": {"api_base": "x", "api_key": "k", "model": "m", "dimension": 8},
            "image": {"enabled": True, "api_base": "x", "api_key": "k", "model": "m", "dimension": 8},
        }
    )
    assert base.image.max_side == 1024
    small = RAGConfig.from_dict(
        {
            "embedding": {"api_base": "x", "api_key": "k", "model": "m", "dimension": 8},
            "image": {"enabled": True, "api_base": "x", "api_key": "k", "model": "m", "dimension": 8, "max_side": 512},
        }
    )
    assert small.image.max_side == 512
    # 缩放参数影响索引内容 → config_hash 必须变化（触发重建）
    assert base.config_hash() != small.config_hash()


def _image_config(max_side: int = 64) -> RAGConfig:
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
                "min_score": 0.0,
                "max_side": max_side,
            },
            "rerank": {"enabled": False},
            "top_n": 3,
        }
    )


class RecordingImageEmbedding(HashEmbedding):
    """Captures the exact bytes handed to the embedding provider."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__(dim=dim, supports_image=True)
        self.last_image_bytes: bytes | None = None

    async def embed_image(self, images):
        self.last_image_bytes = images[0]
        return await super().embed_image(images)


async def test_index_side_normalizes_before_embedding(tmp_path):
    emb = RecordingImageEmbedding()
    engine = RAGEngine(
        _image_config(64),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=HashEmbedding(dim=32),
        image_embedding=emb,
        reranker=None,
    )
    await engine.receive_upload("kb", "big.png", _bsaver(_png_bytes(4000, 3000, "A")))

    async def _indexed() -> bool:
        docs = await _docs(engine)
        return len(docs) == 1 and docs[0]["indexed"]

    await _wait_until(_indexed)
    assert emb.last_image_bytes is not None
    assert max(_size(emb.last_image_bytes)) <= 64
    await engine.close()


async def test_query_side_normalizes_before_embedding(tmp_path):
    emb = RecordingImageEmbedding()
    engine = RAGEngine(
        _image_config(64),
        tmp_path / "rag",
        store=FakeVectorStore(),
        embedding=HashEmbedding(dim=32),
        image_embedding=emb,
        reranker=None,
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "角色A.png").write_bytes(_png_bytes(200, 200, "A"))
    await engine.ingest("kb", [docs / "角色A.png"])

    probe = tmp_path / "probe.png"
    probe.write_bytes(_png_bytes(4000, 3000, "A"))
    await engine.search_image_by_path("kb", probe)
    assert emb.last_image_bytes is not None
    assert max(_size(emb.last_image_bytes)) <= 64
    await engine.close()
