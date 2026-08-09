"""OpenAI-compatible embedding provider (AGENTS.md §11).

POSTs to ``<api_base>/embeddings`` with ``{"model", "input", ...extra}``.
Extra body params (e.g. NVIDIA's ``input_type`` / ``truncate``) are carried
from configuration so asymmetric models work out of the box.

Image embedding is supported through base64 data URIs — the interface used by
multimodal OpenAI-compatible endpoints such as NVIDIA's
``nvidia/llama-nemotron-embed-vl-1b-v2`` (text and image share one vector
space, so the same ``dimension`` applies to both named vectors).
"""

from __future__ import annotations

import asyncio
import base64
import logging

import httpx

from ..exceptions import EmbeddingAPIError
from ._http import post_json
from .base import EmbeddingProvider

logger = logging.getLogger("rag.providers.embedding")

_DEFAULT_INPUT_TYPE = "passage"

_MIME_BY_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"RIFF", "image/webp"),
)


def _guess_image_mime(data: bytes) -> str:
    for magic, mime in _MIME_BY_MAGIC:
        if data.startswith(magic):
            return mime
    return "image/png"


class OpenAICompatibleEmbedding(EmbeddingProvider):
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        dimension: int,
        *,
        batch_size: int = 32,
        concurrency: int = 4,
        timeout: float = 60.0,
        extra_params: dict | None = None,
        input_type: str | None = None,
        supports_image: bool = True,
        transport=None,
    ) -> None:
        if not api_base or not api_key or not model or dimension <= 0:
            raise ValueError("api_base/api_key/model/dimension 必须全部提供")
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._batch_size = max(1, batch_size)
        self._input_type = input_type or _DEFAULT_INPUT_TYPE
        self._supports_image = supports_image
        self._extra_params = dict(extra_params or {})
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    # -- EmbeddingProvider ------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def supports_image(self) -> bool:
        return self._supports_image

    async def embed_text(
        self, texts: list[str], *, input_type: str | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        input_type = input_type or self._input_type
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            out.extend(await self._embed_batch(batch, input_type))
        return out

    async def embed_image(self, images: list[bytes]) -> list[list[float]]:
        if not images:
            return []
        if not self._supports_image:
            raise EmbeddingAPIError(f"模型 {self._model} 不支持图片 Embedding")
        inputs = [
            f"data:{_guess_image_mime(data)};base64,"
            f"{base64.b64encode(data).decode('ascii')}"
            for data in images
        ]
        out: list[list[float]] = []
        for start in range(0, len(inputs), self._batch_size):
            batch = inputs[start : start + self._batch_size]
            out.extend(await self._embed_batch(batch, self._input_type))
        return out

    async def close(self) -> None:
        await self._client.aclose()

    # -- internals --------------------------------------------------------

    async def _embed_batch(self, inputs: list[str], input_type: str) -> list[list[float]]:
        body = {
            "model": self._model,
            "input": inputs,
            "input_type": input_type,
            **self._extra_params,
        }
        async with self._semaphore:
            resp = await post_json(
                self._client,
                f"{self._api_base}/embeddings",
                body,
                error_cls=EmbeddingAPIError,
            )
        try:
            data = resp.json()["data"]
            data.sort(key=lambda d: d.get("index", 0))
            return [item["embedding"] for item in data]
        except (KeyError, ValueError) as exc:
            raise EmbeddingAPIError(
                f"Embedding 响应格式异常: {resp.text[:200]!r}"
            ) from exc
