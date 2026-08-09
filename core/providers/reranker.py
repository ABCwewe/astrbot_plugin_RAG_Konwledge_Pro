"""SiliconFlow-style reranker provider (vLLM Rerank, AGENTS.md §12).

POSTs to ``<api_base>/rerank`` (or ``<api_base>/v1/rerank`` when the base URL
does not already end in ``/v1``) with
``{"model", "query", "documents", "top_n"}`` and reads back
``results: [{index, relevance_score}]``.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from ..exceptions import RerankerAPIError
from ._http import post_json
from .base import RerankResult, RerankerProvider

logger = logging.getLogger("rag.providers.reranker")


class SiliconFlowReranker(RerankerProvider):
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        *,
        concurrency: int = 4,
        timeout: float = 60.0,
        transport=None,
    ) -> None:
        if not api_base or not api_key or not model:
            raise ValueError("api_base/api_key/model 必须全部提供")
        base = api_base.rstrip("/")
        self._rerank_url = f"{base}/rerank" if base.endswith("/v1") else f"{base}/v1/rerank"
        self._model = model
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    @property
    def model_name(self) -> str:
        return self._model

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankResult]:
        if not documents or top_n <= 0:
            return []
        body = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        async with self._semaphore:
            resp = await post_json(
                self._client, self._rerank_url, body, error_cls=RerankerAPIError
            )
        try:
            results = resp.json()["results"]
        except (KeyError, ValueError) as exc:
            raise RerankerAPIError(
                f"Rerank 响应格式异常: {resp.text[:200]!r}"
            ) from exc
        ordered = []
        for item in results:
            index = int(item["index"])
            ordered.append(
                RerankResult(
                    index=index,
                    score=float(item["relevance_score"]),
                    document=documents[index] if 0 <= index < len(documents) else "",
                )
            )
        ordered.sort(key=lambda r: r.score, reverse=True)
        return ordered

    async def close(self) -> None:
        await self._client.aclose()
