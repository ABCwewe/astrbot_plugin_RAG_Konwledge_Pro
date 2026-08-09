"""Shared HTTP plumbing for remote providers (AGENTS.md §29-§30).

Bounded retries with exponential backoff apply to transient failures:
network errors, timeouts, HTTP 5xx and 429. HTTP 4xx (except 429) are
configuration/request errors and fail immediately — retrying them wastes
calls and never succeeds.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from ..exceptions import RAGError

logger = logging.getLogger("rag.providers.http")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRYABLE_EXC = (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError)

_MAX_RETRIES = 3


def _describe_response(resp: httpx.Response, limit: int = 300) -> str:
    """Safe response preview — never includes headers (no API keys leak)."""
    text = (resp.text or "")[:limit]
    return f"HTTP {resp.status_code}: {text}"


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    *,
    error_cls: type[RAGError],
    retries: int = _MAX_RETRIES,
    retry_on: tuple[type[Exception], ...] = _RETRYABLE_EXC,
) -> httpx.Response:
    """POST JSON with bounded retries; raise ``error_cls`` on final failure."""
    attempt = 0
    while True:
        try:
            resp = await client.post(url, json=body)
        except retry_on as exc:
            attempt += 1
            if attempt > retries:
                raise error_cls(f"请求 {url} 失败: {exc!r}") from exc
            await asyncio.sleep(min(2 ** (attempt - 1), 8))
            continue

        if resp.status_code in _RETRYABLE_STATUS:
            attempt += 1
            if attempt > retries:
                raise error_cls(f"请求 {url} 失败: {_describe_response(resp)}")
            await asyncio.sleep(min(2 ** (attempt - 1), 8))
            continue

        if resp.status_code >= 400:
            raise error_cls(f"请求 {url} 失败: {_describe_response(resp)}")
        return resp
