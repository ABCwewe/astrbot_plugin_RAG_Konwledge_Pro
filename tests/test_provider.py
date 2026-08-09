"""Provider tests — request shape, batching, retries, error mapping, image
data URIs, reranker parsing. Uses httpx MockTransport; no real network."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from core.exceptions import EmbeddingAPIError, RerankerAPIError
from core.providers.openai_compatible import OpenAICompatibleEmbedding, _guess_image_mime
from core.providers.reranker import SiliconFlowReranker


def _embedding_transport(*responses: httpx.Response):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        resp = responses[min(len(calls) - 1, len(responses) - 1)]
        return resp

    return httpx.MockTransport(handler), calls


def _ok_embeddings(inputs: list[str]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": [
                {"index": i, "embedding": [float(i + 1), 0.0, 0.0, 0.0]}
                for i in range(len(inputs))
            ]
        },
    )


async def test_embed_text_sends_model_input_and_input_type():
    transport, calls = _embedding_transport(_ok_embeddings(["a", "b"]))
    provider = OpenAICompatibleEmbedding(
        "http://localhost:1/v1", "key", "model-x", 4, transport=transport
    )
    vectors = await provider.embed_text(["a", "b"], input_type="passage")
    assert vectors == [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]
    body = json.loads(calls[0].content)
    assert body["model"] == "model-x"
    assert body["input"] == ["a", "b"]
    assert body["input_type"] == "passage"
    assert calls[0].headers["authorization"] == "Bearer key"


async def test_embed_text_accepts_batch_sizes():
    transport, calls = _embedding_transport(_ok_embeddings(["a"]), _ok_embeddings(["b"]))
    provider = OpenAICompatibleEmbedding(
        "http://localhost:1/v1", "key", "m", 4, batch_size=1, transport=transport
    )
    await provider.embed_text(["a", "b"])
    assert len(calls) == 2


async def test_embed_image_uses_base64_data_uri():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    transport, calls = _embedding_transport(_ok_embeddings([""]))

    provider = OpenAICompatibleEmbedding(
        "http://localhost:1/v1", "key", "m", 4, transport=transport
    )
    await provider.embed_image([png])
    body = json.loads(calls[0].content)
    uri = body["input"][0]
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == png


def test_guess_image_mime():
    assert _guess_image_mime(b"\x89PNG\r\n\x1a\n...") == "image/png"
    assert _guess_image_mime(b"\xff\xd8\xff...") == "image/jpeg"
    assert _guess_image_mime(b"GIF89a...") == "image/gif"
    assert _guess_image_mime(b"\x00\x01\x02") == "image/png"  # unknown fallback


async def test_retries_transient_errors_then_succeeds():
    transport, calls = _embedding_transport(
        httpx.Response(503, json={"error": "unavailable"}),
        httpx.Response(502, json={"error": "bad gateway"}),
        _ok_embeddings(["a"]),
    )
    provider = OpenAICompatibleEmbedding(
        "http://localhost:1/v1", "key", "m", 4, transport=transport
    )
    vectors = await provider.embed_text(["a"])
    assert vectors == [[1.0, 0.0, 0.0, 0.0]]
    assert len(calls) == 3


async def test_persistent_failure_raises_embedding_api_error():
    transport, _ = _embedding_transport(
        httpx.Response(503, json={"error": "x"}),
        httpx.Response(503, json={"error": "x"}),
        httpx.Response(503, json={"error": "x"}),
        httpx.Response(503, json={"error": "x"}),
    )
    provider = OpenAICompatibleEmbedding(
        "http://localhost:1/v1", "key", "m", 4, transport=transport
    )
    with pytest.raises(EmbeddingAPIError):
        await provider.embed_text(["a"])


async def test_4xx_is_immediate_error_without_retry():
    transport, calls = _embedding_transport(
        httpx.Response(400, json={"error": "invalid"})
    )
    provider = OpenAICompatibleEmbedding(
        "http://localhost:1/v1", "key", "m", 4, transport=transport
    )
    with pytest.raises(EmbeddingAPIError):
        await provider.embed_text(["a"])
    assert len(calls) == 1


async def test_reranker_parses_and_orders_results():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "rerank-m"
        assert body["query"] == "q"
        assert body["documents"] == ["doc a", "doc b"]
        assert body["top_n"] == 1
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            },
        )

    reranker = SiliconFlowReranker(
        "http://localhost:1/v1", "key", "rerank-m",
        transport=httpx.MockTransport(handler),
    )
    results = await reranker.rerank("q", ["doc a", "doc b"], 1)
    assert results[0].index == 1
    assert results[0].score == 0.9
    assert results[0].document == "doc b"


async def test_reranker_base_url_path_resolution():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"results": []})

    # Base with /v1 suffix and base without it both hit <base>/v1/rerank.
    r1 = SiliconFlowReranker(
        "http://localhost:1/v1", "k", "m", transport=httpx.MockTransport(handler)
    )
    await r1.rerank("q", ["d"], 1)
    assert seen[-1] == "http://localhost:1/v1/rerank"

    r2 = SiliconFlowReranker(
        "http://localhost:1", "k", "m", transport=httpx.MockTransport(handler)
    )
    await r2.rerank("q", ["d"], 1)
    assert seen[-1] == "http://localhost:1/v1/rerank"


async def test_reranker_persistent_failure_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    reranker = SiliconFlowReranker(
        "http://localhost:1/v1", "k", "m",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RerankerAPIError):
        await reranker.rerank("q", ["d"], 1)
