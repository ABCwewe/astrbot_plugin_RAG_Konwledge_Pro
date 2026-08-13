"""Plugin configuration model.

All configuration flows through :class:`RAGConfig`, a frozen dataclass that is
also responsible for computing the ``config_hash`` used to detect index
rebuild triggers (AGENTS.md §16-§17).

The class can be built from a plain dict (AstrBot's ``AstrBotConfig``, a JSON
file, or test fixtures) via :meth:`RAGConfig.from_dict`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class QdrantConfig:
    url: str = "http://127.0.0.1:6333"
    api_key: str | None = None
    timeout: float = 60.0
    #: Collection 命名空间前缀。多个 client 共享同一 Qdrant 后端时，每个
    #: client 应有唯一前缀以隔离集合（避免互相误删/误认）。空串表示使用
    #: 设备指纹自动生成的命名空间（由 adapter 解析并持久化后注入）。
    collection_prefix: str = ""


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "openai_compatible"
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    dimension: int = 0
    batch_size: int = 32
    concurrency: int = 4
    timeout: float = 60.0
    #: Extra body params forwarded to the embedding endpoint. NVIDIA's
    #: asymmetric models require ``input_type`` and accept ``truncate``.
    extra_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ImageEmbeddingConfig:
    enabled: bool = False
    provider: str = "openai_compatible"
    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None
    dimension: int | None = None
    #: Also search image vectors on ordinary text queries.
    search_always: bool = False
    #: Automatically embed incoming message images and inject the retrieved
    #: context into the LLM request (does not touch native image handling).
    auto_search: bool = True
    #: Minimum vector similarity (cosine) for automatic image retrieval;
    #: 0 disables filtering (noise guard for small knowledge bases).
    min_score: float = 0.0
    #: Normalize images (index + query) so the longest side <= this value
    #: (BOX downscale, deterministic). 0 disables normalization.
    max_side: int = 1024
    batch_size: int = 8
    concurrency: int = 2
    timeout: float = 60.0
    extra_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = True
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    concurrency: int = 4
    timeout: float = 60.0


@dataclass(frozen=True)
class ChunkingConfig:
    separator: str = "\n\n"
    chunk_size: int = 800
    chunk_overlap: int = 100


@dataclass(frozen=True)
class RAGConfig:
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    image: ImageEmbeddingConfig = field(default_factory=ImageEmbeddingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    top_k: int = 30
    top_n: int = 6
    #: Rerank candidate-pool cap. In multi-KB aggregation every KB contributes
    #: up to ``top_k`` candidates, but the merged pool never exceeds this.
    rerank_pool_size: int = 60
    kb_id: str = "default"
    #: Max concurrent index operations (rebuild/incremental sync) across the
    #: engine. Multiple uploads are queued server-side beyond this limit.
    ingest_concurrency: int = 2

    # -- construction -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "RAGConfig":
        """Build a config from a nested plain dict (with defaults filled)."""
        try:
            q = data.get("qdrant", {})
            e = data.get("embedding", {})
            im = data.get("image", {})
            r = data.get("rerank", {})
            c = data.get("chunking", {})
            return cls(
                qdrant=QdrantConfig(
                    url=q.get("url", "http://127.0.0.1:6333"),
                    api_key=q.get("api_key") or None,
                    timeout=q.get("timeout", 60.0),
                    collection_prefix=str(q.get("collection_prefix", "") or ""),
                ),
                embedding=EmbeddingConfig(
                    provider=e.get("provider", "openai_compatible"),
                    api_base=e.get("api_base", ""),
                    api_key=e.get("api_key", ""),
                    model=e.get("model", ""),
                    dimension=int(e.get("dimension", 0)),
                    batch_size=int(e.get("batch_size", 32)),
                    concurrency=int(e.get("concurrency", 4)),
                    timeout=float(e.get("timeout", 60.0)),
                    extra_params=dict(e.get("extra_params", {})),
                ),
                image=ImageEmbeddingConfig(
                    enabled=bool(im.get("enabled", False)),
                    provider=im.get("provider", "openai_compatible"),
                    api_base=im.get("api_base"),
                    api_key=im.get("api_key"),
                    model=im.get("model"),
                    dimension=int(im["dimension"]) if im.get("dimension") else None,
                    search_always=bool(im.get("search_always", False)),
                    auto_search=bool(im.get("auto_search", True)),
                    min_score=float(im.get("min_score", 0.0)),
                    max_side=int(im.get("max_side", 1024)),
                    batch_size=int(im.get("batch_size", 8)),
                    concurrency=int(im.get("concurrency", 2)),
                    timeout=float(im.get("timeout", 60.0)),
                    extra_params=dict(im.get("extra_params", {})),
                ),
                rerank=RerankConfig(
                    enabled=bool(r.get("enabled", True)),
                    api_base=r.get("api_base", ""),
                    api_key=r.get("api_key", ""),
                    model=r.get("model", ""),
                    concurrency=int(r.get("concurrency", 4)),
                    timeout=float(r.get("timeout", 60.0)),
                ),
                chunking=ChunkingConfig(
                    separator=str(c.get("separator", "\n\n")),
                    chunk_size=int(c.get("chunk_size", 800)),
                    chunk_overlap=int(c.get("chunk_overlap", 100)),
                ),
                top_k=int(data.get("top_k", 30)),
                top_n=int(data.get("top_n", 6)),
                rerank_pool_size=int(data.get("rerank_pool_size", 60)),
                kb_id=str(data.get("kb_id", "default")),
                ingest_concurrency=int(data.get("ingest_concurrency", 2)),
            )
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ConfigurationError(f"无效的 RAG 配置: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "RAGConfig":
        """Load config from a JSON file."""
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"无法读取配置文件 {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError(f"配置文件 {path} 必须是 JSON 对象")
        return cls.from_dict(raw)

    def validate(self) -> None:
        """Raise :class:`ConfigurationError` for unusable combinations."""
        e = self.embedding
        if not e.api_base or not e.api_key or not e.model:
            raise ConfigurationError("embedding.api_base/api_key/model 不能为空")
        if e.dimension <= 0:
            raise ConfigurationError("embedding.dimension 必须大于 0")
        if self.chunking.chunk_size <= 0:
            raise ConfigurationError("chunking.chunk_size 必须大于 0")
        if self.chunking.chunk_overlap < 0:
            raise ConfigurationError("chunking.chunk_overlap 不能为负")
        if self.chunking.chunk_overlap >= self.chunking.chunk_size:
            raise ConfigurationError("chunking.chunk_overlap 必须小于 chunk_size")
        if self.image.enabled:
            if not (self.image.api_base and self.image.api_key and self.image.model):
                raise ConfigurationError("image.enabled 时需配置 image.api_base/api_key/model")
            if not self.image.dimension:
                raise ConfigurationError("image.enabled 时需配置 image.dimension")
            if not (0.0 <= self.image.min_score <= 1.0):
                raise ConfigurationError("image.min_score 必须在 0~1 之间")
            if self.image.max_side < 0:
                raise ConfigurationError("image.max_side 不能为负")
        if self.rerank.enabled and (not self.rerank.api_base or not self.rerank.api_key or not self.rerank.model):
            raise ConfigurationError("rerank.enabled 时需配置 rerank.api_base/api_key/model")
        if self.top_k <= 0 or self.top_n <= 0:
            raise ConfigurationError("top_k/top_n 必须大于 0")
        if self.rerank_pool_size < self.top_n:
            raise ConfigurationError("rerank_pool_size 不能小于 top_n")
        if self.ingest_concurrency <= 0:
            raise ConfigurationError("ingest_concurrency 必须大于 0")

    # -- config hash ------------------------------------------------------

    def config_hash(self) -> str:
        """Hash of every setting that affects index content.

        Mirrors AGENTS.md §17: embedding provider/model/dimension, image
        embedding provider/model/dimension, chunk separator/size/overlap.
        """
        payload = {
            "embedding_provider": self.embedding.provider,
            "embedding_model": self.embedding.model,
            "embedding_dimension": self.embedding.dimension,
            "image_provider": self.image.provider if self.image.enabled else None,
            "image_model": self.image.model if self.image.enabled else None,
            "image_dimension": self.image.dimension if self.image.enabled else None,
            "image_max_side": self.image.max_side if self.image.enabled else None,
            "chunk_separator": self.chunking.separator,
            "chunk_size": self.chunking.chunk_size,
            "chunk_overlap": self.chunking.chunk_overlap,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
