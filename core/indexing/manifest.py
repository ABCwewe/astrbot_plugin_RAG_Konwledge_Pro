"""Index manifest + per-KB metadata persistence (AGENTS.md §16, §19).

Every version of a knowledge base's index is described by one
:class:`IndexManifest` file; ``active.json`` points at the READY version.
The document registry (``documents.json``) maps stable document ids to their
content hash so incremental updates can cheaply decide add / modify / delete
without re-parsing unchanged files (AGENTS.md §20-§21).

All writes are atomic (write temp file + ``os.replace``).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import RAGConfig

SCHEMA_VERSION = 1

# Index lifecycle states (AGENTS.md §19).
STATUS_CREATING = "CREATING"
STATUS_BUILDING = "BUILDING"
STATUS_READY = "READY"
STATUS_FAILED = "FAILED"
STATUS_DELETING = "DELETING"

_MANIFEST_PREFIX = "manifest_v"


def normalize_namespace(prefix: str) -> str:
    """Sanitize a collection namespace prefix; empty falls back to the legacy
    ``astrbot_rag`` so old deployments keep working unchanged."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (prefix or ""))
    return safe or "astrbot_rag"


def collection_name_for(prefix: str, kb_id: str, version: int) -> str:
    """Versioned collection name: ``<namespace>_<kb_id>_v<version>``.

    ``prefix`` is the per-deployment namespace (see :mod:`core.naming`) that
    isolates collections when several clients share one Qdrant backend.
    """
    ns = normalize_namespace(prefix)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in kb_id)
    return f"{ns}_{safe}_v{version}"


def document_id_for(kb_id: str, source_identity: str) -> str:
    """Stable document id = SHA-256(kb_id + source identity) (AGENTS.md §43)."""
    import hashlib

    return hashlib.sha256(f"{kb_id}\x00{source_identity}".encode("utf-8")).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@dataclass
class IndexManifest:
    schema_version: int = SCHEMA_VERSION
    kb_id: str = ""
    version: int = 1
    status: str = STATUS_CREATING
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 0
    image_embedding_provider: str | None = None
    image_embedding_model: str | None = None
    image_embedding_dimension: int | None = None
    chunk_separator: str = ""
    chunk_size: int = 0
    chunk_overlap: int = 0
    document_count: int = 0
    chunk_count: int = 0
    collection_name: str = ""
    config_hash: str = ""
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_config(
        cls,
        kb_id: str,
        version: int,
        config: RAGConfig,
        *,
        status: str = STATUS_CREATING,
    ) -> "IndexManifest":
        return cls(
            kb_id=kb_id,
            version=version,
            status=status,
            embedding_provider=config.embedding.provider,
            embedding_model=config.embedding.model,
            embedding_dimension=config.embedding.dimension,
            image_embedding_provider=(
                config.image.provider if config.image.enabled else None
            ),
            image_embedding_model=config.image.model if config.image.enabled else None,
            image_embedding_dimension=(
                config.image.dimension if config.image.enabled else None
            ),
            chunk_separator=config.chunking.separator,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            collection_name=collection_name_for(
                config.qdrant.collection_prefix, kb_id, version
            ),
            config_hash=config.config_hash(),
            created_at=_now(),
            updated_at=_now(),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IndexManifest":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def touch(self) -> None:
        self.updated_at = _now()


@dataclass
class ActiveState:
    kb_id: str
    active_version: int | None
    config_hash: str
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ActiveState":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class DocumentRecord:
    document_id: str
    source: str
    filename: str
    content_hash: str
    indexed_at: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class ManifestStore:
    """File-backed persistence for one knowledge base."""

    def __init__(self, kb_root: str | Path) -> None:
        self.kb_root = Path(kb_root)
        self.kb_root.mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------

    def manifest_path(self, version: int) -> Path:
        return self.kb_root / f"{_MANIFEST_PREFIX}{version}.json"

    @property
    def active_path(self) -> Path:
        return self.kb_root / "active.json"

    @property
    def documents_path(self) -> Path:
        return self.kb_root / "documents.json"

    # -- manifests --------------------------------------------------------

    async def save_manifest(self, manifest: IndexManifest) -> None:
        await asyncio.to_thread(
            _atomic_write_json, self.manifest_path(manifest.version), manifest.to_dict()
        )

    async def load_manifest(self, version: int) -> IndexManifest | None:
        path = self.manifest_path(version)
        if not path.exists():
            return None
        data = await asyncio.to_thread(_read_json, path)
        return IndexManifest.from_dict(data) if data else None

    async def list_manifests(self) -> dict[int, IndexManifest]:
        out: dict[int, IndexManifest] = {}
        for path in self.kb_root.glob(f"{_MANIFEST_PREFIX}*.json"):
            try:
                version = int(path.stem[len(_MANIFEST_PREFIX):])
            except ValueError:
                continue
            data = await asyncio.to_thread(_read_json, path)
            if data:
                out[version] = IndexManifest.from_dict(data)
        return out

    async def delete_manifest(self, version: int) -> None:
        path = self.manifest_path(version)
        if path.exists():
            await asyncio.to_thread(os.remove, path)

    # -- active -----------------------------------------------------------

    async def load_active(self) -> ActiveState | None:
        if not self.active_path.exists():
            return None
        data = await asyncio.to_thread(_read_json, self.active_path)
        return ActiveState.from_dict(data) if data else None

    async def save_active(self, state: ActiveState) -> None:
        if not state.updated_at:
            state.updated_at = _now()
        await asyncio.to_thread(
            _atomic_write_json, self.active_path, state.to_dict()
        )

    # -- document registry ------------------------------------------------

    async def load_documents(self) -> dict[str, DocumentRecord]:
        if not self.documents_path.exists():
            return {}
        data = await asyncio.to_thread(_read_json, self.documents_path)
        if not isinstance(data, dict):
            return {}
        return {
            key: DocumentRecord.from_dict(record)
            for key, record in data.items()
            if isinstance(record, dict)
        }

    async def save_documents(self, records: dict[str, DocumentRecord]) -> None:
        payload = {key: record.to_dict() for key, record in records.items()}
        await asyncio.to_thread(_atomic_write_json, self.documents_path, payload)


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
