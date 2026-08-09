"""Persistent embedding cache (AGENTS.md §22-§23).

Cache keys are ``<kind>:<model>:<content_hash>`` where ``kind`` is ``text`` or
``image`` and the hash is over the payload (chunk text / image bytes). Hits
avoid re-calling remote embedding APIs, which matters for rebuilds, model
switches with overlapping content, duplicate imports and API-failure recovery.

Backed by SQLite; all access is serialized through an ``asyncio.Lock`` and the
connection is created with ``check_same_thread=False`` because the blocking
calls run in worker threads via ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path


class EmbeddingCache:
    def __init__(self, db_path: str | Path) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embedding_cache ("
            "  cache_key TEXT PRIMARY KEY,"
            "  kind TEXT NOT NULL,"
            "  model TEXT NOT NULL,"
            "  content_hash TEXT NOT NULL,"
            "  embedding TEXT NOT NULL"
            ")"
        )
        self._db.commit()
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(kind: str, model: str, content_hash: str) -> str:
        return f"{kind}:{model}:{content_hash}"

    @staticmethod
    def content_hash_of_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def content_hash_of_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    # -- single value -----------------------------------------------------

    async def get(
        self, kind: str, model: str, content_hash: str
    ) -> list[float] | None:
        async with self._lock:
            row = await asyncio.to_thread(
                self._db.execute,
                "SELECT embedding FROM embedding_cache WHERE cache_key = ?",
                (self._key(kind, model, content_hash),),
            )
            row = row.fetchone()
        return json.loads(row[0]) if row else None

    async def set(
        self, kind: str, model: str, content_hash: str, embedding: list[float]
    ) -> None:
        key = self._key(kind, model, content_hash)
        blob = json.dumps(embedding)
        async with self._lock:
            await asyncio.to_thread(
                self._db.execute,
                "INSERT OR REPLACE INTO embedding_cache "
                "(cache_key, kind, model, content_hash, embedding) VALUES (?,?,?,?,?)",
                (key, kind, model, content_hash, blob),
            )
            self._db.commit()

    # -- batch ------------------------------------------------------------

    async def get_many(
        self, kind: str, model: str, content_hashes: list[str]
    ) -> list[list[float] | None]:
        """Return embeddings aligned with ``content_hashes`` (``None`` = miss)."""
        if not content_hashes:
            return []
        keys = [self._key(kind, model, h) for h in content_hashes]
        async with self._lock:
            rows = await asyncio.to_thread(
                self._db.execute,
                f"SELECT cache_key, embedding FROM embedding_cache "
                f"WHERE cache_key IN ({','.join('?' * len(keys))})",
                keys,
            )
            rows = rows.fetchall()
        by_key = {k: json.loads(b) for k, b in rows}
        return [by_key.get(k) for k in keys]

    async def set_many(
        self, kind: str, model: str, content_hashes: list[str], embeddings: list[list[float]]
    ) -> None:
        if not content_hashes:
            return
        async with self._lock:
            await asyncio.to_thread(
                self._db.executemany,
                "INSERT OR REPLACE INTO embedding_cache "
                "(cache_key, kind, model, content_hash, embedding) VALUES (?,?,?,?,?)",
                [
                    (self._key(kind, model, h), kind, model, h, json.dumps(v))
                    for h, v in zip(content_hashes, embeddings)
                ],
            )
            self._db.commit()

    async def stats(self) -> dict:
        async with self._lock:
            row = await asyncio.to_thread(
                self._db.execute, "SELECT COUNT(*), COUNT(DISTINCT model) FROM embedding_cache"
            )
            row = row.fetchone()
        return {"entries": row[0], "models": row[1]}

    def close(self) -> None:
        self._db.close()
