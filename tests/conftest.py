"""Shared fixtures; fakes live in tests.fakes."""

from __future__ import annotations

import pytest

from .fakes import FakeEmbedding, FakeReranker, FakeVectorStore


@pytest.fixture
def fake_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def fake_embedding() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()
