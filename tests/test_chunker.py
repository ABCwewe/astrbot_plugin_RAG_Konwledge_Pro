"""TextChunker tests — AGENTS.md §7 edge cases."""

from __future__ import annotations

import pytest

from core.chunking import TextChunker


def test_empty_text_returns_no_chunks():
    assert TextChunker().split("") == []
    assert TextChunker().split("   \n \t ") == []


def test_consecutive_separators_are_collapsed():
    chunks = TextChunker(separator="\n\n", chunk_size=50, chunk_overlap=0).split(
        "a\n\n\n\nb\n\n\n\n\n\nc"
    )
    assert chunks == ["a\n\nb\n\nc"]


def test_separator_absent_splits_by_size():
    chunker = TextChunker(separator="|||", chunk_size=30, chunk_overlap=0)
    chunks = chunker.split("x" * 100)
    assert len(chunks) == 4  # 30+30+30+10
    assert all(len(c) <= 30 for c in chunks)
    assert "".join(chunks) == "x" * 100


def test_all_chunks_respect_size_bound():
    chunker = TextChunker(separator="\n\n", chunk_size=100, chunk_overlap=20)
    text = "段落" * 200 + "\n\n" + "内容" * 300
    chunks = chunker.split(text)
    assert chunks
    assert all(len(c) <= 100 for c in chunks)


def test_overlap_carries_tail_of_previous_chunk():
    chunker = TextChunker(separator="\n\n", chunk_size=80, chunk_overlap=15)
    chunks = chunker.split(("A" * 30) + "\n\n" + ("B" * 30) + "\n\n" + ("C" * 30))
    assert len(chunks) >= 2
    first = chunks[0]
    second = chunks[1]
    # The second chunk must begin with the last 15 chars of the first.
    assert first[-15:] == second[:15]


def test_last_chunk_may_be_smaller_than_size():
    chunker = TextChunker(separator="\n\n", chunk_size=50, chunk_overlap=5)
    text = "\n\n".join(f"段{i}" * 20 for i in range(3))
    chunks = chunker.split(text)
    assert chunks
    assert len(chunks[-1]) <= 50
    assert all(c.strip() for c in chunks)


def test_oversized_single_segment_is_hard_split():
    chunker = TextChunker(separator="\n\n", chunk_size=10, chunk_overlap=0)
    chunks = chunker.split("A" * 33)
    assert "".join(chunks) == "A" * 33
    assert all(len(c) <= 10 for c in chunks)


def test_validation_rejects_bad_params():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0)
    with pytest.raises(ValueError):
        TextChunker(chunk_overlap=-1)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=10)


def test_full_text_reassembly_loses_nothing():
    chunker = TextChunker(separator="\n\n", chunk_size=37, chunk_overlap=9)
    text = "\n\n".join(f"第{i}章：测试内容" * 7 for i in range(5))
    chunks = chunker.split(text)
    # Every original character appears in at least one chunk.
    original = set(text.replace("\n\n", ""))
    covered = set("".join(chunks).replace("\n\n", ""))
    assert covered == original
