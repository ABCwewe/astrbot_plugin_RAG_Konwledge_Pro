"""Tests for adapter/config_utils.py — masking, coercion, patching."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter.config_utils import (  # noqa: E402
    MASKED_SENTINEL,
    apply_patch,
    coerce_value,
    group_members,
    group_order,
    is_secret_key,
    mask_config,
)

SCHEMA = {
    "qdrant_api_key": {"type": "string", "group": "qdrant"},
    "qdrant_url": {"type": "string", "group": "qdrant"},
    "top_k": {"type": "int", "group": "retrieval"},
    "rerank_enabled": {"type": "bool", "group": "rerank"},
    "chunk_separator": {"type": "string", "group": "chunking"},
}


def test_is_secret_key():
    assert is_secret_key("embedding_api_key")
    assert is_secret_key("qdrant_api_key")
    assert not is_secret_key("qdrant_url")
    assert not is_secret_key("top_k")


def test_mask_config_replaces_secrets_not_empty():
    config = {"embedding_api_key": "sk-real", "qdrant_api_key": "", "top_k": 30}
    masked = mask_config(config)
    assert masked["embedding_api_key"] == MASKED_SENTINEL
    assert masked["qdrant_api_key"] == ""  # empty stays empty
    assert masked["top_k"] == 30
    # original untouched
    assert config["embedding_api_key"] == "sk-real"


def test_coerce_bool_from_strings():
    assert coerce_value("rerank_enabled", "true", SCHEMA) is True
    assert coerce_value("rerank_enabled", "false", SCHEMA) is False
    assert coerce_value("rerank_enabled", "0", SCHEMA) is False
    assert coerce_value("rerank_enabled", "1", SCHEMA) is True
    assert coerce_value("rerank_enabled", True, SCHEMA) is True
    assert coerce_value("rerank_enabled", "yes", SCHEMA) is True


def test_coerce_int_and_string():
    assert coerce_value("top_k", "42", SCHEMA) == 42
    assert coerce_value("qdrant_url", 123, SCHEMA) == "123"


def test_apply_patch_coerces_and_drops_unknown():
    current = {"top_k": 30, "rerank_enabled": True}
    merged = apply_patch(current, {"top_k": "50", "rerank_enabled": "false", "hacker_key": 1}, SCHEMA)
    assert merged["top_k"] == 50
    assert merged["rerank_enabled"] is False
    assert "hacker_key" not in merged
    assert current["top_k"] == 30  # original untouched


def test_apply_patch_sentinel_keeps_secret():
    current = {"qdrant_api_key": "sk-real"}
    merged = apply_patch(current, {"qdrant_api_key": MASKED_SENTINEL}, SCHEMA)
    assert merged["qdrant_api_key"] == "sk-real"
    # new key with sentinel → stays absent
    merged2 = apply_patch({}, {"qdrant_api_key": MASKED_SENTINEL}, SCHEMA)
    assert "qdrant_api_key" not in merged2
    # real new value replaces
    merged3 = apply_patch(current, {"qdrant_api_key": "sk-new"}, SCHEMA)
    assert merged3["qdrant_api_key"] == "sk-new"


def test_groups_follow_schema_order():
    assert group_order(SCHEMA) == ["qdrant", "retrieval", "rerank", "chunking"]
    members = group_members(SCHEMA)
    assert members["qdrant"] == ["qdrant_api_key", "qdrant_url"]
    assert members["chunking"] == ["chunk_separator"]
