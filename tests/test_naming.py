"""Namespace tests — device fingerprint namespace for Qdrant collection
isolation (multi-client shared backend)."""

from __future__ import annotations

import re

from core.naming import compute_device_namespace


def test_namespace_is_short_hex():
    ns = compute_device_namespace()
    assert re.fullmatch(r"[0-9a-f]{16}", ns), ns


def test_namespace_is_deterministic_within_run():
    assert compute_device_namespace() == compute_device_namespace()


def test_namespace_does_not_leak_raw_mac():
    """命名空间是哈希结果，绝不包含可读的 MAC/机器 ID。"""
    ns = compute_device_namespace()
    assert "mac" not in ns
    assert "machine" not in ns
    assert re.fullmatch(r"[0-9a-f]+", ns)
