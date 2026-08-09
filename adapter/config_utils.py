"""Config management helpers for the plugin WebUI (AGENTS.md §36).

Pure logic, no AstrBot imports, so it is unit-testable standalone:

- ``mask_config``    — API keys never leave the backend as plaintext.
- ``coerce_value``   — coerce a frontend string/raw value to the schema type
                       (avoids ``bool("false") == True`` style bugs).
- ``apply_patch``    — merge a frontend patch into the live config, treating
                       the masked sentinel as "keep the old value".
"""

from __future__ import annotations

import json
from pathlib import Path

MASKED_SENTINEL = "********"
"""Value sent to the frontend for secret fields; submitting it back means
'keep the current value'."""

_SECRET_KEY_HINT = "api_key"
"""Config keys containing this substring are treated as secrets."""

_BOOL_TRUE = {"true", "yes", "1", "on"}
_BOOL_FALSE = {"false", "no", "0", "off", ""}


def is_secret_key(key: str) -> bool:
    return _SECRET_KEY_HINT in key.lower()


def mask_config(config: dict) -> dict:
    """Return a copy of ``config`` with secret values replaced by the sentinel.

    Secret keys whose value is empty stay empty (nothing to hide).
    """
    out = dict(config)
    for key, value in out.items():
        if is_secret_key(key) and value not in (None, ""):
            out[key] = MASKED_SENTINEL
    return out


def coerce_value(key: str, value, schema: dict) -> object:
    """Coerce a frontend-supplied value to the schema type of ``key``."""
    meta = schema.get(key, {})
    field_type = meta.get("type", "string")
    if value is None:
        return None
    if field_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        lowered = str(value).strip().lower()
        if lowered in _BOOL_TRUE:
            return True
        if lowered in _BOOL_FALSE:
            return False
        return bool(value)  # last resort
    if field_type == "int":
        return int(value)
    if field_type == "float":
        return float(value)
    return str(value)


def apply_patch(current: dict, patch: dict, schema: dict) -> dict:
    """Merge ``patch`` into a copy of ``current`` with schema coercion.

    - keys not present in the schema are dropped (defense in depth)
    - the masked sentinel on a secret key means "keep current value"
    - values are coerced to the schema type
    """
    merged = dict(current)
    for key, value in patch.items():
        if key not in schema:
            continue
        if is_secret_key(key) and value == MASKED_SENTINEL:
            continue  # keep existing value (may be absent → stays absent)
        merged[key] = coerce_value(key, value, schema)
    return merged


def group_order(schema: dict) -> list[str]:
    """Groups in first-seen order (schema order defines display order)."""
    groups: list[str] = []
    for meta in schema.values():
        group = meta.get("group", "general") if isinstance(meta, dict) else "general"
        if group not in groups:
            groups.append(group)
    return groups


def group_members(schema: dict) -> dict[str, list[str]]:
    """Map group name → ordered list of schema keys."""
    members: dict[str, list[str]] = {}
    for key, meta in schema.items():
        group = meta.get("group", "general") if isinstance(meta, dict) else "general"
        members.setdefault(group, []).append(key)
    return members


def load_search_defaults(path: str | Path) -> list[str]:
    """Load persisted default multi-KB search ids (empty on missing/corrupt)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        ids = raw.get("kb_ids", []) if isinstance(raw, dict) else []
        return [kb for kb in ids if isinstance(kb, str) and kb.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def save_search_defaults(path: str | Path, kb_ids: list[str]) -> list[str]:
    """Persist default multi-KB search ids; returns the deduped/trimmed list."""
    ids: list[str] = []
    for kb in kb_ids:
        kb = str(kb).strip()
        if kb and kb not in ids:
            ids.append(kb)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"kb_ids": ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ids
