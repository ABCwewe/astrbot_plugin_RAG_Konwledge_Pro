"""Per-deployment Qdrant collection namespace (multi-client isolation).

When several AstrBot deployments share one Qdrant backend, collection names
must be namespaced per client so they cannot see / delete each other's data.
The default namespace is derived from stable device characteristics (MAC via
``uuid.getnode()`` + machine id) so it is unique per machine with zero
configuration; the raw identifiers are hashed so nothing identifying leaks
into collection names. The adapter persists the first computed value under
``data/plugin_data/<plugin>/qdrant_namespace`` so the namespace stays stable
even if a device ID later changes (new NIC, VM migration, OS reinstall).
"""

from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from pathlib import Path


def _machine_id() -> str | None:
    """Stable per-OS-install machine identifier (best effort)."""
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography"
            ) as key:
                value, _ = winreg.QueryValueEx(key, "MachineGuid")
                if value:
                    return str(value)
        except Exception:  # noqa: BLE001 - best effort
            pass
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            value = Path(path).read_text(encoding="utf-8").strip()
            if value:
                return value
        except OSError:
            continue
    return None


def compute_device_namespace(length: int = 16) -> str:
    """Opaque, stable per-device namespace seed (hex, ``length`` chars).

    Combines the primary MAC and the platform machine id (whichever are
    available) and hashes them. A machine with no discoverable identity
    falls back to a random value (still persisted by the caller, so it
    remains stable).
    """
    parts: list[str] = []
    try:
        parts.append(f"mac:{uuid.getnode():012x}")
    except Exception:  # noqa: BLE001 - best effort
        pass
    mid = _machine_id()
    if mid:
        parts.append(f"machine:{mid}")
    if not parts:
        parts.append(f"random:{secrets.token_hex(16)}")
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]
