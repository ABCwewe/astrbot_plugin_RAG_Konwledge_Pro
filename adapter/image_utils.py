"""Message image extraction (body first, then quoted/replied messages).

Pure duck-typed logic (no astrbot imports) so it is unit-testable standalone.
Used by the auto image-vector-search injection: when a user message carries an
image, the plugin embeds it and searches the knowledge base, without touching
AstrBot's native image handling (passthrough for vision models / captioning
for non-vision models).
"""

from __future__ import annotations

from typing import Any


def find_message_image(messages: list[Any]) -> Any | None:
    """Return the first usable image component from a message chain.

    Priority: body images first, then the quoted message's chain. An image
    component is usable when it exposes a non-empty ``url`` or ``file``.
    """
    messages = messages or []
    for comp in messages:
        if _is_image(comp):
            return comp
    for comp in messages:
        chain = getattr(comp, "chain", None)
        if chain is None:
            continue
        for sub in chain:
            if _is_image(sub):
                return sub
    return None


def _is_image(comp: Any) -> bool:
    if comp is None:
        return False
    # Duck typing: any component exposing url/file with content qualifies;
    # avoids importing astrbot components (keeps this module import-safe).
    if not getattr(comp, "url", None) and not getattr(comp, "file", None):
        return False
    return True
