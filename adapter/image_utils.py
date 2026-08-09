"""Message image extraction for the LLM tool (mirrors the pattern used by
astrbot_plugin_serpapi_imgsearch.reverse_image_search).

Pure duck-typed logic (no astrbot imports) so it is unit-testable standalone:

- body images first (``Image`` components with ``url``/``file``)
- then quoted/replied messages (``Reply.chain`` images)

The caller (adapter) resolves the component to a usable address via
``convert_to_file_path`` (MediaResolver handles file:// URIs, local paths and
downloads http URLs).
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


def describe_no_image() -> str:
    """Error string returned to the LLM when no image is present."""
    return (
        "error: 未在消息中找到图片。请提示用户直接发送一张图片，"
        "或引用一条包含图片的消息后再试。"
    )
