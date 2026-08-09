"""AstrBot integration adapter package.

``AstrBotRAGAdapter`` is imported lazily (PEP 562) so that importing pure
helper modules (``adapter.config_utils``) does not pull in AstrBot, which
would create AstrBot data directories under the current working directory.
"""

from __future__ import annotations

_LAZY_EXPORTS = ("AstrBotRAGAdapter",)


def __getattr__(name: str):
    if name == "AstrBotRAGAdapter":
        from .astrbot import AstrBotRAGAdapter

        return AstrBotRAGAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
