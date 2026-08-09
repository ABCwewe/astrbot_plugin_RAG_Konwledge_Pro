"""Document parser abstraction (AGENTS.md §8)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import ParsedDocument


class DocumentParser(ABC):
    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Whether this parser can handle the file at ``path``."""

    @abstractmethod
    async def parse(self, path: Path) -> ParsedDocument:
        ...
