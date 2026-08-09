"""Character-based text chunker (AGENTS.md §7).

Splits a document on the configured separator, then greedily merges segments
into chunks no larger than ``chunk_size`` characters. When a chunk overflows,
the next chunk is seeded with the last ``chunk_overlap`` characters of the
previous one so boundary context is preserved.

Edge cases handled: empty text, consecutive separators, a separator that never
appears, segments longer than ``chunk_size``, and a final chunk shorter than
``chunk_size``.
"""

from __future__ import annotations


class TextChunker:
    def __init__(
        self,
        separator: str = "\n\n",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # -- public API -------------------------------------------------------

    def split(self, text: str) -> list[str]:
        """Split ``text`` into a list of non-empty chunks."""
        if not text or not text.strip():
            return []
        separator = self.separator
        raw_segments = text.split(separator) if separator else [text]
        segments = [seg for seg in raw_segments if seg.strip()]
        if not segments:
            return []

        size = self.chunk_size
        overlap = self.chunk_overlap
        chunks: list[str] = []
        buf = ""

        for seg in segments:
            # A single segment larger than chunk_size must be hard-split so
            # every emitted chunk obeys the size bound. Fragments of the same
            # oversized segment are continuations: they join WITHOUT the
            # separator (which would otherwise be injected mid-paragraph).
            oversized = len(seg) > size
            pieces = (
                [seg[i : i + size] for i in range(0, len(seg), size)]
                if oversized
                else [seg]
            )

            for j, piece in enumerate(pieces):
                glue = "" if (oversized and j > 0) else separator
                if buf:
                    joined = buf + glue + piece
                else:
                    joined = piece
                if len(joined) <= size:
                    buf = joined
                    continue

                # Overflow: flush the current buffer.
                chunks.append(buf)
                seed = buf[-overlap:] if overlap > 0 else ""
                buf = self._start_new_chunk(seed, glue, piece, size)

        if buf:
            chunks.append(buf)
        return chunks

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _start_new_chunk(seed: str, glue: str, piece: str, size: int) -> str:
        """Build the next buffer from an overlap seed + glue + piece."""
        candidate = (seed + glue + piece) if seed else piece
        if len(candidate) <= size:
            return candidate
        # Overlap seed is too large to keep alongside this piece; prefer the
        # piece (whole document content) over the seed tail.
        return piece
