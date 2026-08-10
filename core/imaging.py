"""Deterministic image normalization (resize + re-encode) for embedding.

Both index and query images go through the same rule so their vectors live
in a consistent resolution space and the embedding cache key (over the
normalized bytes) is stable. Only oversized images are touched: images within
the limit pass through unchanged, so existing cache entries survive.

Algorithm: BOX downscale (fast, quality close to LANCZOS) preserving aspect,
re-encoded as JPEG q90 for a deterministic byte representation.
"""

from __future__ import annotations

import io

from PIL import Image

_REENCODE_QUALITY = 90


def normalize_image_bytes(data: bytes, max_side: int) -> bytes:
    """Resize so the longest side <= ``max_side`` and re-encode to JPEG.

    - ``max_side <= 0`` disables normalization (input returned unchanged)
    - images already within the limit are returned unchanged (no re-encode,
      existing embedding-cache entries remain valid)
    - undecodable input is passed through unchanged (errors surface later in
      the embed/parse stage)
    """
    if max_side <= 0 or not data:
        return data
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            if max(width, height) <= max_side:
                return data
            scale = max_side / max(width, height)
            new_size = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            resized = img.convert("RGB").resize(new_size, Image.BOX)
            out = io.BytesIO()
            resized.save(out, format="JPEG", quality=_REENCODE_QUALITY)
            return out.getvalue()
    except Exception:  # 非图片/损坏字节：原样透传，由后续阶段报错
        return data
