from __future__ import annotations

import imghdr
import io
from typing import Any

from PIL import Image, ImageOps

from app.core.config import get_settings
from app.core.exceptions import ImageDecodeError


_FORMAT_TO_MIME = {
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


def detect_mime(byte_stream: bytes) -> str | None:
    """Sniff the true image format from magic bytes (don't trust Content-Type)."""
    fmt = imghdr.what(None, h=byte_stream)
    if fmt is None:
        return None
    return _FORMAT_TO_MIME.get(fmt.lower())


async def read_and_validate(file_bytes: bytes) -> str:
    """Return the validated MIME type, raise on unsupported/invalid."""
    settings = get_settings()
    if len(file_bytes) > settings.max_file_size_bytes:
        raise ImageDecodeError(
            f"File too large: {len(file_bytes)} bytes "
            f"(limit {settings.max_file_size_bytes} bytes)"
        )
    mime = detect_mime(file_bytes[:2048])
    if mime is None or mime not in settings.allowed_mime_list:
        raise ImageDecodeError(
            f"Unsupported file type: {mime}. "
            f"Allowed: {settings.allowed_mime_list}"
        )
    return mime


def preprocess(image: Any) -> Image.Image:
    """Normalize an uploaded image before feeding the model.

    - Apply EXIF orientation
    - Convert to RGB
    - Resize down if either dimension exceeds MAX_DIMENSION (preserve aspect)
    """
    settings = get_settings()
    try:
        img = ImageOps.exif_transpose(image)
        img = img.convert("RGB")
        longest = max(img.size)
        if longest > settings.MAX_DIMENSION:
            ratio = settings.MAX_DIMENSION / longest
            new_size = (
                int(img.width * ratio),
                int(img.height * ratio),
            )
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        return img
    except Exception as exc:  # noqa: BLE001
        raise ImageDecodeError(f"Failed to preprocess image: {exc}") from exc


def load_pil_image(file_bytes: bytes) -> Image.Image:
    """Decode raw bytes into a PIL Image."""
    try:
        return Image.open(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        raise ImageDecodeError(f"Invalid image data: {exc}") from exc