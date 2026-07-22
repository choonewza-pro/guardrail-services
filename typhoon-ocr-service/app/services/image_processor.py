import base64
import io
import os
from typing import Tuple
from fastapi import UploadFile
from PIL import Image, ImageOps

from app.core.exceptions import InvalidImageError
from app.core.logging import get_logger

logger = get_logger("typhoon-ocr.image_processor")

DEFAULT_TYPHOON_MAX_IMAGE_SIZE = 1800


def resize_if_needed(img: Image.Image, max_size: int = DEFAULT_TYPHOON_MAX_IMAGE_SIZE) -> Image.Image:
    """
    Resizes image if either width or height exceeds max_size (default: 1800px).
    Preserves original aspect ratio using Lanczos filter.
    """
    width, height = img.size
    if width > max_size or height > max_size:
        if width >= height:
            scale = max_size / float(width)
            new_size = (max_size, int(height * scale))
        else:
            scale = max_size / float(height)
            new_size = (int(width * scale), max_size)

        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
        logger.info("Resized image from %s to %s", (width, height), resized_img.size)
        return resized_img
    return img


def _convert_and_resize(img: Image.Image, max_size: int) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int]]:
    # Apply EXIF rotation if present (JPEG/WEBP)
    img = ImageOps.exif_transpose(img)
    
    # Handle RGBA/LA/P transparency (e.g. WEBP, PNG with alpha channel) over white background
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        alpha_composite = img.convert("RGBA")
        background.paste(alpha_composite, mask=alpha_composite.split()[3])
        rgb_img = background
    else:
        rgb_img = img.convert("RGB")

    original_size = rgb_img.size
    processed_img = resize_if_needed(rgb_img, max_size=max_size)
    processed_size = processed_img.size
    
    # Return an in-memory copy for garbage collection safety
    return processed_img.copy(), original_size, processed_size


async def process_uploaded_image(
    file: UploadFile,
    max_size: int = DEFAULT_TYPHOON_MAX_IMAGE_SIZE,
) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int]]:
    """
    Reads an UploadFile, converts it to RGB PIL Image, resizes if needed,
    and guarantees file handle cleanup and temporary memory cleanup even on error.
    
    Returns: (processed_image, original_size, processed_size)
    """
    contents = None
    try:
        contents = await file.read()
        if not contents:
            raise InvalidImageError("Uploaded file is empty.")

        image_bytes = io.BytesIO(contents)
        try:
            with Image.open(image_bytes) as img:
                return _convert_and_resize(img, max_size=max_size)
        except Exception as exc:
            if isinstance(exc, InvalidImageError):
                raise
            raise InvalidImageError(f"Failed to decode image file: {exc}") from exc
    finally:
        # Guarantee cleanup of uploaded file handle and byte buffer
        try:
            await file.close()
        except Exception as exc:
            logger.warning("Error closing upload file handle: %s", exc)
        del contents


def process_base64_image(
    base64_str: str,
    max_size: int = DEFAULT_TYPHOON_MAX_IMAGE_SIZE,
) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int]]:
    """
    Decodes a Base64 image string (or Data URI), converts to RGB PIL Image,
    resizes if needed, and returns (processed_image, original_size, processed_size).
    """
    if not base64_str or not base64_str.strip():
        raise InvalidImageError("Provided image base64 string is empty.")

    clean_base64 = base64_str.strip()
    if "," in clean_base64:
        clean_base64 = clean_base64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(clean_base64)
        if not image_bytes:
            raise InvalidImageError("Decoded base64 image bytes are empty.")

        bio = io.BytesIO(image_bytes)
        with Image.open(bio) as img:
            return _convert_and_resize(img, max_size=max_size)
    except Exception as exc:
        if isinstance(exc, InvalidImageError):
            raise
        raise InvalidImageError(f"Failed to decode base64 image: {exc}") from exc


def cleanup_temp_file(filepath: str | None) -> None:
    """Helper to safely remove any temporary file from disk if created."""
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
            logger.info("Cleaned up temp file: %s", filepath)
        except Exception as exc:
            logger.warning("Failed to remove temp file %s: %s", filepath, exc)
