from __future__ import annotations

from app.core.config import get_settings
from app.core.exceptions import TextValidationError


def validate_text(text: str) -> str:
    """Validate the incoming Thai text payload.

    - Must not be empty / only whitespace.
    - Must not exceed MAX_TEXT_LENGTH codepoints.
    """
    settings = get_settings()
    if text is None:
        raise TextValidationError("text is required")
    stripped = text.strip()
    if not stripped:
        raise TextValidationError("text must not be empty")
    if len(text) > settings.MAX_TEXT_LENGTH:
        raise TextValidationError(
            f"text length {len(text)} exceeds the {settings.MAX_TEXT_LENGTH} "
            f"character limit"
        )
    return text