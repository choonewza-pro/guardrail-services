from typing import Any, Optional

from pydantic import BaseModel


class EnvelopeResponse(BaseModel):
    success: bool
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[Any] = None