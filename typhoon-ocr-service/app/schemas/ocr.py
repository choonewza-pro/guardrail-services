from typing import Any, Optional
from pydantic import BaseModel, Field


class OCRJSONRequest(BaseModel):
    file: Optional[str] = Field(None, description="Base64 encoded image string or Data URI")
    image_base64: Optional[str] = Field(None, description="Alias for base64 encoded image string")
    question: Optional[str] = Field(None, description="Question or OCR instructions for model")
    system_prompt: Optional[str] = Field(None, description="System prompt instructions (Optional, defaults to Thai document Markdown OCR prompt)")
    temperature: Optional[float] = Field(0.1, description="Sampling temperature 0.0-2.0")
    max_tokens: Optional[int] = Field(4096, description="Max tokens to generate")
    seed: Optional[int] = Field(None, description="Random seed for deterministic output")
    max_retries: Optional[int] = Field(3, description="Maximum retry attempts on failure")
    timeout: Optional[float] = Field(60.0, description="Inference timeout in seconds")
    is_stream: bool = Field(False, description="Enable Server-Sent Events (SSE) streaming")


class ImageSizeInfo(BaseModel):
    original: tuple[int, int]
    processed: tuple[int, int]


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OCRResponseData(BaseModel):
    text: str
    model_used: str
    processing_time_ms: float
    device_used: str
    image_size: Optional[ImageSizeInfo] = None
    usage: Optional[TokenUsage] = None


class OCRSuccessResponse(BaseModel):
    success: bool = True
    data: OCRResponseData


class OCRErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[Any] = None
