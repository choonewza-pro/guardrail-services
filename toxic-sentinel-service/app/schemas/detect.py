from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Thai text to analyze")
    threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Override default TOXIC_THRESHOLD"
    )


class DetectData(BaseModel):
    is_toxic: bool
    score: float
    threshold_used: float
    label: str
    processing_time_ms: float
    device_used: str


class DetectResponse(BaseModel):
    success: bool = True
    data: DetectData


class HealthData(BaseModel):
    status: str
    model_loaded: bool
    device: str


class HealthResponse(BaseModel):
    success: bool = True
    data: HealthData