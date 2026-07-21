from pydantic import BaseModel


class PredictionItem(BaseModel):
    label: str
    score: float


class DetectData(BaseModel):
    is_nsfw: bool
    predictions: list[PredictionItem]
    processing_time_ms: int
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