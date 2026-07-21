from fastapi import APIRouter

from app.schemas.detect import HealthData, HealthResponse
from app.services.model_manager import model_manager

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        data=HealthData(
            status="ready" if model_manager.ready else "not_ready",
            model_loaded=model_manager.ready,
            device=model_manager.device,
        )
    )