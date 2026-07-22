from fastapi import APIRouter
from app.core.config import get_settings
from app.services.model_manager import model_manager

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check():
    """
    Public health check endpoint. Returns model status, loading state, and device info.
    """
    return {
        "status": "ok",
        "model_ready": model_manager.is_loaded,
        "model_loading": model_manager.is_loading,
        "model_name": model_manager.model_name or get_settings().MODEL_NAME,
        "device_used": model_manager.device,
    }
