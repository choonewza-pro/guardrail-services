from fastapi import APIRouter

from app.api.v1.endpoints import health, ocr

root_router = APIRouter()
root_router.include_router(health.router)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(ocr.router, tags=["OCR"])
