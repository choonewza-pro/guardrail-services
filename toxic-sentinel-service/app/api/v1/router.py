from fastapi import APIRouter

from app.api.v1.endpoints import detect, health

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(detect.router)

root_router = APIRouter()
root_router.include_router(health.router)