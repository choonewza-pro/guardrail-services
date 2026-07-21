from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    settings = get_settings()
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"success": False, "error": "Unauthorized: Invalid or missing API key"},
        )
    return api_key