import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.router import api_router, root_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.model_manager import model_manager


def _envelope_error(status: int, message: str, detail: Optional[Any] = None) -> JSONResponse:
    body: dict[str, Any] = {"success": False, "error": message}
    if detail is not None:
        body["detail"] = detail
    return JSONResponse(status_code=status, content=body)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger = get_logger("typhoon-ocr.main")

    settings = get_settings()
    if not settings.API_KEY:
        logger.error("API_KEY is not set. Refusing to start.")
        raise SystemExit("FATAL: API_KEY must be set in .env or environment")

    os.environ["HF_HOME"] = settings.HF_HOME
    logger.info(
        "Starting typhoon-ocr-service (port=%s, model=%s)",
        settings.PORT,
        settings.MODEL_NAME,
    )

    # Launch model loading in background so server boots instantly
    asyncio.create_task(model_manager.load_in_background())

    yield

    logger.info("Shutting down typhoon-ocr-service")
    model_manager.unload()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="typhoon-ocr-service",
        version="1.0.0",
        description="Typhoon OCR 1.5 Document Parsing API Service",
        lifespan=lifespan,
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None,
        openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
    )

    if settings.cors_origins_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Include Routers
    app.include_router(root_router)
    app.include_router(api_router)

    # ---------- Exception Handlers ----------
    @app.exception_handler(StarletteHTTPException)
    async def _http_exc_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return _envelope_error(exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_exc_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _envelope_error(422, "Validation error", detail=exc.errors())

    @app.exception_handler(Exception)
    async def _unhandled_exc_handler(request: Request, exc: Exception) -> JSONResponse:
        logger = get_logger("typhoon-ocr.main")
        logger.exception("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
        return _envelope_error(500, f"Internal Server Error: {exc}")

    return app


app = create_app()
