from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.api.deps import verify_api_key
from app.core.exceptions import ModelNotReadyError, TextValidationError
from app.core.logging import get_logger
from app.schemas.detect import DetectData, DetectRequest, DetectResponse
from app.services.model_manager import model_manager
from app.services.text_processor import validate_text

logger = get_logger("toxic-sentinel.api.detect")

router = APIRouter(tags=["detect"])


@router.post(
    "/detect-toxic",
    response_model=DetectResponse,
    dependencies=[Depends(verify_api_key)],
)
async def detect_toxic(payload: DetectRequest) -> DetectResponse:
    logger.info("detect-toxic request: length=%d", len(payload.text))

    try:
        validate_text(payload.text)
    except TextValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"success": False, "error": str(exc)},
        ) from exc

    try:
        result = await run_in_threadpool(
            model_manager.predict, payload.text, payload.threshold
        )
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=503,
            detail={"success": False, "error": "Model not ready yet"},
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": "Inference failed"},
        ) from exc

    return DetectResponse(data=DetectData(**result))