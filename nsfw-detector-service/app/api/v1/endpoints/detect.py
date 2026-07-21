import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.deps import verify_api_key
from app.core.exceptions import ImageDecodeError, ModelNotReadyError
from app.core.logging import get_logger
from app.schemas.detect import DetectData, DetectResponse
from app.services.image_processor import load_pil_image, preprocess, read_and_validate
from app.services.model_manager import model_manager

logger = get_logger("nsfw-detector.api.detect")

router = APIRouter(tags=["detect"])


@router.post(
    "/detect-nsfw",
    response_model=DetectResponse,
    dependencies=[Depends(verify_api_key)],
)
async def detect_nsfw(image: UploadFile = File(...)) -> DetectResponse:
    file_name = image.filename or "unknown"
    logger.info("detect-nsfw request: filename=%s", file_name)

    data = await image.read()
    if not data:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": "No image file provided"},
        )

    try:
        mime = await read_and_validate(data)
    except ImageDecodeError as exc:
        msg = str(exc)
        if "too large" in msg.lower():
            status = 413
        elif "unsupported" in msg.lower():
            status = 415
        else:
            status = 422
        raise HTTPException(
            status_code=status,
            detail={"success": False, "error": msg},
        ) from exc

    logger.debug("filename=%s validated mime=%s", file_name, mime)

    try:
        raw_img = await run_in_threadpool(load_pil_image, data)
        pil_img = await run_in_threadpool(preprocess, raw_img)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail={"success": False, "error": str(exc)},
        ) from exc

    try:
        result = await run_in_threadpool(model_manager.predict, pil_img)
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