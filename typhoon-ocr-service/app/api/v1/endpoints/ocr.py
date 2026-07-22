import json
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import verify_api_key
from app.core.config import get_settings
from app.core.exceptions import InvalidImageError, OCRProcessingError, OCRTimeoutError
from app.core.logging import get_logger
from app.schemas.ocr import ImageSizeInfo, OCRResponseData, OCRSuccessResponse, TokenUsage
from app.services.image_processor import process_base64_image, process_uploaded_image
from app.services.model_manager import model_manager

router = APIRouter()
logger = get_logger("typhoon-ocr.endpoints.ocr")


@router.post(
    "/ocr",
    response_model=OCRSuccessResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
    summary="Process document image with Typhoon OCR 1.5 (Supports Form-Data and Raw JSON)",
    description=(
        "Protected OCR detection endpoint accepting image via `multipart/form-data` upload "
        "OR `application/json` payload (Base64 encoded string). Supports SSE streaming and automatic cleanup."
    ),
)
async def process_ocr(
    request: Request,
    file: Optional[UploadFile] = File(default=None, description="Document image file for multipart/form-data upload"),
    question: Optional[str] = Form(default=None, description="Question or OCR instructions (Optional)"),
    system_prompt: Optional[str] = Form(default="", description="System prompt instructions (Optional)"),
    temperature: Optional[float] = Form(default=None, description="Sampling temperature 0.0-2.0 (Optional, default: 0.1)"),
    max_tokens: Optional[int] = Form(default=None, description="Max tokens to generate (Optional, default: 4096)"),
    seed: Optional[int] = Form(default=None, description="Random seed for deterministic output (Optional)"),
    max_retries: Optional[int] = Form(default=None, description="Maximum retry attempts on failure (Optional, default: 3)"),
    timeout: Optional[float] = Form(default=None, description="Inference timeout in seconds (Optional, default: 60.0)"),
    is_stream: Optional[bool] = Form(default=None, description="Enable SSE streaming (Optional, default: False)"),
):
    settings = get_settings()
    content_type = (request.headers.get("content-type") or "").lower()

    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"success": False, "error": f"Model '{settings.MODEL_NAME}' is currently downloading/loading in background. Please try again in a few moments."},
        )

    # Default parameter values
    question_val = question or ""
    sys_prompt_val = system_prompt if (system_prompt and system_prompt.strip()) else settings.DEFAULT_SYSTEM_PROMPT
    temp_val = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
    max_tokens_val = max_tokens if max_tokens is not None else settings.DEFAULT_MAX_TOKENS
    seed_val = seed
    retries_val = max_retries if max_retries is not None else settings.DEFAULT_MAX_RETRIES
    timeout_val = timeout if timeout is not None else settings.DEFAULT_TIMEOUT
    stream_val = is_stream if is_stream is not None else False

    img = None
    try:
        # Check if request is Raw JSON (application/json)
        if content_type.startswith("application/json") or file is None:
            try:
                json_data = await request.json()
            except Exception as exc:
                if file is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"success": False, "error": "Missing image file or invalid JSON payload."},
                    )
                json_data = {}

            if json_data:
                b64_image = json_data.get("file") or json_data.get("image_base64")
                if not b64_image:
                    raise InvalidImageError("JSON payload must contain 'file' or 'image_base64' string.")

                question_val = json_data.get("question") or question_val
                sys_prompt_val = json_data.get("system_prompt") if (json_data.get("system_prompt") and str(json_data.get("system_prompt")).strip()) else sys_prompt_val
                if "temperature" in json_data and json_data["temperature"] is not None:
                    temp_val = float(json_data["temperature"])
                if "max_tokens" in json_data and json_data["max_tokens"] is not None:
                    max_tokens_val = int(json_data["max_tokens"])
                if "seed" in json_data:
                    seed_val = json_data["seed"]
                if "max_retries" in json_data and json_data["max_retries"] is not None:
                    retries_val = int(json_data["max_retries"])
                if "timeout" in json_data and json_data["timeout"] is not None:
                    timeout_val = float(json_data["timeout"])
                if "is_stream" in json_data and json_data["is_stream"] is not None:
                    stream_val = bool(json_data["is_stream"])

                img, original_size, processed_size = process_base64_image(b64_image, max_size=settings.MAX_IMAGE_SIZE)
            else:
                img, original_size, processed_size = await process_uploaded_image(file, max_size=settings.MAX_IMAGE_SIZE)
        else:
            # Handle multipart/form-data upload
            img, original_size, processed_size = await process_uploaded_image(file, max_size=settings.MAX_IMAGE_SIZE)

        # ---------------- Non-Streaming Response ----------------
        if not stream_val:
            result = await model_manager.predict_with_retry(
                image=img,
                prompt=question_val,
                system_prompt=sys_prompt_val,
                temperature=temp_val,
                max_tokens=max_tokens_val,
                seed=seed_val,
                max_retries=retries_val,
                timeout=timeout_val,
            )

            response_data = OCRResponseData(
                text=result["text"],
                model_used=result["model_used"],
                processing_time_ms=result["processing_time_ms"],
                device_used=result["device_used"],
                image_size=ImageSizeInfo(original=original_size, processed=processed_size),
                usage=TokenUsage(**result["usage"]),
            )
            return OCRSuccessResponse(success=True, data=response_data)

        # ---------------- SSE Streaming Response ----------------
        else:
            async def sse_event_generator():
                start_t = time.perf_counter()
                try:
                    async for chunk in model_manager.predict_stream(
                        image=img,
                        prompt=question_val,
                        system_prompt=sys_prompt_val,
                        temperature=temp_val,
                        max_tokens=max_tokens_val,
                        seed=seed_val,
                    ):
                        data_payload = json.dumps({"chunk": chunk, "is_final": False}, ensure_ascii=False)
                        yield f"data: {data_payload}\n\n"

                    elapsed_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
                    final_payload = json.dumps(
                        {
                            "chunk": "",
                            "is_final": True,
                            "model_used": model_manager.model_name,
                            "processing_time_ms": elapsed_ms,
                            "device_used": model_manager.device,
                        },
                        ensure_ascii=False,
                    )
                    yield f"data: {final_payload}\n\n"
                except Exception as exc:
                    logger.error("Error during SSE stream: %s", exc)
                    err_payload = json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)
                    yield f"data: {err_payload}\n\n"
                finally:
                    if img:
                        try:
                            img.close()
                        except Exception:
                            pass

            return StreamingResponse(
                sse_event_generator(),
                media_type="text/event-stream",
            )

    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"success": False, "error": f"Invalid image format or data: {exc}"},
        )
    except OCRTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"success": False, "error": str(exc)},
        )
    except OCRProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "error": str(exc)},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /api/v1/ocr: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "error": f"Internal Server Error: {exc}"},
        )
    finally:
        if not stream_val and img:
            try:
                img.close()
            except Exception:
                pass
            del img
