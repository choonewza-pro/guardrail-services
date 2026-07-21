from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/app/.cache")

import threading
import time
from typing import Any

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.core.config import get_settings
from app.core.exceptions import ModelLoadError, ModelNotReadyError
from app.core.logging import get_logger

logger = get_logger("nsfw-detector.services.model_manager")


class ModelManager:
    """Singleton wrapper that loads and runs the NSFW detection model.

    Loaded once during FastAPI lifespan. Predictions are dispatched through a
    thread lock to protect shared model state when concurrency is undesirable.
    """

    def __init__(self) -> None:
        self.image_processor = None
        self.model = None
        self.device: str = "cpu"
        self._ready: bool = False
        self._lock = threading.Lock()
        self._id2label: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------
    def _resolve_device(self) -> str:
        settings = get_settings()
        if settings.DEVICE:
            desired = settings.DEVICE.lower()
        elif settings.USE_GPU:
            desired = "cuda"
        else:
            desired = "cpu"

        if desired == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available -> falling back to CPU"
            )
            return "cpu"
        return desired

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        settings = get_settings()
        start = time.perf_counter()
        self._ready = False
        try:
            self.device = self._resolve_device()
            logger.info(
                "Loading model %s on device=%s", settings.MODEL_NAME, self.device
            )
            self.image_processor = AutoImageProcessor.from_pretrained(
                settings.MODEL_NAME
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                settings.MODEL_NAME
            ).to(self.device)
            self.model.eval()
            self._id2label = dict(self.model.config.id2label)
            self._ready = True
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "Model loaded successfully in %dms (labels=%s)",
                elapsed_ms,
                self._id2label,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Failed to load model: {exc}") from exc

    def unload(self) -> None:
        self.model = None
        self.image_processor = None
        self._ready = False
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        logger.info("Model unloaded")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._ready

    def predict(self, image: Any) -> dict[str, Any]:
        if not self._ready or self.model is None or self.image_processor is None:
            raise ModelNotReadyError("Model is not loaded yet")

        settings = get_settings()
        with self._lock:
            start = time.perf_counter()
            inputs = self.image_processor(images=image, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            predictions: list[dict[str, Any]] = []
            for idx in range(probs.shape[0]):
                label = self._id2label.get(idx, str(idx))
                predictions.append(
                    {"label": label, "score": round(float(probs[idx].item()), 4)}
                )
            predictions.sort(key=lambda p: p["score"], reverse=True)

            nsfw_score = 0.0
            for p in predictions:
                if p["label"].lower() == "nsfw":
                    nsfw_score = p["score"]
                    break

            return {
                "is_nsfw": bool(nsfw_score >= settings.NSFW_THRESHOLD),
                "predictions": predictions,
                "processing_time_ms": elapsed_ms,
                "device_used": self.device,
            }


model_manager = ModelManager()