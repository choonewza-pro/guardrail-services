from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/app/.cache")

import threading
import time
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import get_settings
from app.core.exceptions import ModelLoadError, ModelNotReadyError
from app.core.logging import get_logger

logger = get_logger("toxic-sentinel.services.model_manager")


# Candidate label substrings that identify the "toxic" class for a wide range
# of fine-tuned WangchanBERTa toxic / sentiment classifiers. The first label in
# `id2label` whose lower-cased name matches one of these tokens is treated as
# the toxic class. If none match we fall back to the highest-probability label.
_TOXIC_LABEL_HINTS = (
    "toxic",
    "toxicity",
    "tox",
    "negative",
    "hate",
    "offensive",
    "abusive",
    "bad",
)


class ModelManager:
    """Singleton wrapper that loads and runs the toxic-text classification model.

    Loaded once during FastAPI lifespan. Predictions are dispatched through a
    thread lock to protect shared model state when concurrency is undesirable.
    """

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.device: str = "cpu"
        self._ready: bool = False
        self._lock = threading.Lock()
        self._id2label: dict[int, str] = {}
        self._toxic_label_idx: int | None = None

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------
    def _resolve_device(self) -> str:
        settings = get_settings()
        desired = settings.DEVICE  # already normalized: "auto"|"cuda"|"cpu"

        if desired == "auto":
            desired = "cuda" if torch.cuda.is_available() else "cpu"

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
            self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.MODEL_NAME
            ).to(self.device)
            self.model.eval()
            self._id2label = dict(self.model.config.id2label)
            self._toxic_label_idx = self._find_toxic_label_idx()
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "Model loaded successfully in %dms (labels=%s, toxic_idx=%s)",
                elapsed_ms,
                self._id2label,
                self._toxic_label_idx,
            )
            self._ready = True
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Failed to load model: {exc}") from exc

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self._ready = False
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        logger.info("Model unloaded")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_toxic_label_idx(self) -> int | None:
        for idx, label in self._id2label.items():
            name = str(label).lower()
            if any(hint in name for hint in _TOXIC_LABEL_HINTS):
                return idx
        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._ready

    def predict(self, text: str, threshold: float | None = None) -> dict[str, Any]:
        if not self._ready or self.model is None or self.tokenizer is None:
            raise ModelNotReadyError("Model is not loaded yet")

        settings = get_settings()
        effective_threshold = (
            settings.TOXIC_THRESHOLD if threshold is None else float(threshold)
        )

        with self._lock:
            start = time.perf_counter()
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            elapsed_ms = float((time.perf_counter() - start) * 1000)

            # Pick toxic score/label
            if self._toxic_label_idx is not None:
                toxic_idx = self._toxic_label_idx
            else:
                # Fall back to argmax (treat the top label as the "toxic" one)
                toxic_idx = int(torch.argmax(probs).item())

            toxic_score = float(probs[toxic_idx].item())
            label = str(self._id2label.get(toxic_idx, "toxic"))

            return {
                "is_toxic": bool(toxic_score >= effective_threshold),
                "score": round(toxic_score, 4),
                "threshold_used": round(effective_threshold, 4),
                "label": label,
                "processing_time_ms": round(elapsed_ms, 2),
                "device_used": self.device,
            }


model_manager = ModelManager()