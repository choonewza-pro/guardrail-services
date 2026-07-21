"""Download model weights during Docker build so they're baked into the image.

Run at build time (CPU). After this, /app/.cache contains the weights and the
runtime container never needs to download again.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HOME", "/app/.cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = os.environ.get("MODEL_NAME", "Marqo/nsfw-image-detection-384")

print(f"[preload] Downloading model: {MODEL_NAME}", flush=True)
print(f"[preload] HF_HOME={os.environ.get('HF_HOME')}", flush=True)

try:
    AutoImageProcessor.from_pretrained(MODEL_NAME)
    AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    print("[preload] OK: model weights cached", flush=True)
except Exception as exc:
    print(f"[preload] FAILED: {exc}", flush=True)
    sys.exit(1)