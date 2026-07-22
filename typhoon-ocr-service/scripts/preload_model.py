"""
Download model weights during Docker build so they're cached in HF_HOME.
Uses huggingface_hub snapshot_download to download files efficiently without high RAM usage.
"""
import os
import sys
from huggingface_hub import snapshot_download

os.environ.setdefault("HF_HOME", "/app/.cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

MODEL_NAME = os.environ.get("MODEL_NAME", "typhoon-ai/typhoon-ocr1.5-2b")

print(f"[preload] Downloading model files for: {MODEL_NAME}", flush=True)
print(f"[preload] HF_HOME={os.environ.get('HF_HOME')}", flush=True)

try:
    path = snapshot_download(repo_id=MODEL_NAME)
    print(f"[preload] OK: Model files cached at {path}", flush=True)
except Exception as exc:
    print(f"[preload] FAILED: {exc}", flush=True)
    sys.exit(1)
