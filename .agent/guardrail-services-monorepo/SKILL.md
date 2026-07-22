---
name: guardrail-services-monorepo
description: Monorepo architectural context, service structure, and execution guidelines for AI Content Moderation Microservices (NSFW detection, toxic text, etc.).
---

# Project Context: Guardrail Services (Monorepo)

## Architecture Overview
This project is a **Monorepo** containing independent, containerized AI/ML microservices for automated Content Moderation and Media Inspection.

- **Pattern:** Monorepo with decoupled Microservices.
- **Service Isolation:** Each `*-service/` directory is self-contained — its own Python env, Dockerfiles (CPU/GPU variants), `requirements.txt`, `.env`, and API. Services do **not** import code from each other.
- **Inter-service Communication:** REST APIs only. No shared runtime code. Orchestration via `docker-compose` + the external `km4u-network` bridge network.
- **Shared Network:** All services join the pre-existing external Docker network `km4u-network` (`name: "km4u-network"`, `driver: bridge`, `external: true`) for inter-service discovery.

---

## Repository Structure

```text
guardrail-services/
├── nsfw-detector-service/      # [Service 1] Image NSFW detection API (port 8085)
│   ├── app/                    # FastAPI application
│   │   ├── main.py             # App factory, lifespan, exception handlers
│   │   ├── core/               # config (pydantic-settings), logging, exceptions
│   │   ├── api/                # deps (API key auth), v1/ (router + endpoints)
│   │   ├── schemas/            # Pydantic response models (common, detect)
│   │   └── services/           # model_manager (singleton), image_processor (PIL)
│   ├── scripts/preload_model.py   # Pre-bake HF weights at build time
│   ├── Dockerfile.cpu          # CPU image (python:3.11-slim, ~1.5GB)
│   ├── Dockerfile.gpu          # GPU image (nvidia/cuda:12.1.1, ~3.5GB)
│   ├── docker-compose-cpu.yml  # CPU-only orchestration
│   ├── docker-compose-gpu.yml  # GPU (CUDA) orchestration
│   ├── requirements.txt        # Non-torch deps (torch pinned in Dockerfile)
│   ├── .env.example            # Configuration template
│   ├── README.md               # Service docs + API reference
│   ├── implementation_plan.md  # Full design blueprint
│   └── prompt-python.md        # Original generation prompt
├── toxic-sentinel-service/     # [Service 2] Thai/English toxic text classification (scaffold)
├── AGENTS.md                  # RTK instructions (shared)
└── .agent/
    └── guardrail-services-monorepo/
        └── SKILL.md            # This file
```

> Services are added as new top-level `*-service/` directories. Each must follow the isolation rules above.

---

## Service Blueprint (apply to all services)

Every service in this monorepo follows the same internal layout and conventions. When adding or editing a service, mirror the `nsfw-detector-service` structure.

### Per-Service Directory Layout
```text
<service-name>-service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # create_app() + lifespan + exception handlers
│   ├── core/
│   │   ├── config.py           # Settings(BaseSettings) via pydantic-settings
│   │   ├── logging.py          # setup_logging(), get_logger()
│   │   └── exceptions.py       # custom exception classes
│   ├── api/
│   │   ├── deps.py             # verify_api_key dependency (x-api-key header)
│   │   └── v1/
│   │       ├── router.py       # api_router (prefix /api/v1) + root_router
│   │       └── endpoints/      # one file per endpoint
│   ├── schemas/                # Pydantic models: Envelope + endpoint DTOs
│   └── services/              # model_manager (singleton) + preprocessing
├── scripts/preload_model.py    # Bake model weights into the Docker image
├── Dockerfile.cpu              # CPU build (multi-stage)
├── Dockerfile.gpu              # GPU build (multi-stage, CUDA 12.1)
├── docker-compose-cpu.yml      # CPU stack (external km4u-network)
├── docker-compose-gpu.yml      # GPU stack (external km4u-network)
├── requirements.txt            # non-torch deps; torch installed separately in Dockerfile
├── .env.example
├── .dockerignore
└── README.md
```

---

## Conventions (must follow)

### Configuration (`pydantic-settings`)
- `Settings(BaseSettings)` reads `.env`; `model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")`.
- `get_settings()` is an `@lru_cache` singleton.
- **Always required:** `API_KEY` (non-empty, validated at startup — service refuses to start without it).
- Comma-separated list fields expose `*_list` properties (e.g. `allowed_mime_list`, `cors_origins_list`), never store raw lists in env.
- `HF_HOME=/app/.cache` baked into images; `os.environ["HF_HOME"]` is set in `main.py` lifespan before importing transformers.

### API & Response Envelope
- **Auth:** `x-api-key` header via `APIKeyHeader(name="x-api-key", auto_error=False)`. `verify_api_key` raises `401` with `{"success": false, "error": "Unauthorized: Invalid or missing API key"}`.
- **Standard envelope**, always:
  ```json
  // success
  { "success": true, "data": { ... } }
  // error
  { "success": false, "error": "message", "detail": "optional" }
  ```
- **`/health`**(public, no auth): returns `200` with `model_loaded` flag (never `503` while starting).
- **Protected endpoints** under `/api/v1/...` require `Depends(verify_api_key)`.
- Global exception handlers in `main.py`: `StarletteHTTPException`, `RequestValidationError` (422), and catch-all `Exception` (500, logged).

### Model Management
- **Singleton** pattern: model loaded once in FastAPI `lifespan` (`model_manager.load()`), unloaded on shutdown (`model_manager.unload()` frees CUDA cache).
- **Device resolution:** `settings.DEVICE` ∈ {`auto`|`cuda`|`cpu`} (default `auto` = `cuda` if `torch.cuda.is_available()` else `cpu`; `cuda` auto-falls back to `cpu` when unavailable with a warning log). Never assume GPU availability.
- **Inference** runs in a threadpool (`await run_in_threadpool(model_manager.predict, ...)`) to avoid blocking the async event loop.
- A `threading.Lock` serializes model access when `MAX_CONCURRENT_INFERENCES=1`.
- Weights are **pre-baked** at Docker build time via `scripts/preload_model.py` → runtime uses `TRANSFORMERS_OFFLINE=1`; container start < 5s.

### Docker
- **Multi-stage** Dockerfiles: `deps` → `model-preload` → `final`. App code is copied last for max cache hits.
- **cpu/gpu split:** `Dockerfile.cpu` uses `python:3.11-slim` + torch CPU wheels (~200MB); `Dockerfile.gpu` uses `nvidia/cuda:12.1.1-runtime-ubuntu22.04` + torch `cu121` wheels (~2.5GB).
- **torch/torchvision are pinned in the Dockerfile** (`torch==2.3.1`, `torchvision==0.18.1`), NOT in `requirements.txt`.
- Compose files: read `PORT` from `.env` (default `8085`), attach to `external: true` network `km4u-network`, and add `extra_hosts: host.docker.internal:host-gateway`.
- GPU compose uses `deploy.resources.reservations.devices` (driver: nvidia, count: all, capabilities: [gpu]).
- Build command (pick by host):
  ```bash
  docker compose -f docker-compose-gpu.yml up -d --build   # GPU host
  docker compose -f docker-compose-cpu.yml up -d --build    # CPU-only host
  ```

### Python Style
- `from __future__ import annotations` at the top of modules using modern typing (e.g. `str | None`).
- Logger namespaced: `get_logger("<service>.<area>")` (e.g. `"nsfw-detector.services.model_manager"`).
- No comments in code unless explicitly requested.
- Use `noqa: BLE001` on intentional broad excepts.

---

## Service: nsfw-detector-service

| Item | Value |
|---|---|
| Purpose | Image NSFW detection |
| Port | `8085` (env `PORT`) |
| Model | `Marqo/nsfw-image-detection-384` (HuggingFace) |
| Auth | `x-api-key` header |
| Endpoints | `GET /health` (public), `POST /api/v1/detect-nsfw` (protected, multipart `image`) |
| Image limits | ≤ 10MB; MIME `image/jpeg`, `image/png`, `image/webp`; magic-byte sniffed (not Content-Type) |
| Preprocess | EXIF orientation → RGB → resize if max dim > 1024 (LANCZOS) |
| Threshold | `NSFW_THRESHOLD=0.6` → `is_nsfw` |
| Concurrency | `MAX_CONCURRENT_INFERENCES=1` |
| Docs | `/docs`, `/redoc` toggle via `ENABLE_DOCS` |

Response (`POST /api/v1/detect-nsfw`):
```json
{
  "success": true,
  "data": {
    "is_nsfw": false,
    "predictions": [ { "label": "SFW", "score": 0.9237 }, { "label": "NSFW", "score": 0.0763 } ],
    "processing_time_ms": 188,
    "device_used": "cuda"
  }
}
```

Error codes: `401` (bad key) · `400` (no file) · `413` (>10MB) · `415` (bad MIME) · `422` (bad image) · `503` (model not ready) · `500` (inference fail).

---

## Service: toxic-sentinel-service

Currently an empty scaffold (planned: Thai/English toxic text classification API). When implementing, follow the Service Blueprint above and the `nsfw-detector-service` as the reference template — own `app/`, `Dockerfile.cpu`/`.gpu`, compose files, `requirements.txt`, `.env.example`, and `README.md`.

---

## Working on this repo (agent guidance)

1. **Always work within a single service directory** unless wiring inter-service orchestration. Do not create cross-service imports.
2. **Mirror existing conventions** — when adding endpoints, copy the pattern from `nsfw-detector-service/app/api/v1/endpoints/` (router with `tags`, `response_model`, `dependencies=[Depends(verify_api_key)]`).
3. **Never hardcode ports/keys** — read everything from `Settings` via `get_settings()`.
4. **Keep the envelope consistent** — every response, success or error, uses `{success, data|error}`.
5. **Run via Docker** — these services are designed to run containerized; local non-Docker runs require uncommenting torch in `requirements.txt`.
6. **Prefer `rtk` prefixed commands** for shell ops (see `AGENTS.md`) to save context; use raw commands only when debugging.
7. **Do not add comments** to code unless explicitly asked.
8. When adding a new service, choose a unique port and container name, and attach to the shared `km4u-network`.