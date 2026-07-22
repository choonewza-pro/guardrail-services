<!-- headroom:rtk-instructions -->
# RTK (Rust Token Killer) - Token-Optimized Commands

When running shell commands, **always prefix with `rtk`**. This reduces context
usage by 60-90% with zero behavior change. If rtk has no filter for a command,
it passes through unchanged — so it is always safe to use.

## Key Commands
```bash
# Git (59-80% savings)
rtk git status          rtk git diff            rtk git log

# Files & Search (60-75% savings)
rtk ls <path>           rtk read <file>         rtk grep <pattern>
rtk find <pattern>      rtk diff <file>

# Test (90-99% savings) — shows failures only
rtk pytest tests/       rtk cargo test          rtk test <cmd>

# Build & Lint (80-90% savings) — shows errors only
rtk tsc                 rtk lint                rtk cargo build
rtk prettier --check    rtk mypy                rtk ruff check

# Analysis (70-90% savings)
rtk err <cmd>           rtk log <file>          rtk json <file>
rtk summary <cmd>       rtk deps                rtk env

# GitHub (26-87% savings)
rtk gh pr view <n>      rtk gh run list         rtk gh issue list

# Infrastructure (85% savings)
rtk docker ps           rtk kubectl get         rtk docker logs <c>

# Package managers (70-90% savings)
rtk pip list            rtk pnpm install        rtk npm run <script>
```

## Rules
- In command chains, prefix each segment: `rtk git add . && rtk git commit -m "msg"`
- For debugging, use raw command without rtk prefix
- `rtk proxy <cmd>` runs command without filtering but tracks usage
<!-- /headroom:rtk-instructions -->

# guardrail-services

Monorepo of two sibling FastAPI guardrail microservices (no shared code, no workspace manifest). Each service is a fully self-contained Python project with its own `requirements.txt`, `.env`, and Docker images.

- `nsfw-detector-service/` — NSFW **image** detection (`Marqo/nsfw-image-detection-384`). Python 3.11 base image.
- `toxic-sentinel-service/` — Toxic **Thai text** detection (WangchanBERTa via `AutoModelForSequenceClassification`). Python 3.12 base image.

## Running

Root compose files build and run **both** services at once:

```bash
docker network create km4u-network              # one-time, external network required
docker compose -f docker-compose-gpu.yml up -d --build    # GPU host
docker compose -f docker-compose-cpu.yml up -d --build    # CPU-only host
```

Each service still has its own `docker-compose-cpu.yml` / `docker-compose-gpu.yml` for running it in isolation — prefer those when working on a single service.

## Ports (configured in each service's `.env`, default values)

| Service | Default port |
|---|---|
| nsfw-detector | `8085` |
| toxic-sentinel | `8086` |

Both `.env` files ship with `PORT=8085` by default — changing one service's `.env` does **not** update the other, and both can't bind `8085` at the same time (root compose passes its own port via `NSFW_PORT`/`TOXIC_PORT` env overrides).

## Architecture (identical pattern in both services)

- `app/main.py` — FastAPI app + `lifespan` that loads the model once at startup (singleton). **Refuses to boot if `API_KEY` is empty.**
- `app/core/config.py` — `pydantic-settings` reads `.env`. `API_KEY` has a validator that hard-fails on empty values.
- `app/services/model_manager.py` — Singleton wrapping HuggingFace model. Resolves `DEVICE` (cuda→cpu auto-fallback when `torch.cuda.is_available()` is False). Predictions run under a thread lock.
- `app/api/deps.py` — `x-api-key` header guard → 401 with the envelope `{"success": false, "error": "Unauthorized: Invalid or missing API key"}`.
- `app/api/v1/endpoints/detect.py` — the protected detection endpoint.
- `app/api/v1/endpoints/health.py` — public `GET /health`.
- All errors are wrapped via `main.py` exception handlers into `{"success": false, "error": ...}` envelopes.

## Conventions to preserve when editing

- **Response envelope**: success bodies are `{"success": true, "data": {...}}`; error bodies are `{"success": false, "error": "..."}`. Don't introduce a different shape.
- **Model loading is a singleton** via `lifespan` — never instantiate `ModelManager` or call `.load()` inside a request handler; use the `model_manager` module-level instance.
- **Docker images are multi-stage**: `deps` → `model-preload` (runs `scripts/preload_model.py`, baked into image) → `final` with `TRANSFORMERS_OFFLINE=1`. Runtime does **not** download weights. Changing `MODEL_NAME` requires a rebuild.
- **`torch` is installed in the Dockerfile**, not in `requirements.txt` (different index URLs for CPU vs CUDA). Don't move it into `requirements.txt`.
- **Python version differs per service** — match the base image's version when editing pins in `Dockerfile.*` or `requirements.txt`.
- **`HF_HOME=/app/.cache`** is baked into the Dockerfile env; compose mounts it as a volume in `toxic-sentinel-service`. Don't write cache inside the app code.
- **Tracking `.serena/` is suppressed**: root `.gitignore` already excludes it. Don't add `serena` artifacts to commits.

## Local dev (no Docker)

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --port <PORT> --reload
```

Run from inside the relevant service directory (`nsfw-detector-service/` or `toxic-sentinel-service/`), and copy `.env.example` → `.env` with a real `API_KEY` first.

## Gotchas

- No tests exist in either service — verification is via `curl http://localhost:<PORT>/health` and the protected detection endpoints.
- `toxic-sentinel-service` default `MODEL_NAME=pythainlp/wangchanberta-base-att-spm-uncased` is the **base MLM head**, not a toxic classifier. For real toxic detection, swap in a fine-tuned WangchanBERTa toxic classification model in `.env`. The `model_manager` auto-detects the "toxic" label by substring match on `id2label`, falling back to argmax if no label matches.
- Don't trust the two services' `.env` defaults to stay in sync — they were authored independently.