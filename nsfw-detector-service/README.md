# nsfw-detector-service

NSFW Image Detection API Server ใช้โมเดล `Marqo/nsfw-image-detection-384` ผ่าน HuggingFace Transformers รองรับ GPU (CUDA) และ CPU-only server

---

## คุณสมบัติ

- **FastAPI** Asynchronous API
- **GPU/CUDA** พร้อม fallback CPU อัตโนมัติเมื่อ CUDA ไม่พร้อมใช้งาน
- **CPU-only support** — สร้าง image เล็ก (~1.5GB) สำหรับ server ไม่มี GPU
- **API Key Authentication** ผ่าน Header `x-api-key`
- **Image validation** ตรวจ MIME type จาก magic bytes + จำกัดขนาดไฟล์ 10 MB
- **Singleton model** โหลดครั้งเดียวตอน startup ผ่าน FastAPI lifespan
- **Pre-baked model weights** — ดาวน์โหลดตอน build, container start < 5 วินาที
- **Multi-stage Dockerfile** — self-contained, คำสั่งเดียวจบ

---

## โครงสร้างโปรเจกต์

```
nsfw-detector-service/
├── .env.example              # ตัวอย่าง configuration
├── requirements.txt          # Python dependencies
├── Dockerfile.gpu            # GPU image (CUDA 12.1 + Python 3.11)
├── Dockerfile.cpu             # CPU image (python:3.11-slim)
├── docker-compose-gpu.yml     # GPU server
├── docker-compose-cpu.yml    # CPU-only server
├── scripts/
│   └── preload_model.py       # ดาวน์โหลด model weights ตอน build
└── app/
    ├── main.py               # FastAPI app, lifespan, exception handlers
    ├── core/                 # config, logging, exceptions
    ├── api/                  # deps, routers, endpoints
    ├── schemas/              # Pydantic response models
    └── services/             # model_manager, image_processor
```

โครงสร้างละเอียดเพิ่มเติม: ดู `implementation_plan.md`

---

## การติดตั้งและรัน (Docker เท่านั้น)

### 1. เตรียม `.env`
```bash
cp .env.example .env
# แก้ค่า API_KEY ให้เป็น secret ที่แข็งแรง
```

### 2. เลือก image ตามเครื่อง server

#### 🖥️ กรณีที่ 1: เซิร์ฟเวอร์มี NVIDIA GPU
```bash
docker compose -f docker-compose-gpu.yml up -d --build
```

#### 💻 กรณีที่ 2: เซิร์ฟเวอร์มีแต่ CPU (ไม่มี GPU)
```bash
docker compose -f docker-compose-cpu.yml up -d --build
```

> Build ครั้งแรก: GPU ~15-20 นาที (โหลด torch CUDA 2.5GB) / CPU ~5-8 นาที (โหลด torch CPU 200MB)
> ครั้งต่อไปแก้ code: ~10 วินาที (Docker layer cache)
> Container start: < 5 วินาที (model weights pre-baked ใน image)

### 3. ตรวจสอบสถานะ
```bash
docker compose -f docker-compose-gpu.yml logs -f nsfw-detector
# หรือ
docker compose -f docker-compose-cpu.yml logs -f nsfw-detector

curl http://localhost:8085/health
```

### เมื่อต้องการอัปเดต
| เปลี่ยน | คำสั่ง | เวลา |
|---|---|---|
| App code (`app/`) | `docker compose -f docker-compose-XXX.yml up -d --build` | ~10s |
| requirements.txt | rebuild (Docker cache ถ้า layer ด้านบนไม่เปลี่ยน) | ~3-5min |
| Model (`MODEL_NAME`) | rebuild (preload ใหม่) | ~1min |

---

## API Reference

### `GET /health` *(Public)*

```bash
curl http://localhost:8085/health
```
```json
{
  "success": true,
  "data": {
    "status": "ready",
    "model_loaded": true,
    "device": "cuda"
  }
}
```

### `POST /api/v1/detect-nsfw` *(Protected)*

```bash
curl -X POST http://localhost:8085/api/v1/detect-nsfw \
  -H "x-api-key: your_secret_api_key_here" \
  -F "image=@/path/to/image.jpg"
```

**Success response (200):**
```json
{
  "success": true,
  "data": {
    "is_nsfw": false,
    "predictions": [
      { "label": "SFW", "score": 0.9237 },
      { "label": "NSFW", "score": 0.0763 }
    ],
    "processing_time_ms": 188,
    "device_used": "cuda"
  }
}
```

**Error responses:**

| Status | สาเหตุ | ตัวอย่าง error |
|---|---|---|
| 401 | API key ไม่ถูกต้อง/ไม่มี | `Unauthorized: Invalid or missing API key` |
| 400 | ไม่มีไฟล์ | `No image file provided` |
| 413 | ไฟล์ใหญ่เกิน 10MB | `File too large` |
| 415 | MIME ไม่รองรับ | `Unsupported file type` |
| 422 | ไฟล์ภาพเสีย | `Invalid image data` |
| 503 | โมเดลยังไม่พร้อม | `Model not ready yet` |
| 500 | Inference ล้มเหลว | `Inference failed` |

---

## Configuration (`.env`)

| ตัวแปร | Default | คำอธิบาย |
|---|---|---|
| `PORT` | `8085` | พอร์ต API |
| `API_KEY` | — | **required** secret key |
| `USE_GPU` | `true` | เปิด/ปิด GPU (compose ตั้งไว้ให้) |
| `DEVICE` | auto | override `cuda`/`cpu` |
| `HF_HOME` | `/app/.cache` | HF cache dir (baked ใน image) |
| `MODEL_NAME` | `Marqo/nsfw-image-detection-384` | HuggingFace model |
| `MAX_IMAGE_SIZE_MB` | `10` | ขนาด upload สูงสุด |
| `ALLOWED_MIME` | `image/jpeg,image/png,image/webp` | MIME types ที่อนุญาต |
| `NSFW_THRESHOLD` | `0.6` | threshold ตัดสิน `is_nsfw` |
| `MAX_DIMENSION` | `1024` | resize ขนาดสูงสุด |
| `MAX_CONCURRENT_INFERENCES` | `1` | จำกัด concurrency |
| `CORS_ORIGINS` | — | comma-separated origins |
| `ENABLE_DOCS` | `true` | `/docs` OpenAPI UI |

---

## ข้อกำหนดเครื่อง host

### GPU server
- Docker + NVIDIA Container Toolkit
- NVIDIA GPU driver รองรับ CUDA 12.1+
- ตรวจสอบ GPU visibility:
```bash
docker compose -f docker-compose-gpu.yml exec nsfw-detector \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only')"
```

### CPU-only server
- Docker ติดตั้งแล้ว
- RAM แนะนำขั้นต่ำ 2GB (model ~380MB + inference)
- ไม่ต้องการ NVIDIA driver/toolkit

---

## Performance โดยประมาณ

| | GPU (CUDA) | CPU-only |
|---|---|---|
| Inference ต่อภาพ | ~150-250ms | ~800-2000ms |
| Image size | ~3.5GB | ~1.5GB |
| Build time (cold) | ~15-20 min | ~5-8 min |
| Container start | < 5s | < 5s |

---

## การพัฒนาต่อ

ดู `implementation_plan.md` สำหรับ design specification ฉบับเต็ม (Generated by GML-5.2 Model)