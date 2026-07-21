# toxic-sentinel-service

Toxic Text Detection API Server ใช้โมเดล WangchanBERTa (ผ่าน HuggingFace `transformers`) รองรับ GPU (CUDA) และ CPU-only server

> ค่าตั้งต้นของ `MODEL_NAME` คือ `pythainlp/wangchanberta-base-att-spm-uncased` (base MLM model) — เพื่อใช้งานจริงเป็น toxic classifier ควรเปลี่ยนเป็น fine-tuned สำหรับ toxic classification ที่นำ WangchanBERTa มาต่อหัว SequenceClassification (เช่น Pace, หรือ datasetThai toxic/sentiment) โดยแก้ที่ `.env` ได้เลย  code ใช้ `AutoModelForSequenceClassification` + `AutoTokenizer` กับโมเดลใด ๆ ได้

---

## คุณสมบัติ

- **FastAPI** Asynchronous API
- **GPU/CUDA** พร้อม fallback CPU อัตโนมัติเมื่อ `torch.cuda.is_available()` เป็น False
- **CPU-only support** — สร้าง image เล็กสำหรับ server ไม่มี GPU
- **API Key Authentication** ผ่าน Header `x-api-key`
- **Text validation** — จำกัดความยาว 1,000 ตัวอักษร และห้ามค่าว่าง
- **Singleton model** โหลดครั้งเดียวตอน startup ผ่าน FastAPI lifespan
- **Pre-baked model weights** — ดาวน์โหลดตอน build, container start < 5 วินาที
- **Multi-stage Dockerfile** — self-contained, คำสั่งเดียวจบ

---

## โครงสร้างโปรเจกต์

```
toxic-sentinel-service/
├── .env.example              # ตัวอย่าง configuration
├── requirements.txt          # Python dependencies
├── Dockerfile.gpu            # GPU image (CUDA 12.1 + Python 3.12)
├── Dockerfile.cpu            # CPU image (python:3.12-slim)
├── docker-compose-gpu.yml     # GPU server
├── docker-compose-cpu.yml    # CPU-only server
├── scripts/
│   └── preload_model.py      # ดาวน์โหลด model weights ตอน build
└── app/
    ├── main.py               # FastAPI app, lifespan, exception handlers
    ├── core/                 # config, logging, exceptions
    ├── api/                  # deps, routers, endpoints
    ├── schemas/              # Pydantic request/response models
    └── services/             # model_manager, text_processor
```

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
docker compose -f docker-compose-gpu.yml logs -f toxic-sentinel
# หรือ
docker compose -f docker-compose-cpu.yml logs -f toxic-sentinel

curl http://localhost:8085/health
```

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

### `POST /api/v1/detect-toxic` *(Protected)*

```bash
curl -X POST http://localhost:8085/api/v1/detect-toxic \
  -H "x-api-key: my_super_secret_key_123" \
  -H "Content-Type: application/json" \
  -d '{"text":"ข้อความที่ต้องการตรวจ"}'
```

**Request body:**

| Field | Type | Required | คำอธิบาย |
|---|---|---|---|
| `text` | string | ✅ | ข้อความภาษาไทยที่ต้องการตรวจ (ไม่เป็นค่าว่าง, ไม่เกิน 1,000 ตัวอักษร) |
| `threshold` | float | ❌ | ค่าความไวในการตัดสินผล 0.0-1.0 (หากไม่ส่งมาใช้ `TOXIC_THRESHOLD` จาก `.env`) |

**Success response (200):**
```json
{
  "success": true,
  "data": {
    "is_toxic": true,
    "score": 0.89,
    "threshold_used": 0.5,
    "label": "toxic",
    "processing_time_ms": 15.4,
    "device_used": "cpu"
  }
}
```

**Error responses:**

| Status | สาเหตุ | ตัวอย่าง error |
|---|---|---|
| 401 | API key ไม่ถูกต้อง/ไม่มี | `Unauthorized: Invalid or missing API key` |
| 422 | `text` ว่าง/ยาวเกิน/ผิดรูปแบบ | `text must not be empty` / `text length 1234 exceeds ...` |
| 503 | โมเดลยังไม่พร้อม | `Model not ready yet` |
| 500 | Inference ล้มเหลว | `Inference failed` |

---

## Configuration (`.env`)

| ตัวแปร | Default | คำอธิบาย |
|---|---|---|
| `PORT` | `8085` | พอร์ต API |
| `API_KEY` | — | **required** secret key |
| `DEVICE` | auto | override `cuda`/`cpu` (default: `cuda` ถ้ามี GPU, อย่างอื่น fallback CPU) |
| `HF_HOME` | `/app/.cache` | HF cache dir (baked ใน image + mount เป็น volume) |
| `MODEL_NAME` | `pythainlp/wangchanberta-base-att-spm-uncased` | HuggingFace model |
| `TOXIC_THRESHOLD` | `0.5` | threshold ตัดสิน `is_toxic` |
| `MAX_TEXT_LENGTH` | `1000` | ความยาวข้อความสูงสุด |
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
docker compose -f docker-compose-gpu.yml exec toxic-sentinel \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only')"
```

### CPU-only server
- Docker ติดตั้งแล้ว
- RAM แนะนำขั้นต่ำ 2GB
- ไม่ต้องการ NVIDIA driver/toolkit

---

## การพัฒนาต่อในเครื่อง (ไม่ใช้ Docker)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8085 --reload
```

> ⚠️ WangchanBERTa tokenizer ต้องการ `sentencepiece` (อยู่ใน `requirements.txt` แล้ว)

---

## การอัปเดต

| เปลี่ยน | คำสั่ง | เวลา |
|---|---|---|
| App code (`app/`) | `docker compose -f docker-compose-XXX.yml up -d --build` | ~10s |
| requirements.txt | rebuild (Docker cache ถ้า layer ด้านบนไม่เปลี่ยน) | ~3-5min |
| Model (`MODEL_NAME`) | rebuild (preload ใหม่) | ~1-3min |