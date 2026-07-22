# guardrail-services

Monorepo ของ microservice สำหรับตรวจจับเนื้อหาไม่เหมาะสม ประกอบด้วย service คู่ขนานที่ทำงานอิสระต่อกัน (ไม่มี shared code หรือ workspace manifest) — แต่ service เป็น Python project ครบวงจรของตัวเอง มี `requirements.txt`, `.env`, และ Docker image แยก

| Service | หน้าที่ | โมเดล | Python | Port#
|---|---|---|---|---|
| [`nsfw-detector-service/`](./nsfw-detector-service) | ตรวจจับภาพ NSFW | `Marqo/nsfw-image-detection-384` | 3.11 | `8085` |
| [`toxic-sentinel-service/`](./toxic-sentinel-service) | ตรวจจับข้อความโทรมศักดิ์ (ภาษาไทย) | WangchanBERTa (`AutoModelForSequenceClassification`) | 3.12 | `8086` |

> ค่า `PORT` ใน `.env.example` ของแต่ละ service ตั้งไว้ต่างกัน (8085 / 8086) — แต่ไฟล์ `.env` จริงของแต่ละ service แยกกัน ถ้าเคย copy แล้วไม่ได้แก้ อาจชนกันได้ root compose แก้ด้วยการส่ง `NSFW_PORT` / `TOXIC_PORT` env override ที่ขั้นต้น

---

## โครงสร้าง Monorepo

```
guardrail-services/
├── docker-compose-gpu.yml         # root GPU compose — สั่งรันทั้งสอง service พร้อมกัน
├── docker-compose-cpu.yml         # root CPU-only compose — สั่งรันทั้งสอง service พร้อมกัน
├── nsfw-detector-service/          # self-contained Python project
│   ├── .env.example
│   ├── requirements.txt
│   ├── Dockerfile.gpu / Dockerfile.cpu
│   ├── docker-compose-gpu.yml / docker-compose-cpu.yml   # รัน service เดียวแยก
│   ├── scripts/preload_model.py
│   ├── app/
│   └── README.md
├── toxic-sentinel-service/         # self-contained Python project (เช่นเดียวกัน)
│   ├── ...
│   └── README.md
└── AGENTS.md                       # หมายเหตุสำหรับ AI agent
```

แต่ละ service มี `README.md` ละเอียดของตัวเอง — ดูตามลิงก์ด้านบน

---

## การรัน (Docker)

### 1. เตรียม environment แต่ละ service

```bash
cp nsfw-detector-service/.env.example  nsfw-detector-service/.env
cp toxic-sentinel-service/.env.example  toxic-sentinel-service/.env
# แก้ API_KEY ในทั้งสองไฟล์ให้เป็น secret ที่แข็งแรง (แยกกันหรือใช้ค่าเดียวกันก็ได้)
```

### 2. สร้าง external network (ครั้งเดียว)

```bash
docker network create km4u-network
```

### 3. เลือก compose ตามเครื่อง host

#### 🖥️ เซิร์ฟเวอร์มี NVIDIA GPU

```bash
docker compose -f docker-compose-gpu.yml up -d --build
```

#### 💻 เซิร์ฟเวอร์มีแต่ CPU

```bash
docker compose -f docker-compose-cpu.yml up -d --build
```

> Build ครั้งแรกอาจใช้เวลา 15-20 นาที (GPU) หรือ 5-8 นาที (CPU) เนื่องจากโหลด `torch` + model weights
> ครั้งต่อไปแก้ code เดียว: ~10 วินาที (Docker layer cache)
> Container start: < 5 วินาที (model weights pre-baked ใน image)

### รันเฉพาะ service เดียว

ถ้าทำงานกับ service ใด service หนึ่ง แนะนำให้ใช้ compose ของ service นั้นแทน root compose

```bash
docker compose -f nsfw-detector-service/docker-compose-gpu.yml up -d --build
docker compose -f toxic-sentinel-service/docker-compose-cpu.yml up -d --build
```

### ตรวจสอบสถานะ

```bash
curl http://localhost:8085/health   # nsfw-detector
curl http://localhost:8086/health   # toxic-sentinel
```

---

## API Reference (สรุป)

โครงสร้าง API เหมือนกันทั้งสอง service — มี `GET /health` (public) และ endpoint ตรวจจับ (protected ผ่าน header `x-api-key`)

### nsfw-detector-service

```bash
curl -X POST http://localhost:8085/api/v1/detect-nsfw \
  -H "x-api-key: YOUR_API_KEY" \
  -F "image=@/path/to/image.jpg"
```

### toxic-sentinel-service

```bash
curl -X POST http://localhost:8086/api/v1/detect-toxic \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"ข้อความที่ต้องการตรวจ"}'
```

รายละเอียด field, response shape, และ error table — ดูใน `README.md` ของแต่ละ service

---

## Response Envelope

ทั้งสอง service ใช้รูปแบบ response เดียวกัน

**สำเร็จ:**
```json
{ "success": true, "data": { ... } }
```

**ผิดพลาด:**
```json
{ "success": false, "error": "..." }
```

---

## สถาปัตยกรรม (เหมือนกันในทั้งสอง service)

- `app/main.py` — FastAPI app + `lifespan` โหลดโมเดลครั้งเดียวตอน startup (singleton) — **ปฏิเสธบูตถ้า `API_KEY` ว่าง**
- `app/core/config.py` — `pydantic-settings` อ่าน `.env` มี validator hard-fail ถ้า `API_KEY` ว่าง
- `app/services/model_manager.py` — singleton ครอบ HuggingFace model ตัดสิน `DEVICE` (ค่า: `auto`/`cuda`/`cpu`; `auto` = `cuda` หาก `torch.cuda.is_available()` ไม่งั้น `cpu`; `cuda` auto-fallback เป็น `cpu` เมื่อไม่พร้อม) การทำนายทำภายใต้ thread lock
- `app/api/deps.py` — guard ผ่าน header `x-api-key` → 401 พร้อม envelope
- `app/api/v1/endpoints/detect.py` — endpoint ตรวจจับ (protected)
- `app/api/v1/endpoints/health.py` — `GET /health` (public)
- exception handlers ใน `main.py` ครอบทุก error เป็น envelope `{success:false,error:...}`

---

## Conventions ที่ต้องรักษา

- **Response envelope**: ห้ามเปลี่ยนรูป `{"success": true, "data": {...}}` / `{"success": false, "error": "..."}`
- **Model เป็น singleton** ผ่าน `lifespan` — ห้าม instantiate `ModelManager` หรือเรียก `.load()` ใน request handler
- **Docker image multi-stage**: `deps` → `model-preload` (รัน `scripts/preload_model.py`, baked ใน image) → `final` พร้อม `TRANSFORMERS_OFFLINE=1` — runtime ไม่ดาวน์โหลด weights; ถ้าเปลี่ยน `MODEL_NAME` ต้อง rebuild
- **`torch` อยู่ใน Dockerfile** ไม่ใช่ `requirements.txt` (index URL ต่างกัน CPU vs CUDA)
- **Python version ต่างกัน** ระหว่างสอง service — ดู base image ก่อนแก้ pin
- **`HF_HOME=/app/.cache`** baked ใน Dockerfile env; compose mount เป็น volume สำหรับ toxic-sentinel — ห้ามเขียน cache ใน app code
- **`.serena/` ถูก `.gitignore`** อยู่แล้ว — อย่าเพิ่มเข้า commit

---

## Local Dev (ไม่ใช้ Docker)

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --port <PORT> --reload
```

รันจากใน directory ของ service ที่ต้องการ (`nsfw-detector-service/` หรือ `toxic-sentinel-service/`)  และ copy `.env.example` → `.env` พร้อม `API_KEY` จริงก่อน

---

## ข้อกำหนดเครื่อง host

### GPU server
- Docker + NVIDIA Container Toolkit
- NVIDIA driver รองรับ CUDA 12.1+
- ตรวจสอบ GPU visibility:
```bash
docker compose -f docker-compose-gpu.yml exec nsfw-detector \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only')"
```

### CPU-only server
- Docker ติดตั้งแล้ว
- RAM แนะนำขั้นต่ำ 2GB
- ไม่ต้องการ NVIDIA driver/toolkit

---

## Gotchas

- **ไม่มี test suite** — verification ผ่าน `curl .../health` และ protected detection endpoints
- **`toxic-sentinel-service` ค่า `MODEL_NAME` default** (`pythainlp/wangchanberta-base-att-spm-uncased`) เป็น **base MLM head** ไม่ใช่ toxic classifier — สำหรับงานจริงควรเปลี่ยนเป็น fine-tuned WangchanBERTa toxic classification model ใน `.env`  code `model_manager` จะ auto-detect label "toxic" ผ่าน substring match บน `id2label`; ถ้าไม่ตรงก็ fallback ไป argmax
- **อย่าเชื่อใจ default ของ `.env` สองไฟล์ให้ sync อยู่เสมอ** — ทั้งสอง service เขียนแยกกัน และค่า default อาจเปลี่ยนตามอิสระ