# 🌀 typhoon-ocr-service

FastAPI microservice สำหรับบริการ **Image OCR & Document Parsing** ด้วยโมเดล **Typhoon OCR 1.5** ( default: `typhoon-ai/typhoon-ocr1.5-2b` หรือสลับเป็นโมเดลขนาดใหญ่ขึ้น เช่น `3b`/`8b` ผ่าน `.env`) รองรับการประมวลผลทั้งบน CPU และ NVIDIA GPU (CUDA), ระบบ Authentication ผ่าน API Key, SSE Streaming Response, Image Preprocessing & Auto Scaling, Retry & Timeout handling และ Strict Temporary File Cleanup

---

## 🌟 ฟีเจอร์หลัก (Key Features)

1. **Dynamic Model Switching**: สามารถเปลี่ยนโมเดลในตระกูล Typhoon OCR หรือ Vision-Language Model อื่นๆ บน HuggingFace ผ่านตัวแปร `MODEL_NAME` และ `MODEL_DTYPE` ในไฟล์ `.env` ได้ทันที
2. **Post Endpoint (9 Parameters Support)**:
   - `file`: รูปภาพเอกสาร (PNG, JPG, WEBP)
   - `question`: คำถามหรือ Prompt สำหรับ OCR ( Default: Prompt มาตรฐาน Typhoon OCR 1.5)
   - `system_prompt`: System prompt เพิ่มเติม
   - `temperature`: Sampling temperature (default: `0.1`)
   - `max_tokens`: จำนวน token สูงสุด (default: `4096`)
   - `seed`: Random seed เพื่อผลลัพธ์ที่ควบคุมได้
   - `max_retries`: จำนวนครั้งพยายามใหม่เมื่อเกิดข้อผิดพลาด (default: `3`)
   - `timeout`: เวลา Timeout ของ Request ในหน่วยวินาที (default: `60.0`)
   - `is_stream`: สลับการตอบกลับเป็น Server-Sent Events (SSE) Streaming Response
3. **Automatic Image Preprocessing**: Resize รูปภาพที่มีขนาดเกิน 1,800 px ด้วย Lanczos Filter เพื่อความสมบูรณ์ในการสกัดข้อความ
4. **Strict File & Memory Cleanup**: การันตีการปิดและลบไฟล์ชั่วคราว/หน่วยความจำรูปภาพทันทีหลังประมวลผลเสร็จ หรือเมื่อเกิด Error
5. **API Key Authentication Guard**: ป้องกันการใช้งานผ่าน Header `x-api-key`

---

## 🚀 การติดตั้งและการรันแบบ Local (No Docker)

### 1. สร้าง Virtual Environment
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. ติดตั้ง PyTorch และ Dependencies
```bash
# สำหรับ CPU Only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# สำหรับ NVIDIA GPU (CUDA 12.1):
# pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### 3. ตั้งค่า `.env`
```bash
cp .env.example .env
```
แก้ไขไฟล์ `.env`:
```env
PORT=8087
API_KEY=dev_ocr_secret_key_123
DEVICE=auto
MODEL_NAME=typhoon-ai/typhoon-ocr1.5-2b
MODEL_DTYPE=auto
```

### 4. สั่งรัน Server
```bash
uvicorn app.main:app --port 8087 --reload
```

---

## 🐳 การรันด้วย Docker Compose

ต้องสร้าง Docker Network `km4u-network` ก่อนเป็นครั้งแรก:
```bash
docker network create km4u-network
```

### การรันบน CPU
```bash
docker compose -f docker-compose-cpu.yml up -d --build
```

### การรันบน GPU (NVIDIA CUDA)
```bash
docker compose -f docker-compose-gpu.yml up -d --build
```

---

## 📡 การใช้งาน API Endpoints

### 1. Public Health Check
**`GET /health`**
```bash
curl http://localhost:8087/health
```
**Response Status 200**:
```json
{
  "status": "ok",
  "model_ready": true,
  "model_name": "typhoon-ai/typhoon-ocr1.5-2b",
  "device_used": "cuda"
}
```

---

### 2. Protected OCR Endpoint
**`POST /api/v1/ocr`** ( Header: `x-api-key: dev_ocr_secret_key_123` )

#### แบบที่ A: ส่งแบบ `multipart/form-data` (Binary File Upload):
```bash
curl -X POST "http://localhost:8087/api/v1/ocr" \
  -H "x-api-key: dev_ocr_secret_key_123" \
  -F "file=@document.jpg" \
  -F "temperature=0.1" \
  -F "max_tokens=4096" \
  -F "is_stream=false"
```

#### แบบที่ B: ส่งแบบ `application/json` (Raw JSON + Base64 Image):
```bash
curl -X POST "http://localhost:8087/api/v1/ocr" \
  -H "x-api-key: dev_ocr_secret_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "file": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "temperature": 0.1,
    "max_tokens": 4096,
    "is_stream": false
  }'
```

**JSON Response (Status 200)**:
```json
{
  "success": true,
  "data": {
    "text": "# หัวข้อเอกสาร\n\nเนื้อหาที่สกัดได้แบบ Markdown...",
    "model_used": "typhoon-ai/typhoon-ocr1.5-2b",
    "processing_time_ms": 1420.5,
    "device_used": "cuda",
    "image_size": {
      "original": [2400, 3200],
      "processed": [1350, 1800]
    },
    "usage": {
      "prompt_tokens": 128,
      "completion_tokens": 350,
      "total_tokens": 478
    }
  }
}
```

---

#### ตัวอย่างยิง Request แบบ SSE Streaming Response (`is_stream=true`):
```bash
curl -X POST "http://localhost:8087/api/v1/ocr" \
  -H "x-api-key: dev_ocr_secret_key_123" \
  -F "file=@document.jpg" \
  -F "is_stream=true"
```

**Stream Response (`text/event-stream`)**:
```http
data: {"chunk": "ข้อความ", "is_final": false}

data: {"chunk": "ส่วนถัดไป...", "is_final": false}

data: {"chunk": "", "is_final": true, "model_used": "typhoon-ai/typhoon-ocr1.5-2b", "processing_time_ms": 1420.5, "device_used": "cuda"}
```

---

## 🐍 ตัวอย่างการใช้งานด้วย Python Requests

```python
import requests

url = "http://localhost:8087/api/v1/ocr"
headers = {"x-api-key": "dev_ocr_secret_key_123"}

files = {"file": ("test.png", open("test.png", "rb"), "image/png")}
data = {
    "question": "Extract all text from the image.",
    "temperature": "0.1",
    "max_tokens": "4096",
    "is_stream": "false",
}

response = requests.post(url, headers=headers, files=files, data=data)
print(response.json())
```
