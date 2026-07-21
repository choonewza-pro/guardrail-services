ช่วยเขียนโปรเจกต์ Python (FastAPI) สำหรับทำ Toxic Text Detection API Server ในชื่อโปรเจกต์ "toxic-sentinel" ที่รองรับการประมวลผลด้วยโมเดล WangchanBERTa ทั้งบน CPU และ GPU (CUDA) พร้อมระบบ Authentication, Dockerization ที่แยกสภาพแวดล้อม CPU/GPU และไฟล์สำหรับ Production Deployment โดยมีข้อกำหนดรายละเอียดดังนี้:

### 1. โครงสร้างและชื่อโปรเจกต์:
- ชื่อโปรเจกต์: `toxic-sentinel`
- Framework: Python 3.12+ (เวอร์ชันล่าสุดที่เป็นมาตรฐาน) / FastAPI เวอร์ชันล่าสุด (ใช้ Pydantic v2)
- ใช้ Asynchronous I/O สำหรับระบบ API
- โครงสร้างโปรเจกต์แบบ Modular ปรับแต่งง่ายและอ่านสบาย (เช่น `app/main.py`, `app/core/`, `app/api/`, `app/services/`)

### 2. ระบบ Configuration & Port (อ่านจาก .env):
- ใช้ `pydantic-settings` ในการทำ Configuration Management
- อ่านค่าพอร์ตจากไฟล์ `.env` ผ่านตัวแปร `PORT` โดยกำหนดค่า Default ไว้ที่ `8085`
- อ่านค่า API Key จากไฟล์ `.env` ในตัวแปรชื่อ `API_KEY` (เช่น `API_KEY=my_super_secret_key_123`)
- อ่านค่าสวิตช์อุปกรณ์จาก `.env` เช่น `DEVICE=cuda` หรือ `DEVICE=cpu` ( default ให้ใช้ `cuda` หากพบว่า `torch.cuda.is_available()` มิฉะนั้นให้ Fallback เป็น `cpu`)
- อ่านค่า Model ID เช่น `MODEL_NAME` (default: `pythainlp/wangchanberta-base-att-spm-uncased` หรือโมเดล WangchanBERTa toxic classification ที่เกี่ยวข้อง) และ `TOXIC_THRESHOLD` (default: `0.5`)

### 3. ระบบ Authentication (API Key Guard):
- ใช้ FastAPI `Security` / `APIKeyHeader` Dependency ในการตรวจ Request
- ตรวจสอบ API Key จาก Request Header ชื่อ `x-api-key`
- หากไม่มี Header หรือ API Key ไม่ตรงกับใน `.env` ให้ปฏิเสธด้วย HTTP 401 Unauthorized พร้อม JSON Structure:
  {
    "success": false,
    "error": "Unauthorized: Invalid or missing API key"
  }

### 4. โมเดล AI และ Environment:
- ใช้โมเดล WangchanBERTa ผ่าน `transformers` และ `torch`
- ตรวจสอบอุปกรณ์ประมวลผลแบบไดนามิก: หากตั้งค่าใช้ GPU แต่ระบบไม่มี GPU หรือ `torch.cuda.is_available()` เป็น False ให้ Fallback กลับเป็น `cpu` อัตโนมัติพร้อม Logging แจ้งเตือน
- โหลด/Initialize โมเดลแบบ Singleton Pattern ผ่าน `lifespan` context manager ของ FastAPI เพียงครั้งเดียวตอน Start Server
- กำหนด Cache Directory สำหรับเก็บ Weights โมเดลอย่างชัดเจน ผ่าน Environment Variable `HF_HOME=/app/.cache` เพื่อให้ Mount Volume กับ Docker ได้

### 5. API Endpoints:
- `GET /health` (Public - ไม่ต้องใช้ API Key): คืนค่า Status 200 OK, สถานะโมเดล (Ready/Not Ready), และอุปกรณ์ที่ใช้อยู่ (`cuda` หรือ `cpu`)
- `POST /api/v1/detect-toxic` (Protected - ต้องผ่าน API Key Auth):
  - รับ Request Body เป็น JSON: 
    - `text`: string (ข้อความภาษาไทยที่ต้องการตรวจ, บังคับใส่, ไม่เป็นค่าว่าง)
    - `threshold`: float (Optional, ค่าความไวในการตัดสินผล หากไม่ส่งมาให้ใช้ค่าจาก `.env`)
  - Validate ความยาวข้อความ (ไม่เกิน 1,000 ตัวอักษร)

### 6. Response Format (JSON):
- กรณีสำเร็จ (Status 200):
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
- กรณีเกิด Error (Status 400/422/500): ให้ส่ง JSON Structure `{"success": false, "error": "รายละเอียดข้อผิดพลาด"}`

### 7. ไฟล์และโครงสร้างที่ต้องการทั้งหมด:
1. `.env.example` (ตั้งค่า `PORT=8085`, `API_KEY=your_secret_key`, `DEVICE=cpu`, `TOXIC_THRESHOLD=0.5`, `HF_HOME=/app/.cache`)
2. `requirements.txt` ระบุเวอร์ชันหลักล่าสุดที่เป็นมาตรฐาน:
   - `fastapi`
   - `uvicorn[standard]`
   - `pydantic`
   - `pydantic-settings`
   - `transformers`
   - `torch` (ระบุวิธีติดตั้งที่เหมาะสมสำหรับ CPU และ GPU)
   - `sentencepiece`
   - `python-dotenv`
3. โค้ดโปรเจกต์ในโฟลเดอร์ `app/` (รวม Main App, Lifespan, Config, Service โหลดโมเดล, API Route, และ Auth Guard)
4. Docker Setup แบบแยก Environment ชัดเจน:
   - `Dockerfile.cpu`: ใช้ Base Image `python:3.12-slim` ติดตั้ง `torch` เวอร์ชัน CPU-only เพื่อให้ Image มีขนาดเล็กที่สุดและรันบน CPU ได้รวดเร็ว
   - `docker-compose-cpu.yml`: สำหรับรันบน CPU (พร้อม Volume Mount สำหรับ `HF_HOME`)
   - `Dockerfile.gpu`: ใช้ Base Image `nvidia/cuda:12.x.x-runtime-ubuntu22.04` หรือ `python:3.12-slim` ร่วมกับ PyTorch CUDA 12.x
   - `docker-compose-gpu.yml`: สำหรับรันบน GPU (พร้อมการตั้งค่า `deploy.resources.reservations.devices` สำหรับพาส NVIDIA GPU เข้า Container)
5. `README.md` อธิบายวิธีตั้งค่า, สั่ง Build และการรันทั้งฝั่ง CPU และ GPU