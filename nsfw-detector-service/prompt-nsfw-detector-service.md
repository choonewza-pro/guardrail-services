ช่วยเขียนโปรเจกต์ Python (FastAPI) สำหรับทำ NSFW Image Detection API Server ในชื่อโปรเจกต์ "nsfw-detector-service" ที่รองรับการประมวลผลบน GPU (CUDA) และเขียน Fallback กลับเป็น CPU ได้อย่างมีประสิทธิภาพ พร้อมระบบ Authentication และไฟล์สำหรับ Production Deployment โดยมีข้อกำหนดรายละเอียดดังนี้:

### 1. โครงสร้างและชื่อโปรเจกต์:
- ชื่อโปรเจกต์: `nsfw-detector-service`
- Framework: Python 3.10+ / FastAPI
- ใช้ Asynchronous I/O สำหรับระบบ API
- ใช้โครงสร้าง Modular แยกโฟลเดอร์ชัดเจน (เช่น `app/main.py`, `app/core/`, `app/api/`)

### 2. ระบบ Configuration & Port (อ่านจาก .env):
- ใช้ `pydantic-settings` ในการทำ Configuration Management
- อ่านค่าพอร์ตจากไฟล์ `.env` ผ่านตัวแปร `PORT` โดยกำหนดค่า Default ไว้ที่ `8085`
- อ่านค่า API Key จากไฟล์ `.env` ในตัวแปรชื่อ `API_KEY` (เช่น `API_KEY=my_super_secret_key_123`)
- อ่านค่าสวิตช์เปิด/ปิด GPU จาก `.env` เช่น `USE_GPU=true` หรือ `DEVICE=cuda` (default เป็น cuda หากมีอุปกรณ์พร้อมใช้งาน)

### 3. ระบบ Authentication (API Key Guard):
- ใช้ FastAPI `Security` / `HTTPBearer` หรือ `APIKeyHeader` Dependency ในการตรวจ Request
- ตรวจสอบ API Key จาก Request Header ชื่อ `x-api-key`
- หากไม่มี Header หรือ API Key ไม่ตรงกับใน `.env` ให้ปฏิเสธด้วย HTTP 401 Unauthorized พร้อม JSON:
  {
    "success": false,
    "error": "Unauthorized: Invalid or missing API key"
  }

### 4. โมเดล AI และ GPU Environment:
- ใช้โมเดล: `Marqo/nsfw-image-detection-384` ผ่าน `transformers`, `torch`, `torchvision`
- ตรวจสอบอุปกรณ์ประมวลผลแบบไดนามิก: หาก `USE_GPU=true` และ `torch.cuda.is_available()` ให้เลือกใช้ `cuda` หากไม่พบ ให้ Fallback กลับเป็น `cpu` อัตโนมัติพร้อม Log แจ้งเตือน
- โหลด/Initialize โมเดลแบบ Singleton Pattern ผ่าน `lifespan` context manager ของ FastAPI เพียงครั้งเดียวตอนสั่ง Start Server
- กำหนด Cache Directory สำหรับเก็บ Weights โมเดลอย่างชัดเจน ผ่าน Environment Variable `HF_HOME=/app/.cache` เพื่อให้ Mount Volume กับ Docker ได้

### 5. API Endpoints:
- `GET /health` (Public - ไม่ต้องใช้ API Key): คืนค่า Status 200 OK, สถานะโมเดล (Ready/Not Ready), และอุปกรณ์ที่ใช้อยู่ (`cuda` หรือ `cpu`)
- `POST /api/v1/detect-nsfw` (Protected - ต้องผ่าน API Key Auth):
  - รับไฟล์ภาพผ่าน `multipart/form-data` (ใช้ `UploadFile = File(...)`) ฟิลด์ชื่อ `image`
  - จำกัดขนาดไฟล์อัปโหลดไม่เกิน 10 MB (ทำ Validation Stream/Content-Length)
  - Validate MIME Type รับเฉพาะไฟล์ภาพ (`image/jpeg`, `image/png`, `image/webp`)

### 6. การประมวลผลภาพ (Image Processing Pipeline):
- ใช้ `Pillow` (PIL) ในการอ่านภาพจาก Byte Stream:
  - ย่อขนาดภาพ (Resize) หากความกว้างหรือสูงเกิน 1024px (`LANCZOS` resampling, `preserve aspect ratio`)
  - แปลงภาพทุกฟอร์แมต (รวมถึง .webp) ให้อยู่ในโหมด `RGB` ก่อนส่งเข้าโมเดล
- ส่ง PIL Image เข้าโมเดลเพื่อจำแนกประเภท

### 7. Response Format (JSON):
- กรณีสำเร็จ (Status 200):
  {
    "success": true,
    "data": {
      "is_nsfw": true / false, // Threshold ที่ 0.6
      "predictions": [
        { "label": "nsfw", "score": 0.95 },
        { "label": "normal", "score": 0.05 }
      ],
      "processing_time_ms": 12,
      "device_used": "cuda"
    }
  }
- กรณีเกิด Error อื่นๆ (เช่น ไม่ส่งไฟล์, ไฟล์ใหญ่เกิน, ไฟล์ผิดประเภท - Status 400/500) ให้ส่ง JSON Structure ที่บอกสาเหตุชัดเจน

### 8. โครงสร้างโปรเจกต์และไฟล์ที่ต้องการ:
1. `.env.example` (ตัวอย่างไฟล์ตั้งค่า เช่น `PORT=8085`, `API_KEY=your_secret_api_key_here`, `USE_GPU=true`, `HF_HOME=/app/.cache`)
2. `requirements.txt` ระบุเวอร์ชันหลักล่าสุดที่เป็นมาตรฐาน:
   - `fastapi`
   - `uvicorn[standard]`
   - `pydantic-settings`
   - `python-multipart`
   - `pillow`
   - `transformers`
   - `torch`
   - `torchvision`
   - `python-dotenv`
3. `main.py` หรือโครงสร้าง `app/` (รวม FastAPI App, Lifespan Events, Authentication Logic, Image Pipeline)
4. `Dockerfile` (ใช้ Base Image `nvidia/cuda:12.x` หรือ `python:3.10-slim` พร้อมติดตั้ง Dependencies ที่จำเป็น)
5. `docker-compose.yml` (อ่านค่า PORT จาก `.env`, Mount Volume สำหรับ `/app/.cache` และมีคอนฟิก `deploy.resources.reservations.devices` สำหรับพาส NVIDIA GPU เข้า Container)