ช่วยเขียนโปรเจกต์ Python (FastAPI) สำหรับทำ Image OCR & Document Parsing API Server ในชื่อโปรเจกต์ "typhoon-ocr-service" ที่ประมวลผลด้วยโมเดลตระกูล Typhoon OCR ( default: `typhoon-ai/typhoon-ocr1.5-2b` และ **สามารถสลับเปลี่ยนเป็นโมเดลขนาดใหญ่ขึ้น เช่น 3B/8B ได้ง่ายๆ เพียงแก้ค่าในไฟล์ `.env`**) รองรับการทำงานทั้งบน CPU และ GPU (CUDA) พร้อมระบบ Authentication (API Key), Streaming Response, Image Preprocessing, Retry & Timeout handling, Dockerization แยก CPU/GPU และไฟล์สำหรับ Production Deployment โดยมีรายละเอียดและข้อกำหนดดังนี้:

### 1. โครงสร้างและชื่อโปรเจกต์:

- ชื่อโปรเจกต์: `typhoon-ocr-service`
- Framework: Python 3.12+ / FastAPI (ใช้ Pydantic v2)
- ใช้ Asynchronous I/O สำหรับระบบ API
- โครงสร้างโปรเจกต์แบบ Modular ปรับแต่งง่ายและอ่านสะดวก (`app/main.py`, `app/core/`, `app/api/`, `app/services/`)

### 2. ระบบ Configuration & Dynamic Model Switching (อ่านจาก .env):

- ใช้ `pydantic-settings` สำหรับ Configuration Management
- อ่านค่าพอร์ตจากไฟล์ `.env` ผ่านตัวแปร `PORT` โดยกำหนดค่า Default ไว้ที่ `8087`
- อ่านค่า API Key จากไฟล์ `.env` ในตัวแปรชื่อ `API_KEY` (ตัวเฝ้าระวังบังคับ ห้ามเป็นค่าว่าง)
- อ่านค่าสวิตช์อุปกรณ์จาก `.env` เช่น `DEVICE=auto` (หากพบว่า `torch.cuda.is_available()` ให้ใช้ `cuda` มิฉะนั้น Fallback เป็น `cpu`)
- **Dynamic Model Switching**: อ่านค่า Model ID จาก `.env` ในตัวแปรชื่อ `MODEL_NAME` 
  - Default: `typhoon-ai/typhoon-ocr1.5-2b` (หรือ `scb10x/typhoon-ocr1.5-2b`)
  - **การเปลี่ยนไปใช้โมเดลตัวใหญ่**: ผู้ใช้สามารถเปลี่ยนเป็นโมเดลเวอร์ชันใหญ่ขึ้นได้ทันที เช่น `scb10x/typhoon-ocr1.5-3b` หรือโมเดล vision-language OCR อื่นๆ บน HuggingFace เพียงเปลี่ยนค่า `MODEL_NAME` ใน `.env` โดยไม่ต้องแก้ไขโค้ดโปรเจกต์
- **Model Precision Config**: อ่านค่า `MODEL_DTYPE` จาก `.env` (default: `auto`, รองรับ `float16`, `bfloat16`, `float32`) เพื่อปรับความเหมาะสมของหน่วยความจำเมื่อใช้โมเดลขนาดใหญ่
- อ่านค่า Cache Directory สำหรับโมเดลผ่าน Environment Variable `HF_HOME=/app/.cache`

### 3. ระบบ Authentication (API Key Guard):

- ใช้ FastAPI `Security` / `APIKeyHeader` Dependency ตรวจสอบ Request Header ชื่อ `x-api-key`
- หากไม่มี Header หรือ API Key ไม่ถูกต้อง ให้ตอบกลับด้วย HTTP 401 Unauthorized พร้อม JSON Format:
  ```json
  {
    "success": false,
    "error": "Unauthorized: Invalid or missing API key"
  }
  ```

### 4. โมเดล AI และ Image Preprocessing:

- โหลดโมเดลอย่างยืดหยุ่นตาม `MODEL_NAME` ใน `.env` ผ่าน `transformers` (`AutoModelForImageTextToText`, `AutoProcessor`)
- โหลด/Initialize โมเดลแบบ Singleton Pattern ผ่าน `lifespan` context manager ของ FastAPI เพียงครั้งเดียวเมื่อ Start Server พร้อมรองรับ Thread Lock ในการประมวลผล
- **Image Preprocessing**: ตรวจสอบขนาดรูปภาพ หากความกว้างหรือความสูงเกิน 1,800 pixels ให้ทำการ Resize (Lanczos) โดยคงอัตราส่วนเดิม (Aspect Ratio) ตามข้อแนะนำมาตรฐานของโมเดล Typhoon OCR
- **Default Prompt Structure**: หากผู้ใช้ไม่ได้ระบุคำถาม (`question`) มา ให้ใช้ Default Prompt มาตรฐานของ Typhoon OCR 1.5:
  ```
  Extract all text from the image.

  Instructions:
  - Only return the clean Markdown.
  - Do not include any explanation or extra text.
  - You must include all information on the page.

  Formatting Rules:
  - Tables: Render tables using <table>...</table> in clean HTML format.
  - Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
  - Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:

  <figure>
  Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in Thai.
  </figure>

  - Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
  - Checkboxes: Use ☐ for unchecked and ☑ for checked boxes.
  ```

### 5. API Endpoints & Request Parameters:

- `GET /health` (Public - ไม่ต้องใช้ API Key): คืนค่า Status 200 OK, สถานะโมเดล (`model_ready`: true/false), ชื่อโมเดลปัจจุบัน (`model_name`), และอุปกรณ์ที่ใช้อยู่ (`cuda` หรือ `cpu`)
- `POST /api/v1/ocr` (Protected - ต้องผ่าน API Key Guard):
  - รับข้อมูลในรูปแบบ `multipart/form-data` รองรับ Parameter ทั้งหมด 9 รายการ ดังนี้:
    1. **`file`**: `UploadFile` (รูปภาพไฟล์เอกสาร เช่น PNG, JPG, JPEG, WEBP - บังคับส่ง)
    2. **`question`**: `str` (Form, Optional - คำถามหรือคำสั่งสำหรับ OCR หากไม่ระบุให้ใช้ Default Prompt ของ Typhoon OCR 1.5)
    3. **`system_prompt`**: `str` (Form, Optional - คำสั่งระดับระบบ System Prompt เพิ่มเติม หากไม่ส่งให้ใช้ค่าว่าง)
    4. **`temperature`**: `float` (Form, Optional - ค่า Sampling Temperature เช่น `0.1`, default: `0.1`, ช่วงค่า 0.0 - 2.0)
    5. **`max_tokens`**: `int` (Form, Optional - จำนวน Token สูงสุดที่สร้าง เช่น `4096`, default: `4096`)
    6. **`seed`**: `int` (Form, Optional - ค่า Random Seed เพื่อควบคุมการสุ่มคำตอบ ให้ผลลัพธ์เดิม ซ้ำได้, default: `None`)
    7. **`max_retries`**: `int` (Form, Optional - จำนวนครั้งสูงสุดในการพยายามประมวลผลใหม่เมื่อเกิดข้อผิดพลาด, default: `3`)
    8. **`timeout`**: `float` (Form, Optional - เวลา Timeout ในการประมวลผลต่อ Request หน่วยเป็นวินาที, default: `60.0`)
    9. **`is_stream`**: `bool` (Form, Optional - สวิตช์เลือกการตอบกลับแบบ Streaming Response (SSE), default: `false`)

### 6. Response Format:

#### 6.1 กรณีสำเร็จ และ `is_stream = false` (JSON Standard Status 200):
```json
{
  "success": true,
  "data": {
    "text": "ข้อความ Markdown ที่สกัดได้จากรูปภาพ...",
    "model_used": "typhoon-ai/typhoon-ocr1.5-2b",
    "processing_time_ms": 1250.45,
    "device_used": "cuda",
    "image_size": {
      "original": [2400, 3200],
      "processed": [1350, 1800]
    },
    "usage": {
      "prompt_tokens": 256,
      "completion_tokens": 512,
      "total_tokens": 768
    }
  }
}
```

#### 6.2 กรณีสำเร็จ และ `is_stream = true` (Server-Sent Events: `text/event-stream` Status 200):
ส่งคืนข้อมูลทีละ Chunk ในรูปแบบ SSE:
```http
Content-Type: text/event-stream

data: {"chunk": "ข้อความส่วนที่ 1...", "is_final": false}

data: {"chunk": "ข้อความส่วนที่ 2...", "is_final": false}

data: {"chunk": "", "is_final": true, "model_used": "typhoon-ai/typhoon-ocr1.5-2b", "processing_time_ms": 1250.45, "device_used": "cuda"}
```

#### 6.3 กรณีเกิด Error (HTTP 400 / 422 / 500 / 504 Timeout):
```json
{
  "success": false,
  "error": "รายละเอียดข้อผิดพลาด หรือ Request timeout after 60.0s"
}
```

### 7. ไฟล์และโครงสร้างที่ต้องการทั้งหมด:

1. `.env.example` (ตั้งค่า `PORT=8087`, `API_KEY=your_secret_key`, `DEVICE=auto`, `MODEL_NAME=typhoon-ai/typhoon-ocr1.5-2b`, `MODEL_DTYPE=auto`, `HF_HOME=/app/.cache`)
2. `requirements.txt` ระบุเวอร์ชันมาตรฐาน:
   - `fastapi`
   - `uvicorn[standard]`
   - `pydantic`
   - `pydantic-settings`
   - `transformers`
   - `accelerate`
   - `pillow`
   - `torch` (ระบุวิธีติดตั้งที่เหมาะสมสำหรับ CPU และ CUDA GPU)
   - `python-multipart`
   - `python-dotenv`
3. โค้ดโปรเจกต์ในโฟลเดอร์ `app/` (รวม Main App, Lifespan, Config, Service โหลดและประมวลผลโมเดลตาม `MODEL_NAME`, Image Resizer, API Route, Auth Guard, Streaming Generator และ Handling Retry/Timeout)
4. Docker Setup แบบแยก Environment ชัดเจน:
   - `Dockerfile.cpu`: Base image `python:3.12-slim` ติดตั้ง `torch` CPU-only
   - `docker-compose-cpu.yml`: สำหรับรันบน CPU (พร้อม Volume Mount สำหรับ `HF_HOME`)
   - `Dockerfile.gpu`: Base image `python:3.12-slim` ร่วมกับ PyTorch CUDA 12.x
   - `docker-compose-gpu.yml`: สำหรับรันบน GPU (พร้อม `deploy.resources.reservations.devices` NVIDIA GPU)
5. `README.md` อธิบายการตั้งค่า, การสั่งเปลี่ยนโมเดลใน `.env`, การสั่ง Build/Run ทั้งฝั่ง CPU และ GPU รวมถึงตัวอย่างการส่ง Request ทั้งแบบ Normal JSON และ Streaming (SSE) ผ่าน `curl` หรือ Python `requests`/`httpx`
