from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.gpu", ".env.cpu"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PORT: int = 8087
    LOG_LEVEL: str = "INFO"
    ENABLE_DOCS: bool = True

    API_KEY: str = ""

    DEVICE: str = "auto"
    MODEL_NAME: str = "typhoon-ai/typhoon-ocr1.5-2b"
    MODEL_DTYPE: str = "auto"

    HF_HOME: str = "/app/.cache"

    DEFAULT_MAX_TOKENS: int = 4096
    DEFAULT_TEMPERATURE: float = 0.1
    DEFAULT_TIMEOUT: float = 500.0
    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_SYSTEM_PROMPT: str = (
        "คุณเป็นผู้เชี่ยวชาญด้านการอ่านเอกสารภาษาไทย ให้ถอดรหัสและแปลงเนื้อหาทั้งหมดในเอกสารเป็นรูปแบบ Markdown อย่างถูกต้องแม่นยำ "
        "โดยหากพบข้อมูลประเภทตารางให้จัดรูปแบบเป็นตาราง Markdown (หากมีช่องว่างหรือไม่มีข้อมูลในช่องตารางให้ใส่เครื่องหมาย `-`) "
        "และคงโครงสร้างของข้อความ หัวข้อ หรือเนื้อหาส่วนอื่นๆ ให้อยู่ในรูปแบบ Markdown ที่เหมาะสม"
    )

    MAX_IMAGE_SIZE: int = 1800

    MAX_CONCURRENT_INFERENCES: int = 1
    CORS_ORIGINS: str = ""

    @field_validator("API_KEY")
    @classmethod
    def _validate_api_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "API_KEY must be set. Copy .env.example to .env and fill API_KEY."
            )
        return v.strip()

    @field_validator("DEVICE")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        v = (v or "auto").strip().lower()
        if v not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"DEVICE must be one of: auto|cuda|cpu (got {v!r})")
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        if not self.CORS_ORIGINS.strip():
            return []
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
