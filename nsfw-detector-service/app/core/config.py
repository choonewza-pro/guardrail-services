from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PORT: int = 8085
    LOG_LEVEL: str = "INFO"

    API_KEY: str = ""

    DEVICE: str = "auto"

    HF_HOME: str = "/app/.cache"
    MODEL_NAME: str = "Marqo/nsfw-image-detection-384"

    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_MIME: str = "image/jpeg,image/png,image/webp"

    NSFW_THRESHOLD: float = 0.6
    MAX_DIMENSION: int = 1024

    MAX_CONCURRENT_INFERENCES: int = 1

    CORS_ORIGINS: str = ""
    ENABLE_DOCS: bool = True

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
            raise ValueError(
                f"DEVICE must be one of: auto|cuda|cpu (got {v!r})"
            )
        return v

    @property
    def allowed_mime_list(self) -> list[str]:
        return [m.strip() for m in self.ALLOWED_MIME.split(",") if m.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        if not self.CORS_ORIGINS.strip():
            return []
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_IMAGE_SIZE_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()