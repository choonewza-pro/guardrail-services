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
    MODEL_NAME: str = "airesearch/wangchanberta-base-att-spm-uncased"

    TOXIC_THRESHOLD: float = 0.5
    MAX_TEXT_LENGTH: int = 1000
    TOXIC_LABEL_INDEX: int | None = None

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
    def cors_origins_list(self) -> list[str]:
        if not self.CORS_ORIGINS.strip():
            return []
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()