from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "touch_grass"
    LOCATION_MAX_AGE_SECONDS: int = 60
    GPS_MAX_ACCURACY_METERS: float = 250.0
    GPS_HIGH_CONFIDENCE_METERS: float = 50.0
    GPS_MOVEMENT_MAX_ACCURACY_METERS: float = 250.0
    GPS_MIN_SEGMENT_METERS: float = 25.0
    GPS_MOVEMENT_NOISE_FLOOR_METERS: float = 15.0
    GPS_MOVEMENT_STABILITY_METERS: float = 35.0
    GPS_MOVEMENT_CONFIRMATIONS: int = 2
    GPS_MAX_SPEED_METERS_PER_SECOND: float = 12.0
    MAX_UPLOAD_MB: int = 10
    GEMMA_MODEL: str = "gemma3:1b"
    AI_PROVIDER: Literal["ollama", "openrouter", "cloudflare"] = "ollama"
    AI_FALLBACK_PROVIDER: Literal["none", "openrouter", "ollama", "cloudflare"] = "none"
    CLOUDFLARE_ACCOUNT_ID: str = ""
    CLOUDFLARE_API_TOKEN: str = ""
    CLOUDFLARE_AI_MODEL: str = "@cf/google/gemma-4-26b-a4b-it"
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "google/gemma-4-26b-a4b-it:free"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = Field(default=0.60, ge=0.0, le=1.0)
    TARGET_VOTE_INACTIVE_SECONDS: int = Field(default=90, ge=30)
    DEMO_MODE: bool = False
    OLLAMA_URL: str = "http://localhost:11434"
    GEOCONTEXT_PROVIDER: str = "openstreetmap"
    GEOCONTEXT_NOMINATIM_URL: str = "https://nominatim.openstreetmap.org/reverse"
    GEOCONTEXT_OVERPASS_URL: str = "https://overpass-api.de/api/interpreter"
    GEOCONTEXT_RADIUS_METERS: int = 1200
    GEOCONTEXT_TIMEOUT_SECONDS: float = 10.0
    GEOCONTEXT_RETRIES: int = 2
    GEOCONTEXT_CACHE_TTL_SECONDS: int = 1800
    GEOCONTEXT_REFRESH_DISTANCE_METERS: float = 200.0
    GEOCONTEXT_REFRESH_COOLDOWN_SECONDS: int = 60
    GEOCONTEXT_REFRESH_STABILITY_METERS: float = 150.0
    GEOCONTEXT_REFRESH_CONFIRMATIONS: int = 2
    GEOCONTEXT_USER_AGENT: str = "TouchGrassGame/1.0 (location context)"

    @field_validator("OLLAMA_URL", mode="before")
    @classmethod
    def normalize_ollama_url(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("OLLAMA_URL must not be empty")
        if "://" in normalized:
            scheme, address = normalized.split("://", maxsplit=1)
            address = address.rstrip("/")
            if not address:
                raise ValueError("OLLAMA_URL must include a host")
            return f"{scheme}://{address}"
        normalized = normalized.rstrip("/").strip()
        if not normalized:
            raise ValueError("OLLAMA_URL must not be empty")
        return f"http://{normalized}"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
