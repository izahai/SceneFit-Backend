from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    APP_NAME: str = "Multi-Model AI Backend"
    API_VERSION: str = "v1"
    DEVICE: str = "cuda"
    LOG_LEVEL: str = "INFO"

    IMAGEROUTER_API_KEY: str = Field(..., description="ImageRouter API key")

    model_config = {
        "env_file": ".env",
        "extra": "forbid",
    }

settings = Settings()
