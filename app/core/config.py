from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Multi-Model AI Backend"
    API_VERSION: str = "v1"
    DEVICE: str = "cuda"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()