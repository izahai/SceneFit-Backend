# app/main.py

from fastapi import FastAPI
from app.core.config import settings
from app.core.lifecycle import lifespan
from app.api.v1.api import api_router

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.API_VERSION,
        lifespan=lifespan,
    )

    app.include_router(api_router, prefix="/api/v1")
    return app

app = create_app()
