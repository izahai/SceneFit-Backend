# app/api/v1/api.py

from fastapi import APIRouter
from app.api.v1.endpoints import all_methods_ep
from app.api.v1.endpoints import experiment_ep
from app.api.v1.endpoints import vlm_ep

api_router = APIRouter()


api_router.include_router(vlm_ep.router, prefix="/vlm", tags=["vlm"])