# app/api/v1/api.py

from fastapi import APIRouter
from numpy import clip
from app.api.v1.endpoints import pe_clip_ep
from app.api.v1.endpoints import vlm_ep
from app.api.v1.endpoints import image_edit_ep

api_router = APIRouter()

api_router.include_router(image_edit_ep.router, prefix="/retrieval", tags=["retrieval"])
api_router.include_router(pe_clip_ep.router, prefix="/retrieval", tags=["retrieval"])
