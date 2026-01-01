from fastapi import APIRouter
from app.api.v1.endpoints import clip, diffusion, vlm

api_router = APIRouter()
api_router.include_router(clip.router, prefix="/clip", tags=["CLIP"])
api_router.include_router(diffusion.router, prefix="/diffusion", tags=["Diffusion"])
api_router.include_router(vlm.router, prefix="/vlm", tags=["VLM"])