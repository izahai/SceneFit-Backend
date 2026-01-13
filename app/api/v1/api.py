# app/api/v1/api.py

from fastapi import APIRouter
from app.api.v1.endpoints import diffusion_ep
from app.api.v1.endpoints import mmEmb_ep
from app.api.v1.endpoints import pe_clip_ep
from app.api.v1.endpoints import vlm_ep
from app.api.v1.endpoints import zimage_ep


api_router = APIRouter()
# api_router.include_router(diffusion_ep.router, prefix="", tags=["diffusion"])
# api_router.include_router(mmEmb_ep.router, prefix="/mmEmb", tags=["mmEmb"])
# api_router.include_router(pe_clip_ep.router, prefix="", tags=["pe"])
# api_router.include_router(vlm_ep.router, prefix="", tags=["vlm"])
api_router.include_router(diffusion_ep.router, prefix="", tags=["diffusion"])
api_router.include_router(zimage_ep.router, prefix="", tags=["zimage"])
