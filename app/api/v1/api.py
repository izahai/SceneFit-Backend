# app/api/v1/api.py

from fastapi import APIRouter
from app.api.v1.endpoints import diffusion_ep, vector_db_ep
from app.api.v1.endpoints import mmEmb_ep
from app.api.v1.endpoints import pe_clip_ep
from app.api.v1.endpoints import vlm_ep
from app.api.v1.endpoints import neg_pe_clip_ep
from app.api.v1.endpoints import image_edit_ep
from app.api.v1.endpoints import aesthetic_ep
from app.api.v1.endpoints import asr_ep


api_router = APIRouter()

# Retrieval namespace (all under /retrieval)
api_router.include_router(image_edit_ep.router, prefix="/retrieval", tags=["retrieval"])
#api_router.include_router(pe_clip_ep.router, prefix="/retrieval", tags=["retrieval"])
#api_router.include_router(mmEmb_ep.router, prefix="/retrieval", tags=["retrieval"])
api_router.include_router(vlm_ep.router, prefix="/retrieval", tags=["retrieval"])
# api_router.include_router(asr_ep.router, prefix="/asr", tags=["asr"])
api_router.include_router(vector_db_ep.router, prefix="/retrieval", tags=["retrieval"])
#api_router.include_router(diffusion_ep.router, prefix="/retrieval", tags=["retrieval"])
