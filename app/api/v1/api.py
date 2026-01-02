# app/api/v1/api.py

from fastapi import APIRouter
from app.api.v1.endpoints import mmEmb_ep
from app.api.v1.endpoints import pe_clip_ep

api_router = APIRouter()
#api_router.include_router(mmEmb_ep.router, prefix="/mmEmb", tags=["mmEmb"])
api_router.include_router(pe_clip_ep.router, prefix="/pe", tags=["pe"])