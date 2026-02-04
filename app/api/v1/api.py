# app/api/v1/api.py

from fastapi import APIRouter
from app.api.v1.endpoints import aesthetic_ep


api_router = APIRouter()

api_router.include_router(aesthetic_ep.router, prefix="/retrieval", tags=["retrieval"])
