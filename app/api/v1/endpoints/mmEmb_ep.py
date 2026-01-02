# app/api/v1/endpoints/clip_ep.py

from fastapi import APIRouter
from fastapi import UploadFile, File, Form
from app.schemas.basis_sch import RetrievalResponse
from app.services.model_registry import ModelRegistry

router = APIRouter()

@router.post("/mmEmb", response_model=RetrievalResponse)
def retrieve_by_mmEmb(image: UploadFile = File(...)):
    model = ModelRegistry.get("mmEmb")
    return model.retrieve(image.file)