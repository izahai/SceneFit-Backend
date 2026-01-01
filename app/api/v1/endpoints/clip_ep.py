from fastapi import APIRouter
from app.schemas.clip_model import ClipRequest
from app.services.model_registry import ModelRegistry

router = APIRouter()

@router.post("/encode")
def encode(req: ClipRequest):
    model = ModelRegistry.get("clip")
    return model.infer(req.text)