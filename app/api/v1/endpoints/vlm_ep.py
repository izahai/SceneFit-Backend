# app/api/v1/endpoints/pe_clip_ep.py

import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form

from app.services.img_processor import compose_2d_on_background
from app.services.model_registry import ModelRegistry
from app.schemas.basis_sch import RetrievalResponse

router = APIRouter()

BG_DIR = Path("app/uploads/bg")
BG_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/vlm-txt-suggested-clothes")
def get_suggested_clothes_txt(
    image: UploadFile = File(...),
):
    """
    PE-Core version of image retrieval.
    """
    # -------------------------------------------------
    # 1. Save uploaded background
    # -------------------------------------------------
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{uuid.uuid4().hex}{suffix}"
    bg_path = BG_DIR / bg_filename

    with open(bg_path, "wb") as f:
        f.write(image.file.read())

    model = ModelRegistry.get("vlm")
    res = model.generate_clothing_from_image(bg_path)

    return {
        "res": res,
    }
