# app/api/v1/endpoints/pe_clip_ep.py

import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image

from app.services.img_processor import compose_2d_on_background
from app.services.model_registry import ModelRegistry
from app.schemas.basis_sch import RetrievalResponse
from app.utils.util import load_str_images_from_folder

router = APIRouter()

BG_DIR = Path(__file__).resolve().parents[3] / "uploads/bg"
CLOTHES_DIR = Path(__file__).resolve().parents[3] / "data/2d"
BG_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/vlm-txt-suggested-clothes")
def get_suggested_clothes_txt(
    image: UploadFile = File(...),
):
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
    
@router.post("/vlm-suggested-clothes", response_model=RetrievalResponse)
def get_suggested_clothes(image: UploadFile = File(...)):
    """
    1. Upload image
    2. VLM generates clothing descriptions
    3. PE-CLIP ranks clothes by similarity
    """

    # -------------------------
    # Save uploaded image
    # -------------------------
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{uuid.uuid4().hex}{suffix}"
    bg_path = BG_DIR / bg_filename

    with open(bg_path, "wb") as f:
        f.write(image.file.read())

    # -------------------------
    # Generate clothing text (VLM)
    # -------------------------
    vlm = ModelRegistry.get("vlm")
    descriptions = vlm.generate_clothing_from_image(bg_path)

    # -------------------------
    # Load clothes images
    # -------------------------
    clothes_images = load_str_images_from_folder(CLOTHES_DIR)
    clothes = [
        (img_path.stem, Image.open(img_path).convert("RGB"))
        for img_path in clothes_images
    ]

    # -------------------------
    # CLIP matching
    # -------------------------
    matcher = ModelRegistry.get("pe_clip_matcher")
    results = matcher.match_clothes(
        descriptions=descriptions,
        clothes=clothes,
        top_k=10,
    )

    # -------------------------
    # Response
    # -------------------------
    return {
        "query": descriptions,
        "results": results,
    }