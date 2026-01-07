# app/api/v1/endpoints/pe_clip_ep.py

import json
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from PIL import Image
import time

from app.services.model_registry import ModelRegistry
from app.utils.util import load_str_images_from_folder
from app.services.clothes_captions import generate_clothes_captions_json

router = APIRouter()

BG_DIR = Path("app/uploads/bg")
CLOTHES_DIR = Path("app/data/2d")
CLOTHES_CAPTION = Path("app/data/clothes_captions.json")
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
    
@router.post("/vlm-suggested-clothes")
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
    bg_filename = f"{time.time_ns().hex}{suffix}"
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
    
@router.get("/vlm-clothes-captions")
def vlm_clothes_captions():
    """
    Return clothes captions JSON.
    If it doesn't exist, generate it first.
    """

    # -------------------------
    # Load cached JSON if exists
    # -------------------------
    if CLOTHES_CAPTION.exists():
        with open(CLOTHES_CAPTION, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------
    # Generate + save JSON
    # -------------------------
    data = generate_clothes_captions_json(CLOTHES_CAPTION)

    return data