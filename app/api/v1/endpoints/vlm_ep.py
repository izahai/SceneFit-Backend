# app/api/v1/endpoints/vlm_ep.py

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

def _save_upload(image: UploadFile) -> Path:
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{time.time_ns()}{suffix}"
    bg_path = BG_DIR / bg_filename
    with open(bg_path, "wb") as f:
        f.write(image.file.read())
    return bg_path

def _load_clothes_captions() -> dict:
    if CLOTHES_CAPTION.exists():
        with open(CLOTHES_CAPTION, "r", encoding="utf-8") as f:
            return json.load(f)
    return generate_clothes_captions_json()

def _tournament_select(vlm, background_caption: str, clothes_captions: dict) -> str | None:
    if not clothes_captions:
        return None
    candidates = sorted(clothes_captions.items())
    while len(candidates) > 1:
        next_round: list[tuple[str, str]] = []
        for i in range(0, len(candidates), 10):
            batch = candidates[i:i + 10]
            if len(batch) == 1:
                next_round.append(batch[0])
                continue
            best_name = vlm.choose_best_clothes(background_caption, batch)
            best_caption = dict(batch).get(best_name, batch[0][1])
            next_round.append((best_name, best_caption))
        candidates = next_round
    return candidates[0][0]

def _baseline_suggested_clothes(vlm, bg_path: Path):
    descriptions = vlm.generate_clothing_from_image(bg_path)
    clothes_images = load_str_images_from_folder(CLOTHES_DIR)
    clothes = [
        (img_path.stem, Image.open(img_path).convert("RGB"))
        for img_path in clothes_images
    ]
    matcher = ModelRegistry.get("pe_clip_matcher")
    results = matcher.match_clothes(
        descriptions=descriptions,
        clothes=clothes,
        top_k=1,
    )
    return descriptions, (results[0] if results else None)


@router.post("/vlm-txt-suggested-clothes")
def get_suggested_clothes_txt(
    image: UploadFile = File(...),
):
    bg_path = _save_upload(image)

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
    bg_path = _save_upload(image)

    # -------------------------
    # Generate clothing text (VLM)
    # -------------------------
    vlm = ModelRegistry.get("vlm")
    descriptions = vlm.generate_clothing_from_image(bg_path)
    clothes_images = load_str_images_from_folder(CLOTHES_DIR)
    clothes = [
        (img_path.stem, Image.open(img_path).convert("RGB"))
        for img_path in clothes_images
    ]
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
    return _load_clothes_captions()


@router.post("/vlm-tournament-selection")
def vlm_bg_best_clothes(image: UploadFile = File(...)):
    """
    1. Upload image
    2. VLM generates background caption
    3. VLM selects best clothes from captions in batches of 10
    4. Repeat until a single winner remains
    """

    # -------------------------
    # Save uploaded image
    # -------------------------
    bg_path = _save_upload(image)

    # -------------------------
    # Background caption
    # -------------------------
    vlm = ModelRegistry.get("vlm")
    background_caption = vlm.generate_clothes_caption(
        str(bg_path),
        vlm.bg_caption,
    )

    # -------------------------
    # Load clothes captions
    # -------------------------
    clothes_captions = _load_clothes_captions()
    best_clothes = _tournament_select(vlm, background_caption, clothes_captions)
    if best_clothes is None:
        return {
            "background_caption": background_caption,
            "best_clothes": None,
        }
    print(f"[VLM] Best clothes: {best_clothes}", flush=True)

    return {
        "background_caption": background_caption,
        "best_clothes": best_clothes,
    }


@router.post("/vlm-best-clothes-baselines")
def vlm_best_clothes_baselines(image: UploadFile = File(...)):
    """
    Baseline 1: suggested-clothes (VLM descriptions + PE-CLIP top-1)
    Baseline 2: tournament selection (VLM background caption + captions JSON)
    """

    bg_path = _save_upload(image)

    vlm = ModelRegistry.get("vlm")

    # -------------------------
    # Baseline 1: suggested-clothes
    # -------------------------
    descriptions, baseline1 = _baseline_suggested_clothes(vlm, bg_path)

    # -------------------------
    # Baseline 2: tournament selection
    # -------------------------
    background_caption = vlm.generate_clothes_caption(
        str(bg_path),
        vlm.bg_caption,
    )

    clothes_captions = _load_clothes_captions()
    best_clothes = _tournament_select(vlm, background_caption, clothes_captions)

    return {
        "baseline_1": {
            "query": descriptions,
            "result": baseline1,
        },
        "baseline_2": {
            "background_caption": background_caption,
            "best_clothes": best_clothes,
        },
    }
