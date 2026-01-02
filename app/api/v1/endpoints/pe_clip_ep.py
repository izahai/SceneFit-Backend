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


@router.post("/pe", response_model=RetrievalResponse)
def retrieve_best_matched_figures_pe(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    scale: float = Form(1.0),
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

    # -------------------------------------------------
    # 2. Compose candidate images
    # -------------------------------------------------
    images = compose_2d_on_background(
        bg_path=bg_path,
        fg_dir="app/clothes/2d",
        scale=scale,
        return_format="pil",
    )

    # -------------------------------------------------
    # 3. Score using PE-Core model
    # -------------------------------------------------
    model = ModelRegistry.get("pe_clip")
    scores = model.score_images(images)

    # -------------------------------------------------
    # 4. Top-K
    # -------------------------------------------------
    return {
        "count": min(top_k, len(scores)),
        "results": scores[:top_k],
    }
