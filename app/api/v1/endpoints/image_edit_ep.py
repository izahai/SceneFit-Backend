import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form

from app.services.img_processor import compose_2d_on_background
from app.services.model_registry import ModelRegistry
from app.services.image_edit import edit_image_scene_img

router = APIRouter()

BG_DIR = Path("app/uploads/bg")
BG_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/image_edit")
def retrieve_clothes_image_edit(
    image: UploadFile = File(...),
    top_k: int = Form(5),
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
    # 2. Get GPT edited images
    # -------------------------------------------------
    edit_image_scene_img(bg_path, save_result=False)

    # -------------------------------------------------
    # 3. Score using PE-Core model
    # -------------------------------------------------
    model = ModelRegistry.get("pe")
    scores = model.score_images(items)

    # -------------------------------------------------
    # 4. Top-K
    # -------------------------------------------------
    return {
        "count": min(top_k, len(scores)),
        "results": scores[:top_k],
    }
