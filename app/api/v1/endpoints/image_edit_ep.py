import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Request, Depends

from app.services.img_processor import compose_2d_on_background
from app.services.model_registry import ModelRegistry
from app.services.image_edit import edit_image_scene_img

router = APIRouter()

BG_DIR = Path("app/uploads/bg")
BG_DIR.mkdir(parents=True, exist_ok=True)


def get_vector_db(request: Request):
    return request.app.state.vector_db


@router.post("/image_edit")
def retrieve_clothes_image_edit(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    vector_db = Depends(get_vector_db),
):
    """
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
    print("[image_edit_ep] Editing image via GPT...")
    edit_result = edit_image_scene_img(bg_path, save_result=False)
    edited_image_path = edit_result.get("edited_path", bg_path)

    # -------------------------------------------------
    # 3. Score using PE-Core model
    # -------------------------------------------------
    print("[image_edit_ep] Retrieving best matched clothes via vector DB...")
    scores = vector_db.search_by_image(edited_image_path, top_k=top_k)

    # -------------------------------------------------
    # 4. Top-K
    # -------------------------------------------------
    print("[image_edit_ep] Returning results...")
    return {
        "edited_image_path": str(edited_image_path),
        "count": min(top_k, len(scores)),
        "results": [
            {
                "id": s["id"],
                "score": s["score"],
                "clothes_path": s.get("metadata"),
            }
            for s in scores[:top_k]
        ],
    }
