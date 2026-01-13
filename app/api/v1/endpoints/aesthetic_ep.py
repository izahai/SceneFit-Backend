from fastapi import APIRouter, UploadFile, File, Form
import uuid
from pathlib import Path
from app.services.img_processor import compose_2d_on_background
from app.services.model_registry import ModelRegistry

router = APIRouter()

BG_DIR = Path("app/uploads/bg")
BG_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/aesthetic")
def retrieve_best_fit_aesthetic(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{uuid.uuid4().hex}{suffix}"
    bg_path = BG_DIR / bg_filename

    with open(bg_path, "wb") as f:
        f.write(image.file.read())

    items = compose_2d_on_background(
        bg_path=bg_path,
        fg_dir="app/data/2d",
        return_format="pil",
    )

    model = ModelRegistry.get("aesthetic")
    scores = model.score_images(items)

    return {
        "count": min(top_k, len(scores)),
        "results": scores[:top_k],
    }