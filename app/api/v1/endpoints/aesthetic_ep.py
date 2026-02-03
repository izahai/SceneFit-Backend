from fastapi import APIRouter, UploadFile, File, Form
import uuid
from pathlib import Path
from app.services.img_processor import compose_2d_on_background
from app.services.model_registry import ModelRegistry
import time

router = APIRouter()

BG_DIR = Path("app/uploads/bg")
BG_DIR.mkdir(parents=True, exist_ok=True)

def _save_bg_upload(image: UploadFile) -> Path:
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{time.time_ns()}{suffix}"
    bg_path = BG_DIR / bg_filename
    
    if not bg_path.exists():
        with open(bg_path, "wb") as f:
            f.write(image.file.read())
    return bg_path

def score_outfits(rg_head, bg_path: Path, top_k: int = 5):
    items = compose_2d_on_background(
        bg_path=bg_path,
        fg_dir="app/data/2d",
        return_format="pil",
    )
    print(f"[AESTHETIC] Scoring {len(items)} outfits ...")
    scores = rg_head.score_images(items)

    return {
        "count": min(top_k, len(scores)),
        "results": scores[:top_k],
    }

@router.post("/aesthetic")
def retrieve_best_fit_aesthetic(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    bg_path = _save_bg_upload(image)
    model = ModelRegistry.get("aesthetic")
    results = score_outfits(model, bg_path, top_k)
    return results["results"]
