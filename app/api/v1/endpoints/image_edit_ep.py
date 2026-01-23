import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Request, Depends

from app.services.image_edit import edit_image_scene_img

router = APIRouter()

BG_DIR = Path("app/uploads/bg")
BG_DIR.mkdir(parents=True, exist_ok=True)
ALL_BG_DIR = Path("app/data/bg")
RETRIEVAL_RESULTS_DIR = Path("app/retrieval_results/image_edit")

def get_vector_db(request: Request):
    return request.app.state.vector_db

def _extract_outfit_name(meta: str | None) -> str | None:
    if not meta:
        return None
    return Path(meta).stem  # filename without extension
    
@router.post("/image-edit")
def retrieve_clothes_image_edit(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    gender: str = Form("male"),
    crop_clothes: bool = Form(True),
    return_metadata: bool = Form(True),
    vector_db = Depends(get_vector_db),
):
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
    edit_result = edit_image_scene_img(bg_path, save_result=False, gender=gender, crop_clothes=crop_clothes)
    processed_image_path = edit_result.get("cropped_path") if crop_clothes else edit_result.get("edited_path")
    print(f"[image_edit_ep] Processed image saved to: {processed_image_path}")

    # -------------------------------------------------
    # 3. Score using PE-Core model
    # -------------------------------------------------
    print("[image_edit_ep] Retrieving best matched clothes via vector DB...")
    scores = vector_db.search_by_image(processed_image_path, top_k=top_k)



    response = {
        "method": "image-edit",
        "gender": gender,
        "edited_image_path": str(processed_image_path),
        "count": min(top_k, len(scores)),
        "results": [
            {
                "outfit_name": _extract_outfit_name(s.get("metadata")),
                "score": s["score"],
                "clothes_path": s.get("metadata"),
            }
            for s in scores[:top_k]
        ],
    }

    if not return_metadata:
        for item in response["results"]:
            item.pop("clothes_path", None)
    print("[image_edit_ep] Returning results...")
    return response

@router.post("/image-edit/retrieve-all")
def retrieve_all_backgrounds(
    top_k: int = Form(5),
    crop_clothes: bool = Form(True),
    return_metadata: bool = Form(True),
    vector_db = Depends(get_vector_db),
):
    if not RETRIEVAL_RESULTS_DIR.exists():
        os.makedirs(RETRIEVAL_RESULTS_DIR)

    results = []
    for bg_file in ALL_BG_DIR.glob("*"):
        for gender in ['male', 'female']:
            # ------------------------------
            # 1. Edit image via GPT
            # ------------------------------

            print(f"[image_edit_ep] Editing image via GPT for background: {bg_file} with gender: {gender}...")
            edit_result = edit_image_scene_img(bg_file, save_result=False, gender=gender, crop_clothes=crop_clothes)
            processed_image_path = edit_result.get("cropped_path") if crop_clothes else edit_result.get("edited_path")
            print(f"[image_edit_ep] Processed image saved to: {processed_image_path}")

            # ------------------------------
            # 2. Score using PE-Core model
            # ------------------------------

            print("[image_edit_ep] Retrieving best matched clothes via vector DB...")
            scores = vector_db.search_by_image(processed_image_path, top_k=top_k)

            # ------------------------------
            # 3. Top-K
            # ------------------------------
            print("[image_edit_ep] Returning results...")
            entry = {
                "method": "image-edit",
                "gender": gender,
                "bg_path": str(bg_file),
                "edited_image_path": str(processed_image_path),
                "count": min(top_k, len(scores)),
                "results": [
                    {
                        "outfit_name": _extract_outfit_name(s.get("metadata")),
                        "score": s["score"],
                        "clothes_path": s.get("metadata"),
                    }
                    for s in scores[:top_k]
                ],
            }

            if not return_metadata:
                for item in entry["results"]:
                    item.pop("clothes_path", None)

            results.append(entry)

    with open(RETRIEVAL_RESULTS_DIR / "retrieval_results.json", "w") as f:
        import json
        json.dump(results, f, indent=2)

    return results