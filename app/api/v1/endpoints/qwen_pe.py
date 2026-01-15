from fastapi import APIRouter, UploadFile, File
from PIL import Image

from app.services.model_registry import ModelRegistry
from app.services.compose import compose

router = APIRouter()


@router.post("/retrieve")
def retrieve(image: UploadFile = File(...)):
    model = ModelRegistry.get("qwen_pe")

    # -------------------------------------------------
    # 1. Load background image
    # -------------------------------------------------
    bg_image = Image.open(image.file).convert("RGBA")

    # -------------------------------------------------
    # 2. Scene understanding (Qwen-VL)
    # -------------------------------------------------
    scene = model.parse_scene(bg_image)

    # -------------------------------------------------
    # 3. Generate positive / negative clothing prompts
    # -------------------------------------------------
    pos_text, neg_text = model.generate_pos_neg(scene)

    # -------------------------------------------------
    # 4. Stage-1 Recall (FAISS)
    # -------------------------------------------------
    candidates = model.recall(pos_text)

    # -------------------------------------------------
    # 5. Stage-2 Rerank (NO RENDER)
    # -------------------------------------------------
    ranked = model.fast_rerank(candidates, pos_text, neg_text)

    # -------------------------------------------------
    # 6. Stage-3 Rerank (RENDER + PE-Core)
    # -------------------------------------------------
    results = []
    for meta, _ in ranked[:10]:
        cloth = Image.open(meta["file"]).convert("RGBA")
        composed = compose(bg_image, cloth)

        score = model.pe.score_images(
            [(meta["id"], composed)],
            pos_prompt=pos_text,
            neg_prompt=neg_text,
        )

        results.append(
            {
                "id": meta["id"],
                "score": float(score),
            }
        )

    return {
        "results": results[:5],
        "scene": scene,
        "positive_prompt": pos_text,
        "negative_prompt": neg_text,
    }
