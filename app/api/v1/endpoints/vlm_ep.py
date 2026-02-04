# app/api/v1/endpoints/vlm_ep.py

import json
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Body
from PIL import Image
import time
import torch
import torch.nn.functional as F
import gc

from app.services.model_registry import ModelRegistry
from app.utils.util import load_str_images_from_folder
from app.services.clothes_captions import generate_clothes_captions_json
from app.api.v1.endpoints.aesthetic_ep import score_outfits
router = APIRouter()

BG_DIR = Path("app/data/bg")
CLOTHES_DIR = Path("app/data/2d")
CLOTHES_CAPTION = Path("app/data/clothes_captions.json")
RESULTS_DIR = Path("app/uploads/results")
BG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _save_bg_upload(image: UploadFile) -> Path:
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{time.time_ns()}{suffix}"
    bg_path = BG_DIR / bg_filename
    with open(bg_path, "wb") as f:
        f.write(image.file.read())
    return bg_path

def _get_clothes_captions() -> dict:
    if CLOTHES_CAPTION.exists():
        with open(CLOTHES_CAPTION, "r", encoding="utf-8") as f:
            return json.load(f)
    return generate_clothes_captions_json()


def _write_result(payload: dict) -> Path:
    filename = f"{time.time_ns()}_{uuid.uuid4().hex}.json"
    output_path = RESULTS_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path

def _select_clothes_via_tournament(vlm, background_caption: str, clothes_captions: dict) -> str | None:
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

def _method_image_match(vlm, bg_path: Path):
    descriptions = _generate_clothes_descriptions(vlm, bg_path)
    results = _rank_clothes_by_image(descriptions, top_k=1)
    return descriptions, (results[0] if results else None)

def _method_caption_match(vlm, descriptions):
    results = _rank_clothes_by_caption(descriptions, top_k=1)
    return descriptions, (results[0] if results else None)

def _generate_clothes_descriptions(vlm, bg_path: Path) -> list[str]:
    return vlm.generate_clothing_from_image(bg_path)

def _rank_clothes_by_image(descriptions: list[str], top_k: int = 10):
    clothes_images = load_str_images_from_folder(CLOTHES_DIR)
    clothes = [
        (img_path.stem, Image.open(img_path).convert("RGB"))
        for img_path in clothes_images
    ]
    matcher = ModelRegistry.get("pe_clip_matcher")
    return matcher.match_clothes(
        descriptions=descriptions,
        query_emb=None,
        clothes=clothes,
        top_k=top_k,
    )

def _rank_clothes_by_caption(descriptions: list[str],
                            matcher_name: str = "text_matcher",
                            top_k: int = 10):
    clothes_captions = _get_clothes_captions()
    matcher = ModelRegistry.get(matcher_name)
    return matcher.match_clothes_captions(
        descriptions=descriptions,
        clothes_captions=clothes_captions,
        top_k=top_k,
    )

def _build_query_embedding(
    color_outfits: list[str],
    scene_caption: str,
    bg_image_emb: torch.Tensor,
    matcher,
    n_good: int = 7,
    weights=(0.40, 0.4, 0.2),
    domain_suffix: str = ", 3D rendered character model, game asset style"
):
    w_semantic, w_scene, w_img = weights

    color_outfits = [desc + domain_suffix for desc in color_outfits]

    # Encode
    semantic_emb = matcher.encode_text(color_outfits)   # (10, D)
    scene_emb = matcher.encode_text([scene_caption]) # (1, D)

    good_emb = semantic_emb[:n_good]      # (7, D)
    bad_emb  = semantic_emb[n_good:]      # (3, D)

    bad_vec = bad_emb.mean(dim=0, keepdim=True)
    bad_unit = F.normalize(bad_vec, p=2, dim=-1) # Hat b
        
        # 3. Orthogonal Rejection: Remove 'bad' component from 'good'
        # Projection = (good @ bad.T) * bad
    projection = (good_emb @ bad_unit.T) * bad_unit
    clean_good = good_emb - projection # Strictly orthogonal now

    # Build 7 contrastive queries (diverse)
    queries = (
        w_semantic * clean_good +
        w_scene * scene_emb +
        w_img * bg_image_emb
    )

    return F.normalize(queries, dim=-1)  # (7, D)


def _rank_clothes_by_feedback(descriptions: list[str],
                            matcher_name: str = "text_matcher",
                            top_k: int = 10,
                            fb_text: str | None = None):
    clothes_captions = _get_clothes_captions()
    matcher = ModelRegistry.get(matcher_name)
    return matcher.match_clothes_captions(
        descriptions=descriptions,
        clothes_captions=clothes_captions,
        fb_text=fb_text,
        top_k=top_k,
    )

@router.post("/vlm-generated-clothes-captions")
def get_vlm_descriptions(
    image: UploadFile = File(...),
):
    bg_path = _save_bg_upload(image)

    model = ModelRegistry.get("vlm")
    res = model.generate_clothing_from_image(bg_path)

    return {
        "res": res,
    }
    
@router.post("/vlm-clip-images-matching")
def get_clothes_by_image_match(image: UploadFile = File(...)):
    """
    1. Upload image
    2. VLM generates clothing descriptions
    3. PE-CLIP ranks clothes by similarity
    """

    # -------------------------
    # Save uploaded image
    # -------------------------
    bg_path = _save_bg_upload(image)

    # -------------------------
    # Generate clothing text (VLM)
    # -------------------------
    vlm = ModelRegistry.get("vlm")
    descriptions = _generate_clothes_descriptions(vlm, bg_path)
    results = _rank_clothes_by_image(descriptions, top_k=10)

    # -------------------------
    # Response
    # -------------------------
    return {
        "query": descriptions,
        "results": results,
    }

@router.post("/vlm-clip-caption-matching")
def get_clothes_by_image_match_captions(image: UploadFile = File(...)):
    """
    1. Upload image
    2. VLM generates clothing descriptions
    3. Matcher ranks clothes by similarity using captions JSON
    """

    # -------------------------
    # Save uploaded image
    # -------------------------
    bg_path = _save_bg_upload(image)

    # -------------------------
    # Generate clothing text (VLM)
    # -------------------------
    vlm = ModelRegistry.get("vlm")
    descriptions = _generate_clothes_descriptions(vlm, bg_path)

    # -------------------------
    # Load clothes captions + match
    # -------------------------
    results = _rank_clothes_by_caption(descriptions, top_k=10)

    # -------------------------
    # Response
    # -------------------------
    return {
        "query": descriptions,
        "results": results,
    }
    
@router.get("/clothes-captions")
def get_clothes_captions():
    """
    Return clothes captions JSON.
    If it doesn't exist, generate it first.
    """

    # -------------------------
    # Load cached JSON if exists
    # -------------------------
    return _get_clothes_captions()


@router.post("/vlm-tournament-selection")
def get_best_clothes_by_tournament(image: UploadFile = File(...)):
    """
    1. Upload image
    2. VLM generates background caption
    3. VLM selects best clothes from captions in batches of 10
    4. Repeat until a single winner remains
    """

    # -------------------------
    # Save uploaded image
    # -------------------------
    bg_path = _save_bg_upload(image)

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
    clothes_captions = _get_clothes_captions()
    best_clothes = _select_clothes_via_tournament(vlm, background_caption, clothes_captions)
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


@router.post("/all-methods")
def get_clothes_all_methods(image: UploadFile = File(...)):
    """
    Baseline 1: suggested-clothes (VLM descriptions + PE-CLIP top-1)
    Baseline 2: tournament selection (VLM background caption + captions JSON)
    Baseline 3: captions matching (VLM descriptions + captions JSON)
    """

    bg_path = _save_bg_upload(image)

    vlm = ModelRegistry.get("vlm")

    # -------------------------
    # Baseline 1: suggest-clothes-img-matching
    # -------------------------
    descriptions, res1 = _method_image_match(vlm, bg_path)
    
    # -------------------------
    # Baseline 2: suggest-clothes-img-matching
    # -------------------------
    descriptions, res2 = _method_caption_match(vlm, descriptions)

    # -------------------------
    # Baseline 3: tournament selection
    # -------------------------
    background_caption = vlm.generate_clothes_caption(
        str(bg_path),
        vlm.bg_caption,
    )
    clothes_captions = _get_clothes_captions()
    res3 = _select_clothes_via_tournament(vlm, background_caption, clothes_captions)
    res3 = {"name_clothes": res3, "similarity": 0, "best_description": "",}

    # -------------------------
    # Baseline 4: Aesthetic Predictor
    # -------------------------
    aes_model = ModelRegistry.get("aesthetic")
    aes_pred = score_outfits(aes_model, bg_path)
    res4 = aes_pred["results"][0]

    response_payload = {
        "img_name": image.filename,
        "approach_1": {
            "bg_caption": "",
            "query": descriptions,
            "result": res1,
        },
        "approach_2": {
            "bg_caption": "",
            "query": descriptions,
            "result": res2,
        },
        "approach_3": {
            "bg_caption": background_caption,
            "query": [],
            "result": res3
        },
        "approach_4": {
            "bg_caption": "",
            "query": [],
            "result": res4
        },
    }

@router.post("/vlm-faiss-composed-retrieval")
def composed_retrieval(image: UploadFile = File(...), top_k: int = 10):
    bg_path = _save_bg_upload(image)

    vlm = ModelRegistry.get("vlm")
    

    # -------------------------
    # Extract query signals
    # -------------------------
    signals = vlm.extract_query_signals(str(bg_path))
    print("Got 0")
    ModelRegistry.release("vlm")
    print("Got 1")
    matcher = ModelRegistry.get("pe_clip_matcher")
    print("Got 2")
    # -------------------------
    # Background image embedding
    # -------------------------
    bg_img = Image.open(bg_path).convert("RGB")
    bg_emb = matcher.encode_image([bg_img])  # (1, D)

    # -------------------------
    # Build fused query
    # -------------------------
    query_emb = _build_query_embedding(
        signals["color_outfits"],
        signals["scene_caption"],
        bg_emb,
        matcher,
    )
    print("Query Embedding Shape: ",query_emb.shape)
    
    print("Got 3")

    # -------------------------
    # FAISS retrieval (coarse)  
    # -------------------------
    candidates = matcher.match_clothes(
        query_emb=query_emb,
        top_k=top_k,
    )
    ModelRegistry.release("pe_clip_matcher")
    print("Got 4")
    reranker = ModelRegistry.get("qwen_reranker")
    print("Got 5")
    # Attach captions + images for reranker
    for c in candidates:
        c["image"] = Image.open(
            Path("app/data/2d") / f"{c['outfit_name']}"
        ).convert("RGB")
        #c["caption"] = c.get("caption", "")

    # -------------------------
    # Qwen3-VL reranking (fine)
    # -------------------------
    reranked = reranker.rerank(
        query_text=signals["scene_caption"],
        candidates=candidates,
    )
    ModelRegistry.release("qwen_reranker")
    # REMOVE non-serializable fields
    for c in reranked:
        c.pop("image", None)
    
    return {
        "method": "vlm-faiss-composed-retrieval",
        "count": len(reranked),
        "scene_caption": signals["scene_caption"],
        "results": reranked,
        "best": reranked[0] if reranked else None,
    }


@router.post("/vlm-caption-feedback")
def get_clothes_by_feedback(payload: dict = Body(...)):

    descriptions = payload["descriptions"]
    fb_text = payload.get("fb_text")

    results = _rank_clothes_by_feedback(
        descriptions=descriptions,
        top_k=10,
        fb_text=fb_text,
    )

    return {
        "query": descriptions,
        "feedback": fb_text,
        "results": results,
    }
