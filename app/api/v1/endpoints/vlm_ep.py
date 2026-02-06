# app/api/v1/endpoints/vlm_ep.py

import json
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Body, Form
from PIL import Image
import time
import torch
import torch.nn.functional as F
import gc
import numpy as np
from app.services.model_registry import ModelRegistry
from app.utils.util import load_str_images_from_folder
from app.services.clothes_captions import generate_clothes_captions_json
from app.api.v1.endpoints.aesthetic_ep import score_outfits
from app.utils.util import convert_filename_to_url
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
    print(color_outfits)
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
def _diversify_with_mmr(
    candidates: list[dict],
    faiss_meta: dict,
    lambda_param: float = 0.5,
    top_k: int = 10
) -> list[dict]:
    """
    Select diverse candidates using Maximal Marginal Relevance.
    Uses pre-computed embeddings from FAISS metadata.
    
    lambda_param: 0 = max diversity, 1 = max relevance
    """
    if len(candidates) <= top_k:
        return candidates
    
    # Get pre-computed embeddings for candidates
    filename_to_idx = {name: idx for idx, name in enumerate(faiss_meta["filenames"])}
    all_embeddings = torch.from_numpy(faiss_meta["embeddings"]).float()
    
    candidate_embs = []
    for c in candidates:
        idx = filename_to_idx[c["outfit_name"]]
        candidate_embs.append(all_embeddings[idx])
    
    candidate_embs = torch.stack(candidate_embs)  # (N, D)
    
    selected = []
    selected_embs = []
    remaining_indices = list(range(len(candidates)))
    
    # Select first item (highest relevance)
    first_idx = 0
    selected.append(candidates[first_idx])
    selected_embs.append(candidate_embs[first_idx])
    remaining_indices.remove(first_idx)
    
    # Iteratively select diverse items
    while len(selected) < top_k and remaining_indices:
        best_score = -float('inf')
        best_idx = None
        
        for idx in remaining_indices:
            # Relevance to query (already computed by FAISS)
            relevance = candidates[idx]["similarity"]
            
            # Max similarity to already selected items
            if selected_embs:
                selected_tensor = torch.stack(selected_embs)
                # Cosine similarity (embeddings are already normalized)
                similarity_to_selected = (candidate_embs[idx] @ selected_tensor.T).max().item()
                diversity = 1 - similarity_to_selected
            else:
                diversity = 1.0
            
            # MMR score
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        selected.append(candidates[best_idx])
        selected_embs.append(candidate_embs[best_idx])
        remaining_indices.remove(best_idx)
    
    return selected
from sklearn.cluster import KMeans

def _diversify_with_clustering(
    candidates: list[dict],
    faiss_meta: dict,
    n_clusters: int = 5,
    samples_per_cluster: int = 2
) -> list[dict]:
    """
    Cluster candidates and sample from each cluster.
    Uses pre-computed embeddings.
    """
    if len(candidates) <= n_clusters:
        return candidates
    
    # Get pre-computed embeddings
    filename_to_idx = {name: idx for idx, name in enumerate(faiss_meta["filenames"])}
    all_embeddings = faiss_meta["embeddings"]
    
    candidate_embs = np.array([
        all_embeddings[filename_to_idx[c["outfit_name"]]]
        for c in candidates
    ])
    
    # Cluster
    kmeans = KMeans(n_clusters=min(n_clusters, len(candidates)), random_state=42)
    cluster_labels = kmeans.fit_predict(candidate_embs)
    
    # Sample from each cluster
    diverse_results = []
    for cluster_id in range(kmeans.n_clusters):
        cluster_candidates = [
            candidates[i] for i in range(len(candidates))
            if cluster_labels[i] == cluster_id
        ]
        # Sort by relevance within cluster
        cluster_candidates.sort(key=lambda x: x["similarity"], reverse=True)
        diverse_results.extend(cluster_candidates[:samples_per_cluster])
    
    # Sort final results by relevance
    diverse_results.sort(key=lambda x: x["similarity"], reverse=True)
    return diverse_results
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
@router.post("/vlm-faiss-composed-retrieval")
def composed_retrieval(
    image: UploadFile = File(...), 
    top_k: int = 10,
    diversity_method: str = "mmr",  # "mmr", "cluster", or "none"
    lambda_param: float = 0.6
):
    bg_path = _save_bg_upload(image)
    vlm = ModelRegistry.get("vlm")
    
    signals = vlm.extract_query_signals(str(bg_path))
    print("Got 0")
    ModelRegistry.release("vlm")
    print("Got 1")
    matcher = ModelRegistry.get("pe_clip_matcher")
    print("Got 2")
    bg_img = Image.open(bg_path).convert("RGB")
    bg_emb = matcher.encode_image([bg_img])
    
    query_emb = _build_query_embedding(
        signals["color_outfits"],
        signals["scene_caption"],
        bg_emb,
        matcher,
    )
    
    # Retrieve MORE candidates than needed (2-3x)
    initial_k = top_k * 3
    candidates = matcher.match_clothes(
        query_emb=query_emb,
        top_k=initial_k,
    )
    
    # Apply diversity strategy (NO IMAGE LOADING NEEDED!)
    if diversity_method == "mmr":
        candidates = _diversify_with_mmr(
            candidates, 
            matcher.faiss_meta,  # ← Pass metadata with embeddings
            lambda_param, 
            top_k
        )
    elif diversity_method == "cluster":
        candidates = _diversify_with_clustering(
            candidates, 
            matcher.faiss_meta,
            n_clusters=5, 
            samples_per_cluster=2
        )
    else:
        candidates = candidates[:top_k]
    print("Got 3")
    
    ModelRegistry.release("pe_clip_matcher")
    
    # NOW load images only for reranking
    print("Got 4")
    reranker = ModelRegistry.get("qwen_reranker")
    print("Got 5")
    for c in candidates:
        c["image"] = Image.open(
            Path("app/data/2d") / f"{c['outfit_name']}"
        ).convert("RGB")
    
    reranked = reranker.rerank(
        query_text=signals["scene_caption"],
        candidates=candidates,
    )
    ModelRegistry.release("qwen_reranker")
    
    # Format results
    for c in reranked:
        c.pop("image", None)
    
    formatted_results = [
        {
            "name": c["outfit_name"],
            "score": float(c["score"]),
            "image_url": convert_filename_to_url(c["outfit_name"]),
        }
        for c in reranked
    ]
    
    return formatted_results


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


@router.post("/vlm-suggest-outfit")
def suggest_outfit(
    bg_image: UploadFile = File(...),
    preference_text: str | None = Form(None),
    feedback_text: str | None = Form(None)
):
    bg_path = _save_bg_upload(bg_image)

    vlm = ModelRegistry.get("vlm")

    outfit_desc = vlm.suggest_outfit_from_bg(str(bg_path), preference_text=preference_text, feedback_text=feedback_text)
    print(f"[vlm_ep] Outfit suggestion: {outfit_desc}")

    return {
        "bg_image": bg_image.filename,
        "outfit_description": outfit_desc,
    }