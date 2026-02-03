from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from app.api.v1.endpoints.pe_clip_ep import retrieve_scene_outfit
from app.services.all_methods import *

router = APIRouter()

@router.post("/all-methods")
def retrieve_all_methods(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Endpoint that combines multiple retrieval methods for demonstration purposes.
    
    {
        "<method_name>": [
            {"name": str, "score": float},
        ],
    }
    """
    
    image_edit_results = get_image_edit_results(image, top_k)
    vlm_results = get_vlm_results(image, top_k)
    clip_results = get_clip_results(image, top_k)
    aes_results = get_aes_results(image, top_k)
    
    return {
        "image_edit": image_edit_results,
        "vlm": vlm_results,
        "clip": clip_results,
        "aesthetic": aes_results,
    }
    
@router.post("/clip")
def retrieve_clip_method(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Endpoint for CLIP-based retrieval method.
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    return get_clip_results(image, top_k)

@router.post("/image-edit")
def retrieve_image_edit_method(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Endpoint for Image Edit-based retrieval method.
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    return get_image_edit_results(image, top_k)

@router.post("/vlm")
def retrieve_vlm_method(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Endpoint for VLM-based retrieval method.
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    return get_vlm_results(image, top_k)

@router.post("/aesthetic")
def retrieve_aesthetic_method(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Endpoint for Aesthetic-based retrieval method.
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    return get_aes_results(image, top_k)