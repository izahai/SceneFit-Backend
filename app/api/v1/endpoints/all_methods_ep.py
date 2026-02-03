from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from app.api.v1.endpoints.pe_clip_ep import retrieve_scene_outfit
from app.services.all_methods import *

router = APIRouter()

@router.post("/all-methods", tags=["all_methods"])
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