import asyncio
import io
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from app.api.v1.endpoints.pe_clip_ep import retrieve_scene_outfit
from app.services.all_methods import *

router = APIRouter()

@router.post("/all-methods")
async def retrieve_all_methods(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Endpoint that combines multiple retrieval methods for demonstration purposes.
    Returns partial results even if some services fail.
    
    {
        "<method_name>": [
            {"name": str, "score": float},
        ] | {"error": true, "message": str} (if service failed),
    }
    """
    
    # Read file content once to avoid conflicts in parallel execution
    image_content = await image.read()
    filename = image.filename
    content_type = image.content_type
    
    # Execute all retrieval methods in parallel with error handling
    image_edit_results, vlm_results, clip_results, aes_results = await asyncio.gather(
        asyncio.to_thread(get_image_edit_results, image_content, filename, content_type, top_k, False),
        asyncio.to_thread(get_vlm_results, image_content, filename, content_type, top_k, False),
        asyncio.to_thread(get_clip_results, image_content, filename, content_type, top_k, False),
        asyncio.to_thread(get_aes_results, image_content, filename, content_type, top_k, False),
        return_exceptions=True
    )
    
    response = {
        "imageEdit": image_edit_results,
        "vlm": vlm_results,
        "clip": clip_results,
        "aesthetic": aes_results,
    }
    
    # Count how many services succeeded vs failed
    success_count = sum(1 for v in response.values() if not isinstance(v, dict) or not v.get("error"))
    failed_count = 4 - success_count
    print(f"[ALL-METHODS] {success_count}/4 services succeeded, {failed_count} failed")
    
    return response
    
@router.post("/clip")
async def retrieve_clip_method(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Endpoint for CLIP-based retrieval method.
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    image_content = await image.read()
    return get_clip_results(image_content, image.filename, image.content_type, top_k, mock=False)

@router.post("/image-edit")
async def retrieve_image_edit_method(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Endpoint for Image Edit-based retrieval method.
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    image_content = await image.read()
    return get_image_edit_results(image_content, image.filename, image.content_type, top_k, mock=False)

@router.post("/vlm")
async def retrieve_vlm_method(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Endpoint for VLM-based retrieval method.
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    image_content = await image.read()
    return get_vlm_results(image_content, image.filename, image.content_type, top_k, mock=False)

@router.post("/aesthetic")
async def retrieve_aesthetic_method(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Endpoint for Aesthetic-based retrieval method.
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    image_content = await image.read()
    return get_aes_results(image_content, image.filename, image.content_type, top_k, mock=False)