"""
Safe wrapper functions for all retrieval methods.
These functions catch exceptions and return error objects instead of raising.
"""
from app.services.all_methods import (
    get_clip_results,
    get_image_edit_results,
    get_vlm_results,
    get_aes_results,
)


def get_clip_results_safe(image_content: bytes, filename: str, content_type: str, top_k: int, mock=True):
    """
    Safe version that returns error object on failure instead of raising.
    """
    try:
        return get_clip_results(image_content, filename, content_type, top_k, mock)
    except Exception as e:
        error_msg = str(e)
        print(f"[CLIP] Service error: {error_msg[:100]}")
        return {"error": True, "message": error_msg}


def get_image_edit_results_safe(image_content: bytes, filename: str, content_type: str, top_k: int, mock=True):
    """
    Safe version that returns error object on failure instead of raising.
    """
    try:
        return get_image_edit_results(image_content, filename, content_type, top_k, mock)
    except Exception as e:
        error_msg = str(e)
        print(f"[IMAGE_EDIT] Service error: {error_msg[:100]}")
        return {"error": True, "message": error_msg}


def get_vlm_results_safe(image_content: bytes, filename: str, content_type: str, top_k: int, mock=True):
    """
    Safe version that returns error object on failure instead of raising.
    """
    try:
        return get_vlm_results(image_content, filename, content_type, top_k, mock)
    except Exception as e:
        error_msg = str(e)
        print(f"[VLM] Service error: {error_msg[:100]}")
        return {"error": True, "message": error_msg}


def get_aes_results_safe(image_content: bytes, filename: str, content_type: str, top_k: int, mock=True):
    """
    Safe version that returns error object on failure instead of raising.
    """
    try:
        return get_aes_results(image_content, filename, content_type, top_k, mock)
    except Exception as e:
        error_msg = str(e)
        print(f"[AESTHETIC] Service error: {error_msg[:100]}")
        return {"error": True, "message": error_msg}
