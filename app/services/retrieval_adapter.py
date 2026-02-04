"""
Example of how to conditionally use mock functions in your API endpoints.

This module provides a clean way to switch between real and mock implementations
based on an environment variable or configuration setting.
"""

import os
from fastapi import UploadFile

# Check environment variable to determine if we should use mock data
USE_MOCK_DATA = os.getenv("SCENEFIT_USE_MOCK", "false").lower() in ("true", "1", "yes")

if USE_MOCK_DATA:
    print("🧪 Using MOCK implementations (no remote API calls)")
    from app.services.all_methods_mock import (
        get_clip_results_mock as get_clip_results,
        get_image_edit_results_mock as get_image_edit_results,
        get_vlm_results_mock as get_vlm_results,
        get_aes_results_mock as get_aes_results,
    )
else:
    print("🌐 Using REAL implementations (making remote API calls)")
    from app.services.all_methods import (
        get_clip_results,
        get_image_edit_results,
        get_vlm_results,
        get_aes_results,
    )


# Export the functions (they'll be either real or mock depending on config)
__all__ = [
    "get_clip_results",
    "get_image_edit_results",
    "get_vlm_results",
    "get_aes_results",
    "USE_MOCK_DATA",
]


def get_all_results(image: UploadFile, top_k: int = 10):
    """
    Get results from all retrieval methods.
    
    Args:
        image: Uploaded image file
        top_k: Number of results to retrieve from each method
    
    Returns:
        Dictionary with results from all methods
    """
    return {
        "clip": get_clip_results(image, top_k),
        "image_edit": get_image_edit_results(image, top_k),
        "vlm": get_vlm_results(image, top_k),
        "aesthetic": get_aes_results(image, top_k),
        "mock_mode": USE_MOCK_DATA
    }


# Example usage in an API endpoint:
"""
from fastapi import FastAPI, UploadFile, File
from app.services.retrieval_adapter import get_clip_results, USE_MOCK_DATA

app = FastAPI()

@app.post("/api/v1/retrieve/clip")
async def retrieve_clip(
    image: UploadFile = File(...),
    top_k: int = 10
):
    results = get_clip_results(image, top_k)
    return {
        "results": results,
        "mock_mode": USE_MOCK_DATA
    }
"""
