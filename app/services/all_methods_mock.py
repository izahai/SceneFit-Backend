"""
Mock functions for testing without making remote API calls.
Returns mock data with outfit names matching those in app/data/data/2d/
"""

import random
from pathlib import Path
from fastapi import UploadFile


# Cache the list of outfit names from the 2d directory
_OUTFIT_CACHE = None


def _get_outfit_names():
    """
    Get list of outfit names from the app/data/data/2d directory.
    Caches the result for performance.
    """
    global _OUTFIT_CACHE
    
    if _OUTFIT_CACHE is None:
        data_dir = Path(__file__).parent.parent / "data" / "data" / "2d"
        
        if data_dir.exists():
            # Get all .png files and remove the extension
            _OUTFIT_CACHE = [
                f.stem for f in data_dir.glob("*.png")
                if f.is_file() and not f.name.startswith('.')
            ]
        else:
            # Fallback outfit names if directory doesn't exist
            _OUTFIT_CACHE = [
                "avatars_00320df985504a278f628f6b168d2495",
                "avatars_00b29fcff3164e5ea6ee6b7e4da87fee",
                "m1_brown_10", "m1_brown_4", "m1_dark_1", 
                "m1_light_5", "m5_brown_3", "m5_dark_7",
                "m5_light_2", "m6_brown_6", "m6_dark_4", "m6_light_8"
            ]
    
    return _OUTFIT_CACHE


def _generate_mock_results(top_k: int, method_name: str):
    """
    Generate mock results with random outfit names and scores.
    
    Args:
        top_k: Number of results to generate
        method_name: Name of the method (used for score variation)
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    outfit_names = _get_outfit_names()
    
    # Randomly select outfits without replacement (if possible)
    num_outfits = min(top_k, len(outfit_names))
    selected_outfits = random.sample(outfit_names, num_outfits)
    
    # Generate scores (higher scores first, decreasing)
    # Different methods have different score ranges
    if method_name == "clip":
        base_score = 0.95
        decay = 0.05
    elif method_name == "image_edit":
        base_score = 0.92
        decay = 0.04
    elif method_name == "vlm":
        base_score = 0.88
        decay = 0.06
    elif method_name == "aesthetic":
        base_score = 0.90
        decay = 0.03
    else:
        base_score = 0.85
        decay = 0.05
    
    results = []
    for i, outfit_name in enumerate(selected_outfits):
        # Add some randomness to the scores
        score = base_score - (i * decay) + random.uniform(-0.02, 0.02)
        score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
        
        results.append({
            "name": outfit_name,
            "score": round(score, 4)
        })
    
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"[{method_name.upper()}] MOCK - Generated {len(results)} mock results")
    
    return results


def get_clip_results_mock(image: UploadFile, top_k: int):
    """
    Mock version: Retrieve using naive CLIP embeddings and vector database.
    Returns mock data without making remote API calls.
    
    Args:
        image: Uploaded image file (not used in mock)
        top_k: Number of results to retrieve
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    print(f"[CLIP] MOCK - Simulating retrieval with top_k={top_k}")
    return _generate_mock_results(top_k, "clip")


def get_image_edit_results_mock(image: UploadFile, top_k: int):
    """
    Mock version: Retrieve using image edit model.
    Returns mock data without making remote API calls.
    
    Args:
        image: Uploaded image file (not used in mock)
        top_k: Number of results to retrieve
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    print(f"[IMAGE_EDIT] MOCK - Simulating retrieval with top_k={top_k}")
    return _generate_mock_results(top_k, "image_edit")


def get_vlm_results_mock(image: UploadFile, top_k: int):
    """
    Mock version: Retrieve using Vision-Language Model.
    Returns mock data without making remote API calls.
    
    Args:
        image: Uploaded image file (not used in mock)
        top_k: Number of results to retrieve
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    print(f"[VLM] MOCK - Simulating retrieval with top_k={top_k}")
    return _generate_mock_results(top_k, "vlm")


def get_aes_results_mock(image: UploadFile, top_k: int):
    """
    Mock version: Retrieve using aesthetic predictor.
    Returns mock data without making remote API calls.
    
    Args:
        image: Uploaded image file (not used in mock)
        top_k: Number of results to retrieve
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    print(f"[AESTHETIC] MOCK - Simulating retrieval with top_k={top_k}")
    return _generate_mock_results(top_k, "aesthetic")


# Helper function to clear the outfit cache (useful for testing)
def clear_outfit_cache():
    """Clear the cached outfit names, forcing a reload on next call."""
    global _OUTFIT_CACHE
    _OUTFIT_CACHE = None
