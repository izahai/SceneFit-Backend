

from pathlib import Path
import yaml
import httpx
import random
import os
from fastapi import HTTPException, UploadFile


# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / "config" / "retrieval_methods.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RETRIEVAL_CONFIG = config["retrieval_methods"]
TIMEOUT = config.get("timeout", 60)
RETRY_CONFIG = config.get("retry", {"max_attempts": 3, "delay_seconds": 1})

# Mock mode flag - set to True to use mock data instead of real API calls
USE_MOCK_DATA = False

# Cache for outfit names to avoid reading directory multiple times
_OUTFIT_NAMES_CACHE = None


def _get_available_outfit_names():
    """
    Get list of available outfit names from the 2d directory.
    Returns outfit names without the .png extension.
    """
    global _OUTFIT_NAMES_CACHE
    
    if _OUTFIT_NAMES_CACHE is not None:
        return _OUTFIT_NAMES_CACHE
    
    # Path to 2d images directory
    data_2d_path = Path(__file__).parent.parent / "data" / "2d"
    
    outfit_names = []
    if data_2d_path.exists():
        for file in data_2d_path.iterdir():
            if file.is_file() and file.suffix == ".png":
                # Remove .png extension to get outfit name
                outfit_names.append(file.stem)
    
    # Cache the results
    _OUTFIT_NAMES_CACHE = outfit_names
    print(f"[MOCK] Loaded {len(outfit_names)} outfit names from {data_2d_path}")
    
    return outfit_names


def generate_mock_results(top_k: int, method_name: str = "mock") -> list:
    """
    Generate mock retrieval results with names matching those in app/data/2d/
    
    Args:
        top_k: Number of results to generate
        method_name: Name of the method (for logging)
    
    Returns:
        List of dicts with format [{"name": str, "score": float}]
    """
    outfit_names = _get_available_outfit_names()
    
    if not outfit_names:
        print(f"[{method_name.upper()}] WARNING - No outfit names found, using placeholder data")
        outfit_names = [f"outfit_{i}" for i in range(100)]
    
    # Randomly select outfit names (without replacement if possible)
    num_to_select = min(top_k, len(outfit_names))
    selected_names = random.sample(outfit_names, num_to_select)
    
    # Generate results with descending scores
    results = []
    for i, name in enumerate(selected_names):
        # Generate scores in descending order (0.95 to 0.50)
        score = 0.95 - (i * 0.45 / max(1, top_k - 1))
        results.append({
            "name": name,
            "score": round(score, 4)
        })
    
    print(f"[{method_name.upper()}] Generated {len(results)} mock results")
    return results


def _make_request(method_name: str, image: UploadFile, top_k: int):
    """
    Generic function to make HTTP request to a retrieval method endpoint.
    
    Args:
        method_name: Name of the method in config (clip, image_edit, vlm, aesthetic)
        image: Uploaded image file
        top_k: Number of results to retrieve
    
    Returns:
        List of results in format [{"name": str, "score": float}]
    """
    if method_name not in RETRIEVAL_CONFIG:
        raise HTTPException(
            status_code=500,
            detail=f"Method '{method_name}' not found in configuration"
        )
    
    method_config = RETRIEVAL_CONFIG[method_name]
    url = f"{method_config['url']}{method_config['endpoint']}"
    
    print(f"[{method_name.upper()}] Calling endpoint: {url}")
    print(f"[{method_name.upper()}] Parameters - image: {image.filename}, top_k: {top_k}")
    
    try:
        # Reset file pointer to beginning
        image.file.seek(0)
        
        # Prepare multipart form data
        files = {"image": (image.filename, image.file, image.content_type)}
        data = {"top_k": top_k}
        
        # Make HTTP request with retry logic
        for attempt in range(RETRY_CONFIG["max_attempts"]):
            try:
                print(f"[{method_name.upper()}] Attempt {attempt + 1}/{RETRY_CONFIG['max_attempts']} - Sending request...")
                
                with httpx.Client(timeout=TIMEOUT) as client:
                    response = client.post(url, files=files, data=data)
                    response.raise_for_status()
                    
                    result = response.json()
                    result_count = len(result) if isinstance(result, list) else len(result.get('results', []))
                    print(f"[{method_name.upper()}] ✓ SUCCESS - Received {result_count} results (status: {response.status_code})")
                    
                    return result
                    
            except httpx.HTTPStatusError as e:
                print(f"[{method_name.upper()}] ✗ HTTP ERROR - Status {e.response.status_code}: {e.response.text[:100]}")
                if attempt == RETRY_CONFIG["max_attempts"] - 1:
                    print(f"[{method_name.upper()}] FAILED - Max retries reached")
                    raise HTTPException(
                        status_code=e.response.status_code,
                        detail=f"Error from {method_name} service: {e.response.text}"
                    )
                    
            except httpx.RequestError as e:
                print(f"[{method_name.upper()}] ✗ CONNECTION ERROR - {str(e)[:100]}")
                if attempt == RETRY_CONFIG["max_attempts"] - 1:
                    print(f"[{method_name.upper()}] FAILED - Service unreachable after {RETRY_CONFIG['max_attempts']} attempts")
                    raise HTTPException(
                        status_code=503,
                        detail=f"Unable to reach {method_name} service at {url}: {str(e)}"
                    )
            
            # Wait before retry
            if attempt < RETRY_CONFIG["max_attempts"] - 1:
                print(f"[{method_name.upper()}] Retrying in {RETRY_CONFIG['delay_seconds']}s...")
                import time
                time.sleep(RETRY_CONFIG["delay_seconds"])
                
    except Exception as e:
        print(f"[{method_name.upper()}] ✗ UNEXPECTED ERROR - {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request for {method_name}: {str(e)}"
        )


def get_clip_results(image: UploadFile, top_k: int, mock=True):
    """
    Retrieve using naive CLIP embeddings and vector database.
    """
    if mock:
        return generate_mock_results(top_k, "clip")
    return _make_request("clip", image, top_k)


def get_image_edit_results(image: UploadFile, top_k: int, mock=True):
    """
    Retrieve using image edit model.
    """
    if mock:
        return generate_mock_results(top_k, "image_edit")
    return _make_request("image_edit", image, top_k)


def get_vlm_results(image: UploadFile, top_k: int, mock=True):
    """
    Retrieve using Vision-Language Model.
    """
    if mock:
        return generate_mock_results(top_k, "vlm")
    return _make_request("vlm", image, top_k)


def get_aes_results(image: UploadFile, top_k: int, mock=True):
    """
    Retrieve using aesthetic predictor.
    """
    if mock:
        return generate_mock_results(top_k, "aesthetic")
    return _make_request("aesthetic", image, top_k)