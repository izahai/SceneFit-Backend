

from pathlib import Path
import yaml
import httpx
import random
import os
import time
from app.services.post_processing import shuffle_retrieval_results
from fastapi import HTTPException, UploadFile
from app.utils.util import convert_filename_to_url

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / "config" / "retrieval_methods.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RETRIEVAL_CONFIG = config["retrieval_methods"]
TIMEOUT = config.get("timeout", 60)
RETRY_CONFIG = config.get("retry", {"max_attempts": 3, "delay_seconds": 1})

# Oversampling: ask upstream for more items than requested to allow better shuffle diversity.
OVERSAMPLE_FACTOR = int(os.getenv("SCENEFIT_OVERSAMPLE_FACTOR", "20"))
MAX_REQUEST_TOP_K = int(os.getenv("SCENEFIT_MAX_REQUEST_TOP_K", "100"))

# Mock mode flag - set to True to use mock data instead of real API calls
USE_MOCK_DATA = False

# Cache for outfit names to avoid reading directory multiple times
_OUTFIT_NAMES_CACHE = None


def _strip_png_extension(filename: str) -> str:
    """Remove a trailing .png extension from a filename."""
    return filename[:-4] if isinstance(filename, str) and filename.lower().endswith(".png") else filename


def _ensure_png_extension(filename: str) -> str:
    """Guarantee the filename ends with .png for URL generation."""
    if not isinstance(filename, str):
        return filename
    return filename if filename.lower().endswith(".png") else f"{filename}.png"


def _normalize_result_item(item: dict) -> dict | None:
    """Coerce a single result into {name, score, image_url} with required extension rules."""
    if not isinstance(item, dict):
        return None

    raw_name = item.get("name") or item.get("image_name") or item.get("file_name")
    if not raw_name:
        return None

    filename_with_ext = _ensure_png_extension(raw_name)
    clean_name = _strip_png_extension(raw_name)

    score_val = item.get("score")
    try:
        score = float(score_val) if score_val is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0

    existing_url = item.get("image_url")
    image_url = existing_url if isinstance(existing_url, str) and existing_url.lower().endswith(".png") else convert_filename_to_url(filename_with_ext)

    return {
        "name": clean_name,
        "score": score,
        "image_url": image_url,
    }


def _normalize_and_shuffle_results(raw_results, top_k: int):
    """Normalize remote/mock payloads and enforce the required response shape."""
    if isinstance(raw_results, dict) and raw_results.get("error"):
        return raw_results

    results_list = raw_results.get("results") if isinstance(raw_results, dict) and "results" in raw_results else raw_results
    if not isinstance(results_list, list):
        return raw_results

    normalized = []
    for item in results_list:
        normalized_item = _normalize_result_item(item)
        if normalized_item is not None:
            normalized.append(normalized_item)

    return shuffle_retrieval_results(normalized, top_k)


def _get_request_top_k(top_k: int) -> int:
    """Compute how many items to request upstream for better shuffle diversity."""
    if top_k <= 0:
        return 0
    oversampled = top_k * max(1, OVERSAMPLE_FACTOR)
    return min(MAX_REQUEST_TOP_K, max(top_k, oversampled))


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
        List of dicts with format [{"name": str, "score": float, "image_url": str}]
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
        score = 0.95 - (i * 0.45 / max(1, top_k - 1))
        filename_with_ext = _ensure_png_extension(name)
        results.append({
            "name": _strip_png_extension(name),
            "score": round(score, 4),
            "image_url": convert_filename_to_url(filename_with_ext),
        })
    
    print(f"[{method_name.upper()}] Generated {len(results)} mock results")
    return results


def _make_request(method_name: str, image_content: bytes, filename: str, content_type: str, top_k: int):
    """
    Generic function to make HTTP request to a retrieval method endpoint.
    
    Args:
        method_name: Name of the method in config (clip, image_edit, vlm, aesthetic)
        image_content: Image file content as bytes
        filename: Original filename
        content_type: MIME type of the image
        top_k: Number of results to retrieve
    
    Returns:
        List of results in format [{"name": str, "score": float, "image_url": str}]
    """
    if method_name not in RETRIEVAL_CONFIG:
        raise HTTPException(
            status_code=500,
            detail=f"Method '{method_name}' not found in configuration"
        )
    
    method_config = RETRIEVAL_CONFIG[method_name]
    url = f"{method_config['url']}{method_config['endpoint']}"
    
    print(f"[{method_name.upper()}] Calling endpoint: {url}")
    print(f"[{method_name.upper()}] Parameters - image: {filename}, top_k: {top_k}")
    
    try:
        # Create BytesIO object from content
        import io
        image_file = io.BytesIO(image_content)
        
        # Prepare multipart form data
        files = {"image": (filename, image_file, content_type)}
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


def get_clip_results(image_content: bytes, filename: str, content_type: str, top_k: int, mock=True):
    """
    Retrieve using naive CLIP embeddings and vector database, returning normalized name/score/image_url entries.
    """
    request_top_k = _get_request_top_k(top_k)
    start = time.perf_counter()
    fell_back = False
    if mock:
        results = generate_mock_results(request_top_k, "clip")
    else:
        try:
            results = _make_request("clip", image_content, filename, content_type, request_top_k)
        except Exception as exc:  # Fallback to mock if the request fails
            print(f"[CLIP] Falling back to mock results: {str(exc)[:120]}")
            results = generate_mock_results(request_top_k, "clip")
            fell_back = True

    elapsed = time.perf_counter() - start
    mode = "mock" if mock or fell_back else "real"
    print(f"[CLIP] Completed in {elapsed:.3f}s (mode={mode}, request_top_k={request_top_k}, return_top_k={top_k})")

    return _normalize_and_shuffle_results(results, top_k)


def get_image_edit_results(image_content: bytes, filename: str, content_type: str, top_k: int, mock=True):
    """
    Retrieve using image edit model, returning normalized name/score/image_url entries.
    """
    request_top_k = _get_request_top_k(top_k)
    start = time.perf_counter()
    fell_back = False
    if mock:
        results = generate_mock_results(request_top_k, "image_edit")
    else:
        try:
            results = _make_request("image_edit", image_content, filename, content_type, request_top_k)
        except Exception as exc:  # Fallback to mock if the request fails
            print(f"[IMAGE_EDIT] Falling back to mock results: {str(exc)[:120]}")
            results = generate_mock_results(request_top_k, "image_edit")
            fell_back = True

    elapsed = time.perf_counter() - start
    mode = "mock" if mock or fell_back else "real"
    print(f"[IMAGE_EDIT] Completed in {elapsed:.3f}s (mode={mode}, request_top_k={request_top_k}, return_top_k={top_k})")

    return _normalize_and_shuffle_results(results, top_k)


def get_vlm_results(image_content: bytes, filename: str, content_type: str, top_k: int, mock=True):
    """
    Retrieve using Vision-Language Model, returning normalized name/score/image_url entries.
    """
    request_top_k = _get_request_top_k(top_k)
    start = time.perf_counter()
    fell_back = False
    if mock:
        results = generate_mock_results(request_top_k, "vlm")
    else:
        try:
            results = _make_request("vlm", image_content, filename, content_type, request_top_k)
        except Exception as exc:  # Fallback to mock if the request fails
            print(f"[VLM] Falling back to mock results: {str(exc)[:120]}")
            results = generate_mock_results(request_top_k, "vlm")
            fell_back = True

    elapsed = time.perf_counter() - start
    mode = "mock" if mock or fell_back else "real"
    print(f"[VLM] Completed in {elapsed:.3f}s (mode={mode}, request_top_k={request_top_k}, return_top_k={top_k})")

    return _normalize_and_shuffle_results(results, top_k)


def get_aes_results(image_content: bytes, filename: str, content_type: str, top_k: int, mock=True):
    """
    Retrieve using aesthetic predictor, returning normalized name/score/image_url entries.
    """
    request_top_k = _get_request_top_k(top_k)
    start = time.perf_counter()
    fell_back = False
    if mock:
        results = generate_mock_results(request_top_k, "aesthetic")
    else:
        try:
            results = _make_request("aesthetic", image_content, filename, content_type, request_top_k)
        except Exception as exc:  # Fallback to mock if the request fails
            print(f"[AESTHETIC] Falling back to mock results: {str(exc)[:120]}")
            results = generate_mock_results(request_top_k, "aesthetic")
            fell_back = True

    elapsed = time.perf_counter() - start
    mode = "mock" if mock or fell_back else "real"
    print(f"[AESTHETIC] Completed in {elapsed:.3f}s (mode={mode}, request_top_k={request_top_k}, return_top_k={top_k})")

    return _normalize_and_shuffle_results(results, top_k)