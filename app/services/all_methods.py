

from pathlib import Path
import yaml
import httpx
from fastapi import HTTPException, UploadFile


# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / "config" / "retrieval_methods.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RETRIEVAL_CONFIG = config["retrieval_methods"]
TIMEOUT = config.get("timeout", 60)
RETRY_CONFIG = config.get("retry", {"max_attempts": 3, "delay_seconds": 1})


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


def get_clip_results(image: UploadFile, top_k: int):
    """
    Retrieve using naive CLIP embeddings and vector database.
    """
    return _make_request("clip", image, top_k)


def get_image_edit_results(image: UploadFile, top_k: int):
    """
    Retrieve using image edit model.
    """
    return _make_request("image_edit", image, top_k)


def get_vlm_results(image: UploadFile, top_k: int):
    """
    Retrieve using Vision-Language Model.
    """
    return _make_request("vlm", image, top_k)


def get_aes_results(image: UploadFile, top_k: int):
    """
    Retrieve using aesthetic predictor.
    """
    return _make_request("aesthetic", image, top_k)