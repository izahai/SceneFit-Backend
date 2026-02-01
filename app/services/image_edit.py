from io import BytesIO
import sys
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
import base64
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from fastapi import HTTPException
import mimetypes
import ssl

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.services.model_registry import ModelRegistry
from app.utils.util import crop_clothes_region

load_dotenv()

SAVE_DIR = Path('app/retrieval_results/image_edit/edited_image/')
CROPPED_SAVE_DIR = Path('app/retrieval_results/image_edit/cropped_clothes/')
REF_IMAGE_PATH_MAN = Path('app/data/ref_images/man.png')
REF_IMAGE_PATH_WOMAN = Path('app/data/ref_images/woman.png')

API_KEY = os.getenv("IMAGEROUTER_API_KEY")
URL = "https://api.imagerouter.io/v1/openai/images/edits"
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "https://nondepressed-semipneumatically-eveline.ngrok-free.dev")
VLM_VERIFY_SSL = os.getenv("VLM_VERIFY_SSL", "true").lower() != "false"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}
MODEL = "openai/gpt-image-1.5:free"

def save_to_file(result, save_result, crop_clothes=True):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    img_id = sum(1 for _ in SAVE_DIR.glob("*.jpg")) + 1

    if save_result:
        res_path = SAVE_DIR / f"{img_id}_response.json"
        with open(res_path, 'w') as f:
            json.dump(result, f, indent=2)

    filepath = SAVE_DIR / f"{img_id}.jpg"
    cropped_filepath = None
    image_data = requests.get(result['data'][0]['url']).content
    with open(filepath, 'wb') as f:
        f.write(image_data)
    if crop_clothes:
        with Image.open(BytesIO(image_data)) as img:
            cropped_img = crop_clothes_region(img)
            # Also save cropped image separately
            CROPPED_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            cropped_filepath = CROPPED_SAVE_DIR / f"{img_id}_cropped.jpg"
            cropped_img.save(cropped_filepath)
            
    return {
        "edited_path": str(filepath),
        "cropped_path": str(cropped_filepath) if cropped_filepath else None
    }

def save_edited_image(edited_image: Image.Image, cropped_image: Image.Image | None = None) -> dict:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    img_id = sum(1 for _ in SAVE_DIR.glob("*.jpg")) + 1
    filepath = SAVE_DIR / f"{img_id}.jpg"
    edited_image.save(filepath)
    result = {"edited_path": str(filepath)}

    if cropped_image:
        CROPPED_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        cropped_filepath = CROPPED_SAVE_DIR / f"{img_id}_cropped.jpg"
        cropped_image.save(cropped_filepath)
        result["cropped_path"] = str(cropped_filepath)

    return result

def format_prompt(scene_description):
    prompt = (
        "Change the outfit of this person into an outfit that is suitable "
        f"for a travel to {scene_description}. "
        "Do not change the background, just change the outfit."
    )
    return prompt

def edit_image_scene_desc(scene_description, save_result=True, gender='male', crop_clothes=True):

    prompt = format_prompt(scene_description)

    if gender == 'male':
        REF_IMAGE_PATH = REF_IMAGE_PATH_MAN
    elif gender == 'female':
        REF_IMAGE_PATH = REF_IMAGE_PATH_WOMAN
    else:
        raise ValueError("Gender must be 'male' or 'female'")

    if not REF_IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Reference image not found: {REF_IMAGE_PATH}")

    data = {
        "model": MODEL,
        "prompt": prompt,
        "output_format": "jpg",
        "quality": "auto",
        "size": "auto"
    }

    with REF_IMAGE_PATH.open("rb") as ref_image:
        files = {
            "image": (REF_IMAGE_PATH.name, ref_image, f"image/{REF_IMAGE_PATH.suffix.lstrip('.')}")
        }

        response = requests.post(URL, headers=HEADERS, files=files, data=data)
        result = response.json()
        result['prompt'] = prompt

    if result.get('error'):
        raise RuntimeError(f"API Error: {result['error']}")
    
    paths = save_to_file(result=result, save_result=save_result, crop_clothes=crop_clothes)
    return {"result": result, "edited_path": paths["edited_path"], "cropped_path": paths["cropped_path"]}


def edit_image_scene_img(
    scene_path,
    save_result=True,
    gender='male',
    crop_clothes=True,
    preference_text: str | None = None,
    feedback_text: str | None = None,
    ref_image_path: Path | None = None,
):
    prompt_parts = [
        "Change the outfit of the given person into an outfit that matches the scene.",
        "Return the image of such person in the original background of that person.",
        "Do not add the person into the scene image.",
    ]
    if preference_text:
        prompt_parts.append(f"User preference: {preference_text}.")
    if feedback_text:
        prompt_parts.append(f"Additional feedback: {feedback_text}.")

    prompt = " ".join(prompt_parts)

    if not ref_image_path:
        if gender == 'male':
            REF_IMAGE_PATH = REF_IMAGE_PATH_MAN
        elif gender == 'female':    
            REF_IMAGE_PATH = REF_IMAGE_PATH_WOMAN
        else:
            raise ValueError("Gender must be 'male' or 'female'")
    else:
        REF_IMAGE_PATH = Path(ref_image_path)

    if not REF_IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Reference image not found: {REF_IMAGE_PATH}")

    if not Path(scene_path).is_file():
        raise FileNotFoundError(f"Scene image not found: {scene_path}")

    data = {
        "model": MODEL,
        "prompt": prompt,
        "output_format": "jpg",
        "quality": "auto",
        "size": "auto"
    }

    files = [
        (
            "image",
            (REF_IMAGE_PATH.name, open(REF_IMAGE_PATH, "rb"), f"image/{REF_IMAGE_PATH.suffix.lstrip('.')}")
        ),
        (
            "image",
            (Path(scene_path).name, open(scene_path, "rb"), f"image/{Path(scene_path).suffix.lstrip('.')}")
        )
    ]

    response = requests.post(URL, headers=HEADERS, files=files, data=data)
    result = response.json()

    if result.get("error"):
        raise RuntimeError(f"API Error: {result['error']}")

    result["prompt"] = prompt
    paths = save_to_file(result=result, save_result=save_result, crop_clothes=crop_clothes)
    return {"result": result, "edited_path": paths["edited_path"], "cropped_path": paths["cropped_path"], "ref_path": str(REF_IMAGE_PATH)}

def edit_image_outfit_desc(
    outfit_description: str,
    save_result=True,
    gender='male',
    crop_clothes=True,
    preference_text: str | None = None,
    ref_image_path: Path | None = None,
    model_name='image_edit_flux'
):
    model = ModelRegistry.get(f"{model_name}")

    edited_image = model.edit_outfit_desc(
        outfit_description=outfit_description,
        # save_result=save_result,
        gender=gender,
        # crop_clothes=crop_clothes,
        preference_text=preference_text,
        ref_image_path=ref_image_path
    )

    if crop_clothes:
        cropped_image = crop_clothes_region(edited_image)

    paths = save_edited_image(
        edited_image=edited_image,
        cropped_image=cropped_image if crop_clothes else None
    )

    return paths
    

def get_outfit_suggestion_remote(bg_path: Path, verify_ssl: bool | None = None) -> str:
    """Call remote VLM service to get outfit suggestion for a background image."""
    url = f"{VLM_BASE_URL}/api/v1/retrieval/vlm-suggest-outfit"
    mime_type, _ = mimetypes.guess_type(bg_path)
    mime_type = mime_type or "image/png"
    verify = VLM_VERIFY_SSL if verify_ssl is None else verify_ssl

    # Ngrok-free requires this header to bypass the browser warning page
    headers = {"ngrok-skip-browser-warning": "true"}

    # Create a custom SSL context that's more lenient for ngrok
    class SSLContextAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            ctx = create_urllib3_context()
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            kwargs['ssl_context'] = ctx
            return super().init_poolmanager(*args, **kwargs)

    def _send(verify_flag: bool, target_url: str = url):
        session = requests.Session()
        if not verify_flag:
            # Use custom adapter for better ngrok compatibility
            session.mount('https://', SSLContextAdapter())
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        with open(bg_path, "rb") as fp:
            files = {"bg_image": (bg_path.name, fp, mime_type)}
            return session.post(target_url, files=files, headers=headers, timeout=60, verify=verify_flag)

    try:
        resp = _send(verify)
    except requests.exceptions.SSLError as ssl_err:
        # If SSL fails and verification is enabled, retry once with verification disabled
        if verify:
            try:
                resp = _send(False)
            except Exception as retry_err:
                raise HTTPException(status_code=502, detail=f"VLM SSL error (verify=True): {ssl_err}; Retry with verify=False failed: {retry_err}") from ssl_err
        else:
            # SSL protocol error even with verify=False - report it clearly
            raise HTTPException(status_code=502, detail=f"VLM SSL protocol error: {ssl_err}. Check that the VLM server is running and ngrok tunnel is active.") from ssl_err
    except requests.exceptions.RequestException as req_err:
        raise HTTPException(status_code=502, detail=f"VLM connection error: {req_err}") from req_err

    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"VLM service error {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=502, detail=f"Invalid VLM response: {resp.text}") from exc

    outfit_desc = data.get("outfit_description")
    if not outfit_desc:
        raise HTTPException(status_code=502, detail="VLM response missing outfit_description")

    return outfit_desc

# -----------------------------------------------------
# For Debugging: only run when executed directly
# -----------------------------------------------------

def main():
    bg_image_path = Path('app/data/bg/1.png')

    print("Requesting outfit suggestion from remote VLM...")
    # Disable SSL verification for local testing against ngrok if cert chain breaks
    outfit_desc = get_outfit_suggestion_remote(bg_image_path, verify_ssl=False)
    print("Outfit suggestion:", outfit_desc)

if __name__ == "__main__":
    main()
