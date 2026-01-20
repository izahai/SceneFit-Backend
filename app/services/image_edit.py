import requests
import base64
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SAVE_DIR = Path('app/data/edited_image/')
REF_IMAGE_PATH = Path('app/data/man.png')
API_KEY = os.getenv("IMAGEROUTER_API_KEY")
URL = "https://api.imagerouter.io/v1/openai/images/edits"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}
MODEL = "openai/gpt-image-1.5:free"

def save_to_file(result, save_result):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    img_id = sum(1 for _ in SAVE_DIR.glob("*.jpg")) + 1


    if save_result:
        res_path = SAVE_DIR / f"{img_id}_response.json"
        with open(res_path, 'w') as f:
            json.dump(result, f, indent=2)

    filepath = SAVE_DIR / f"{img_id}.jpg"
    image_data = requests.get(result['data'][0]['url']).content
    with open(filepath, 'wb') as f:
        f.write(image_data)

    return filepath


def format_prompt(scene_description):
    prompt = (
        "Change the outfit of this person into an outfit that is suitable "
        f"for a travel to {scene_description}. "
        "Do not change the background, just change the outfit"
    )
    return prompt

def edit_image_scene_desc(scene_description, save_result=True):

    prompt = format_prompt(scene_description)

    if not REF_IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Reference image not found: {REF_IMAGE_PATH}")

    data = {
        "model": "openai/gpt-image-1.5:free",
        "prompt": prompt,
        "output_format": "jpg",
        "quality": "auto",
        "size": "auto"
    }

    with REF_IMAGE_PATH.open("rb") as ref_image:
        files = {
            "image": (REF_IMAGE_PATH.name, ref_image, "image/png")
        }

        response = requests.post(URL, headers=HEADERS, files=files, data=data)
        result = response.json()
        result['prompt'] = prompt

    if result.get('error'):
        raise RuntimeError(f"API Error: {result['error']}")
    
    save_to_file(result=result, save_result=save_result)
    return result

import requests
from pathlib import Path

def edit_image_scene_img(scene_path, save_result=True):
    prompt = (
        "Change the outfit of this the given into an outfit that matches the scene. "
        "Return the image of such person in the original background of that person (the blue and purple one). "
        "Do not add the person into the scene image."
    )

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
            (REF_IMAGE_PATH.name, open(REF_IMAGE_PATH, "rb"), "image/png")
        ),
        (
            "image",
            (Path(scene_path).name, open(scene_path, "rb"), "image/png")
        )
    ]

    response = requests.post(URL, headers=HEADERS, files=files, data=data)
    result = response.json()

    if result.get("error"):
        raise RuntimeError(f"API Error: {result['error']}")

    result["prompt"] = prompt
    edited_path = save_to_file(result=result, save_result=save_result)
    return {"result": result, "edited_path": edited_path}

def main():
    # For testing
    scene_description = "A snowy mountain village with wooden houses and pine trees"
    bg_image_path = 'data/bg/7.png'
    # edit_image_scene_desc(scene_description=scene_description, save_result=True)
    edit_image_scene_img(scene_path=bg_image_path, save_result=False)

if __name__ == "__main__":
    main()

