# app/utils/util.py

import yaml
from pathlib import Path
from functools import lru_cache
from typing import List
import json
from pathlib import Path
from PIL import Image

_PROMPT_CONFIG_PATH = Path("app/config/prompts2.yaml")
_CLOTHES_JSON = Path("app/data/clothes.json")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def load_prompts(task: str) -> list[str]:
    with open(_PROMPT_CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)

    if task not in data:
        raise KeyError(f"Prompt task '{task}' not found in prompts.yaml")

    prompts = data[task]
    return [prompts["positive"], prompts["negative"]]

def load_prompt_by_key(key: str) -> str:
    with open(_PROMPT_CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)

    if key not in data:
        raise KeyError(f"Prompt key '{key}' not found in prompts.yaml")

    return data[key]

@lru_cache(maxsize=1)
def get_clothes_list() -> List[str]:
    with open(_CLOTHES_JSON, "r") as f:
        return json.load(f)

def load_images_from_folder(
    folder: str | Path,
    recursive: bool = True,
    max_images: int | None = None,
) -> List[Image.Image]:
    """
    Load all images from a folder into a list of PIL Images.

    Args:
        folder: Path to folder containing images
        recursive: Whether to search subfolders
        max_images: Optional cap to limit number of images loaded

    Returns:
        List[PIL.Image.Image]
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    pattern = "**/*" if recursive else "*"

    images: List[Image.Image] = []
    for path in sorted(folder.glob(pattern)):
        if path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue

        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            # Skip corrupted images but don’t crash
            print(f"[WARN] Failed to load image {path}: {e}")

        if max_images is not None and len(images) >= max_images:
            break

    if not images:
        raise RuntimeError(f"No valid images found in {folder}")

    return images