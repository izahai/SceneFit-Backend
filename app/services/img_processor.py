# app/services/img_processor.py

from rembg import remove
from PIL import Image
import numpy as np
import json
from typing import Tuple, List
import os

def remove_background(input_path, output_path):
    with open(input_path, "rb") as f:
        input_image = f.read()

    output_image = remove(input_image)

    with open(output_path, "wb") as f:
        f.write(output_image)

def paste_centered(
    fg_path: str,
    bg_path: str,
    output_path: str,
    scale: float = 1.0,
):
    """
    Paste a foreground image (with transparency) centered on a background image.

    Args:
        fg_path: Path to foreground image (RGBA, background removed)
        bg_path: Path to background image
        output_path: Where to save the final image
        scale: Optional scale factor for foreground (e.g. 0.7 to make it smaller)
    """
    fg = Image.open(fg_path).convert("RGBA")
    bg = Image.open(bg_path).convert("RGBA")

    # Optionally scale foreground
    if scale != 1.0:
        fg_w, fg_h = fg.size
        fg = fg.resize(
            (int(fg_w * scale), int(fg_h * scale)),
            Image.LANCZOS,
        )

    bg_w, bg_h = bg.size
    fg_w, fg_h = fg.size

    # Compute centered position
    x = (bg_w - fg_w) // 2
    y = (bg_h - fg_h) // 2

    # Paste using alpha channel as mask
    bg.paste(fg, (x, y), fg)

    bg.save(output_path)

def compose_2d_on_background(
    bg_path: str,
    fg_dir: str = "app/data/2d",
    fg_files: List[str] | None = None,
    clothes_json: str = "app/data/clothes.json",
    scale: float = 1.0,
    return_format: str = "pil",  # "pil" | "numpy"
    output_dir: str = "app/outputs/composed",
) -> List[Tuple[str, Image.Image]]:
    """
    Paste foreground images centered on the background image.
    Uses fg_files when provided, otherwise loads from clothes_json.
    Saves all composed images into outputs/composed using the figure name.
    Returns (filename, image).
    """

    bg_original = Image.open(bg_path).convert("RGBA")
    results = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------
    # Load fg list — use all images in fg_dir when not provided
    # -------------------------------------------------
    if fg_files is None:
        if not os.path.isdir(fg_dir):
            raise RuntimeError(f"fg_dir does not exist: {fg_dir}")

        all_files = sorted(os.listdir(fg_dir))
        IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif")
        fg_files = [f for f in all_files if f.lower().endswith(IMAGE_EXTS)]

        # If none found in directory, fall back to clothes_json (if exists)
        if not fg_files:
            if os.path.isfile(clothes_json):
                with open(clothes_json, "r") as f:
                    fg_files = json.load(f)
            if not fg_files:
                raise RuntimeError(f"No image files found in {fg_dir} and clothes.json is empty")
    else:
        fg_files = [str(p) for p in fg_files]
        if not fg_files:
            raise RuntimeError("fg_files is empty")

    # -------------------------------------------------
    # Process each foreground image
    # -------------------------------------------------
    for fg_file in fg_files:
        fg_path = os.path.join(fg_dir, fg_file)

        # Skip missing files
        if not os.path.isfile(fg_path):
            continue

        fg = Image.open(fg_path).convert("RGBA")
        bg = bg_original.copy()

        if scale != 1.0:
            fg_w, fg_h = fg.size
            fg = fg.resize(
                (int(fg_w * scale), int(fg_h * scale)),
                Image.LANCZOS
            )

        bg_w, bg_h = bg.size
        fg_w, fg_h = fg.size

        x = (bg_w - fg_w) // 2
        y = (bg_h - fg_h) // 2

        bg.paste(fg, (x, y), fg)

        # -------------------------------------------------
        # SAVE composed image
        # -------------------------------------------------
        # output_path = os.path.join(output_dir, fg_file)
        # bg.save(output_path)

        # -------------------------------------------------
        # RETURN formats
        # -------------------------------------------------
        if return_format == "pil":
            results.append((fg_file, bg))

        elif return_format == "numpy":
            results.append((fg_file, np.array(bg.convert("RGB"))))

        else:
            raise ValueError("return_format must be 'pil' or 'numpy'")

    if not results:
        raise RuntimeError(
            "No valid foreground images found (all files missing?)"
        )

    return results
