# app/services/img_processor.py

# from rembg import remove
from PIL import Image
import numpy as np
import json
from typing import Tuple, List
import os

# def remove_background(input_path, output_path):
#     with open(input_path, "rb") as f:
#         input_image = f.read()

#     output_image = remove(input_image)

#     with open(output_path, "wb") as f:
#         f.write(output_image)

# --- CLI Test ---
# python - << 'EOF'
# from app.services.img_processor import remove_background
# import os

# input_path = "app/clothes/1.png"
# output_path = "test.png"

# remove_background(input_path, output_path)

# assert os.path.exists(output_path), "[Error] Output file was not created"
# print("[Complete] Background removal test passed")
# EOF

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

# --- CLI Test ---
# python - << 'EOF'
# from app.services.img_processor import paste_centered

# paste_centered(
#     fg_path="test.png",          # output from rembg
#     bg_path="bg_test.png",
#     output_path="final.png",
#     scale=0.8
# )

# print("[Complete] Image compositing passed")
# EOF

def compose_2d_on_background(
    bg_path: str,
    fg_dir: str = "app/data/2d",
    clothes_json: str = "app/data/clothes.json",
    scale: float = 1.0,
    return_format: str = "pil",  # "pil" | "numpy"
    output_dir: str = "app/outputs/composed",
) -> List[Tuple[str, Image.Image]]:
    """
    Paste foreground images listed in clothes.json centered on the background image.
    Saves all composed images into outputs/composed using the figure name.
    Returns (filename, image).
    """

    bg_original = Image.open(bg_path).convert("RGBA")
    results = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------
    # Load clothes.json (list[str])
    # -------------------------------------------------
    with open(clothes_json, "r") as f:
        fg_files: List[str] = json.load(f)

    if not fg_files:
        raise RuntimeError("clothes.json is empty")

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