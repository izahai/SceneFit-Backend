# build_peclip_faiss.py

import faiss
import torch
import pickle
import numpy as np
from pathlib import Path
from PIL import Image

from app.models.pe_clip_matcher import PEClipMatcher


@torch.no_grad()
def main():
    device = "cuda"
    clothes_dir = Path("app/data/2d")
    output_dir = Path("app/data/faiss")

    output_dir.mkdir(parents=True, exist_ok=True)

    matcher = PEClipMatcher(device=device, load_faiss=False)

    image_paths = sorted(
        p for p in clothes_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    assert len(image_paths) > 0, "No clothing images found."

    # -------- SINGLE PASS IMAGE ENCODING --------
    images = [Image.open(p).convert("RGB") for p in image_paths]
    image_embs = matcher.encode_image(images)  # (N, D)

    image_embs = image_embs.cpu().numpy().astype("float32")
    dim = image_embs.shape[1]

    # -------- FAISS INDEX --------
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(image_embs)

    faiss.write_index(index, str(output_dir / "clothes_image.index"))

    # -------- METADATA --------
    with open(output_dir / "clothes_image_meta.pkl", "wb") as f:
        pickle.dump(
            {
                "filenames": [p.name for p in image_paths]
            },
            f,
        )

    print(f"[OK] Indexed {len(image_paths)} clothing images.")


if __name__ == "__main__":
    main()
