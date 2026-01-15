import faiss, json
import numpy as np
from pathlib import Path
from PIL import Image
from app.models.qwen_pe.qwen_embed import QwenVLEmbedder

embedder = QwenVLEmbedder()
paths = list(Path("app/data/assets/clothes").glob("*.png"))

embs, meta = [], []

for p in paths:
    emb = embedder.encode_image(Image.open(p))
    embs.append(emb)
    meta.append({"id": p.stem, "file": str(p)})

embs = np.stack(embs).astype("float32")
dim = embs.shape[1]

index = faiss.IndexIVFPQ(
    faiss.IndexFlatIP(dim),
    dim,
    2048,
    16,
    8,
)
index.train(embs)
index.add(embs)

faiss.write_index(index, "app/data/faiss/qwen.index")
json.dump(meta, open("app/data/faiss/qwen_meta.json", "w"))
