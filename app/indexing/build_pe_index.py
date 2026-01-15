import faiss, json
import numpy as np
from pathlib import Path
from PIL import Image
from app.models.negative_generator import NegativePEModel

pe = NegativePEModel()
paths = list(Path("app/data/assets/clothes").glob("*.png"))

imgs = [Image.open(p) for p in paths]
embs = pe.encode_image(imgs).cpu().numpy().astype("float32")

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

faiss.write_index(index, "app/data/faiss/pe.index")
json.dump(
    [{"id": p.stem, "file": str(p)} for p in paths],
    open("app/data/faiss/pe_meta.json", "w"),
)
