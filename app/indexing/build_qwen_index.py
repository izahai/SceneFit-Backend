import faiss, json
import numpy as np
from pathlib import Path
from PIL import Image
from app.models.qwen_pe.qwen_embed import QwenVLEmbedder
import multiprocessing as mp
def main():

    embedder = QwenVLEmbedder()
    embedder.load()
    paths = list(Path("app/data/2d").glob("*.png"))

    embs, meta = [], []

    embs = embedder.encode_batch(paths)

    meta = [{"id": p.stem, "file": str(p)} for p in paths]

    # 5. Normalization (Crucial for Cosine Similarity)
    embs = embs.astype("float32")
    faiss.normalize_L2(embs) # In-place normalization

    # 6. FAISS Indexing
    dim = embs.shape[1]
    print(f"Embedding dimension: {dim}")

    # Use IndexFlatIP for exact search (best for < 100k items)
    # It calculates Dot Product. Since vectors are normalized, Dot Product == Cosine Similarity
    index = faiss.IndexFlatIP(dim) 
    
    index.add(embs) # No training needed for Flat index

    # 7. Save
    output_dir = Path("app/data/faiss")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(output_dir / "qwen.index"))
    with open(output_dir / "qwen_meta.json", "w") as f:
        json.dump(meta, f)
        
    print("Indexing complete.")

if __name__ == "__main__":
    main()
