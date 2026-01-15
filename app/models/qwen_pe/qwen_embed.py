import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

class QwenVLEmbedder:
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-Embedding-8B"
        )
        self.model = AutoModel.from_pretrained(
            "Qwen/Qwen3-VL-Embedding-8B"
        ).to(device).eval()

    def encode_image(self, img: Image.Image) -> np.ndarray:
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model(**inputs).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model(**inputs).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]
