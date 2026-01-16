import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

class QwenVLEmbedder:
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-Embedding-2B"
        )
        self.model = AutoModel.from_pretrained(
            "Qwen/Qwen3-VL-Embedding-2B"
        ).to(device).eval()

    def encode_image(self, img: Image.Image):
        inputs = self.processor(
            text="<image>",
            images=img,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Qwen3-VL embedding head
        emb = outputs.image_embeds  # or outputs.pooler_output depending on config
        return emb[0].cpu().numpy()


    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model(**inputs).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]
