# app/models/mmemb_model.py 

import torch
import torch.nn.functional as F
from transformers import AutoModel
from app.utils.device import resolve_device, resolve_dtype


class MmEmbModel:
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.model_name = model_name

        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)

        self.model = None

    def load(self):
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )

        self.model.to(self.device)
        self.model.eval()

    # -------------------------
    # Text encoding
    # -------------------------
    def encode_text(self, texts):
        with torch.no_grad():
            embeddings = self.model.encode_text(
                texts=texts,
                task="retrieval",
                prompt_name="query",
            )
        return self._normalize(embeddings)

    # -------------------------
    # Image encoding
    # -------------------------
    def encode_image(self, images):
        with torch.no_grad():
            embeddings = self.model.encode_image(
                images=images,
                task="retrieval",
            )
        return self._normalize(embeddings)

    # -------------------------
    # Cosine similarity
    # -------------------------
    def similarity(self, text_embeddings, image_embeddings):
        return text_embeddings @ image_embeddings.T

    @staticmethod
    def _normalize(x):
        return F.normalize(x, p=2, dim=-1)