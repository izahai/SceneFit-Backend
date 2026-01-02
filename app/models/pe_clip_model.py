# app/models/pe_clip_model.py

import torch
import torch.nn.functional as F
from PIL import Image

import app.services.pe_core.vision_encoder.pe as pe
import app.services.pe_core.vision_encoder.transforms as transforms

from app.utils.device import resolve_device
from app.utils.util import load_prompts


class PEClipModel:
    """
    PE-Core CLIP-based image-text similarity model.
    API-compatible with MmEmbModel.
    """

    def __init__(
        self,
        config_name: str = "PE-Core-L14-336",
        device: str | None = None,
        autocast: bool = True,
    ):
        self.device = resolve_device(device)
        self.autocast = autocast

        # -------------------------
        # Load model
        # -------------------------
        self.model = pe.CLIP.from_config(
            config_name,
            pretrained=True,
        ).to(self.device)
        self.model.eval()

        # -------------------------
        # Transforms
        # -------------------------
        self.image_transform = transforms.get_image_transform(
            self.model.image_size
        )
        self.text_tokenizer = transforms.get_text_tokenizer(
            self.model.context_length
        )

        # -------------------------
        # Load prompts & encode once
        # -------------------------
        self.prompts = load_prompts("outfit_match")
        self.prompt_embeddings = self.encode_text(self.prompts)

    # -------------------------------------------------
    # Text encoding
    # -------------------------------------------------
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.text_tokenizer(texts).to(self.device)

        with torch.no_grad(), torch.autocast(
            device_type=self.device,
            enabled=self.autocast,
        ):
            _, text_features, _ = self.model(
                image=None,
                text=tokens,
            )

        return self._normalize(text_features)

    # -------------------------------------------------
    # Image encoding
    # -------------------------------------------------
    def encode_image(self, images) -> torch.Tensor:
        """
        images: PIL.Image | list[PIL.Image]
        """
        if isinstance(images, Image.Image):
            images = [images]

        image_tensor = torch.stack(
            [self.image_transform(img) for img in images]
        ).to(self.device)

        with torch.no_grad(), torch.autocast(
            device_type=self.device,
            enabled=self.autocast,
        ):
            image_features, _, _ = self.model(
                image=image_tensor,
                text=None,
            )

        return self._normalize(image_features)

    # -------------------------------------------------
    # Similarity
    # -------------------------------------------------
    @staticmethod
    def similarity(text_embeddings, image_embeddings):
        return text_embeddings @ image_embeddings.T

    @staticmethod
    def _normalize(x):
        return F.normalize(x, p=2, dim=-1)

    # -------------------------------------------------
    # Scoring helpers (same API as MmEmbModel)
    # -------------------------------------------------
    def score_image(self, image: Image.Image):
        image_emb = self.encode_image(image)

        pos_emb = self.prompt_embeddings[0]
        neg_emb = self.prompt_embeddings[1]

        pos_score = (image_emb @ pos_emb.T).item()
        neg_score = (image_emb @ neg_emb.T).item()

        return {
            "positive_score": pos_score,
            "negative_score": neg_score,
            "confidence": pos_score - neg_score,
        }

    def score_images(self, images: list[Image.Image]):
        image_embeddings = self.encode_image(images)

        pos_emb = self.prompt_embeddings[0]
        neg_emb = self.prompt_embeddings[1]

        results = []

        for idx, img_emb in enumerate(image_embeddings):
            pos_score = (img_emb @ pos_emb.T).item()
            neg_score = (img_emb @ neg_emb.T).item()
            confidence = pos_score - neg_score

            results.append({
                "index": idx,
                "positive_score": pos_score,
                "negative_score": neg_score,
                "confidence": confidence,
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
