# app/models/pe_clip_matcher.py

import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple
from pathlib import Path

import app.services.pe_core.vision_encoder.pe as pe
import app.services.pe_core.vision_encoder.transforms as transforms

from app.utils.device import resolve_device, resolve_autocast, resolve_dtype


class PEClipMatcher:
    """
    Standalone PE-CLIP matcher:
    - Loads PE CLIP internally
    - Encodes AI-generated text
    - Encodes clothing images
    - Ranks clothes by cosine similarity
    """

    def __init__(
        self,
        config_name: str = "PE-Core-B16-224",
        device: str | None = None,
        autocast: bool = True,
    ):
        # -------------------------
        # Device / autocast
        # -------------------------
        self.device = resolve_device(device)
        self.device_type = self.device.type
        self.use_autocast = autocast and resolve_autocast(self.device)
        self.autocast_dtype = resolve_dtype(self.device)

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

    # -------------------------------------------------
    # Text encoding
    # -------------------------------------------------
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.text_tokenizer(texts).to(self.device)

        with torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=self.use_autocast,
        ):
            _, text_features, _ = self.model(
                image=None,
                text=tokens,
            )

        return F.normalize(text_features, dim=-1)

    # -------------------------------------------------
    # Image encoding
    # -------------------------------------------------
    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        image_tensor = torch.stack(
            [self.image_transform(img) for img in images]
        ).to(self.device)

        with torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=self.use_autocast,
        ):
            image_features, _, _ = self.model(
                image=image_tensor,
                text=None,
            )

        return F.normalize(image_features, dim=-1)

    # -------------------------------------------------
    # Matching
    # -------------------------------------------------
    @torch.no_grad()
    def match_clothes(
        self,
        descriptions: list[str],
        clothes: list[tuple[str, Image.Image]],
        top_k: int | None = None,
    ):
        """
        descriptions: list of AI-generated text strings
        clothes: [(name, PIL.Image)]
        """

        # -------------------------
        # Encode text (per string)
        # -------------------------
        text_embs = self.encode_text(descriptions)      # (N_text, D)
        text_embs = F.normalize(text_embs, dim=-1)

        # -------------------------
        # Encode images
        # -------------------------
        names, images = zip(*clothes)
        image_embs = self.encode_image(list(images))    # (N_img, D)
        image_embs = F.normalize(image_embs, dim=-1)

        # -------------------------
        # Similarity: image ↔ all texts
        # -------------------------
        # (N_img, D) @ (D, N_text) → (N_img, N_text)
        sim_matrix = image_embs @ text_embs.T

        # For each image, take the best matching description
        best_scores, best_text_idx = sim_matrix.max(dim=1)

        results = [
            {
                "name_clothes": names[i],
                "similarity": float(best_scores[i]),
                "best_description": descriptions[best_text_idx[i]],
            }
            for i in range(len(names))
        ]

        results.sort(key=lambda x: x["similarity"], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    @torch.no_grad()
    def match_clothes_captions(
        self,
        descriptions: list[str],
        clothes_captions: dict[str, str],
        top_k: int | None = None,
    ):
        """
        descriptions: list of AI-generated text strings
        clothes_captions: {image_name: caption}
        """

        if not clothes_captions:
            return []

        names = list(clothes_captions.keys())
        captions = [clothes_captions[name] for name in names]

        # -------------------------
        # Encode text (per string)
        # -------------------------
        query_embs = self.encode_text(descriptions)     # (N_text, D)
        query_embs = F.normalize(query_embs, dim=-1)

        # -------------------------
        # Encode clothes captions
        # -------------------------
        caption_embs = self.encode_text(captions)       # (N_img, D)
        caption_embs = F.normalize(caption_embs, dim=-1)

        # -------------------------
        # Similarity: caption ↔ all descriptions
        # -------------------------
        # (N_caption, D) @ (D, N_text) → (N_caption, N_text)
        sim_matrix = caption_embs @ query_embs.T

        # For each caption, take the best matching description
        best_scores, best_text_idx = sim_matrix.max(dim=1)

        results = [
            {
                "name_clothes": Path(names[i]).stem,
                "similarity": float(best_scores[i]),
                "best_description": descriptions[best_text_idx[i]],
            }
            for i in range(len(names))
        ]

        results.sort(key=lambda x: x["similarity"], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results
