from __future__ import annotations

from pathlib import Path
import os
import logging
from typing import Optional

from PIL import Image
import torch

from app.utils.device import resolve_device, resolve_dtype
from app.services.zimage.config import (
    DEFAULT_CFG_TRUNCATION,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_WIDTH,
)
from app.services.zimage.utils import (
    AttentionBackend,
    ensure_model_weights,
    load_from_local_dir,
    set_attention_backend,
)
from app.services.zimage.zimage import generate as zimage_generate


logger = logging.getLogger(__name__)


class ZimageModel:
    """
    Z-Image native PyTorch inference wrapper.
    """

    def __init__(
        self,
        model_path: str | Path = "ckpts/Z-Image-Turbo",
        repo_id: str = "Tongyi-MAI/Z-Image-Turbo",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        text_encoder_device: str | None = "cpu",
        attention_backend: str | AttentionBackend | None = None,
        compile: bool = False,
        verify_weights: bool = False,
    ):
        self.model_path = Path(model_path)
        self.repo_id = repo_id
        self.device = resolve_device(device)
        self.dtype = dtype or resolve_dtype(self.device)
        self.text_encoder_device = (
            torch.device(text_encoder_device) if text_encoder_device is not None else None
        )
        self.compile = compile
        self.verify_weights = verify_weights
        self.attention_backend = attention_backend or os.getenv("ZIMAGE_ATTENTION", "_native_flash")

        self._components: Optional[dict] = None

    def load(self):
        if self._components is not None:
            return None
        model_dir = ensure_model_weights(
            str(self.model_path),
            repo_id=self.repo_id,
            verify=self.verify_weights,
        )
        self._components = load_from_local_dir(
            model_dir,
            device=self.device,
            dtype=self.dtype,
            compile=self.compile,
        )
        text_encoder = self._components.get("text_encoder")
        if text_encoder is not None and self.text_encoder_device is not None:
            # Keep text encoder on CPU to reduce VRAM usage during generation.
            text_encoder.to(self.text_encoder_device)
            if self.device.type == "cuda" and self.text_encoder_device.type == "cpu":
                torch.cuda.empty_cache()
        set_attention_backend(self.attention_backend)
        logger.info("Z-Image loaded on %s with dtype=%s", self.device, self.dtype)
        return None

    def generate(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        seed: int | None = None,
        num_images_per_prompt: int = 1,
        cfg_truncation: float = DEFAULT_CFG_TRUNCATION,
        cfg_normalization: bool = False,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        output_type: str = "pil",
    ) -> list[Image.Image] | torch.Tensor:
        if self._components is None:
            self.load()

        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        return zimage_generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            cfg_normalization=cfg_normalization,
            cfg_truncation=cfg_truncation,
            max_sequence_length=max_sequence_length,
            output_type=output_type,
            **self._components,
        )
