# app/models/sd15_model.py

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import random
import sys
import gc
from pathlib import Path

import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class SD15Model:
    """
    Stand-alone SD1.5 loader/inference class using the original CompVis repo.

    Required environment variables (if not passed to __init__):
    - SD15_REPO_PATH: path to the cloned CompVis/stable-diffusion repo
    - SD15_CKPT_PATH: path to the SD1.5 checkpoint (e.g. v1-5-pruned-emaonly.ckpt)
    - SD15_CONFIG_PATH: optional path to the config yaml
    """

    def __init__(
        self,
        ckpt_path: str | None = None,
        config_path: str | None = None,
        repo_path: str | None = None,
        device: str | None = None,
        precision: str = "autocast",
        steps: int = 30,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        eta: float = 0.0,
        seed: int | None = None,
        sampler: str = "ddim",
    ):
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.repo_path = repo_path
        self.device = self._resolve_device(device)
        self.precision = precision
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.eta = eta
        self.seed = seed
        self.sampler_name = sampler.lower()

        self._latent_channels = 4
        self._downsample_factor = 8
        self._model = None
        self._sampler = None

    def load(self) -> None:
        if self._model is not None:
            return

        repo_path = self._resolve_repo_path(self.repo_path)
        config_path = self._resolve_config_path(self.config_path, repo_path)
        ckpt_path = self._resolve_ckpt_path(self.ckpt_path, repo_path)

        self.repo_path = str(repo_path)
        self.config_path = str(config_path)
        self.ckpt_path = str(ckpt_path)

        self._ensure_repo_on_path(repo_path)
        self._model = self._load_model(config_path, ckpt_path)
        self._sampler = self._create_sampler(self.sampler_name, self._model)

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        eta: float | None = None,
    ) -> Image.Image:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        if self._model is None or self._sampler is None:
            self.load()

        steps = self.steps if steps is None else int(steps)
        guidance_scale = self.guidance_scale if guidance_scale is None else float(guidance_scale)
        width = self.width if width is None else int(width)
        height = self.height if height is None else int(height)
        eta = self.eta if eta is None else float(eta)

        width, height = self._normalize_size(width, height)
        seed = self._resolve_seed(seed)

        self._seed_everything(seed)

        shape = [
            self._latent_channels,
            height // self._downsample_factor,
            width // self._downsample_factor,
        ]

        neg_prompt = negative_prompt or ""

        try:
            with torch.inference_mode():
                with self._precision_context():
                    with self._ema_scope():
                        conditioning = self._model.get_learned_conditioning([prompt])
                        uncond = None
                        if guidance_scale != 1.0:
                            uncond = self._model.get_learned_conditioning([neg_prompt])

                        samples, _ = self._sampler.sample(
                            S=steps,
                            conditioning=conditioning,
                            batch_size=1,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=guidance_scale,
                            unconditional_conditioning=uncond,
                            eta=eta,
                        )

                        decoded = self._model.decode_first_stage(samples)
                        decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)

            return self._tensor_to_pil(decoded[0])
        finally:
            try:
                del conditioning
                del uncond
                del samples
                del decoded
            except Exception:
                pass
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def generate_from_image(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        strength: float = 0.75,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        noise_seed: int | None = None,
        eta: float | None = None,
        return_intermediates: bool = False,
    ) -> Image.Image | tuple[Image.Image, list[Image.Image], Image.Image | None]:
        if image is None:
            raise ValueError("Image is required for img2img denoising.")
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        if self._model is None or self._sampler is None:
            self.load()

        if not hasattr(self._sampler, "stochastic_encode"):
            raise RuntimeError("Sampler does not support img2img; use ddim.")

        steps = self.steps if steps is None else int(steps)
        guidance_scale = self.guidance_scale if guidance_scale is None else float(guidance_scale)
        eta = self.eta if eta is None else float(eta)

        if strength < 0.0 or strength > 1.0:
            raise ValueError("strength must be between 0.0 and 1.0")

        width, height = self._resolve_init_size(image, width, height)
        seed = self._resolve_seed(seed)
        noise_seed = seed if noise_seed is None else int(noise_seed)

        self._seed_everything(seed)

        device = self.device
        batch_size = 1
        try:
            init_tensor = self._prepare_init_image(image, width, height).to(device)

            with torch.inference_mode():
                with self._precision_context():
                    with self._ema_scope():
                        init_latent = self._model.get_first_stage_encoding(
                            self._model.encode_first_stage(init_tensor)
                        )

                        self._sampler.make_schedule(
                            ddim_num_steps=steps,
                            ddim_eta=eta,
                            verbose=False,
                        )
                        t_enc = int(strength * steps)

                        noise_gen = torch.Generator(device=device).manual_seed(noise_seed)
                        noise = torch.randn(
                            init_latent.shape,
                            generator=noise_gen,
                            device=device,
                            dtype=init_latent.dtype,
                        )
                        t_tensor = torch.tensor([t_enc] * batch_size, device=device)
                        z_enc = self._sampler.stochastic_encode(
                            init_latent,
                            t_tensor,
                            noise=noise,
                        )

                        uc = None
                        if guidance_scale != 1.0:
                            uc = self._model.get_learned_conditioning([negative_prompt or ""])
                        c = self._model.get_learned_conditioning([prompt])

                        if return_intermediates:
                            noise_decoded = self._model.decode_first_stage(z_enc)
                            noise_decoded = torch.clamp(
                                (noise_decoded + 1.0) / 2.0, 0.0, 1.0
                            )
                            noise_image = self._tensor_to_pil(noise_decoded[0])
                            del noise_decoded

                            samples, step_images = self._decode_with_intermediates(
                                z_enc,
                                c,
                                t_enc,
                                unconditional_guidance_scale=guidance_scale,
                                unconditional_conditioning=uc,
                            )
                        else:
                            samples = self._sampler.decode(
                                z_enc,
                                c,
                                t_enc,
                                unconditional_guidance_scale=guidance_scale,
                                unconditional_conditioning=uc,
                            )
                            step_images = None
                            noise_image = None

                        decoded = self._model.decode_first_stage(samples)
                        decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)

            result_image = self._tensor_to_pil(decoded[0])
            if return_intermediates:
                return result_image, step_images or [], noise_image
            return result_image
        finally:
            try:
                del init_tensor
                del init_latent
                del noise
                del z_enc
                del samples
                del decoded
            except Exception:
                pass
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _decode_with_intermediates(
        self,
        x_latent: torch.Tensor,
        cond,
        t_start: int,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning=None,
        use_original_steps: bool = False,
    ) -> tuple[torch.Tensor, list[Image.Image]]:
        timesteps = (
            np.arange(self._sampler.ddpm_num_timesteps)
            if use_original_steps
            else self._sampler.ddim_timesteps
        )
        t_start = max(0, min(int(t_start), timesteps.shape[0]))
        if t_start == 0:
            return x_latent, []

        timesteps = timesteps[:t_start]
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        x_dec = x_latent
        step_images: list[Image.Image] = []

        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            ts = torch.full(
                (x_latent.shape[0],),
                step,
                device=x_latent.device,
                dtype=torch.long,
            )
            x_dec, pred_x0 = self._sampler.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
            decoded = self._model.decode_first_stage(pred_x0)
            decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
            step_images.append(self._tensor_to_pil(decoded[0]))
            del pred_x0
            del decoded

        return x_dec, step_images

    def _precision_context(self):
        if self.precision == "autocast" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda")
        return contextlib.nullcontext()

    def _ema_scope(self):
        ema_scope = getattr(self._model, "ema_scope", None)
        if callable(ema_scope):
            return ema_scope()
        return contextlib.nullcontext()

    def _load_model(self, config_path: Path, ckpt_path: Path):
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency omegaconf. Install stable-diffusion requirements."
            ) from exc

        try:
            config = OmegaConf.load(config_path)
            model = self._instantiate_from_config(config.model)
            load_device = self.device if self.device.type == "cuda" else torch.device("cpu")
            if load_device.type == "cuda":
                model.to(load_device)
            checkpoint = self._load_checkpoint(ckpt_path, load_device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("SD15 missing keys: %s", missing)
            if unexpected:
                logger.warning("SD15 unexpected keys: %s", unexpected)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Missing dependency {exc.name}. Install stable-diffusion requirements."
            ) from exc
        finally:
            try:
                del checkpoint
                del state_dict
            except Exception:
                pass
            gc.collect()

        if self.precision == "fp16" and self.device.type == "cuda":
            model = model.half()

        model.to(self.device)
        model.eval()
        return model

    def _create_sampler(self, sampler_name: str, model):
        if sampler_name == "plms":
            from ldm.models.diffusion.plms import PLMSSampler

            return PLMSSampler(model)

        if sampler_name != "ddim":
            raise ValueError(f"Unsupported sampler: {sampler_name}")

        from ldm.models.diffusion.ddim import DDIMSampler

        return DDIMSampler(model)

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _resolve_seed(seed: int | None) -> int:
        if seed is not None:
            return int(seed)
        return random.randint(0, 2**32 - 1)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _ensure_repo_on_path(repo_path: Path) -> None:
        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    @staticmethod
    def _resolve_repo_path(repo_path: str | None) -> Path:
        candidates = []
        if repo_path:
            candidates.append(Path(repo_path))

        env_repo = os.getenv("SD15_REPO_PATH") or os.getenv("STABLE_DIFFUSION_REPO")
        if env_repo:
            candidates.append(Path(env_repo))

        app_root = Path(__file__).resolve().parents[1]
        base_root = Path(__file__).resolve().parents[2]
        for base in (Path.cwd(), base_root):
            candidates.append(base / "stable-diffusion")
            candidates.append(base / "stable_diffusion")
            candidates.append(base / "app/services/stable-diffusion")
            candidates.append(base / "app/services/stable_diffusion")
        candidates.append(app_root / "services/stable-diffusion")
        candidates.append(app_root / "services/stable_diffusion")

        for candidate in candidates:
            if candidate and candidate.exists() and (candidate / "ldm").exists():
                return candidate.resolve()

        raise FileNotFoundError(
            "Stable Diffusion repo not found. Set SD15_REPO_PATH to the cloned "
            "CompVis/stable-diffusion directory."
        )

    @staticmethod
    def _resolve_config_path(config_path: str | None, repo_path: Path) -> Path:
        if config_path:
            path = Path(config_path)
        else:
            env_path = os.getenv("SD15_CONFIG_PATH")
            if env_path:
                path = Path(env_path)
            else:
                path = repo_path / "configs/stable-diffusion/v1-inference.yaml"

        if not path.exists():
            raise FileNotFoundError(f"SD1.5 config not found: {path}")

        return path.resolve()

    @staticmethod
    def _resolve_ckpt_path(ckpt_path: str | None, repo_path: Path) -> Path:
        if ckpt_path:
            path = Path(ckpt_path)
        else:
            env_keys = (
                "SD15_CKPT_PATH",
                "SD15_WEIGHTS",
                "SD15_MODEL_PATH",
                "SD15_CKPT",
            )
            path = None
            for key in env_keys:
                value = os.getenv(key)
                if value:
                    path = Path(value)
                    break
            if path is None:
                default_paths = [
                    repo_path / "models/ldm/stable-diffusion-v1/model.ckpt",
                    repo_path / "models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt",
                    repo_path / "models/ldm/stable-diffusion-v1/v1-5-pruned.ckpt",
                    repo_path / "v1-5-pruned-emaonly.ckpt",
                    repo_path / "v1-5-pruned.ckpt",
                    repo_path / "v1-5-pruned-emaonly.safetensors",
                ]
                for candidate in default_paths:
                    if candidate.exists():
                        path = candidate
                        break

            if path is None:
                path = SD15Model._auto_download_ckpt(repo_path)

        if path is None or not path.exists():
            raise FileNotFoundError(
                "SD1.5 checkpoint not found. Set SD15_CKPT_PATH to the .ckpt file "
                "downloaded from https://huggingface.co/stable-diffusion-v1-5/"
                "stable-diffusion-v1-5."
            )

        return path.resolve()

    @staticmethod
    def _auto_download_ckpt(repo_path: Path) -> Path | None:
        repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        filename = os.getenv("SD15_CKPT_FILENAME", "v1-5-pruned-emaonly.ckpt")
        target_dir = repo_path / "models/ldm/stable-diffusion-v1"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename

        if target_path.exists():
            return target_path

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency huggingface_hub for SD1.5 auto-download."
            ) from exc

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to download SD1.5 checkpoint from Hugging Face. "
                "Set SD15_CKPT_PATH manually or ensure HF_TOKEN is valid."
            ) from exc

        return Path(downloaded)

    @staticmethod
    def _load_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
        if ckpt_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency safetensors for .safetensors checkpoint."
                ) from exc
            return load_file(str(ckpt_path), device=str(device))
        return torch.load(ckpt_path, map_location=device, weights_only=False)

    @staticmethod
    def _resolve_init_size(
        image: Image.Image,
        width: int | None,
        height: int | None,
    ) -> tuple[int, int]:
        if width is None or height is None:
            img_w, img_h = image.size
            width = img_w if width is None else width
            height = img_h if height is None else height
        return SD15Model._normalize_size_to_multiple(width, height, multiple=32)

    @staticmethod
    def _normalize_size_to_multiple(
        width: int,
        height: int,
        multiple: int = 32,
    ) -> tuple[int, int]:
        width = max(multiple, width)
        height = max(multiple, height)
        if width % multiple != 0 or height % multiple != 0:
            new_width = width - (width % multiple)
            new_height = height - (height % multiple)
            logger.info(
                "Adjusting size to multiples of %s: %sx%s -> %sx%s",
                multiple,
                width,
                height,
                new_width,
                new_height,
            )
            width, height = new_width, new_height
        return width, height

    @staticmethod
    def _prepare_init_image(
        image: Image.Image,
        width: int,
        height: int,
    ) -> torch.Tensor:
        image = image.convert("RGB")
        if (width, height) != image.size:
            image = image.resize((width, height), Image.LANCZOS)
        image_array = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        return tensor * 2.0 - 1.0

    @staticmethod
    def _normalize_size(width: int, height: int) -> tuple[int, int]:
        width = max(64, width)
        height = max(64, height)
        if width % 8 != 0 or height % 8 != 0:
            new_width = width - (width % 8)
            new_height = height - (height % 8)
            logger.info(
                "Adjusting size to multiples of 8: %sx%s -> %sx%s",
                width,
                height,
                new_width,
                new_height,
            )
            width, height = new_width, new_height
        return width, height

    @staticmethod
    def _get_obj_from_str(target: str):
        module_path, cls_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)

    def _instantiate_from_config(self, config):
        if "target" not in config:
            raise KeyError("Expected key `target` to instantiate.")
        params = config.get("params", {})
        return self._get_obj_from_str(config["target"])(**params)

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.detach().float().cpu()
        array = tensor.permute(1, 2, 0).numpy()
        array = (array * 255.0).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)
