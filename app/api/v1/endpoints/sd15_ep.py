# app/api/v1/endpoints/sd15_ep.py

import base64
import random
import threading
from io import BytesIO

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import Response
from PIL import Image

from app.models.sd15_model import SD15Model

router = APIRouter()

_model = None
_model_lock = threading.Lock()


def _get_model() -> SD15Model:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SD15Model()
                _model.load()
    return _model


def _image_to_png_bytes(image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@router.get("/sd15")
def generate_sd15(
    prompt: str = Query(..., min_length=1),
    negative_prompt: str | None = Query(None),
    steps: int = Query(30, ge=1, le=150),
    guidance_scale: float = Query(7.5, ge=0.0, le=20.0),
    width: int = Query(512, ge=64, le=2048),
    height: int = Query(512, ge=64, le=2048),
    seed: int | None = Query(None),
    return_base64: bool = Query(False),
):
    """
    Generate an image using Stable Diffusion 1.5.
    """
    model = _get_model()
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    try:
        image = model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if return_base64:
        png_bytes = _image_to_png_bytes(image)
        return {
            "seed": seed,
            "image_base64": base64.b64encode(png_bytes).decode("ascii"),
        }

    return Response(content=_image_to_png_bytes(image), media_type="image/png")


@router.post("/sd15/denoise")
def denoise_sd15(
    prompt: str = Form(...),
    negative_prompt: str | None = Form(None),
    image: UploadFile | None = File(None),
    width: int | None = Form(None),
    height: int | None = Form(None),
    steps: int = Form(50),
    guidance_scale: float = Form(7.5),
    strength: float = Form(0.75),
    eta: float = Form(0.0),
    seed: int | None = Form(None),
    noise_seed: int | None = Form(None),
    return_base64: bool = Form(False),
):
    """
    Img2img-style denoising: add fixed noise then denoise with prompts.
    """
    model = _get_model()
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    if image is None:
        width = 512 if width is None else width
        height = 512 if height is None else height
        init_image = Image.new("RGB", (width, height), color=(127, 127, 127))
    else:
        try:
            init_image = Image.open(BytesIO(image.file.read())).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid image upload.") from exc

    try:
        if return_base64:
            result, step_images, noise_image = model.generate_from_image(
                image=init_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                strength=strength,
                width=width,
                height=height,
                seed=seed,
                noise_seed=noise_seed,
                eta=eta,
                return_intermediates=True,
            )
        else:
            result = model.generate_from_image(
                image=init_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                strength=strength,
                width=width,
                height=height,
                seed=seed,
                noise_seed=noise_seed,
                eta=eta,
            )
            step_images = []
            noise_image = None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if return_base64:
        png_bytes = _image_to_png_bytes(result)
        step_images_base64 = [
            base64.b64encode(_image_to_png_bytes(step_image)).decode("ascii")
            for step_image in step_images
        ]
        return {
            "seed": seed,
            "image_base64": base64.b64encode(png_bytes).decode("ascii"),
            "step_images_base64": step_images_base64,
            "noise_image_base64": (
                base64.b64encode(_image_to_png_bytes(noise_image)).decode("ascii")
                if noise_image is not None
                else None
            ),
        }

    return Response(content=_image_to_png_bytes(result), media_type="image/png")
