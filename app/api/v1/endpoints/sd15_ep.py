# app/api/v1/endpoints/sd15_ep.py

import base64
import random
import threading
from io import BytesIO

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

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
