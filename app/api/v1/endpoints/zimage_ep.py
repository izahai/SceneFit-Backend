# app/api/v1/endpoints/zimage_ep.py

from io import BytesIO

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.services.model_registry import ModelRegistry
from app.services.zimage.config import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_WIDTH,
)

router = APIRouter()


class ZimageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str | None = None
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    seed: int | None = None


@router.post("/zimage", response_class=Response)
def generate_zimage(request: ZimageRequest):
    model = ModelRegistry.get("zimage")
    images = model.generate(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        height=request.height,
        width=request.width,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
        output_type="pil",
    )

    image = images[0]
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
