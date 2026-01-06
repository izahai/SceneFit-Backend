# app/core/lifecycle.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch

from app.services.model_registry import ModelRegistry
from app.models.mmemb_model import MmEmbModel
from app.models.pe_clip_model import PEClipModel
from app.models.vl_model import VLModel
from app.models.pe_clip_matcher import PEClipMatcher

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # ---------- Startup ----------
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[START] Backend started")

    print("[START] Loading models...")

    #  ---------- Jina ----------
    # print("[START] Loading Jina Model ...")
    # ModelRegistry.register(
    #     name="mmEmb",
    #     model=MmEmbModel(),
    # )

    #  ---------- PE ----------
    # print("[START] Loading Perception Model ...")
    # ModelRegistry.register(
    #     name="pe",
    #     model=PEClipModel(),
    # )
    
    #  ---------- Qwen3 ----------
    print("[START] Loading Qwen3 ...")
    ModelRegistry.register(
        name="vlm",
        model=VLModel(),
    )
    
    #  ---------- PE Matcher ----------
    print("[START] Loading PE Matcher ...")
    ModelRegistry.register(
        name="pe_clip_matcher",
        model=PEClipMatcher(),
    )
    
    print("[START] Models loaded")
    print("[START] Backend started")

    yield  # Application runs here

    # ---------- Shutdown ----------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Shutdown] Backend shutdown")