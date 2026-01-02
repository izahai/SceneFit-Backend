# app/core/lifecycle.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch

from app.services.model_registry import ModelRegistry
from app.models.mmemb_model import MmEmbModel
from app.models.pe_clip_model import PEClipModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # ---------- Startup ----------
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[START] Backend started")


    print("[START] Loading models...")

    # Initialize Jina model
    # ModelRegistry.register(
    #     name="mmEmb",
    #     model=MmEmbModel(),
    # )

    print("[START] Loading Perception Model ...")
    ModelRegistry.register(
        name="pe",
        model=PEClipModel(),
    )

    print("[START] Models loaded")
    print("[START] Backend started")

    yield  # Application runs here

    # ---------- Shutdown ----------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Shutdown] Backend shutdown")