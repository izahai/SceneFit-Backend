# app/core/lifecycle.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch

from app.core.vector_db import VectorDatabase
from app.models.image_edit_model import ImageEditFlux
from app.services.model_registry import ModelRegistry
from app.models.mmemb_model import MmEmbModel
from app.models.pe_clip_model import PEClipModel
from app.models.vl_model import VLModel
from app.models.pe_clip_matcher import PEClipMatcher
from app.models.diffusion_model import DiffusionModel
# from app.models.vqvae_model import VQVAEModel
from app.core.vector_db import VectorDatabase

# Expose vector DB for downstream endpoints
vector_db: VectorDatabase | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # ---------- Startup ----------
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[START] Backend started")

    print("[START] Loading models...")

    # ----- Aesthetic Predictor -----
    ModelRegistry.get("aesthetic")
    print("[START] Models loaded")
    print("[START] Backend started")

    yield  # Application runs here

    # ---------- Shutdown ----------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Shutdown] Backend shutdown")
