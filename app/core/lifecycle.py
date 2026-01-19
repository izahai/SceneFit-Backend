# app/core/lifecycle.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch

from app.core.vector_db import VectorDatabase
from app.services.model_registry import ModelRegistry
from app.models.mmemb_model import MmEmbModel
from app.models.pe_clip_model import PEClipModel
from app.models.vl_model import VLModel
from app.models.pe_clip_matcher import PEClipMatcher
from app.models.diffusion_model import DiffusionModel
from app.models.vqvae_model import VQVAEModel

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
    
    # ---------- Qwen3 ----------
    # print("[START] Loading Qwen3 ...")
    # ModelRegistry.register(
    #     name="vlm",
    #     model=VLModel(),
    # )
    
    #  ---------- PE Matcher ----------
    # print("[START] Loading PE Matcher ...")
    # ModelRegistry.register(
    #     name="pe_clip_matcher",
    #     model=PEClipMatcher(),
    # )
    
    # ---------- Diffusion ----------
    # print("[START] Loading Diffusion ...")
    # ModelRegistry.register(
    #     name="diffusion",
    #     model=DiffusionModel(),
    # )

    # ---------- VQVAE ----------
    # print("[START] Loading VQVAE Model ...")
    # ModelRegistry.register(
    #     name="vqvae",
    #     model=VQVAEModel(),
    # )
    
    vector_db = VectorDatabase(
        embedding_model="pe",
        use_gpu=True,  # set False if you prefer CPU
        index_path="app/data/indexes/pe_clothes.index",            # optional explicit paths
        metadata_path="app/data/indexes/pe_clothes.index.meta.json",
        data_dir="app/data/2d",                            # where to build from if missing
        auto_prepare=True,                                 # triggers ensure_ready()
    )
    app.state.vector_db = vector_db
    print("[START] Models loaded")
    print("[START] Backend started")

    yield  # Application runs here

    # ---------- Shutdown ----------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Shutdown] Backend shutdown")