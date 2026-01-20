# app/core/lifecycle.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch

from app.services.model_registry import ModelRegistry


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
    # ModelRegistry.get("jina-v4")

    #  ---------- PE ----------
    # print("[START] Loading Perception Model ...")
    # ModelRegistry.get("pe")
    
    # ---------- Qwen3 ----------
    # print("[START] Loading Qwen3 ...")
    #ModelRegistry.get("vlm")
    #ModelRegistry.get("qwen_reranker")
    #  ---------- PE Matcher ----------
    # print("[START] Loading PE Matcher ...")
    #ModelRegistry.get("pe_clip_matcher")
    #ModelRegistry.get("negative_pe")
    #ModelRegistry.get("qwen_pe")
    # ---------- Diffusion ----------
    print("[START] Loading Diffusion ...")
    # ModelRegistry.get("diffusion")
    
    print("[START] Models loaded")
    print("[START] Backend started")

    yield  # Application runs here

    # ---------- Shutdown ----------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Shutdown] Backend shutdown")
