# app/services/model_registry.py

from typing import Dict, Any
from app.models.mmemb_model import MmEmbModel
from app.models.pe_clip_model import PEClipModel 
from app.models.vl_model import VLModel
from app.models.pe_clip_matcher import PEClipMatcher
from app.models.diffusion_model import DiffusionModel

class ModelRegistry:
    _models: Dict[str, object] = {}

    @classmethod
    def register(cls, name: str, model: Any):
        cls._models[name] = model

    @classmethod
    def get(cls, name: str):
        if name not in cls._models:
            cls._models[name] = cls._load(name)
        return cls._models[name]

    @staticmethod
    def _load(name: str):
        if name == "jina-v4":
            model = MmEmbModel()
        elif name == "pe":
            model = PEClipModel()
        elif name == "vlm":
            model = VLModel()
        elif name == "pe_clip_matcher":
            model = PEClipMatcher()
        elif name == "diffusion":
            model = DiffusionModel()
        else:
            raise ValueError(f"Unknown model: {name}")

        model.load()
        return model