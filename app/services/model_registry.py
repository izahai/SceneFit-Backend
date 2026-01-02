# app/services/model_registry.py

from typing import Dict
from app.models.mmemb_model import MmEmbModel

class ModelRegistry:
    _models: Dict[str, object] = {}

    @classmethod
    def get(cls, name: str):
        if name not in cls._models:
            cls._models[name] = cls._load(name)
        return cls._models[name]

    @staticmethod
    def _load(name: str):
        if name == "mmEmb":
            model = MmEmbModel()
        else:
            raise ValueError(f"Unknown model: {name}")

        model.load()
        return model
