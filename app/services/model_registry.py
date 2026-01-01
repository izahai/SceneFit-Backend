from typing import Dict
from app.models.clip_model import CLIPModel
from app.models.diffusion_model import DiffusionModel
from app.models.vl_model import VLMModel

class ModelRegistry:
    _models: Dict[str, object] = {}

    @classmethod
    def get(cls, name: str):
        if name not in cls._models:
            cls._models[name] = cls._load(name)
        return cls._models[name]

    @staticmethod
    def _load(name: str):
        if name == "clip":
            model = CLIPModel()
        # elif name == "diffusion":
        #     model = DiffusionModel()
        # elif name == "vlm":
        #     model = VLMModel()
        else:
            raise ValueError(f"Unknown model: {name}")

        model.load()
        return model