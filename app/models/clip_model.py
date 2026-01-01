from app.models.base_model import BaseModel

class CLIPModel(BaseModel):

    def load(self):
        # load CLIP weights here
        self.model = "clip_loaded"

    def infer(self, text: str):
        return {
            "embedding_dim": 512,
            "text": text
        }
