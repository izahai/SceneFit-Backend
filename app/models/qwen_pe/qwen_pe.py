from PIL import Image
import faiss
import json
import yaml
import numpy as np

from models.qwen_pe.qwen_embed import QwenVLEmbedder
from models.negative_generator import NegativePEModel
from models.vl_model import VLModel


class QwenPE:
    def __init__(self):
        # -------- Models --------
        self.vlm = VLModel()                  # Qwen-VL for generation / reasoning
        self.qwen = QwenVLEmbedder()           # Qwen3-VL embedding model
        self.pe = NegativePEModel()            # PE-Core model

        # -------- FAISS indices --------
        self.qwen_index = faiss.read_index("app/data/faiss/qwen.index")
        self.pe_index = faiss.read_index("app/data/faiss/pe.index")

        self.qwen_meta = json.load(open("app/data/faiss/qwen_meta.json"))
        self.pe_meta = json.load(open("app/data/faiss/pe_meta.json"))

        # -------- Prompts --------
        with open("app/prompts/prompts.yaml") as f:
            self.prompts = yaml.safe_load(f)

    # -------------------------------------------------
    # Scene understanding (Qwen-VL, JSON output)
    # -------------------------------------------------
    def parse_scene(self, bg_image: Image.Image) -> dict:
        prompt = self.prompts["scene"]

        raw = self.vlm.generate(
            image=bg_image,
            system_prompt=prompt["system"],
            user_prompt=prompt["user"],
        )

        # Expect strict JSON from prompt
        return json.loads(raw)

    # -------------------------------------------------
    # Generate positive / negative clothing descriptions
    # -------------------------------------------------
    def generate_pos_neg(self, scene: dict) -> tuple[str, str]:
        scene_text = (
            f"Environment: {scene.get('environment')}. "
            f"Mood and color tone: {scene.get('mood')}."
        )

        pos_prompt = self.prompts["positive_clothing"]
        neg_prompt = self.prompts["negative_clothing"]

        pos = self.vlm.generate(
            system_prompt=pos_prompt["system"],
            user_prompt=pos_prompt["user"].format(scene=scene_text),
        )

        neg = self.vlm.generate(
            system_prompt=neg_prompt["system"],
            user_prompt=neg_prompt["user"].format(scene=scene_text),
        )

        return pos, neg

    # -------------------------------------------------
    # Stage-1 Recall (Qwen embedding + PE text probe)
    # -------------------------------------------------
    def recall(self, positive_text: str, k: int = 300) -> list[dict]:
        # Qwen semantic recall
        q_emb = self.qwen.encode_text(positive_text).astype("float32")
        _, q_ids = self.qwen_index.search(q_emb[None], k)

        # PE-Core text probe recall
        p_emb = self.pe.encode_text([positive_text]).cpu().numpy()
        _, p_ids = self.pe_index.search(p_emb, k)

        ids = set(q_ids[0]) | set(p_ids[0])

        return [
            self.qwen_meta[i]
            for i in ids
            if i >= 0 and i < len(self.qwen_meta)
        ]

    # -------------------------------------------------
    # Stage-2 Rerank (NO RENDER, contrastive)
    # -------------------------------------------------
    def fast_rerank(
        self,
        candidates: list[dict],
        pos_text: str,
        neg_text: str,
    ):
        images = [Image.open(c["file"]) for c in candidates]

        img_emb = self.pe.encode_image(images)
        pos_emb = self.pe.encode_text([pos_text])
        neg_emb = self.pe.encode_text([neg_text])

        pos_sim = (img_emb @ pos_emb.T).squeeze(1)
        neg_sim = (img_emb @ neg_emb.T).squeeze(1)

        score = pos_sim / (pos_sim + self.pe.lambda_neg * neg_sim)

        return sorted(
            zip(candidates, score.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
