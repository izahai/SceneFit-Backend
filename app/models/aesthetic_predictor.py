import torch
import torch.nn as nn
from urllib.request import urlretrieve
import os
from os.path import expanduser
from app.utils.device import resolve_device
from app.models.pe_clip_model import PEClipModel

class AestheticPredictor:
    def __init__(self, device=None):
        self.device = resolve_device(device)
        self.pe = PEClipModel(device=self.device, autocast=False)
        self.head = self.get_aesthetic_model().to(self.device)
        self.head.eval()
        
    def get_aesthetic_model(self, clip_model="vit_l_14"):
        home = expanduser("~")
        cache_folder = home + "/.cache/emb_reader"
        path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = (
                "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
            )
            urlretrieve(url_model, path_to_model)
        if clip_model == "vit_l_14":
            m = nn.Linear(768, 1)
        # elif clip_model == "vit_b_32":
            # m = nn.Linear(512, 1)
        else:
            raise ValueError()
        s = torch.load(path_to_model)
        m.load_state_dict(s)
        m.eval()
        return m

    @torch.no_grad()
    def score_images(self, items):
        names, images = zip(*items)
        embeddings = self.pe.encode_image(list(images))
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        preds = self.head(embeddings).squeeze(-1).tolist()
        results = [{"name_clothes": n, "aesthetic_score": p} for n, p in zip(names, preds)]
        results.sort(key=lambda x: x["aesthetic_score"], reverse=True)
        return results
