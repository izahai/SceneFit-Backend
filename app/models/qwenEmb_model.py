import torch
from torch import Tensor
import torch.nn.functional as F
from app.utils.device import resolve_device, resolve_dtype, resolve_autocast
from transformers import AutoTokenizer, AutoModel

class Qwen3Emb_Model():
    def __init__(self, device: str | None = None, config_name='Qwen/Qwen3-Embedding-0.6B'):
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device)
        self.autocast = resolve_autocast(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(config_name, padding_side="left")
        self.model = AutoModel.from_pretrained(
            config_name,
            torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    def last_token_pool(self,
                        last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        max_length: int = 512,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Returns:Tensor of shape (len(texts), hidden_dim)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            with self.autocast:
                outputs = self.model(**batch)
                embeddings = self.last_token_pool(
                    outputs.last_hidden_state, batch["attention_mask"]
                )

            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)        