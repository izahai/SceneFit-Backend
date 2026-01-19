import torch
from torch import Tensor
import torch.nn.functional as F
from app.utils.device import resolve_device, resolve_dtype, resolve_autocast
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

class TextMatcherModel():
    def __init__(self, device: str | None = None, config_name='Qwen/Qwen3-Embedding-0.6B'):
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device)

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

            outputs = self.model(**batch)
            embeddings = self.last_token_pool(
                outputs.last_hidden_state, batch["attention_mask"]
            )

            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)   
    
    @torch.no_grad()
    def encode_normalized(self, texts: list[str]):
        embs = self.encode(texts)
        return F.normalize(embs, dim=-1)
    
    @torch.no_grad()
    def apply_feedback(
        self,
        query_embs: torch.Tensor,   # (N_text, D)
        instruction_text: str | None,
        beta: float = 0.2,
    ):
        if instruction_text is None:
            return query_embs

        instr_emb = self.encode_normalized([instruction_text])  # (1, D)
        return F.normalize(query_embs + beta * instr_emb, dim=-1)

    
    @torch.no_grad
    def get_reformulated_query(self, descriptions: list[str],
                        topk_captions: list[str],
                        fb_text: str,
                        threshold: float,
                        alpha: float,
                        beta: float = 1.0,
                        gamma: float = 1.0):
        
        query_embs = self.encode_normalized(descriptions)
        caption_embs = self.encode_normalized(topk_captions)
        fb_embs = self.encode_normalized(fb_text)           # (M, D) or (D,)
        
        if fb_embs.dim() == 1:
            fb_embs = fb_embs.unsqueeze(0)

        # Similarity matrix: (N_captions, N_feedback)
        best_sim_matrix = caption_embs @ fb_embs.T

        relevant_mask = best_sim_matrix > threshold
        irrelevant_mask = best_sim_matrix <= threshold

        relevant_embs = caption_embs[relevant_mask.any(dim=1)]
        irrelevant_embs = caption_embs[irrelevant_mask.any(dim=1)]

        # Means (guard against empty sets)
        rel_mean = relevant_embs.mean(dim=0) if relevant_embs.numel() > 0 else 0.0
        irrel_mean = irrelevant_embs.mean(dim=0) if irrelevant_embs.numel() > 0 else 0.0

        reform_q = (
            alpha * query_embs +
            beta * rel_mean -
            gamma * irrel_mean
        )
        return reform_q
    
    @torch.no_grad()
    def get_clothes_feedback(
        self,
        descriptions: list[str],
        clothes_captions: dict[str, str],
        fb_text: str,
        top_k: int | None = None,
        tau: float = 0.1,        # temperature (important)
    ):
        if not clothes_captions:
            return []

        # -------------------------
        # Encode descriptions (Q)
        # -------------------------
        Q = self.encode_normalized(descriptions)      # (N_desc, D)

        # -------------------------
        # Encode feedback (f)
        # -------------------------
        f = self.encode_normalized([fb_text])[0]      # (D,)

        # -------------------------
        # Attention over descriptions
        # -------------------------
        # similarity between feedback and each description
        attn_logits = (Q @ f) / tau                   # (N_desc,)
        weights = torch.softmax(attn_logits, dim=0)   # (N_desc,)

        # -------------------------
        # Collapse to single query
        # -------------------------
        q_fb = torch.sum(weights[:, None] * Q, dim=0) # (D,)
        q_fb = F.normalize(q_fb, dim=0)

        # -------------------------
        # Encode clothes captions
        # -------------------------
        names = list(clothes_captions.keys())
        captions = [clothes_captions[n] for n in names]
        caption_embs = self.encode_normalized(captions)  # (N_caps, D)

        # -------------------------
        # Final scoring
        # -------------------------
        scores = caption_embs @ q_fb                  # (N_caps,)

        results = [
            {
                "name_clothes": Path(names[i]).stem,
                "similarity": float(scores[i]),
                "best_description": descriptions[int(weights.argmax())],
            }
            for i in range(len(names))
        ]

        results.sort(key=lambda x: x["similarity"], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results
    
    @torch.no_grad()
    def get_clothes_feedback_rocchio(
        self,
        descriptions: list[str],
        clothes_captions: dict[str, str],
        topk_captions: list[str],
        fb_text: str,
        top_k: int | None = None,
        threshold: float = 0.3,     # relevance threshold
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.25,
    ):
        if not clothes_captions:
            return []

        # -------------------------
        # Prepare data
        # -------------------------
        names = list(clothes_captions.keys())
        captions = [clothes_captions[n] for n in names]

        # -------------------------
        # Reformulate queries (Rocchio)
        # -------------------------
        query_embs = self.get_reformulated_query(
            descriptions=descriptions,
            topk_captions=topk_captions,
            fb_text=fb_text,
            threshold=threshold,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        query_embs = F.normalize(query_embs, dim=-1)   # (N_desc, D)

        # -------------------------
        # Encode captions
        # -------------------------
        caption_embs = self.encode_normalized(captions)  # (N_cap, D)

        # -------------------------
        # Similarity & ranking
        # -------------------------
        sim_matrix = caption_embs @ query_embs.T
        best_scores, best_text_idx = sim_matrix.max(dim=1)

        results = [
            {
                "name_clothes": Path(names[i]).stem,
                "similarity": float(best_scores[i]),
                "best_description": descriptions[int(best_text_idx[i])],
            }
            for i in range(len(names))
        ]

        results.sort(key=lambda x: x["similarity"], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    
    @torch.no_grad()
    def match_clothes_captions(
        self,
        descriptions: list[str],
        clothes_captions: dict[str, str],
        # fb_text: str | None = None,
        # beta: float = 0.2,
        top_k: int | None = None,
    ):
        """
        descriptions: list of AI-generated text strings
        clothes_captions: {image_name: caption}
        """
        if not clothes_captions:
            return []

        names = list(clothes_captions.keys())
        captions = [clothes_captions[name] for name in names]
        
        query_embs = self.encode_normalized(descriptions)
        # query_embs = self.apply_feedback(query_embs, fb_text, beta)

        caption_embs = self.encode_normalized(captions)

        # Similarity: caption ↔ all descriptions
        # (N_caption, D) @ (D, N_text) → (N_caption, N_text)
        sim_matrix = caption_embs @ query_embs.T

        # For each caption, take the best matching description
        best_scores, best_text_idx = sim_matrix.max(dim=1)
        results = [
            {
                "name_clothes": Path(names[i]).stem,
                "similarity": float(best_scores[i]),
                "best_description": descriptions[int(best_text_idx[i])],
            }
            for i in range(len(names))
        ]

        results.sort(key=lambda x: x["similarity"], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results