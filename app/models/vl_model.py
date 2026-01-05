# app/models/vl_model.py

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

from app.utils.device import resolve_device, resolve_dtype
from app.utils.util import load_prompt_by_key


class VLModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 384,
    ):
        self.model_name = model_name
        #self.device = resolve_device(device)
        #self.dtype = resolve_dtype(self.device)
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype="auto",
            device_map="auto",
        )
        self.model.eval()

        self.system_role = load_prompt_by_key("system_role")
        self.vlm_task = load_prompt_by_key("vlm_task")

    # -------------------------
    # Core generation helper
    # -------------------------
    @torch.no_grad()
    def _generate(self, messages: list[dict]) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        print("Generating ...")
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # Remove prompt tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[-1]:]

        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    # -------------------------
    # Image → clothing captions
    # -------------------------
    def generate_clothing_from_image(self, image_path: str) -> list[str]:
        """
        Given an image URL, generate 3 clothing descriptions.
        """
        
        image = Image.open(image_path).convert("RGB")


        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_role,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": self.vlm_task,
                    },
                ],
            },
        ]

        output = self._generate(messages).strip()
        print(output)

        paragraphs = [p.strip() for p in output.split("\n") if p.strip()]
        print(paragraphs)

        # Ensure exactly 3 outputs
        if len(paragraphs) > 3:
            paragraphs = paragraphs[:3]
        elif len(paragraphs) < 3:
            paragraphs += [""] * (3 - len(paragraphs))

        return paragraphs
