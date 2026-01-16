from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image
import numpy as np
from PIL import Image

class QwenVLEmbedder:
    def __init__(self):
        self.llm = LLM(
            model="Qwen/Qwen3-VL-Embedding-2B",
            runner="pooling",
            dtype="bfloat16",
            trust_remote_code=True,
        )

    def encode_image(self, image: Image.Image):
        inputs = [{
            "prompt": self.llm.llm_engine.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": ""}
                    ]}
                ],
                tokenize=False,
                add_generation_prompt=True,
            ),
            "multi_modal_data": {"image": image},
        }]

        output = self.llm.embed(inputs)[0]
        return np.array(output.outputs.embedding)
