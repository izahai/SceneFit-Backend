from zipfile import Path
from app.services.pe_core import args
from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image
import numpy as np
from PIL import Image

class QwenVLEmbedder:
    def __init__(self):
        self.llm = None

    def load(self):
        self.llm = LLM(
            engine_args=EngineArgs(
                model="Qwen/Qwen3-VL-Embedding-2B",
                runner="pooling",
                dtype=args.dtype,
                trust_remote_code=True
            )
        )

    def encode_batch(self, image_paths: list[Path]):
        """
        Process images in a single batch call.
        """
        inputs = []
        
        # 1. Pre-construct all prompts
        for p in image_paths:
            # Check if file exists/is valid image to prevent crashes
            try:
                img = Image.open(p).convert("RGB")
                
                # Construct conversation structure
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Describe this image for retrieval."}
                        ]
                    }
                ]
                
                # Apply template
                prompt = self.llm.llm_engine.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": img},
                })
            except Exception as e:
                print(f"Skipping {p}: {e}")
                continue

        # 2. Run Batch Inference
        # vLLM handles the internal batching based on available GPU memory
        outputs = self.llm.embed(inputs)
        
        # 3. Extract Embeddings
        # Note: output.outputs is a list, we take the first element's embedding
        embeddings = [out.outputs.embedding for out in outputs]
        return np.array(embeddings)        