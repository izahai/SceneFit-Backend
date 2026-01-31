from abc import ABC
from pathlib import Path
from  diffusers import Flux2KleinPipeline
import torch
from io import BytesIO
from PIL import Image

REF_IMAGE_PATH_MAN = Path('app/data/ref_images/man.png')
REF_IMAGE_PATH_WOMAN = Path('app/data/ref_images/woman.png')

class ImageEditModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def edit_image_scene_img():
        raise NotImplementedError("Subclasses must implement this method.")
    
    def edit_outfit_desc(
        self,
        outfit_description: str,
        save_result=True,
        gender='male',
        crop_clothes=True,
        preference_text: str | None = None,
        feedback_text: str | None = None,
        ref_image_path: Path | None = None
    ):
        raise NotImplementedError("Subclasses must implement this method.")
    
class ImageEditFlux(ImageEditModel):
    def __init__(self):
        super().__init__(model_name="ImageEditFlux")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16).to(self.device)

    def _get_prompt(self, outfit_description: str, preference_text: str) -> str:
        prompt = (
            "Change the outfit of this person into the outfit described as: "
            f"{outfit_description}. "
            "Consider these preferences: "
            f"{preference_text}"
        )
        return prompt
    
    def edit_outfit_desc(
        self,
        outfit_description: str,
        gender='male',
        preference_text: str | None = None,
        ref_image_path: Path | None = None
    ) -> Image.Image:
        
        if not ref_image_path or not ref_image_path.is_file():
            if gender == 'male':
                ref_image_path = REF_IMAGE_PATH_MAN
            elif gender == 'female':
                ref_image_path = REF_IMAGE_PATH_WOMAN
            else:
                raise ValueError("Gender must be 'male' or 'female'")
        img = Image.open(ref_image_path).convert("RGB")
        img = img.resize((512,512))
        images = self.model(
            image=img,
            prompt = self._get_prompt(outfit_description, preference_text or ""),
            height=512,
            width=512,
            num_inference_steps=30,
            guidance_scale=2.0,
            generator=torch.Generator(device=self.device).manual_seed(0)
        )

        edited_image = images[0]
        return edited_image