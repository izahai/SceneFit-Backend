
import json
import time
import os
import google.generativeai as genai
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional

# ---------------------------------------------------------
# 0. ROBUST GEMINI CLASS
# ---------------------------------------------------------
class GeminiModel:
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided")

        genai.configure(api_key=self.api_key)
        self.model_name = self._resolve_model_name(model_name)
        print(f"✅ Initialized Gemini SDK with resolved model: {self.model_name}")

        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={"response_mime_type": "application/json"}
        )

    def _resolve_model_name(self, requested_name: str) -> str:
        """Finds the best available model (Flash or Pro)."""
        try:
            available = [m.name.replace("models/", "") for m in genai.list_models()]
            if requested_name in available: return requested_name
            # Fallback to any available flash/pro model
            for m in available:
                if "flash" in m and "2.5" in m: return m
            for m in available:
                if "pro" in m and "2.5" in m: return m
            return requested_name
        except:
            return requested_name

    def generate_content(self, contents: list):
        try:
            return self.model.generate_content(contents)
        except Exception as e:
            time.sleep(2) # Retry once
            try:
                return self.model.generate_content(contents)
            except:
                return None

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
GEMINI_API_KEY ="AIzaSyB9kk9ev5wlPluRz963El86IAm4jy0GuAY"  # <--- PASTE NEW KEY HERE
CLOTHES_DIR = Path("/content/app/data/2d")
BG_PATH = Path("/content/app/data/bg/bg002.png")

# Input Data
RESULTS_DATA = {
  "method": "vlm-faiss-composed-retrieval",
  "count": 10,
  "scene_caption": "This vibrant, sunlit outdoor scene features a field of purple flowers under a bright blue sky, surrounded by cascading purple trees and rocky terrain, suggesting a casual, cheerful spring or summer day perfect for light, colorful, and breathable clothing that complements the vivid purple and green tones.",
  "results": [
    {
      "name_clothes": "avatars_ad56d7730bfd427aaf9677c0a2832786.png",
      "similarity": 0.2713354825973511,
      "rerank_score": 0.33645108342170715
    },
    {
      "name_clothes": "avatars_7f8a319e60a442a09b3363b0efc63de5.png",
      "similarity": 0.26883774995803833,
      "rerank_score": 0.3333745300769806
    },
    {
      "name_clothes": "avatars_c4e36d88a50849e38f271f9570f04cf7.png",
      "similarity": 0.27215754985809326,
      "rerank_score": 0.3129391372203827
    },
    {
      "name_clothes": "avatars_1f19e7485f824d95ba1236b22fce8620.png",
      "similarity": 0.2811240553855896,
      "rerank_score": 0.3053753077983856
    },
    {
      "name_clothes": "avatars_25c9a73c21084d8c9de08c24a22479c4.png",
      "similarity": 0.2754352390766144,
      "rerank_score": 0.2988414168357849
    },
    {
      "name_clothes": "avatars_0897ecda4c0249339070911a3a969206.png",
      "similarity": 0.2830156683921814,
      "rerank_score": 0.2913392186164856
    },
    {
      "name_clothes": "avatars_44943d65275a4b29b08e692e9d91ce95.png",
      "similarity": 0.26985442638397217,
      "rerank_score": 0.28558698296546936
    },
    {
      "name_clothes": "avatars_909f9eb4a670471b8365b42034b7e045.png",
      "similarity": 0.2690908908843994,
      "rerank_score": 0.27889981865882874
    },
    {
      "name_clothes": "avatars_addb93897bec4bd2b9488b01ceb0bf9e.png",
      "similarity": 0.27695006132125854,
      "rerank_score": 0.26413965225219727
    },
    {
      "name_clothes": "avatars_c57f656e23954722a64a62997787b03c.png",
      "similarity": 0.26888835430145264,
      "rerank_score": 0.2464873045682907
    }
  ],
  "best": {
    "name_clothes": "avatars_ad56d7730bfd427aaf9677c0a2832786.png",
    "similarity": 0.2713354825973511,
    "rerank_score": 0.33645108342170715
  }
}

# ---------------------------------------------------------
# 1. COMPOSITOR
# ---------------------------------------------------------
def create_composite(bg_path: Path, avatar_path: Path) -> Image.Image:
    if not bg_path.exists(): raise FileNotFoundError(f"Bg not found: {bg_path}")
    if not avatar_path.exists(): raise FileNotFoundError(f"Avatar not found: {avatar_path}")

    bg = Image.open(bg_path).convert("RGBA")
    avatar = Image.open(avatar_path).convert("RGBA")

    target_height = int(bg.height * 0.75)
    aspect_ratio = avatar.width / avatar.height
    new_width = int(target_height * aspect_ratio)

    avatar = avatar.resize((new_width, target_height), Image.Resampling.LANCZOS)

    x_pos = (bg.width - new_width) // 2
    y_pos = bg.height - target_height - int(bg.height * 0.05)

    composite = Image.new("RGBA", bg.size)
    composite.paste(bg, (0, 0))
    composite.paste(avatar, (x_pos, y_pos), avatar)
    return composite.convert("RGB")

# ---------------------------------------------------------
# 2. GEMINI JUDGE (Pure Visual)
# ---------------------------------------------------------
def get_gemini_score(client: GeminiModel, image: Image.Image) -> dict:
    prompt = """
    You are an expert fashion stylist and visual critic.
    You are viewing a COMPOSITE image where a 3D character has been superimposed onto a background scene.

    **TASK:**
    Analyze the background environment.
    Evaluate the **Stylistic and Aesthetic** fit of the character's outfit.

    **CRITICAL SCORING RULES:**
    1. **Visual Harmony != Same Color:** - Do NOT give high scores just because the outfit matches the background color (e.g., a Green shirt in a Green forest is BAD/CAMOUFLAGE).
       - Reward **Complementary Colors** and **Contrast** that make the character distinct but stylistically cohesive.
    2. **Ignore Artifacts:** Ignore the "pasted" look, floating feet, or lighting mismatches. Focus on the concept.
    3. **Be Strict:** Use the full integer range (e.g., 12, 48, 87).

    **SCORING CRITERIA (0-100):**
    - **Occasion Fit:** Is the outfit logically appropriate for the activity implied by the scene? (e.g., Hiking gear for mountains = High; Suit for mountains = Low).
    - **Visual Harmony:** Does the outfit look aesthetically pleasing?
       - *Penalty:* -10 points if the character blends into the background (Camouflage).
       - *Bonus:* +10 points for complementary palettes (e.g., Earth tones in a forest, White on a beach).
    - **Seasonality:** Does the clothing warmth match the weather?

    **OUTPUT JSON:**
    {
      "occasion_score": <int>,
      "visual_score": <int>,
      "season_score": <int>,
      "overall_score": <int>,
      "reasoning": "<Concise explanation, explicitly mentioning if colors clash or blend in>"
    }
    """

    response = client.generate_content([prompt, image])
    if response:
        return json.loads(response.text)
    return None

# ---------------------------------------------------------
# 3. MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print(f"--- Starting Visual Evaluation of {len(RESULTS_DATA['results'])} Items ---")

    # Auto-resolves to 2.5-flash or pro if available
    client = GeminiModel(api_key=GEMINI_API_KEY, model_name="gemini-2.5-flash")
    eval_results = []

    for i, item in enumerate(RESULTS_DATA["results"]):
        filename = item["name_clothes"]
        avatar_path = CLOTHES_DIR / filename

        print(f"\n[{i+1}/10] Processing: {filename}")

        try:
            composite_img = create_composite(BG_PATH, avatar_path)
        except Exception as e:
            print(f"   [!] Error compositing: {e}")
            continue

        scores = get_gemini_score(client, composite_img)

        if scores:
            overall = scores.get('overall_score')
            print(f"   -> Score: {overall} | Occasion: {scores.get('occasion_score')} | Visual: {scores.get('visual_score')}")
            print(f"   -> Reason: {scores.get('reasoning')[:60]}...")

            record = {
                "rank": i + 1,
                "clothing_file": filename,
                "scores": scores
            }
            eval_results.append(record)
            time.sleep(1.2) # Rate limit safety
        else:
            print("   [!] Failed to get score")

    # Save
    with open("evaluation_results_full.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n✅ Done! Saved to 'evaluation_results_full.json'")

if __name__ == "__main__":
    main()