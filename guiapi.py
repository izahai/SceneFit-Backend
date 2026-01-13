import base64
import io
import requests
import streamlit as st
from PIL import Image

API_URL = "https://proconciliation-tien-erythemal.ngrok-free.dev/api/v1/sd15/denoise"

st.set_page_config(layout="wide")
st.title("SD 1.5 Denoise Viewer")

# ---- Inputs ----
prompt = st.text_input(
    "Prompt",
    "beautiful high-quality image, visually pleasing composition, harmonious colors, soft lighting, gentle shadows, balanced contrast, clean details, smooth gradients, aesthetic atmosphere, calm and inviting mood, polished professional look, high resolution, sharp focus, artistic, elegant, visually stunning",
)
negative_prompt = st.text_input(
    "Negative Prompt",
    "ugly, low quality, blurry, noisy, harsh lighting, flat lighting, oversaturated, muddy colors, distorted shapes, cluttered composition, text, watermark, logo, artifacts",
)
steps = st.slider("Steps", 1, 150, 50)
guidance_scale = st.slider("Guidance Scale", 0.0, 20.0, 7.5)
strength = st.slider("Strength", 0.0, 1.0, 0.75)
eta = st.slider("ETA", 0.0, 1.0, 0.0)
seed = st.number_input("Seed (0 = random)", min_value=0, value=0)

uploaded_image = st.file_uploader("Upload Image (optional)", type=["png", "jpg", "jpeg"])

# ---- Submit ----
if st.button("Run Denoise"):
    with st.spinner("Generating..."):
        files = {}
        if uploaded_image:
            files["image"] = uploaded_image

        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "eta": eta,
            "seed": None if seed == 0 else seed,
            "return_base64": True,
        }

        response = requests.post(API_URL, data=data, files=files)
        response.raise_for_status()
        result = response.json()

    # ---- Final Image ----
    st.subheader("Final Image")
    final_image = Image.open(
        io.BytesIO(base64.b64decode(result["image_base64"]))
    )
    st.image(final_image, use_container_width=True)

    # ---- Step Images ----
    step_images = result.get("step_images_base64", [])
    if step_images:
        st.subheader(f"Step Images ({len(step_images)})")

        cols = st.columns(4)
        for i, img_b64 in enumerate(step_images):
            img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
            with cols[i % 4]:
                st.image(img, caption=f"Step {i+1}", use_container_width=True)
