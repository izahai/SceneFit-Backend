import base64
import io
import requests
import streamlit as st
from PIL import Image

# -----------------------------
# Config
# -----------------------------
API_URL = "https://proconciliation-tien-erythemal.ngrok-free.dev/api/v1/sd15/denoise"
DISPLAY_SIZE = 256      # uniform display size (px)
NUM_COLS = 4            # images per row

st.set_page_config(layout="wide")
st.title("SD 1.5 Denoise Viewer")

# -----------------------------
# Helpers
# -----------------------------
def b64_to_image(b64_str, size):
    img = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
    return img.resize((size, size), Image.BICUBIC)

# -----------------------------
# Sidebar Inputs
# -----------------------------
with st.sidebar:
    st.header("Generation Settings")

    prompt = st.text_area(
        "Prompt",
        "beautiful high-quality image, visually pleasing composition, harmonious colors, soft lighting, gentle shadows, balanced contrast, clean details, smooth gradients, aesthetic atmosphere, calm and inviting mood, polished professional look",
        height=150,
    )

    negative_prompt = st.text_area(
        "Negative Prompt",
        "ugly, low quality, blurry, noisy, harsh lighting, oversaturated, distorted, artifacts, text, watermark",
        height=120,
    )

    steps = st.slider("Steps", 1, 150, 30)
    guidance_scale = st.slider("Guidance Scale", 0.0, 20.0, 7.5)
    strength = st.slider("Strength", 0.0, 1.0, 0.25)
    eta = st.slider("ETA", 0.0, 1.0, 0.0)
    seed = st.number_input("Seed (0 = random)", min_value=0, value=1234)

# -----------------------------
# Image Upload
# -----------------------------
uploaded_image = st.file_uploader(
    "Upload Image (optional)", type=["png", "jpg", "jpeg"]
)

# -----------------------------
# Run Button
# -----------------------------
if st.button("🚀 Run Denoise"):
    with st.spinner("Generating images..."):

        files = {}
        original_image = None

        if uploaded_image:
            image_bytes = uploaded_image.getvalue()
            files["image"] = (
                uploaded_image.name,
                image_bytes,
                uploaded_image.type,
            )
            original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        data = {
            "prompt": prompt,
            "steps": str(steps),
            "guidance_scale": str(guidance_scale),
            "strength": str(strength),
            "eta": str(eta),
            "return_base64": "true",
        }

        if negative_prompt.strip():
            data["negative_prompt"] = negative_prompt

        if seed != 0:
            data["seed"] = str(seed)

        response = requests.post(API_URL, data=data, files=files)

        if response.status_code != 200:
            st.error("API Error")
            st.code(response.text)
            st.stop()

        result = response.json()

    # -----------------------------
    # Collect Images
    # -----------------------------
    images = []

    if original_image is not None:
        images.append((
            "Original",
            original_image.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC),
        ))

    if result.get("noise_image_base64"):
        images.append((
            "Noise",
            b64_to_image(result["noise_image_base64"], DISPLAY_SIZE),
        ))

    for i, step_b64 in enumerate(result.get("step_images_base64", [])):
        images.append((
            f"Step {i + 1}",
            b64_to_image(step_b64, DISPLAY_SIZE),
        ))

    images.append((
        "Final",
        b64_to_image(result["image_base64"], DISPLAY_SIZE),
    ))

    # -----------------------------
    # Display Grid
    # -----------------------------
    st.divider()
    st.subheader("Results")

    rows = [images[i:i + NUM_COLS] for i in range(0, len(images), NUM_COLS)]

    for row in rows:
        cols = st.columns(len(row))
        for col, (label, img) in zip(cols, row):
            with col:
                st.image(img, caption=label, use_container_width=False)

    # -----------------------------
    # Metadata
    # -----------------------------
    st.divider()
    st.caption(f"Seed used: {result.get('seed')}")
