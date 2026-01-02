# app/services/img_processor.py

from rembg import remove
from PIL import Image

def remove_background(input_path, output_path):
    with open(input_path, "rb") as f:
        input_image = f.read()

    output_image = remove(input_image)

    with open(output_path, "wb") as f:
        f.write(output_image)

# python - << 'EOF'
# from app.services.img_processor import remove_background
# import os

# input_path = "app/clothes/1.png"
# output_path = "test.png"

# remove_background(input_path, output_path)

# assert os.path.exists(output_path), "[Error] Output file was not created"
# print("[Complete] Background removal test passed")
# EOF
