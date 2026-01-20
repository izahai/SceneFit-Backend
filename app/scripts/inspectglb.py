from pygltflib import GLTF2
import json

# Path to your Ready Player Me GLB
GLB_PATH = "avatar.glb"
OUTPUT_JSON = "avatar_inspected.json"

# Load the GLB file
gltf = GLTF2().load(GLB_PATH)

# Convert to a Python dictionary
gltf_dict = gltf.to_dict()

# Save everything to JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(gltf_dict, f, indent=2)

print(f"GLB data exported to {OUTPUT_JSON}")