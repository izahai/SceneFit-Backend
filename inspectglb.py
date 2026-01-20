from pygltflib import GLTF2
import json

# Path to your Ready Player Me GLB
GLB_PATH = "app/data/glb/m1_brown_4.glb"
OUTPUT_JSON = "avatar_inspected.json"

# Load the GLB file
gltf = GLTF2().load(GLB_PATH)

# Serialize using pygltflib's JSON encoder to handle Attributes objects
gltf_json = gltf.to_json(indent=2)

# Save everything to JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    f.write(gltf_json)

print(f"GLB data exported to {OUTPUT_JSON}")
