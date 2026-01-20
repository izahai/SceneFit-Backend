import trimesh
import pyrender
import numpy as np
from PIL import Image
from rembg import remove
import io



def look_at(camera_position, target, up=np.array([0, 1, 0])):
    forward = camera_position - target
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    true_up = np.cross(forward, right)

    pose = np.eye(4)
    pose[:3, :3] = np.vstack([right, true_up, forward]).T
    pose[:3, 3] = camera_position
    return pose


def render_glb_front_view(
    glb_path,
    output_image_path="render.png",
    image_size=(600, 1200),
    camera_distance_factor=1.0,
    bg_color=(255, 255, 255, 255),  # 👈 NEW: background color
    remove_bg=False                 # 👈 NEW: control background removal
):
    mesh = trimesh.load(glb_path)

    scene_trimesh = mesh if isinstance(mesh, trimesh.Scene) else trimesh.Scene(mesh)

    # Normalize bg_color for pyrender (expects 0–1 floats)
    bg_color_norm = [c / 255.0 for c in bg_color]

    render_scene = pyrender.Scene(
        bg_color=bg_color_norm,
        ambient_light=[0.3, 0.3, 0.3]
    )

    for geom in scene_trimesh.geometry.values():
        render_scene.add(pyrender.Mesh.from_trimesh(geom, smooth=True))

    # Bounding box
    bbox = scene_trimesh.bounds
    min_corner, max_corner = bbox
    size = np.linalg.norm(max_corner - min_corner)

    # Target (upper body)
    height = max_corner[1] - min_corner[1]
    center = (min_corner + max_corner) / 2
    center[1] = min_corner[1] + height * 0.65

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3)

    camera_distance = size * camera_distance_factor
    camera_height = size * 0.3

    camera_position = center + np.array([
        0.0,
        camera_height,
        camera_distance
    ])

    camera_pose = look_at(camera_position, center)
    render_scene.add(camera, pose=camera_pose)

    # Light follows camera
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    render_scene.add(light, pose=camera_pose)

    # Render
    renderer = pyrender.OffscreenRenderer(
        viewport_width=image_size[0],
        viewport_height=image_size[1]
    )

    color, _ = renderer.render(render_scene)
    renderer.delete()

    img = Image.fromarray(color).convert("RGBA")

    if remove_bg:
        # Remove background (returns transparent PNG)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        output_bytes = remove(buffer.read())

        # Composite onto chosen background color
        fg = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        bg = Image.new("RGBA", fg.size, bg_color)
        final_img = Image.alpha_composite(bg, fg)
    else:
        final_img = img

    final_img.convert("RGB").save(output_image_path)

    print(f"Saved render to: {output_image_path}")

