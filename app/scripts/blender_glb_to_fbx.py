import argparse
import os
import sys

import bpy


def _parse_args() -> argparse.Namespace:
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1 :]
    else:
        args = []

    parser = argparse.ArgumentParser(description="Convert a GLB file to FBX using Blender.")
    parser.add_argument("--input", required=True, help="Path to the source .glb file")
    parser.add_argument("--output", required=True, help="Path to the output .fbx file")
    return parser.parse_args(args)


def main() -> None:
    args = _parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Start clean
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import GLB
    bpy.ops.import_scene.gltf(filepath=input_path)

    # --------------------------------------------------
    # REMOVE SPHERE / STATIC MESHES
    # --------------------------------------------------
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            has_armature = any(
                mod.type == "ARMATURE" for mod in obj.modifiers
            )

            # Delete meshes not skinned to skeleton
            if not has_armature:
                bpy.data.objects.remove(obj, do_unlink=True)
                
    for mat in bpy.data.materials:
        mat.use_nodes = False


    # --------------------------------------------------
    # EXPORT FBX
    # --------------------------------------------------
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=False,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_ALL",
        bake_space_transform=True,
        add_leaf_bones=False,
    )



if __name__ == "__main__":
    main()
