import argparse
import os
import shutil
import subprocess
from pathlib import Path


def _resolve_blender_path(cli_blender: str | None) -> str:
    if cli_blender:
        return cli_blender

    env_blender = os.environ.get("BLENDER")
    if env_blender:
        return env_blender

    blender_in_path = shutil.which("blender")
    if blender_in_path:
        return blender_in_path

    raise FileNotFoundError(
        "Blender executable not found. Provide --blender or set BLENDER env var."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert .glb files to .fbx via Blender."
    )
    parser.add_argument(
        "--input-dir",
        default="app/data/glb",
        help="Folder containing .glb files",
    )
    parser.add_argument(
        "--output-dir",
        default="app/data/fbx",
        help="Folder to write .fbx files",
    )
    parser.add_argument(
        "--blender",
        default=None,
        help="Path to Blender executable (or use BLENDER env var)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .fbx outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    blender_path = _resolve_blender_path(args.blender)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converter_script = Path("app/scripts/blender_glb_to_fbx.py").resolve()
    glb_files = sorted(input_dir.glob("*.glb"))

    if not glb_files:
        print(f"No .glb files found in {input_dir}")
        return 0

    for glb_file in glb_files:
        output_path = output_dir / f"{glb_file.stem}.fbx"
        if output_path.exists() and not args.force:
            print(f"Skipping existing {output_path}")
            continue

        cmd = [
            blender_path,
            "-b",
            "-P",
            str(converter_script),
            "--",
            "--input",
            str(glb_file.resolve()),
            "--output",
            str(output_path.resolve()),
        ]
        print(f"Converting {glb_file.name} -> {output_path.name}")
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python3 app/scripts/convert_glb_to_fbx.py --blender "/Applications/Blender.app/Contents/MacOS/Blender"