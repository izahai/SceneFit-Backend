import argparse
import os
import subprocess
from pathlib import Path


def _resolve_blender_python(cli_blender: str | None) -> Path:
    """
    Resolve Blender's bundled Python executable on macOS.
    """

    # 1. Explicit Blender.app path
    if cli_blender:
        blender_app = Path(cli_blender)
    else:
        env_blender = os.environ.get("BLENDER")
        if env_blender:
            blender_app = Path(env_blender)
        else:
            blender_app = Path("/Applications/Blender.app")

    if blender_app.is_dir() and blender_app.suffix == ".app":
        python_path = (
            blender_app
            / "Contents"
            / "Resources"
            / "4.0"
            / "python"
            / "bin"
            / "python3"
        )
        if python_path.exists():
            return python_path

    raise FileNotFoundError(
        "Blender bundled Python not found. "
        "Provide --blender pointing to Blender.app or set BLENDER env var."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert .glb files to .fbx using Blender's bundled Python."
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
        help="Path to Blender.app (or use BLENDER env var)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .fbx outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    blender_python = _resolve_blender_python(args.blender)

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
            print(f"Skipping existing {output_path.name}")
            continue

        cmd = [
            str(blender_python),
            str(converter_script),
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
