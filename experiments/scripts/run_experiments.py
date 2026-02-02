#!/usr/bin/env python3
"""
Batch evaluation script for GeminiVisionEvaluator.

Example:
python run_eval.py \
  --results_dir app/results \
  --bg_dir app/data/bg \
  --clothes_dir app/data/clothes \
  --output_dir experiments/gemini_eval \
  --model gemini-2.5-flash \
  --rate_limit 1.2
"""

import argparse
import json
import time
from pathlib import Path
from dotenv import load_dotenv

from gemini_evaluator import GeminiVisionEvaluator

load_dotenv()


# -----------------------------
# Utilities
# -----------------------------
def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def match_bg_image(json_path: Path, bg_dir: Path) -> Path:
    """
    bg_001.json -> bg_001.png
    """
    bg_name = json_path.stem + ".png"
    bg_path = bg_dir / bg_name
    if not bg_path.exists():
        raise FileNotFoundError(f"Missing background image: {bg_path}")
    return bg_path


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Gemini Vision Batch Evaluator")

    parser.add_argument("--results_dir", required=True, type=Path)
    parser.add_argument("--bg_dir", required=True, type=Path)
    parser.add_argument("--clothes_dir", required=True, type=Path)
    parser.add_argument("--output_dir", default="experiments/results", type=Path)

    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--rate_limit", type=float, default=1.2)
    parser.add_argument("--max_files", type=int, default=None)

    args = parser.parse_args()

    # Validate paths
    for p in [args.results_dir, args.bg_dir, args.clothes_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")

    json_files = sorted(args.results_dir.glob("*.json"))
    if args.max_files:
        json_files = json_files[: args.max_files]

    print(f"🔍 Found {len(json_files)} JSON files")

    evaluator = GeminiVisionEvaluator(
        model_name=args.model,
        results_dir=str(args.output_dir),
    )

    # Global batch containers
    all_inputs = []
    all_outputs = []
    all_method_names = []

    # -----------------------------
    # Build batch
    # -----------------------------
    for json_path in json_files:
        print(f"\n📄 Loading {json_path.name}")
        data = load_json(json_path)

        bg_path = match_bg_image(json_path, args.bg_dir)

        scene_caption = data.get("scene_caption", "")
        method_name = data.get("method", "unknown-method")

        for result in data["results"]:
            input_data = {
                "background_path": str(bg_path),
                "clothes_dir": str(args.clothes_dir),
                "scene_caption": scene_caption,
            }

            all_inputs.append(input_data)
            all_outputs.append(result)
            all_method_names.append(method_name)

    if not all_inputs:
        print("⚠️ No evaluation samples found.")
        return

    # -----------------------------
    # Run evaluation
    # -----------------------------
    print("\n🚀 Starting Gemini evaluation")
    print(f"   Samples: {len(all_inputs)}")
    print(f"   Model:   {args.model}")

    # We group by method name to keep result files clean
    results_by_method = {}

    for method in sorted(set(all_method_names)):
        idxs = [i for i, m in enumerate(all_method_names) if m == method]

        inputs = [all_inputs[i] for i in idxs]
        outputs = [all_outputs[i] for i in idxs]

        print(f"\n🧪 Evaluating method: {method} ({len(inputs)} samples)")

        results = evaluator.evaluate_batch(
            inputs=inputs,
            method_outputs=outputs,
            method_name=method,
            rate_limit_delay=args.rate_limit,
            save_results=True,
        )

        results_by_method[method] = results

    print("\n✅ All evaluations completed")


if __name__ == "__main__":
    main()
