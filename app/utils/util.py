# app/utils/util.py

import yaml
from pathlib import Path

_PROMPT_CONFIG_PATH = Path("config/prompts.yaml")

def load_prompts(task: str) -> list[str]:
    with open(_PROMPT_CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)

    if task not in data:
        raise KeyError(f"Prompt task '{task}' not found in prompts.yaml")

    prompts = data[task]
    return [prompts["positive"], prompts["negative"]]
