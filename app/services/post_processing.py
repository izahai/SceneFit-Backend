


import os
import uuid


def shuffle_retrieval_results(original_filename: str) -> str:
    """
    Generate a shuffled filename to avoid collisions.
    """
    name, ext = os.path.splitext(original_filename)
    random_suffix = uuid.uuid4().hex
    new_filename = f"{name}_{random_suffix}{ext}"
    return new_filename