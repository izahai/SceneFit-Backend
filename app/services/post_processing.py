import os
import uuid
import random
from difflib import SequenceMatcher

def _get_distributions(num_results: int, top_k: int) -> list[int]:
    """
    Distribute num_results into top_k bins with increasing values.

    Guarantees all bins are at least 1 (when top_k <= num_results) to avoid
    downstream zero-sized buckets.
    """
    print("Calling _get_distributions")
    if top_k <= 0 or num_results <= 0:
        return []

    # Never create more buckets than available results; this keeps per-bucket
    # allocation positive.
    top_k = min(top_k, num_results)

    if top_k == 1:
        return [num_results]

    weights = [(i + 1) ** 2 for i in range(top_k)]
    total_weight = sum(weights)

    # Start with 1 in every bin so none are zero, then distribute the rest.
    remaining = num_results - top_k
    distributions = [1] * top_k
    cumulative = 0

    for idx, weight in enumerate(weights):
        extra = int((weight / total_weight) * remaining)
        distributions[idx] += extra
        cumulative += extra

    # Fix rounding by assigning any leftover to the last bin.
    distributions[-1] += remaining - cumulative

    return distributions


def _default_name_similarity(a: str | None, b: str | None) -> float:
    """Lightweight string similarity for cases where we only have names available."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def rerank_with_soft_penalty(
    results: list[dict[str, float]],
    alpha: float = 0.25,
    similarity_threshold: float = 0.85,
    similarity_fn=None,
) -> list[dict[str, float]]:
    """
    Diversity-aware re-ranking with a soft penalty on very similar items.

    Each selection step chooses the item with the highest adjusted score:
    adjusted_score = relevance - alpha * max_similarity_to_already_picked

    Similarity defaults to a string ratio on names; plug in a custom similarity
    function (e.g., cosine on embeddings) via `similarity_fn` for better results.
    """
    if not results:
        return []

    sim_fn = similarity_fn or _default_name_similarity
    # Start from relevance order to bias toward high scores.
    remaining = sorted(results, key=lambda x: x.get("score", 0.0) or 0.0, reverse=True)
    picked: list[dict[str, float]] = []

    while remaining:
        best_idx = None
        best_adjusted = None

        for idx, candidate in enumerate(remaining):
            name = str(candidate.get("name", ""))
            base_score = float(candidate.get("score", 0.0) or 0.0)

            max_sim = 0.0
            for chosen in picked:
                sim = sim_fn(name, str(chosen.get("name", "")))
                if sim > max_sim:
                    max_sim = sim
                if max_sim >= similarity_threshold:
                    # Already too close; further comparisons won't help this candidate much.
                    break

            adjusted = base_score - alpha * max_sim

            if best_adjusted is None or adjusted > best_adjusted:
                best_adjusted = adjusted
                best_idx = idx

        # Select the best candidate under the penalty and continue.
        picked.append(remaining.pop(best_idx))

    return picked


def shuffle_retrieval_results(
    results: list[dict[str, float]],
    k: int,
    apply_soft_penalty: bool = False,
    penalty_alpha: float = 0.25,
    penalty_similarity_threshold: float = 0.85,
    similarity_fn=None,
) -> list[dict[str, float]]:
    """
    Post-process retrieval results by optional diversity re-ranking and bucket shuffle.
    
    Steps:
      1) (Optional) Apply soft diversity penalty to spread out near-duplicates.
      2) Shuffle within size-increasing buckets to add mild randomness while
         keeping earlier items more likely to stay near the top.
    """
    if not results:
        return []

    if k >= len(results):
        # If k is greater than or equal to total results, return all results
        return results[:k]

    if apply_soft_penalty:
        results = rerank_with_soft_penalty(
            results,
            alpha=penalty_alpha,
            similarity_threshold=penalty_similarity_threshold,
            similarity_fn=similarity_fn,
        )
    
    # Get the distribution of bucket sizes
    num_results = len(results)
    distributions = _get_distributions(num_results, min(k, num_results))
    print(distributions)
    
    # Split results into buckets and shuffle within each bucket
    shuffled_results = []
    start_idx = 0
    
    for bucket_size in distributions:
        # Get the current bucket
        end_idx = start_idx + bucket_size
        bucket = results[start_idx:end_idx]
        
        print(f"bucket with {start_idx=}, {end_idx=}, {bucket_size=}")
        # Randomly shuffle within the bucket
        random.shuffle(bucket)
        
        # Add shuffled bucket to results
        shuffled_results.append(bucket[0])
        start_idx = end_idx
    
    # Return top k results
    return shuffled_results[:k]
    
    
if __name__ == "__main__":
    # Test _get_distributions
    print("Distribution tests:")
    print(f"_get_distributions(100, 5) = {_get_distributions(100, 5)}")
    print(f"_get_distributions(60, 5) = {_get_distributions(60, 5)}")
    print(f"_get_distributions(45, 4) = {_get_distributions(45, 4)}")
    
    # Test shuffle_retrieval_results
    print("\n\nShuffle retrieval results test:")
    test_results = [
        {"name": f"item_{i}", "score": 1.0 - (i * 0.01)}
        for i in range(100)
    ]
    
    print(f"Original results (first 5): {test_results[:5]}")
    shuffled = shuffle_retrieval_results(test_results, apply_soft_penalty=False, k=5)
    print(f"\nShuffled results (k=5):")
    for i, item in enumerate(shuffled):
        print(f"  {i+1}. {item['name']} (score: {item['score']:.2f})")
    
    # Run multiple times to show randomness
    # print("\n\nSecond shuffle (to demonstrate randomness):")
    # shuffled2 = shuffle_retrieval_results(test_results, k=5)
    # for i, item in enumerate(shuffled2):
    #     print(f"  {i+1}. {item['name']} (score: {item['score']:.2f})")
        
    
    