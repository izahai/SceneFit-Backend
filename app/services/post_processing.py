import os
import uuid
import random

def _get_distributions(num_results: int, top_k: int) -> list[int]:
    """
    Distribute num_results into top_k bins with increasing values.
    
    Examples:
        _get_distributions(45, 4) -> [4, 9, 13, 19] (sum=45)
        _get_distributions(60, 5) -> [4, 10, 14, 16, 16] (sum=60)
    """
    if top_k <= 0:
        return []
    if top_k == 1:
        return [num_results]
    
    # Use quadratic weighted distribution where position i gets weight (i+1)^2
    # This makes left values smaller and increases more rapidly
    total_weight = sum((i + 1) ** 2 for i in range(top_k))  # 1^2 + 2^2 + 3^2 + ... + top_k^2
    
    distributions = []
    cumulative = 0
    
    for i in range(top_k):
        weight = (i + 1) ** 2  # Quadratic weight for rapid growth
        # Calculate proportional value
        value = int((weight / total_weight) * num_results)
        distributions.append(value)
        cumulative += value
    
    # Adjust the last element to ensure exact sum
    distributions[-1] += (num_results - cumulative)
    
    return distributions
    

def shuffle_retrieval_results(results: list[dict[str, float]], k: int) -> list[dict[str, float]]:
    """
    Post-process retrieval results by shuffling within buckets of increasing sizes.
    
    Args:
        results: List of retrieval results with format [{"name": str, "score": float}, ...]
        k: Expected number of results to return
    
    Returns:
        List of k results after bucket-based random selection
    
    Example:
        results = [{"name": "item1", "score": 0.9}, {"name": "item2", "score": 0.8}, ...]
        shuffle_retrieval_results(results, k=5) -> returns 5 randomly selected items
    """
    if not results:
        return []
    
    if k >= len(results):
        # If k is greater than or equal to total results, return all results
        return results[:k]
    
    # Get the distribution of bucket sizes
    num_results = len(results)
    distributions = _get_distributions(num_results, min(k, num_results))
    
    # Split results into buckets and shuffle within each bucket
    shuffled_results = []
    start_idx = 0
    
    for bucket_size in distributions:
        # Get the current bucket
        end_idx = start_idx + bucket_size
        bucket = results[start_idx:end_idx]
        
        # Randomly shuffle within the bucket
        random.shuffle(bucket)
        
        # Add shuffled bucket to results
        shuffled_results.extend(bucket)
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
        for i in range(20)
    ]
    
    print(f"Original results (first 5): {test_results[:5]}")
    
    shuffled = shuffle_retrieval_results(test_results, k=10)
    print(f"\nShuffled results (k=10):")
    for i, item in enumerate(shuffled):
        print(f"  {i+1}. {item['name']} (score: {item['score']:.2f})")
    
    # Run multiple times to show randomness
    print("\n\nSecond shuffle (to demonstrate randomness):")
    shuffled2 = shuffle_retrieval_results(test_results, k=10)
    for i, item in enumerate(shuffled2):
        print(f"  {i+1}. {item['name']} (score: {item['score']:.2f})")
        
    
    