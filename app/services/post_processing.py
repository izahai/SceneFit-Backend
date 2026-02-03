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
    

def shuffle_retrieval_results(results: list[dict[str, float]], top_k=5) -> str:
    """
    Generate a shuffled filename to avoid collisions.
    results: [
        {"name": str, "score": float},
        ...
    ]
    """
    pass
    
    
    
if __name__ == "__main__":
    # Example usage
    print(_get_distributions(100, 5))  # Example output: [4, 9, 13, 19]
    print(_get_distributions(60, 5))  # Example output: [4, 10, 14, 16, 16]
        
    
    