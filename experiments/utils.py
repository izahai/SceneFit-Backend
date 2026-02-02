"""
Utility functions for experiment workflows.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


def load_test_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load test dataset from JSON file.
    
    Args:
        dataset_path: Path to JSON file containing test data
        
    Returns:
        List of test samples
    """
    with open(dataset_path, 'r') as f:
        return json.load(f)


def save_results_to_csv(results: Dict[str, Any], output_path: str) -> None:
    """
    Save evaluation results to CSV for analysis.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save CSV file
    """
    # Extract individual results
    individual_results = results.get("individual_results", [])
    
    # Flatten results for CSV
    rows = []
    for result in individual_results:
        row = {
            "sample_id": result.get("sample_id"),
            "success": result.get("success")
        }
        
        # Add metrics
        metrics = result.get("metrics", {})
        for metric_name, value in metrics.items():
            row[f"metric_{metric_name}"] = value
        
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def compare_results_csv(result_files: List[str], output_path: str) -> None:
    """
    Compare multiple result files and create comparison CSV.
    
    Args:
        result_files: List of paths to result JSON files
        output_path: Path to save comparison CSV
    """
    comparison_data = []
    
    for file_path in result_files:
        with open(file_path, 'r') as f:
            results = json.load(f)
            
        method_name = results.get("method_name")
        aggregate = results.get("aggregate_metrics", {})
        
        row = {"method": method_name}
        row.update(aggregate)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df.to_csv(output_path, index=False)
    print(f"Comparison saved to {output_path}")


def generate_experiment_report(results_dir: str, output_path: str) -> None:
    """
    Generate a markdown report from all results in a directory.
    
    Args:
        results_dir: Directory containing result JSON files
        output_path: Path to save markdown report
    """
    results_path = Path(results_dir)
    result_files = list(results_path.glob("*.json"))
    
    with open(output_path, 'w') as f:
        f.write("# Experiment Results Report\n\n")
        f.write(f"Total evaluations: {len(result_files)}\n\n")
        
        for result_file in result_files:
            with open(result_file, 'r') as rf:
                results = json.load(rf)
            
            f.write(f"## {results.get('method_name')} - {results.get('evaluator')}\n\n")
            f.write(f"**Timestamp:** {results.get('timestamp')}\n\n")
            f.write(f"**Samples:** {results.get('num_samples')}\n\n")
            
            f.write("### Aggregate Metrics\n\n")
            aggregate = results.get('aggregate_metrics', {})
            for metric, value in aggregate.items():
                if isinstance(value, float):
                    f.write(f"- **{metric}**: {value:.4f}\n")
                else:
                    f.write(f"- **{metric}**: {value}\n")
            
            f.write("\n---\n\n")
    
    print(f"Report generated: {output_path}")


def create_test_dataset_template(output_path: str, num_samples: int = 10) -> None:
    """
    Create a template test dataset JSON file.
    
    Args:
        output_path: Path to save template
        num_samples: Number of sample entries to create
    """
    template = []
    
    for i in range(num_samples):
        sample = {
            "id": i,
            "query": f"Sample query {i}",
            "parameters": {
                "style": "modern",
                "complexity": "medium"
            },
            "metadata": {
                "category": "test",
                "tags": ["sample"]
            }
        }
        template.append(sample)
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Test dataset template created: {output_path}")
