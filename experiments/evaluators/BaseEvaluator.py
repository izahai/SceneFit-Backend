"""
Base evaluator class for VLM-based evaluation of method outputs.
Provides common functionality for evaluating different methods using Vision-Language Models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import json
import time
from pathlib import Path
from datetime import datetime


class BaseEvaluator(ABC):
    """
    Abstract base class for VLM evaluators.
    
    Subclasses should implement the _evaluate_single method to provide
    specific VLM evaluation logic.
    """
    
    def __init__(
        self,
        name: str,
        model_name: str,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        results_dir: Optional[str] = None
    ):
        """
        Initialize the base evaluator.
        
        Args:
            name: Name identifier for this evaluator
            model_name: Name/identifier of the VLM model being used
            max_tokens: Maximum tokens for model output
            temperature: Sampling temperature for the model
            results_dir: Directory to save evaluation results
        """
        self.name = name
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.results_dir = Path(results_dir) if results_dir else Path("experiments/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.evaluation_history = []
    
    @abstractmethod
    def _evaluate_single(
        self,
        input_data: Dict[str, Any],
        method_output: Any,
        criteria: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single method output using the VLM.
        
        This method should be implemented by subclasses to provide
        specific VLM evaluation logic.
        
        Args:
            input_data: Original input data (images, text, etc.)
            method_output: Output from the method being evaluated
            criteria: Evaluation criteria/metrics to assess
            
        Returns:
            Dictionary containing evaluation results with metrics and scores
        """
        pass
    
    def evaluate_batch(
        self,
        inputs: List[Dict[str, Any]],
        method_outputs: List[Any],
        method_name: str,
        criteria: Optional[Dict[str, str]] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of method outputs.
        
        Args:
            inputs: List of input data dictionaries
            method_outputs: List of method outputs
            method_name: Name of the method being evaluated
            criteria: Evaluation criteria
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary containing aggregated results and individual evaluations
        """
        if len(inputs) != len(method_outputs):
            raise ValueError("Number of inputs must match number of outputs")
        
        results = {
            "method_name": method_name,
            "evaluator": self.name,
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(inputs),
            "individual_results": [],
            "aggregate_metrics": {}
        }
        
        # Evaluate each sample
        for idx, (input_data, output) in enumerate(zip(inputs, method_outputs)):
            try:
                eval_result = self._evaluate_single(input_data, output, criteria)
                eval_result["sample_id"] = idx
                eval_result["success"] = True
                results["individual_results"].append(eval_result)
            except Exception as e:
                results["individual_results"].append({
                    "sample_id": idx,
                    "success": False,
                    "error": str(e)
                })
        
        # Calculate aggregate metrics
        results["aggregate_metrics"] = self._compute_aggregate_metrics(
            results["individual_results"]
        )
        
        # Track evaluation
        self.evaluation_history.append({
            "method": method_name,
            "timestamp": results["timestamp"],
            "num_samples": len(inputs)
        })
        
        # Save results
        if save_results:
            self._save_results(results, method_name)
        
        return results
    
    def _compute_aggregate_metrics(
        self,
        individual_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute aggregate metrics from individual evaluation results.
        
        Args:
            individual_results: List of individual evaluation results
            
        Returns:
            Dictionary of aggregated metrics
        """
        successful_results = [r for r in individual_results if r.get("success", False)]
        
        if not successful_results:
            return {"error": "No successful evaluations"}
        
        aggregate = {
            "total_samples": len(individual_results),
            "successful_samples": len(successful_results),
            "success_rate": len(successful_results) / len(individual_results)
        }
        
        # Collect all metric keys from first successful result
        if successful_results:
            sample_metrics = successful_results[0].get("metrics", {})
            for metric_name in sample_metrics.keys():
                values = []
                for result in successful_results:
                    metric_value = result.get("metrics", {}).get(metric_name)
                    if isinstance(metric_value, (int, float)):
                        values.append(metric_value)
                
                if values:
                    aggregate[f"{metric_name}_mean"] = sum(values) / len(values)
                    aggregate[f"{metric_name}_min"] = min(values)
                    aggregate[f"{metric_name}_max"] = max(values)
        
        return aggregate
    
    def _save_results(self, results: Dict[str, Any], method_name: str) -> None:
        """
        Save evaluation results to disk.
        
        Args:
            results: Results dictionary to save
            method_name: Name of the method evaluated
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{method_name}_{self.name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def compare_methods(
        self,
        method_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare results across multiple methods.
        
        Args:
            method_results: Dictionary mapping method names to their evaluation results
            
        Returns:
            Comparison summary
        """
        comparison = {
            "evaluator": self.name,
            "timestamp": datetime.now().isoformat(),
            "methods": list(method_results.keys()),
            "comparison": {}
        }
        
        # Extract aggregate metrics for each method
        for method_name, results in method_results.items():
            aggregate = results.get("aggregate_metrics", {})
            comparison["comparison"][method_name] = aggregate
        
        # Determine best performing method for each metric
        comparison["best_performers"] = self._find_best_performers(
            method_results
        )
        
        return comparison
    
    def _find_best_performers(
        self,
        method_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Find the best performing method for each metric.
        
        Args:
            method_results: Dictionary of method results
            
        Returns:
            Dictionary mapping metrics to best performing method names
        """
        best_performers = {}
        
        # Collect all metrics
        all_metrics = set()
        for results in method_results.values():
            metrics = results.get("aggregate_metrics", {})
            all_metrics.update(metrics.keys())
        
        # Find best for each metric
        for metric in all_metrics:
            if metric in ["total_samples", "successful_samples", "error"]:
                continue
            
            best_method = None
            best_value = None
            
            for method_name, results in method_results.items():
                value = results.get("aggregate_metrics", {}).get(metric)
                if value is not None and isinstance(value, (int, float)):
                    if best_value is None or value > best_value:
                        best_value = value
                        best_method = method_name
            
            if best_method:
                best_performers[metric] = {
                    "method": best_method,
                    "value": best_value
                }
        
        return best_performers
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all evaluations performed.
        
        Returns:
            Summary dictionary
        """
        return {
            "evaluator": self.name,
            "model": self.model_name,
            "total_evaluations": len(self.evaluation_history),
            "history": self.evaluation_history
        }
