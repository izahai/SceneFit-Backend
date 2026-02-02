"""
Example experiment script demonstrating how to use the evaluation framework.

This script shows:
1. How to configure method endpoints
2. How to call methods using MethodClient
3. How to evaluate outputs using VLM evaluators
4. How to compare methods
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from method_client import MethodClient, MethodOrchestrator
from evaluators.vlm_evaluators import GPT4VisionEvaluator, CustomVLMEvaluator
from config_loader import get_config


def example_single_method_evaluation():
    """Example: Evaluate a single method on one input."""
    print("=" * 60)
    print("Example 1: Single Method Evaluation")
    print("=" * 60)
    
    # Initialize method client
    client = MethodClient("method1")
    
    # Prepare input
    input_data = {
        "query": "Generate a realistic bedroom scene",
        "parameters": {
            "style": "modern",
            "lighting": "natural"
        }
    }
    
    # Call method
    print(f"\nCalling {client.method_name}...")
    try:
        output = client.generate(input_data)
        print(f"✓ Method output received")
    except Exception as e:
        print(f"✗ Error calling method: {e}")
        return
    
    # Evaluate with VLM
    print("\nEvaluating output with GPT-4 Vision...")
    try:
        # Note: Set your OpenAI API key in environment or pass it here
        evaluator = GPT4VisionEvaluator()
        
        # Define evaluation criteria
        criteria = {
            "relevance": "How well does the output match the input query?",
            "quality": "Overall visual and technical quality",
            "coherence": "Logical consistency and realism",
            "creativity": "Originality and aesthetic appeal"
        }
        
        # Evaluate
        result = evaluator.evaluate_batch(
            inputs=[input_data],
            method_outputs=[output],
            method_name=client.method_name,
            criteria=criteria
        )
        
        print("\n✓ Evaluation completed!")
        print(f"\nAggregate Metrics:")
        for metric, value in result["aggregate_metrics"].items():
            print(f"  {metric}: {value}")
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")


def example_batch_evaluation():
    """Example: Evaluate multiple inputs on a single method."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Evaluation")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = MethodOrchestrator(method_names=["method1"])
    
    # Prepare batch of inputs
    batch_inputs = [
        {"query": "Modern living room with plants"},
        {"query": "Minimalist bedroom with warm lighting"},
        {"query": "Industrial style kitchen"}
    ]
    
    # Run method on batch
    print(f"\nProcessing batch of {len(batch_inputs)} inputs...")
    try:
        outputs = orchestrator.run_batch_on_method(
            method_name="method1",
            endpoint="generate",
            batch_data=batch_inputs
        )
        print(f"✓ Batch processing completed")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Evaluate batch
    print("\nEvaluating batch with custom VLM...")
    try:
        evaluator = CustomVLMEvaluator(
            name="custom_evaluator",
            model_name="my_vlm_model"
        )
        
        result = evaluator.evaluate_batch(
            inputs=batch_inputs,
            method_outputs=outputs,
            method_name="method1"
        )
        
        print("\n✓ Batch evaluation completed!")
        print(f"Success rate: {result['aggregate_metrics']['success_rate']:.2%}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_compare_methods():
    """Example: Compare multiple methods on the same inputs."""
    print("\n" + "=" * 60)
    print("Example 3: Compare Multiple Methods")
    print("=" * 60)
    
    # Initialize orchestrator with all methods
    orchestrator = MethodOrchestrator()
    
    # Check health of all methods
    print("\nChecking method availability...")
    health_status = orchestrator.health_check_all()
    for method, is_healthy in health_status.items():
        status = "✓ Available" if is_healthy else "✗ Unavailable"
        print(f"  {method}: {status}")
    
    # Prepare test inputs
    test_inputs = [
        {"query": "Cozy reading corner with bookshelves"},
        {"query": "Bright office space with natural light"}
    ]
    
    # Run all methods on same inputs
    print(f"\nRunning {len(test_inputs)} inputs on all methods...")
    all_results = {}
    
    for method_name in orchestrator.clients.keys():
        try:
            outputs = orchestrator.run_batch_on_method(
                method_name=method_name,
                endpoint="generate",
                batch_data=test_inputs
            )
            all_results[method_name] = outputs
            print(f"  ✓ {method_name} completed")
        except Exception as e:
            print(f"  ✗ {method_name} failed: {e}")
    
    # Evaluate all methods
    print("\nEvaluating all methods...")
    evaluator = CustomVLMEvaluator(
        name="comparative_evaluator",
        model_name="comparison_vlm"
    )
    
    evaluation_results = {}
    for method_name, outputs in all_results.items():
        try:
            result = evaluator.evaluate_batch(
                inputs=test_inputs,
                method_outputs=outputs,
                method_name=method_name,
                save_results=True
            )
            evaluation_results[method_name] = result
            print(f"  ✓ {method_name} evaluated")
        except Exception as e:
            print(f"  ✗ {method_name} evaluation failed: {e}")
    
    # Compare methods
    if evaluation_results:
        print("\nComparing methods...")
        comparison = evaluator.compare_methods(evaluation_results)
        
        print("\nBest Performers:")
        for metric, info in comparison.get("best_performers", {}).items():
            print(f"  {metric}: {info['method']} (score: {info['value']:.2f})")
    
    # Show statistics
    print("\nMethod Statistics:")
    stats = orchestrator.get_all_stats()
    for method, stat in stats.items():
        print(f"  {method}:")
        print(f"    Requests: {stat['request_count']}")
        print(f"    Errors: {stat['error_count']}")
        print(f"    Error Rate: {stat['error_rate']:.2%}")


def example_custom_criteria():
    """Example: Use custom evaluation criteria."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Evaluation Criteria")
    print("=" * 60)
    
    # Define domain-specific criteria
    custom_criteria = {
        "scene_realism": "How realistic and believable is the generated scene?",
        "composition": "Quality of spatial arrangement and visual balance",
        "lighting_quality": "Accuracy and aesthetics of lighting",
        "detail_level": "Richness of details and textures",
        "prompt_adherence": "How well does output follow the input prompt?"
    }
    
    print("\nUsing custom criteria:")
    for criterion, description in custom_criteria.items():
        print(f"  • {criterion}: {description}")
    
    # Your evaluation code here with custom criteria
    print("\n✓ Custom criteria evaluation can be implemented similarly to previous examples")


def show_configuration():
    """Display current configuration."""
    print("\n" + "=" * 60)
    print("Current Configuration")
    print("=" * 60)
    
    config = get_config()
    
    print("\nConfigured Methods:")
    methods = config.get_all_methods()
    for method_name, method_config in methods.items():
        print(f"\n  {method_name}:")
        print(f"    Name: {method_config['name']}")
        print(f"    URL: {method_config['base_url']}")
        print(f"    Endpoints: {', '.join(method_config['endpoints'].keys())}")
    
    print("\n\nConfigured VLM Evaluators:")
    vlms = config.get_all_vlms()
    for vlm_name, vlm_config in vlms.items():
        print(f"\n  {vlm_name}:")
        print(f"    Model: {vlm_config['model']}")
        print(f"    API Type: {vlm_config['api_type']}")
    
    print("\n\nEvaluation Metrics:")
    metrics = config.get_metrics()
    print(f"  {', '.join(metrics)}")


def main():
    """Main function to run examples."""
    print("\n" + "=" * 60)
    print("Experiment Framework - Usage Examples")
    print("=" * 60)
    
    # Show configuration
    show_configuration()
    
    # Note: Uncomment the examples you want to run
    # Make sure to update config.yaml with your actual ngrok URLs first!
    
    # example_single_method_evaluation()
    # example_batch_evaluation()
    # example_compare_methods()
    # example_custom_criteria()
    
    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)
    print("\nTo run examples:")
    print("1. Update experiments/config.yaml with your ngrok URLs")
    print("2. Set API keys in environment variables:")
    print("   - OPENAI_API_KEY for GPT-4 Vision")
    print("   - ANTHROPIC_API_KEY for Claude")
    print("3. Uncomment the example functions you want to run")
    print("4. Run: python experiments/scripts/run_experiments.py")


if __name__ == "__main__":
    main()
