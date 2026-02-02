# Experiments Framework

This directory contains the evaluation framework for comparing different research methods.

## Structure

```
experiments/
├── config.yaml                 # Configuration for method URLs and VLM evaluators
├── config_loader.py           # Configuration loader utility
├── method_client.py           # HTTP client for calling method endpoints
├── utils.py                   # Utility functions for data processing
├── evaluators/
│   ├── BaseEvaluator.py      # Abstract base class for evaluators
│   └── vlm_evaluators.py     # VLM evaluator implementations
├── scripts/
│   └── run_experiments.py    # Example experiment scripts
└── results/                   # Evaluation results (auto-generated)
```

## Quick Start

### 1. Configure Method Endpoints

Edit `config.yaml` to add your ngrok URLs:

```yaml
methods:
  method1:
    name: "Method 1"
    base_url: "https://your-ngrok-url-1.ngrok.io"
    endpoints:
      generate: "/api/generate"
      process: "/api/process"
```

### 2. Set API Keys

For VLM evaluators, set environment variables:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your-openai-key"
$env:ANTHROPIC_API_KEY = "your-anthropic-key"

# Or in .env file
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

### 3. Run Experiments

```python
from method_client import MethodClient
from evaluators.vlm_evaluators import GPT4VisionEvaluator

# Call a method
client = MethodClient("method1")
output = client.generate({"query": "Generate a scene"})

# Evaluate output
evaluator = GPT4VisionEvaluator()
results = evaluator.evaluate_batch(
    inputs=[{"query": "Generate a scene"}],
    method_outputs=[output],
    method_name="method1"
)
```

## Usage Examples

### Single Method Evaluation

```python
from method_client import MethodClient
from evaluators.vlm_evaluators import GPT4VisionEvaluator

# Initialize client
client = MethodClient("method1")

# Prepare input
input_data = {
    "query": "Modern living room",
    "parameters": {"style": "minimalist"}
}

# Call method
output = client.generate(input_data)

# Evaluate
evaluator = GPT4VisionEvaluator()
result = evaluator.evaluate_batch(
    inputs=[input_data],
    method_outputs=[output],
    method_name="method1"
)

print(result["aggregate_metrics"])
```

### Compare Multiple Methods

```python
from method_client import MethodOrchestrator
from evaluators.vlm_evaluators import CustomVLMEvaluator

# Initialize orchestrator
orchestrator = MethodOrchestrator()

# Run same input on all methods
input_data = {"query": "Generate bedroom"}
results = orchestrator.run_on_all_methods("generate", input_data)

# Evaluate all methods
evaluator = CustomVLMEvaluator("my_evaluator", "my_model")
evaluation_results = {}

for method_name, output in results.items():
    eval_result = evaluator.evaluate_batch(
        inputs=[input_data],
        method_outputs=[output],
        method_name=method_name
    )
    evaluation_results[method_name] = eval_result

# Compare
comparison = evaluator.compare_methods(evaluation_results)
print(comparison["best_performers"])
```

### Batch Processing

```python
from method_client import MethodOrchestrator

orchestrator = MethodOrchestrator()

# Prepare batch
batch_inputs = [
    {"query": "Living room"},
    {"query": "Bedroom"},
    {"query": "Kitchen"}
]

# Process batch on a method
outputs = orchestrator.run_batch_on_method(
    method_name="method1",
    endpoint="generate",
    batch_data=batch_inputs
)
```

## Custom VLM Evaluator

Create your own evaluator by extending `BaseEvaluator`:

```python
from evaluators.BaseEvaluator import BaseEvaluator

class MyCustomEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(
            name="my_evaluator",
            model_name="my_model"
        )
    
    def _evaluate_single(self, input_data, method_output, criteria=None):
        # Your evaluation logic here
        metrics = {
            "relevance": 8.5,
            "quality": 9.0
        }
        
        return {
            "metrics": metrics,
            "explanations": {...}
        }
```

## Configuration Reference

### Method Configuration

```yaml
methods:
  method_name:
    name: "Display Name"
    description: "Method description"
    base_url: "https://your-url.ngrok.io"
    endpoints:
      generate: "/api/generate"
      process: "/api/process"
      health: "/health"
```

### VLM Configuration

```yaml
vlm_evaluators:
  evaluator_name:
    model: "model-identifier"
    api_type: "openai|anthropic|google"
    max_tokens: 1000
    temperature: 0.0
```

## API Reference

### MethodClient

- `generate(input_data)` - Call the generate endpoint
- `process(input_data)` - Call the process endpoint
- `health_check()` - Check if server is healthy
- `get_stats()` - Get request statistics

### MethodOrchestrator

- `get_client(method_name)` - Get client for specific method
- `health_check_all()` - Check health of all methods
- `run_on_all_methods(endpoint, input_data)` - Run on all methods
- `run_batch_on_method(method_name, endpoint, batch_data)` - Batch processing

### BaseEvaluator

- `evaluate_batch(inputs, outputs, method_name)` - Evaluate batch
- `compare_methods(method_results)` - Compare multiple methods
- `get_evaluation_summary()` - Get evaluation statistics

## Results

Results are automatically saved to `experiments/results/` as JSON files:

```json
{
  "method_name": "method1",
  "evaluator": "gpt4v",
  "timestamp": "2024-01-15T10:30:00",
  "num_samples": 10,
  "individual_results": [...],
  "aggregate_metrics": {
    "relevance_mean": 8.5,
    "quality_mean": 7.8,
    "success_rate": 1.0
  }
}
```

## Utilities

```python
from experiments.utils import (
    load_test_dataset,
    save_results_to_csv,
    generate_experiment_report
)

# Load test data
test_data = load_test_dataset("data/test_set.json")

# Export to CSV
save_results_to_csv(results, "output.csv")

# Generate markdown report
generate_experiment_report("experiments/results", "report.md")
```

## Troubleshooting

### Method Connection Issues

1. Verify ngrok URLs are active: `curl https://your-url.ngrok.io/health`
2. Check firewall settings
3. Verify endpoint paths in config.yaml

### API Key Issues

```python
import os
print(os.getenv("OPENAI_API_KEY"))  # Should not be None
```

### Import Errors

Make sure you're running from the project root:
```bash
cd d:\projects\SceneFit\SceneFit-Backend
python experiments/scripts/run_experiments.py
```

## Advanced Usage

### Custom Evaluation Criteria

```python
criteria = {
    "scene_realism": "How realistic is the scene?",
    "composition": "Quality of spatial arrangement",
    "lighting": "Lighting quality and accuracy"
}

result = evaluator.evaluate_batch(
    inputs=inputs,
    method_outputs=outputs,
    method_name="method1",
    criteria=criteria
)
```

### Update Method URLs Programmatically

```python
from config_loader import get_config

config = get_config()
config.update_method_url("method1", "https://new-url.ngrok.io")
```

## License

Same as parent project.
