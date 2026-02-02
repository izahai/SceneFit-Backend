"""
Example VLM evaluator implementations.
Provides concrete implementations of BaseEvaluator for different VLMs.
"""

from typing import Dict, Any, Optional
import os
from evaluators.BaseEvaluator import BaseEvaluator


class GPT4VisionEvaluator(BaseEvaluator):
    """Evaluator using GPT-4 Vision model."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        results_dir: Optional[str] = None
    ):
        """
        Initialize GPT-4 Vision evaluator.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env variable
            max_tokens: Maximum tokens for output
            temperature: Sampling temperature
            results_dir: Directory to save results
        """
        super().__init__(
            name="gpt4v",
            model_name="gpt-4-vision-preview",
            max_tokens=max_tokens,
            temperature=temperature,
            results_dir=results_dir
        )
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")
    
    def _evaluate_single(
        self,
        input_data: Dict[str, Any],
        method_output: Any,
        criteria: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single method output using GPT-4 Vision.
        
        Args:
            input_data: Original input containing images/text
            method_output: Output from the method being evaluated
            criteria: Evaluation criteria
            
        Returns:
            Evaluation results with metrics and explanations
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(input_data, method_output, criteria)
        
        # Prepare messages for GPT-4 Vision
        messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator assessing the quality of generated outputs. "
                          "Provide detailed, objective evaluations with numerical scores."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Add images if present
        if "image_url" in input_data or "image_path" in input_data:
            # Handle image content (would need base64 encoding for actual implementation)
            pass
        
        # Call GPT-4 Vision API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        evaluation_text = response.choices[0].message.content
        
        # Parse the evaluation response
        return self._parse_evaluation_response(evaluation_text, criteria)
    
    def _build_evaluation_prompt(
        self,
        input_data: Dict[str, Any],
        method_output: Any,
        criteria: Optional[Dict[str, str]] = None
    ) -> str:
        """Build evaluation prompt for the VLM."""
        prompt = "Please evaluate the following method output:\n\n"
        
        # Add input context
        if "query" in input_data:
            prompt += f"Input Query: {input_data['query']}\n\n"
        
        # Add method output
        prompt += f"Method Output:\n{method_output}\n\n"
        
        # Add evaluation criteria
        if criteria:
            prompt += "Evaluate based on the following criteria:\n"
            for criterion, description in criteria.items():
                prompt += f"- {criterion}: {description}\n"
        else:
            prompt += "Evaluate based on: relevance, quality, coherence, and accuracy.\n"
        
        prompt += "\nFor each criterion, provide:\n"
        prompt += "1. A score from 1-10\n"
        prompt += "2. A brief explanation\n\n"
        prompt += "Format your response as JSON with this structure:\n"
        prompt += '{\n  "metrics": {"criterion_name": score, ...},\n'
        prompt += '  "explanations": {"criterion_name": "explanation", ...}\n}'
        
        return prompt
    
    def _parse_evaluation_response(
        self,
        response_text: str,
        criteria: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Parse the VLM evaluation response."""
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return {
                    "metrics": parsed.get("metrics", {}),
                    "explanations": parsed.get("explanations", {}),
                    "raw_response": response_text
                }
            except json.JSONDecodeError:
                pass
        
        # Fallback: return raw response
        return {
            "metrics": {},
            "explanations": {},
            "raw_response": response_text,
            "parsing_error": "Failed to parse structured response"
        }


class ClaudeVisionEvaluator(BaseEvaluator):
    """Evaluator using Claude 3 Vision model."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        results_dir: Optional[str] = None
    ):
        """
        Initialize Claude Vision evaluator.
        
        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env variable
            max_tokens: Maximum tokens for output
            temperature: Sampling temperature
            results_dir: Directory to save results
        """
        super().__init__(
            name="claude_vision",
            model_name="claude-3-opus-20240229",
            max_tokens=max_tokens,
            temperature=temperature,
            results_dir=results_dir
        )
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install anthropic package: pip install anthropic")
    
    def _evaluate_single(
        self,
        input_data: Dict[str, Any],
        method_output: Any,
        criteria: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single method output using Claude Vision.
        
        Args:
            input_data: Original input containing images/text
            method_output: Output from the method being evaluated
            criteria: Evaluation criteria
            
        Returns:
            Evaluation results with metrics and explanations
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(input_data, method_output, criteria)
        
        # Call Claude API
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        evaluation_text = message.content[0].text
        
        # Parse the evaluation response
        return self._parse_evaluation_response(evaluation_text, criteria)
    
    def _build_evaluation_prompt(
        self,
        input_data: Dict[str, Any],
        method_output: Any,
        criteria: Optional[Dict[str, str]] = None
    ) -> str:
        """Build evaluation prompt for Claude."""
        prompt = "Please evaluate the following method output:\n\n"
        
        if "query" in input_data:
            prompt += f"Input Query: {input_data['query']}\n\n"
        
        prompt += f"Method Output:\n{method_output}\n\n"
        
        if criteria:
            prompt += "Evaluate based on the following criteria:\n"
            for criterion, description in criteria.items():
                prompt += f"- {criterion}: {description}\n"
        else:
            prompt += "Evaluate based on: relevance, quality, coherence, and accuracy.\n"
        
        prompt += "\nFor each criterion, provide a score from 1-10 and an explanation.\n"
        prompt += "Format your response as JSON."
        
        return prompt
    
    def _parse_evaluation_response(
        self,
        response_text: str,
        criteria: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Parse the Claude evaluation response."""
        import json
        import re
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return {
                    "metrics": parsed.get("metrics", {}),
                    "explanations": parsed.get("explanations", {}),
                    "raw_response": response_text
                }
            except json.JSONDecodeError:
                pass
        
        return {
            "metrics": {},
            "explanations": {},
            "raw_response": response_text,
            "parsing_error": "Failed to parse structured response"
        }


class CustomVLMEvaluator(BaseEvaluator):
    """
    Template for custom VLM evaluator.
    Extend this class to implement your own VLM evaluation logic.
    """
    
    def __init__(
        self,
        name: str,
        model_name: str,
        api_endpoint: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        results_dir: Optional[str] = None
    ):
        """
        Initialize custom VLM evaluator.
        
        Args:
            name: Name for this evaluator
            model_name: Model identifier
            api_endpoint: Custom API endpoint if needed
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            results_dir: Results directory
        """
        super().__init__(
            name=name,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            results_dir=results_dir
        )
        self.api_endpoint = api_endpoint
    
    def _evaluate_single(
        self,
        input_data: Dict[str, Any],
        method_output: Any,
        criteria: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Implement your custom evaluation logic here.
        
        This is a template - replace with your actual VLM API calls.
        """
        # Example structure - replace with your implementation
        metrics = {
            "relevance": 8.0,
            "quality": 7.5,
            "coherence": 9.0,
            "accuracy": 8.5
        }
        
        explanations = {
            "relevance": "Output is highly relevant to the input query.",
            "quality": "Good quality but could be improved.",
            "coherence": "Excellent coherence and logical flow.",
            "accuracy": "Accurate representation of the task."
        }
        
        return {
            "metrics": metrics,
            "explanations": explanations,
            "raw_response": "Custom evaluation completed"
        }
