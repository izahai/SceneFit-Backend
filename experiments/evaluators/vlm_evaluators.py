"""
Example VLM evaluator implementations.
Provides concrete implementations of BaseEvaluator for different VLMs.
"""
from typing import Dict, Any, Optional, List
import os
import time
import json
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the base class from your friend's code
# Adjust this import based on your actual project structure
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

"""
Gemini Vision Evaluator implementation.
Integrates Gemini 2.5 Flash/Pro for visual fashion evaluation.
"""




class GeminiVisionEvaluator(BaseEvaluator):
    """
    Evaluator using Gemini 2.5 Vision model for fashion/outfit assessment.
    
    This evaluator specializes in evaluating outfit-background compatibility
    by creating composite images and using Gemini's vision capabilities.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        max_tokens: int = 1000,
        temperature: float = 0.0,
        results_dir: Optional[str] = None
    ):
        """
        Initialize Gemini Vision evaluator.
        
        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY in .env file
                    or environment variable
            model_name: Gemini model name (e.g., "gemini-2.5-flash" or "gemini-2.5-pro")
            max_tokens: Maximum tokens for output
            temperature: Sampling temperature
            results_dir: Directory to save results
        """
        super().__init__(
            name="gemini_vision",
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            results_dir=results_dir
        )
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Please either:\n"
                "1. Create a .env file with: GEMINI_API_KEY=your_key_here\n"
                "2. Set environment variable: export GEMINI_API_KEY=your_key_here\n"
                "3. Pass api_key parameter to GeminiVisionEvaluator(api_key='your_key')"
            )
        
        try:
            genai.configure(api_key=self.api_key)
            self.model_name = self._resolve_model_name(model_name)
            print(f"✅ Initialized Gemini SDK with resolved model: {self.model_name}")
            
            self.model = genai.GenerativeModel(
                self.model_name,
                generation_config={"response_mime_type": "application/json"}
            )
        except ImportError:
            raise ImportError("Please install google-generativeai package: pip install google-generativeai")
    
    def _resolve_model_name(self, requested_name: str) -> str:
        """
        Finds the best available model (Flash or Pro).
        Falls back gracefully if requested model is unavailable.
        
        Args:
            requested_name: The requested model name
            
        Returns:
            The resolved/available model name
        """
        try:
            available = [m.name.replace("models/", "") for m in genai.list_models()]
            
            if requested_name in available:
                return requested_name
            
            # Fallback to any available flash model
            for m in available:
                if "flash" in m and "2.5" in m:
                    return m
            
            # Fallback to any available pro model
            for m in available:
                if "pro" in m and "2.5" in m:
                    return m
            
            # Last resort: return requested name and let API handle it
            return requested_name
        except Exception:
            return requested_name
    
    def _create_composite_image(
        self,
        background_path: Path,
        avatar_path: Path
    ) -> Image.Image:
        """
        Create a composite image by overlaying avatar on background.
        
        Args:
            background_path: Path to background image
            avatar_path: Path to avatar/clothing image
            
        Returns:
            Composite PIL Image
        """
        if not background_path.exists():
            raise FileNotFoundError(f"Background not found: {background_path}")
        if not avatar_path.exists():
            raise FileNotFoundError(f"Avatar not found: {avatar_path}")
        
        # Load images
        bg = Image.open(background_path).convert("RGBA")
        avatar = Image.open(avatar_path).convert("RGBA")
        
        # Resize avatar to 75% of background height
        target_height = int(bg.height * 0.75)
        aspect_ratio = avatar.width / avatar.height
        new_width = int(target_height * aspect_ratio)
        
        avatar = avatar.resize((new_width, target_height), Image.Resampling.LANCZOS)
        
        # Position avatar (centered horizontally, bottom with 5% margin)
        x_pos = (bg.width - new_width) // 2
        y_pos = bg.height - target_height - int(bg.height * 0.05)
        
        # Create composite
        composite = Image.new("RGBA", bg.size)
        composite.paste(bg, (0, 0))
        composite.paste(avatar, (x_pos, y_pos), avatar)
        
        return composite.convert("RGB")
    
    def _build_evaluation_prompt(self) -> str:
        """
        Build the evaluation prompt for Gemini.
        
        Returns:
            Evaluation prompt string
        """
        return """
You are an expert fashion stylist and visual critic.
You are viewing a COMPOSITE image where a 3D character has been superimposed onto a background scene.

**TASK:**
Analyze the background environment.
Evaluate the **Stylistic and Aesthetic** fit of the character's outfit.

**CRITICAL SCORING RULES:**
1. **Visual Harmony != Same Color:** 
   - Do NOT give high scores just because the outfit matches the background color (e.g., a Green shirt in a Green forest is BAD/CAMOUFLAGE).
   - Reward **Complementary Colors** and **Contrast** that make the character distinct but stylistically cohesive.
2. **Ignore Artifacts:** Ignore the "pasted" look, floating feet, or lighting mismatches. Focus on the concept.
3. **Be Strict:** Use the full integer range (e.g., 12, 48, 87).

**SCORING CRITERIA (0-100):**
- **Occasion Fit:** Is the outfit logically appropriate for the activity implied by the scene? (e.g., Hiking gear for mountains = High; Suit for mountains = Low).
- **Visual Harmony:** Does the outfit look aesthetically pleasing?
   - *Penalty:* -10 points if the character blends into the background (Camouflage).
   - *Bonus:* +10 points for complementary palettes (e.g., Earth tones in a forest, White on a beach).
- **Seasonality:** Does the clothing warmth match the weather?

**OUTPUT JSON:**
{
  "occasion_score": <int>,
  "visual_score": <int>,
  "season_score": <int>,
  "overall_score": <int>,
  "reasoning": "<Concise explanation, explicitly mentioning if colors clash or blend in>"
}
"""
    
    def _evaluate_single(
        self,
        input_data: Dict[str, Any],
        method_output: Any,
        criteria: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single method output using Gemini Vision.
        
        Args:
            input_data: Original input containing:
                - background_path: Path to background image
                - scene_caption: Optional scene description
            method_output: Dictionary containing:
                - name_clothes: Filename of clothing/avatar image
                - similarity: Optional similarity score
                - rerank_score: Optional rerank score
                OR a Path to the avatar image
            criteria: Evaluation criteria (optional, uses default if not provided)
            
        Returns:
            Evaluation results with metrics and explanations
        """
        try:
            # Extract paths
            background_path = Path(input_data.get("background_path"))
            
            # Handle method_output as either dict or Path
            if isinstance(method_output, dict):
                avatar_filename = method_output.get("name_clothes")
                clothes_dir = Path(input_data.get("clothes_dir", "."))
                avatar_path = clothes_dir / avatar_filename
                
                # Store original scores for reference
                original_similarity = method_output.get("similarity")
                original_rerank = method_output.get("rerank_score")
            else:
                avatar_path = Path(method_output)
                original_similarity = None
                original_rerank = None
            
            # Create composite image
            composite_img = self._create_composite_image(background_path, avatar_path)
            
            # Build prompt
            prompt = self._build_evaluation_prompt()
            
            # Call Gemini API with retry
            response = self._call_gemini_with_retry(composite_img, prompt)
            
            if response:
                scores = json.loads(response.text)
                
                # Extract metrics in the format expected by BaseEvaluator
                metrics = {
                    "occasion_fit": scores.get("occasion_score", 0),
                    "visual_harmony": scores.get("visual_score", 0),
                    "seasonality": scores.get("season_score", 0),
                    "overall_score": scores.get("overall_score", 0)
                }
                
                # Add original scores if available
                if original_similarity is not None:
                    metrics["original_similarity"] = original_similarity
                if original_rerank is not None:
                    metrics["original_rerank"] = original_rerank
                
                return {
                    "metrics": metrics,
                    "explanations": {
                        "overall": scores.get("reasoning", "")
                    },
                    "raw_response": scores
                }
            else:
                raise Exception("Failed to get response from Gemini API")
                
        except Exception as e:
            raise Exception(f"Evaluation failed: {str(e)}")
    
    def _call_gemini_with_retry(
        self,
        image: Image.Image,
        prompt: str,
        max_retries: int = 2
    ) -> Any:
        """
        Call Gemini API with retry logic.
        
        Args:
            image: PIL Image to evaluate
            prompt: Evaluation prompt
            max_retries: Maximum number of retry attempts
            
        Returns:
            Gemini response object
        """
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content([prompt, image])
                if response:
                    return response
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)  # Exponential backoff
                    print(f"⚠️  API call failed, retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
                else:
                    raise
        
        return None
    
    def evaluate_batch(
        self,
        inputs: List[Dict[str, Any]],
        method_outputs: List[Any],
        method_name: str,
        criteria: Optional[Dict[str, str]] = None,
        save_results: bool = True,
        rate_limit_delay: float = 1.2
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of method outputs with rate limiting.
        
        This override adds rate limiting to prevent API quota issues.
        
        Args:
            inputs: List of input data dictionaries
            method_outputs: List of method outputs
            method_name: Name of the method being evaluated
            criteria: Evaluation criteria
            save_results: Whether to save results to disk
            rate_limit_delay: Delay in seconds between API calls
            
        Returns:
            Dictionary containing aggregated results and individual evaluations
        """
        if len(inputs) != len(method_outputs):
            raise ValueError("Number of inputs must match number of outputs")
        
        results = {
            "method_name": method_name,
            "evaluator": self.name,
            "model": self.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(inputs),
            "individual_results": [],
            "aggregate_metrics": {}
        }
        
        # Evaluate each sample with rate limiting
        for idx, (input_data, output) in enumerate(zip(inputs, method_outputs)):
            print(f"\n[{idx+1}/{len(inputs)}] Evaluating sample {idx}...")
            
            try:
                eval_result = self._evaluate_single(input_data, output, criteria)
                eval_result["sample_id"] = idx
                eval_result["success"] = True
                results["individual_results"].append(eval_result)
                
                # Print brief summary
                overall_score = eval_result["metrics"].get("overall_score", 0)
                print(f"   ✓ Score: {overall_score}")
                
            except Exception as e:
                print(f"   ✗ Error: {e}")
                results["individual_results"].append({
                    "sample_id": idx,
                    "success": False,
                    "error": str(e)
                })
            
            # Rate limiting (skip on last iteration)
            if idx < len(inputs) - 1:
                time.sleep(rate_limit_delay)
        
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