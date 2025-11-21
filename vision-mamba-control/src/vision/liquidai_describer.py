#!/usr/bin/env python3
"""
LiquidAI Scene Describer
-----------------------
Uses LiquidAI LFM2 model to generate natural language descriptions
of detected objects and scenes from CCTV/vision analysis.
"""

import threading
import time
from typing import Dict, List, Optional


class LiquidAIDescriber:
    """
    LiquidAI-powered scene description generator.

    Takes detection results and generates human-readable descriptions
    using the LFM2 language model.
    """

    def __init__(self, model_id: str = "LiquidAI/LFM2-1.2B", device: str = "cpu"):
        """
        Initialize the LiquidAI describer.

        Args:
            model_id: HuggingFace model ID
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.loading = False
        self._lock = threading.Lock()
        self.last_description = ""
        self.last_description_time = 0
        self.description_interval = 5.0  # Generate new description every 5 seconds

    def load_model(self) -> bool:
        """
        Load the LiquidAI model (lazy loading).

        Returns:
            True if loaded successfully
        """
        if self.loaded:
            return True

        if self.loading:
            return False

        with self._lock:
            if self.loaded:
                return True

            self.loading = True

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                print(f"Loading LiquidAI model: {self.model_id}")

                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True
                )

                # Load model on CPU (for GTX 1050 Ti compatibility)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
                self.model.eval()

                self.loaded = True
                self.loading = False
                print(f"LiquidAI model loaded successfully on {self.device}")
                return True

            except Exception as e:
                print(f"Failed to load LiquidAI model: {e}")
                self.loading = False
                return False

    def _build_prompt(self, detections: List[Dict], analytics: Dict) -> str:
        """
        Build a prompt from detection results.

        Args:
            detections: List of detected objects
            analytics: Analytics data from CCTV monitor

        Returns:
            Formatted prompt string
        """
        # Count objects by class
        object_counts = {}
        for det in detections:
            cls = det.get('class', 'object')
            object_counts[cls] = object_counts.get(cls, 0) + 1

        # Build object description
        objects_desc = []
        for cls, count in object_counts.items():
            if count == 1:
                objects_desc.append(f"1 {cls}")
            else:
                objects_desc.append(f"{count} {cls}s")

        objects_str = ", ".join(objects_desc) if objects_desc else "nothing detected"

        # Analytics info
        total_persons = analytics.get('total_persons', 0)
        loitering = analytics.get('loitering', [])
        close_persons = analytics.get('close_persons', [])

        # Build prompt
        prompt = f"""Describe this CCTV scene briefly:
Objects: {objects_str}
People count: {total_persons}
Loitering alerts: {len(loitering)}
Close proximity alerts: {len(close_persons)}

Scene description:"""

        return prompt

    def describe_scene(
        self,
        detections: List[Dict],
        analytics: Dict,
        max_tokens: int = 50
    ) -> str:
        """
        Generate a description of the current scene.

        Args:
            detections: List of detected objects
            analytics: Analytics data
            max_tokens: Maximum tokens to generate

        Returns:
            Generated scene description
        """
        # Rate limiting - don't generate too frequently
        current_time = time.time()
        if current_time - self.last_description_time < self.description_interval:
            return self.last_description

        # Check if model is loaded
        if not self.loaded:
            if not self.load_model():
                return self._generate_simple_description(detections, analytics)

        try:
            import torch

            prompt = self._build_prompt(detections, analytics)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the generated part
            description = full_response[len(prompt):].strip()

            # Clean up
            if description:
                # Take first sentence if too long
                if '.' in description:
                    description = description.split('.')[0] + '.'
                description = description[:200]  # Max 200 chars
            else:
                description = self._generate_simple_description(detections, analytics)

            self.last_description = description
            self.last_description_time = current_time

            return description

        except Exception as e:
            print(f"LiquidAI description error: {e}")
            return self._generate_simple_description(detections, analytics)

    def _generate_simple_description(
        self,
        detections: List[Dict],
        analytics: Dict
    ) -> str:
        """
        Generate a simple rule-based description (fallback).

        Args:
            detections: List of detected objects
            analytics: Analytics data

        Returns:
            Simple description string
        """
        total_persons = analytics.get('total_persons', 0)
        loitering = analytics.get('loitering', [])
        close_persons = analytics.get('close_persons', [])

        parts = []

        if total_persons == 0:
            parts.append("No people detected in the scene")
        elif total_persons == 1:
            parts.append("1 person detected")
        else:
            parts.append(f"{total_persons} people detected")

        if loitering:
            parts.append(f"{len(loitering)} loitering alert(s)")

        if close_persons:
            parts.append(f"{len(close_persons)} proximity warning(s)")

        if not loitering and not close_persons and total_persons > 0:
            parts.append("Normal activity")

        return ". ".join(parts) + "."

    def get_status(self) -> Dict:
        """
        Get the status of the LiquidAI model.

        Returns:
            Status dictionary
        """
        return {
            'loaded': self.loaded,
            'loading': self.loading,
            'model_id': self.model_id,
            'device': self.device,
            'last_description': self.last_description
        }


# Singleton instance
_describer_instance = None


def get_describer(model_id: str = "LiquidAI/LFM2-1.2B") -> LiquidAIDescriber:
    """
    Get the singleton LiquidAI describer instance.

    Args:
        model_id: HuggingFace model ID

    Returns:
        LiquidAIDescriber instance
    """
    global _describer_instance

    if _describer_instance is None:
        _describer_instance = LiquidAIDescriber(model_id=model_id, device="cpu")

    return _describer_instance


if __name__ == "__main__":
    # Test the describer
    describer = LiquidAIDescriber()

    test_detections = [
        {'class': 'person', 'confidence': 0.95, 'distance_depth': 3.2},
        {'class': 'person', 'confidence': 0.87, 'distance_depth': 5.1},
    ]

    test_analytics = {
        'total_persons': 2,
        'loitering': [],
        'close_persons': []
    }

    print("Testing LiquidAI Scene Describer...")
    print()

    # First test with simple description (before model loads)
    simple_desc = describer._generate_simple_description(test_detections, test_analytics)
    print(f"Simple description: {simple_desc}")
    print()

    # Test with model
    print("Loading model and generating AI description...")
    ai_desc = describer.describe_scene(test_detections, test_analytics)
    print(f"AI description: {ai_desc}")
