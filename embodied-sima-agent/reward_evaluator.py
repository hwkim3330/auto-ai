#!/usr/bin/env python3
"""
Reward Evaluator - LLM-based episode evaluation
================================================

"LLM이 성공 여부를 판단한다"

Instead of hand-crafted reward functions, use LLM to:
1. Evaluate episode success/quality
2. Identify failure reasons
3. Suggest improvements
4. Score performance (0-1)

This enables self-supervised learning without human labeling
"""

import sys
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add paths for LLM
sys.path.append(str(Path(__file__).parent.parent / "streaming-agi"))
sys.path.append(str(Path(__file__).parent.parent / "emotional-agi"))


# ============================================================================
# Evaluation Result
# ============================================================================

@dataclass
class EvaluationResult:
    """
    Result of LLM evaluation

    Used for reinforcement learning and curriculum generation
    """
    # Score
    success: bool              # Overall success
    score: float               # Quality score (0-1)

    # Analysis
    strengths: List[str]       # What went well
    weaknesses: List[str]      # What went wrong
    suggestions: List[str]     # How to improve

    # Metadata
    task_goal: str
    num_steps: int
    total_reward: float
    evaluation_time: float

    def to_dict(self) -> Dict:
        """Convert to dict"""
        return {
            'success': self.success,
            'score': self.score,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'suggestions': self.suggestions,
            'task_goal': self.task_goal,
            'num_steps': self.num_steps,
            'total_reward': self.total_reward,
        }


# ============================================================================
# Reward Evaluator
# ============================================================================

class RewardEvaluator:
    """
    LLM-based episode evaluator

    Uses language model to assess episode quality

    This replaces hand-crafted reward functions with learned evaluation
    """

    def __init__(self, llm_model: str = "qwen2.5:3b"):
        self.llm_model = llm_model

        # Import LLM
        try:
            from streaming_continuous_agi import StreamingLLM
            self.llm = StreamingLLM(model=llm_model)
            print(f"[RewardEvaluator] Initialized with {llm_model}")
        except Exception as e:
            print(f"[RewardEvaluator] LLM not available: {e}")
            self.llm = None

        # Evaluation history
        self.evaluations: List[EvaluationResult] = []

    def evaluate_episode(
        self,
        task_goal: str,
        episode_log: List[Dict],
        final_state: Optional[Dict] = None
    ) -> EvaluationResult:
        """
        Evaluate episode using LLM

        Args:
            task_goal: What was the task
            episode_log: List of steps (from memory)
            final_state: Final environment state

        Returns:
            Evaluation result
        """
        start_time = time.time()

        # Format episode for LLM
        prompt = self._format_evaluation_prompt(task_goal, episode_log, final_state)

        # Get LLM evaluation
        if self.llm is not None:
            evaluation_text = self._query_llm(prompt)
            result = self._parse_evaluation(evaluation_text, task_goal, episode_log)
        else:
            # Fallback: simple heuristic
            result = self._heuristic_evaluation(task_goal, episode_log)

        result.evaluation_time = time.time() - start_time

        # Store
        self.evaluations.append(result)

        return result

    def _format_evaluation_prompt(
        self,
        task_goal: str,
        episode_log: List[Dict],
        final_state: Optional[Dict]
    ) -> str:
        """Format episode into LLM prompt"""

        lines = []
        lines.append("You are an expert evaluator of AI agent performance.")
        lines.append("Evaluate the following episode and provide a structured assessment.")
        lines.append("")
        lines.append(f"TASK GOAL: {task_goal}")
        lines.append("")
        lines.append("EPISODE LOG:")

        # Limit to last 20 steps to avoid token limit
        log_subset = episode_log[-20:] if len(episode_log) > 20 else episode_log

        for i, step in enumerate(log_subset):
            skill = step.get('skill', 'unknown')
            reward = step.get('reward', 0)
            success = step.get('success', True)
            status = "✓" if success else "✗"

            lines.append(f"  Step {i+1}: {skill} {status} (reward: {reward:.2f})")

        lines.append("")
        if final_state:
            lines.append("FINAL STATE:")
            for key, value in final_state.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        lines.append("Please evaluate this episode and provide:")
        lines.append("")
        lines.append("1. SUCCESS: Did the agent achieve the task goal? (YES/NO)")
        lines.append("2. SCORE: Quality of execution (0.0 to 1.0)")
        lines.append("3. STRENGTHS: What did the agent do well? (list 1-3 items)")
        lines.append("4. WEAKNESSES: What went wrong? (list 1-3 items)")
        lines.append("5. SUGGESTIONS: How to improve? (list 1-3 items)")
        lines.append("")
        lines.append("Format your response as:")
        lines.append("SUCCESS: YES/NO")
        lines.append("SCORE: 0.XX")
        lines.append("STRENGTHS:")
        lines.append("- ...")
        lines.append("WEAKNESSES:")
        lines.append("- ...")
        lines.append("SUGGESTIONS:")
        lines.append("- ...")

        return "\n".join(lines)

    def _query_llm(self, prompt: str) -> str:
        """Query LLM for evaluation"""
        if self.llm is None:
            return ""

        response = ""
        for token in self.llm.generate_stream(prompt):
            response += token

        return response

    def _parse_evaluation(
        self,
        evaluation_text: str,
        task_goal: str,
        episode_log: List[Dict]
    ) -> EvaluationResult:
        """
        Parse LLM response into structured result

        Extracts success, score, strengths, weaknesses, suggestions
        """
        lines = evaluation_text.split('\n')

        success = False
        score = 0.5
        strengths = []
        weaknesses = []
        suggestions = []

        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith('SUCCESS:'):
                success = 'YES' in line.upper()
            elif line.startswith('SCORE:'):
                try:
                    score = float(line.split(':')[1].strip())
                except:
                    score = 0.5
            elif line.startswith('STRENGTHS:'):
                current_section = 'strengths'
            elif line.startswith('WEAKNESSES:'):
                current_section = 'weaknesses'
            elif line.startswith('SUGGESTIONS:'):
                current_section = 'suggestions'
            elif line.startswith('-') or line.startswith('•'):
                item = line.lstrip('-•').strip()
                if current_section == 'strengths':
                    strengths.append(item)
                elif current_section == 'weaknesses':
                    weaknesses.append(item)
                elif current_section == 'suggestions':
                    suggestions.append(item)

        # Calculate metrics
        num_steps = len(episode_log)
        total_reward = sum(step.get('reward', 0) for step in episode_log)

        return EvaluationResult(
            success=success,
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            task_goal=task_goal,
            num_steps=num_steps,
            total_reward=total_reward,
            evaluation_time=0.0
        )

    def _heuristic_evaluation(
        self,
        task_goal: str,
        episode_log: List[Dict]
    ) -> EvaluationResult:
        """
        Fallback heuristic evaluation when LLM unavailable

        Simple rule-based scoring
        """
        num_steps = len(episode_log)
        total_reward = sum(step.get('reward', 0) for step in episode_log)
        success_rate = sum(1 for step in episode_log if step.get('success', True)) / max(num_steps, 1)

        # Simple heuristics
        success = total_reward > 0.5 and success_rate > 0.7
        score = (total_reward + success_rate) / 2

        strengths = []
        weaknesses = []
        suggestions = []

        if success_rate > 0.8:
            strengths.append("High success rate on individual steps")
        else:
            weaknesses.append("Many failed steps")
            suggestions.append("Review failing skills and improve execution")

        if num_steps < 20:
            strengths.append("Efficient - completed in few steps")
        elif num_steps > 50:
            weaknesses.append("Too many steps taken")
            suggestions.append("Find more direct path to goal")

        return EvaluationResult(
            success=success,
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            task_goal=task_goal,
            num_steps=num_steps,
            total_reward=total_reward,
            evaluation_time=0.0
        )

    def get_success_rate(self, last_n: Optional[int] = None) -> float:
        """
        Get success rate over recent evaluations

        Args:
            last_n: How many recent episodes (None = all)

        Returns:
            Success rate (0-1)
        """
        if not self.evaluations:
            return 0.0

        subset = self.evaluations[-last_n:] if last_n else self.evaluations
        successes = sum(1 for eval in subset if eval.success)

        return successes / len(subset)

    def get_average_score(self, last_n: Optional[int] = None) -> float:
        """
        Get average score over recent evaluations

        Args:
            last_n: How many recent episodes (None = all)

        Returns:
            Average score (0-1)
        """
        if not self.evaluations:
            return 0.0

        subset = self.evaluations[-last_n:] if last_n else self.evaluations
        total_score = sum(eval.score for eval in subset)

        return total_score / len(subset)

    def get_common_weaknesses(self, last_n: int = 10) -> List[Tuple[str, int]]:
        """
        Get most common weaknesses from recent evaluations

        Used for curriculum generation - focus on weak areas

        Returns:
            List of (weakness, count) tuples
        """
        if not self.evaluations:
            return []

        subset = self.evaluations[-last_n:]

        weakness_counts = {}
        for eval in subset:
            for weakness in eval.weaknesses:
                weakness_counts[weakness] = weakness_counts.get(weakness, 0) + 1

        # Sort by count
        common = sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)

        return common[:5]  # Top 5


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate reward evaluator"""
    print("\n" + "="*70)
    print("REWARD EVALUATOR - Demo")
    print("="*70)

    evaluator = RewardEvaluator()

    # Simulate episode
    task_goal = "Move to workbench and craft an axe"

    episode_log = [
        {'skill': 'move_to_target', 'reward': 0.1, 'success': True},
        {'skill': 'interact_with_object', 'reward': 0.2, 'success': True},
        {'skill': 'craft_item', 'reward': 0.5, 'success': True},
    ]

    print(f"\n[Demo] Evaluating episode:")
    print(f"  Task: {task_goal}")
    print(f"  Steps: {len(episode_log)}")

    # Evaluate
    result = evaluator.evaluate_episode(task_goal, episode_log)

    print(f"\n[Demo] Evaluation result:")
    print(f"  Success: {result.success}")
    print(f"  Score: {result.score:.2f}")
    print(f"  Strengths: {result.strengths}")
    print(f"  Weaknesses: {result.weaknesses}")
    print(f"  Suggestions: {result.suggestions}")

    # Statistics
    print(f"\n[Demo] Statistics:")
    print(f"  Success rate: {evaluator.get_success_rate():.2%}")
    print(f"  Average score: {evaluator.get_average_score():.2f}")

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    demo()
