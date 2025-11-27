#!/usr/bin/env python3
"""
SIMA-style Embodied Agent - Complete integration
=================================================

"생각하고, 기억하고, 행동하고, 평가하고, 학습한다"

Complete SIMA2-style architecture integrating all components:

1. Environment Adapter → Unified interface
2. High-level Planner (Streaming AGI) → What to do
3. Skill Library → How to do it
4. Low-level Controller (NCP) → Execution
5. Memory System → Remember experiences
6. Reward Evaluator (LLM + Emotions) → Self-assess
7. Learning Loop → Improve over time

"오픈소스 SIMA2 - 게임/시뮬레이터를 학습하는 체화 AGI"
"""

import sys
from pathlib import Path
import time
import json
from typing import Dict, List, Optional, Any

# Add paths
sys.path.append(str(Path(__file__).parent.parent / "streaming-agi"))
sys.path.append(str(Path(__file__).parent.parent / "emotional-agi"))
sys.path.append(str(Path(__file__).parent.parent / "computer-use-ncp"))

# Import all components
from env_adapter import create_env, BaseEnvAdapter, Observation, Action
from skill_library import SkillLibrary
from memory_system import MemoryManager
from reward_evaluator import RewardEvaluator

# Import existing AGI components
from streaming_continuous_agi import ParallelThinkingAGI
from emotional_agi import EmotionalAGI
from computer_agent import ComputerUseAgent


# ============================================================================
# SIMA-style Embodied Agent
# ============================================================================

class EmbodiedAgent:
    """
    Complete SIMA2-style embodied agent

    "환경과 상호작용하며 학습하는 체화 AGI"

    Architecture:
        Environment ← Agent → Experience
             ↓                     ↓
        Observation            Memory
             ↓                     ↓
        Planner (LLM)         Consolidation
             ↓                     ↓
        Skills                Learned Knowledge
             ↓                     ↓
        Low-level Policy      Evaluator (LLM + Emotions)
             ↓                     ↓
        Actions               Self-assessment
             ↓                     ↓
        Environment          Curriculum Generation
    """

    def __init__(
        self,
        env_config: Optional[Dict] = None,
        agent_config: Optional[Dict] = None
    ):
        print("\n" + "="*70)
        print("SIMA-STYLE EMBODIED AGENT - Initializing")
        print("="*70)

        env_config = env_config or {'type': 'screen'}
        agent_config = agent_config or {}

        # Component initialization
        print("\n[EmbodiedAgent] Loading components...")

        # 1. Environment
        self.env = create_env(env_config)
        print(f"  ✓ Environment: {env_config.get('type', 'screen')}")

        # 2. High-level Planner (Streaming AGI)
        model = agent_config.get('llm_model', 'qwen2.5:3b')
        self.planner = ParallelThinkingAGI(model=model)
        print(f"  ✓ Planner: {model}")

        # 3. Skill Library
        self.skills = SkillLibrary()
        print(f"  ✓ Skills: {len(self.skills.skills)} skills")

        # 4. Memory System
        self.memory = MemoryManager()
        print(f"  ✓ Memory: episodic + semantic")

        # 5. Reward Evaluator
        self.evaluator = RewardEvaluator(llm_model=model)
        print(f"  ✓ Evaluator: LLM-based")

        # 6. Emotional System (optional)
        use_emotions = agent_config.get('use_emotions', True)
        if use_emotions:
            self.emotions = EmotionalAGI()
            print(f"  ✓ Emotions: 7 emotions")
        else:
            self.emotions = None

        # Statistics
        self.total_episodes = 0
        self.total_steps = 0
        self.total_reward = 0.0

        print("\n" + "="*70)
        print("SIMA Agent Ready!")
        print("="*70)

    def execute_task(
        self,
        task_description: str,
        max_steps: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Execute complete task

        Full loop:
        1. Plan using LLM
        2. Execute skills
        3. Store in memory
        4. Evaluate episode
        5. Learn from experience

        Args:
            task_description: What to do
            max_steps: Max steps per episode
            verbose: Print progress

        Returns:
            Episode results
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🎯 TASK: {task_description}")
            print(f"{'='*70}\n")

        start_time = time.time()

        # Reset environment
        obs = self.env.reset(task_spec={'goal': task_description})

        # Start new episode in memory
        self.memory.start_new_episode()

        # Episode log
        episode_log = []
        step = 0

        # Main execution loop
        while step < max_steps:
            step += 1

            if verbose:
                print(f"\n[Step {step}] Planning...")

            # === 1. High-level Planning ===
            plan = self._plan_next_actions(task_description, obs)

            if verbose:
                print(f"[Step {step}] Plan: {plan}")

            # === 2. Skill Execution ===
            skill_results = self._execute_plan(plan, verbose=verbose)

            # === 3. Store in Memory ===
            for skill_result in skill_results:
                self.memory.store_experience(
                    observation=obs,
                    action=None,  # Skill-level, not low-level action
                    reward=0.1 if skill_result.success else -0.1,
                    next_observation=None,  # TODO: Get next obs
                    skill_used=skill_result.skill_name,
                    plan_step=plan,
                    emotion_state=self.emotions.emotions.__dict__ if self.emotions else None,
                    success=skill_result.success
                )

                episode_log.append({
                    'step': step,
                    'skill': skill_result.skill_name,
                    'success': skill_result.success,
                    'reward': 0.1 if skill_result.success else -0.1,
                })

            # === 4. Check if task complete ===
            # Use emotional satisfaction as termination condition
            if self.emotions:
                if not self.emotions.emotions.should_continue_learning():
                    if verbose:
                        print(f"\n😌 [Emotions] Satisfied - Stopping")
                    break

            # Simple step limit
            if step >= max_steps:
                if verbose:
                    print(f"\n⏱️  [Limit] Reached max steps")
                break

        # === 5. Episode Evaluation ===
        if verbose:
            print(f"\n{'='*70}")
            print("📊 EPISODE EVALUATION")
            print(f"{'='*70}\n")

        evaluation = self.evaluator.evaluate_episode(
            task_goal=task_description,
            episode_log=episode_log
        )

        if verbose:
            print(f"Success: {'✓' if evaluation.success else '✗'}")
            print(f"Score: {evaluation.score:.2f}")
            print(f"Strengths: {evaluation.strengths}")
            print(f"Weaknesses: {evaluation.weaknesses}")
            print(f"Suggestions: {evaluation.suggestions}")

        # === 6. Memory Consolidation ===
        if verbose:
            print(f"\n[Memory] Consolidating knowledge...")

        self.memory.consolidate_knowledge(llm=self.planner.llm)

        # === 7. Emotional Experience ===
        if self.emotions:
            novelty = 0.5  # TODO: Calculate based on memory
            self.emotions.experience(
                content=task_description,
                novelty=novelty,
                success=evaluation.success
            )

        # Update statistics
        self.total_episodes += 1
        self.total_steps += step
        self.total_reward += sum(log['reward'] for log in episode_log)

        duration = time.time() - start_time

        result = {
            'task': task_description,
            'success': evaluation.success,
            'score': evaluation.score,
            'steps': step,
            'duration': duration,
            'episode_log': episode_log,
            'evaluation': evaluation.to_dict(),
            'memory_summary': self.memory.get_memory_summary(),
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ EPISODE COMPLETE")
            print(f"{'='*70}")
            print(f"Duration: {duration:.2f}s")
            print(f"Steps: {step}")
            print(f"Success: {'✓' if evaluation.success else '✗'}")
            print(f"Score: {evaluation.score:.2f}")
            print(f"{'='*70}\n")

        return result

    def _plan_next_actions(
        self,
        task_description: str,
        current_obs: Observation
    ) -> str:
        """
        Use LLM to plan next actions

        Returns high-level plan (list of skills to execute)
        """
        # Format prompt with memory context
        memory_context = self.memory.format_for_llm(include_recent=5, include_knowledge=3)

        # Build prompt
        prompt = f"""You are an AI agent controlling a computer/game character.

{memory_context}

CURRENT TASK: {task_description}

CURRENT OBSERVATION:
- Position: {current_obs.position}
- Inventory: {current_obs.inventory}
- Mission state: {current_obs.mission_state}

Plan the next 1-3 high-level actions to make progress on the task.
Use natural language instructions that can be mapped to skills.

Available skills:
- move to [target]
- interact with [object]
- craft [item]
- use [tool] on [target]
- wait for [duration]
- observe [target]

Format your response as a numbered list:
1. ...
2. ...
3. ...

Keep it concise and executable.
"""

        # Get LLM response
        plan_text = ""
        for token in self.planner.llm.generate_stream(prompt, system="You are a helpful planning assistant."):
            plan_text += token

        # Extract first instruction (simple parsing)
        lines = plan_text.split('\n')
        for line in lines:
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                # Found instruction
                instruction = line.strip().lstrip('0123456789.-) ').strip()
                if instruction:
                    return instruction

        return "observe environment"  # Fallback

    def _execute_plan(self, plan: str, verbose: bool = True) -> List:
        """
        Execute plan using skill library

        Args:
            plan: High-level instruction

        Returns:
            List of skill execution results
        """
        results = []

        # Parse plan into skills
        skill_result = self.skills.execute_instruction(self, plan)
        results.append(skill_result)

        if verbose and skill_result.success:
            print(f"  ✓ Executed: {skill_result.skill_name}")
        elif verbose:
            print(f"  ✗ Failed: {skill_result.error_message}")

        return results

    def get_statistics(self) -> Dict:
        """Get agent statistics"""
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
            'avg_steps_per_episode': self.total_steps / max(self.total_episodes, 1),
            'avg_reward_per_episode': self.total_reward / max(self.total_episodes, 1),
            'success_rate': self.evaluator.get_success_rate(),
            'avg_score': self.evaluator.get_average_score(),
            'memory': self.memory.get_memory_summary(),
        }


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate SIMA-style embodied agent"""
    print("\n" + "="*70)
    print("SIMA-STYLE EMBODIED AGENT - Demo")
    print("="*70)
    print()
    print("Complete integration:")
    print("  • Environment adapter (screen-based)")
    print("  • LLM planner (Streaming AGI)")
    print("  • Skill library")
    print("  • Memory system")
    print("  • LLM evaluator")
    print("  • Emotional system")
    print()
    print("="*70)

    # Create agent
    agent = EmbodiedAgent(
        env_config={'type': 'screen'},
        agent_config={'llm_model': 'qwen2.5:3b', 'use_emotions': True}
    )

    # Execute task
    task = "Open a text editor and type 'Hello World'"

    result = agent.execute_task(task, max_steps=10, verbose=True)

    # Show statistics
    print("\n" + "="*70)
    print("AGENT STATISTICS")
    print("="*70)

    stats = agent.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    print("\n✓ Demo complete!")
    print("\n\"생각하고, 기억하고, 행동하고, 평가하고, 학습한다\"")
    print("\"Think, Remember, Act, Evaluate, and Learn\"")


if __name__ == "__main__":
    demo()
