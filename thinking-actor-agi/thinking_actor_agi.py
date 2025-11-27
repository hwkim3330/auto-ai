#!/usr/bin/env python3
"""
Thinking Actor AGI - AGI that thinks and acts simultaneously
=============================================================

"생각하면서 동시에 행동한다"

Architecture:
    Query → Streaming AGI (think) → Parse tokens → Computer Agent (act)
                ↓                                        ↓
            Stream output                          Execute actions
                ↓                                        ↓
            Observe results ←─────────────────────────────┘
                ↓
            Learn & Improve

Key Features:
1. Streaming Thinking: Token-by-token reasoning with Ollama
2. Parallel Acting: Execute actions while thinking continues
3. Action Parsing: Extract action commands from thinking tokens
4. Remote Control: HTTP API for remote operation
5. Continuous Learning: Learn from action results

"생각과 행동이 동시에 일어난다"
"""

import sys
from pathlib import Path
import json
import re
import time
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass
import threading
from queue import Queue

# Add paths
sys.path.append(str(Path(__file__).parent.parent / "streaming-agi"))
sys.path.append(str(Path(__file__).parent.parent / "computer-use-ncp"))

from streaming_continuous_agi import ParallelThinkingAGI, StreamingLLM
from computer_agent import ComputerUseAgent, Action, ActionType


# ============================================================================
# Action Command Parser
# ============================================================================

@dataclass
class ActionCommand:
    """Parsed action command from thinking token"""
    type: str
    params: Dict
    reasoning: str
    timestamp: float


class ActionParser:
    """
    Parse action commands from AGI thinking tokens

    Recognizes patterns like:
    - [ACTION: click(100, 200)] - Click at coordinates
    - [ACTION: type("hello")] - Type text
    - [ACTION: move(x=50, y=100)] - Move mouse
    - [ACTION: wait(0.5)] - Wait duration
    """

    def __init__(self):
        # Action patterns
        self.patterns = {
            'click': r'\[ACTION:\s*click\((\d+),\s*(\d+)\)\]',
            'type': r'\[ACTION:\s*type\("([^"]+)"\)\]',
            'move': r'\[ACTION:\s*move\(x=(\d+),\s*y=(\d+)\)\]',
            'key': r'\[ACTION:\s*key\("([^"]+)"\)\]',
            'wait': r'\[ACTION:\s*wait\(([0-9.]+)\)\]',
        }

    def parse(self, token: str) -> Optional[ActionCommand]:
        """
        Parse action command from token

        Args:
            token: Thinking token from AGI

        Returns:
            ActionCommand if found, None otherwise
        """
        for action_type, pattern in self.patterns.items():
            match = re.search(pattern, token)
            if match:
                params = self._extract_params(action_type, match)
                return ActionCommand(
                    type=action_type,
                    params=params,
                    reasoning=token.replace(match.group(0), '').strip(),
                    timestamp=time.time()
                )
        return None

    def _extract_params(self, action_type: str, match) -> Dict:
        """Extract parameters from regex match"""
        if action_type == 'click':
            return {'x': int(match.group(1)), 'y': int(match.group(2))}
        elif action_type == 'type':
            return {'text': match.group(1)}
        elif action_type == 'move':
            return {'x': int(match.group(1)), 'y': int(match.group(2))}
        elif action_type == 'key':
            return {'key': match.group(1)}
        elif action_type == 'wait':
            return {'duration': float(match.group(1))}
        return {}


# ============================================================================
# Thinking Actor AGI
# ============================================================================

class ThinkingActorAGI:
    """
    AGI that thinks and acts simultaneously

    "생각하는 동안 행동한다"

    Architecture:
        Streaming AGI → Parse tokens → Computer Agent
             ↓                             ↓
        Thinking output            Action execution
             ↓                             ↓
        Learn from results ←───────────────┘
    """

    def __init__(self, model: str = "qwen2.5:3b"):
        print("\n" + "="*70)
        print("THINKING ACTOR AGI - Initializing")
        print("="*70)

        # Components
        self.agi = ParallelThinkingAGI(model=model)
        self.agent = ComputerUseAgent()
        self.parser = ActionParser()

        # Action queue for parallel execution
        self.action_queue = Queue()
        self.action_thread = None
        self.running = False

        # Statistics
        self.total_thoughts = 0
        self.total_actions = 0
        self.successful_actions = 0

        print("\n[ThinkingActor] Ready!")
        print(f"  AGI Model: {model}")
        print(f"  Computer Agent: {self.agent.vision.feature_dim} vision features")
        print(f"  NCP Brain: {self.agent.ncp.wiring.total_neurons} neurons")
        print("="*70)

    def _action_executor_thread(self):
        """Background thread for executing actions"""
        while self.running:
            try:
                if not self.action_queue.empty():
                    action_cmd = self.action_queue.get(timeout=0.1)

                    # Convert ActionCommand to Action
                    action = self._convert_to_action(action_cmd)

                    # Execute
                    print(f"\n[Action] Executing: {action.type.value} - {action_cmd.reasoning}")
                    success = self.agent.act(action)

                    self.total_actions += 1
                    if success:
                        self.successful_actions += 1

                    print(f"[Action] {'✓' if success else '✗'} Success rate: {self.successful_actions}/{self.total_actions}")

            except Exception as e:
                continue

    def _convert_to_action(self, cmd: ActionCommand) -> Action:
        """Convert ActionCommand to Agent Action"""
        type_mapping = {
            'click': ActionType.MOUSE_CLICK,
            'move': ActionType.MOUSE_MOVE,
            'type': ActionType.KEYBOARD_TYPE,
            'key': ActionType.KEYBOARD_KEY,
            'wait': ActionType.WAIT,
        }

        return Action(
            type=type_mapping[cmd.type],
            params=cmd.params,
            timestamp=cmd.timestamp
        )

    def think_and_act(self, query: str, max_depth: int = 1, verbose: bool = True) -> Dict:
        """
        Think about query while executing actions

        Args:
            query: Question or task
            max_depth: Thinking depth
            verbose: Print thinking process

        Returns:
            Result with thoughts and actions
        """
        print(f"\n{'='*70}")
        print(f"💭 Query: {query}")
        print(f"{'='*70}\n")

        # Start action executor thread
        self.running = True
        self.action_thread = threading.Thread(target=self._action_executor_thread, daemon=True)
        self.action_thread.start()

        # Stream thinking
        thought_tokens = []
        action_count = 0

        try:
            # Use streaming LLM directly for action control
            system_prompt = """You are an AI that can think and control a computer simultaneously.

When you want to perform an action, include it in your thinking like this:
- [ACTION: click(100, 200)] - Click at coordinates
- [ACTION: type("hello")] - Type text
- [ACTION: move(x=50, y=100)] - Move mouse
- [ACTION: key("Return")] - Press key
- [ACTION: wait(0.5)] - Wait duration

Think step-by-step and include actions as needed."""

            for token in self.agi.llm.generate_stream(query, system=system_prompt):
                # Print thinking token
                if verbose:
                    print(token, end='', flush=True)

                thought_tokens.append(token)

                # Parse for actions
                combined_text = ''.join(thought_tokens[-20:])  # Check last 20 tokens
                action_cmd = self.parser.parse(combined_text)

                if action_cmd:
                    # Queue action for execution
                    self.action_queue.put(action_cmd)
                    action_count += 1

                self.total_thoughts += 1

        finally:
            # Stop action executor
            self.running = False
            if self.action_thread:
                self.action_thread.join(timeout=2)

        result = {
            'query': query,
            'total_tokens': len(thought_tokens),
            'total_actions': action_count,
            'executed_actions': self.total_actions,
            'successful_actions': self.successful_actions,
            'thinking': ''.join(thought_tokens)
        }

        print(f"\n\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Tokens generated: {result['total_tokens']}")
        print(f"Actions found: {result['total_actions']}")
        print(f"Actions executed: {result['executed_actions']}")
        print(f"Success rate: {self.successful_actions}/{self.total_actions}")
        print(f"{'='*70}")

        return result


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate Thinking Actor AGI"""
    print("\n" + "="*70)
    print("THINKING ACTOR AGI - Demo")
    print("="*70)
    print()
    print("AGI that thinks and acts simultaneously")
    print("생각하면서 동시에 행동한다")
    print()
    print("="*70)

    # Create AGI
    agi = ThinkingActorAGI(model="qwen2.5:3b")

    # Example task
    task = """I need to open a text editor.
Please think about how to do this and execute the necessary actions.
Include [ACTION: ...] commands in your thinking."""

    # Think and act
    result = agi.think_and_act(task, max_depth=0, verbose=True)

    print("\n✓ Demo complete!")
    print(f"\nFinal stats:")
    print(f"  Total thoughts: {agi.total_thoughts}")
    print(f"  Total actions: {agi.total_actions}")
    print(f"  Success rate: {agi.successful_actions}/{agi.total_actions}")


if __name__ == "__main__":
    demo()
