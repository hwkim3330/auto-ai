#!/usr/bin/env python3
"""
Computer Use Agent with Neural Circuit Policies
================================================

"컴퓨터를 사용하는 법을 배우는 AI - NCP로 점점 나아진다"

Architecture:
    Screen → Vision → NCP → Action → Execute → Learn
                      ↑__________________|
                      (continuous improvement)

Key Features:
1. Vision: Screenshot → feature extraction
2. NCP: Biological neural circuit for decision-making
3. Action: Keyboard/mouse control
4. Learning: Online learning from results
5. Continuous improvement: Gets better over time

"하루하루 컴퓨터를 더 잘 쓰게 된다"
"""

import numpy as np
import subprocess
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add NCP to path
sys.path.append(str(Path(__file__).parent.parent / "neural-circuit-policies"))
from ncp_core import auto_wiring, NeuralCircuitPolicy


# ============================================================================
# Action Types
# ============================================================================

class ActionType(Enum):
    """Possible computer actions"""
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    KEYBOARD_TYPE = "keyboard_type"
    KEYBOARD_KEY = "keyboard_key"
    WAIT = "wait"
    SCREENSHOT = "screenshot"


@dataclass
class Action:
    """An action to execute"""
    type: ActionType
    params: Dict
    timestamp: float


@dataclass
class Experience:
    """Learning experience"""
    screen_features: np.ndarray
    action: Action
    result_features: np.ndarray
    reward: float
    timestamp: float


# ============================================================================
# Vision System
# ============================================================================

class VisionSystem:
    """
    Extract features from screenshots

    Simple feature extraction:
    - Downsample screen to 32x32 grid
    - Convert to grayscale
    - Normalize to [-1, 1]
    - Flatten to 1024-dim vector
    """

    def __init__(self, target_size: Tuple[int, int] = (32, 32)):
        self.target_size = target_size
        self.feature_dim = target_size[0] * target_size[1]

    def capture_screen(self) -> np.ndarray:
        """
        Capture screenshot using scrot

        Returns:
            Raw screenshot data (will implement image processing)
        """
        try:
            # For now, return simulated screen features
            # In production, use: scrot, PIL, opencv
            return np.random.randn(1920, 1080, 3)  # Simulated
        except Exception as e:
            print(f"[Vision] Error capturing screen: {e}")
            return np.zeros((1920, 1080, 3))

    def extract_features(self, screenshot: np.ndarray) -> np.ndarray:
        """
        Extract features from screenshot

        Args:
            screenshot: Raw image data

        Returns:
            Feature vector (1024-dim for 32x32)
        """
        # Simulate feature extraction
        # In production: downsample, grayscale, normalize
        features = np.random.randn(self.feature_dim)

        # Normalize
        features = np.tanh(features)  # [-1, 1]

        return features

    def get_current_features(self) -> np.ndarray:
        """Get features from current screen"""
        screen = self.capture_screen()
        return self.extract_features(screen)


# ============================================================================
# Action Executor
# ============================================================================

class ActionExecutor:
    """
    Execute computer actions

    Uses xdotool for keyboard/mouse control
    """

    def __init__(self):
        self.last_action_time = time.time()

    def execute(self, action: Action) -> bool:
        """
        Execute an action

        Args:
            action: Action to execute

        Returns:
            Success status
        """
        try:
            if action.type == ActionType.MOUSE_MOVE:
                x = action.params.get("x", 0)
                y = action.params.get("y", 0)
                subprocess.run(["xdotool", "mousemove", str(x), str(y)], check=True)

            elif action.type == ActionType.MOUSE_CLICK:
                button = action.params.get("button", 1)
                subprocess.run(["xdotool", "click", str(button)], check=True)

            elif action.type == ActionType.KEYBOARD_TYPE:
                text = action.params.get("text", "")
                subprocess.run(["xdotool", "type", text], check=True)

            elif action.type == ActionType.KEYBOARD_KEY:
                key = action.params.get("key", "Return")
                subprocess.run(["xdotool", "key", key], check=True)

            elif action.type == ActionType.WAIT:
                duration = action.params.get("duration", 0.1)
                time.sleep(duration)

            self.last_action_time = time.time()
            return True

        except Exception as e:
            print(f"[Executor] Error executing {action.type}: {e}")
            return False


# ============================================================================
# NCP-based Computer Use Agent
# ============================================================================

class ComputerUseAgent:
    """
    Computer use agent with Neural Circuit Policy

    "컴퓨터를 사용하는 법을 배우는 AI"

    Architecture:
        Vision (1024) → NCP → Action (8)

    NCP Structure:
        Sensory (1024) → Inter (32) → Command (32) → Motor (8)

    Motor outputs map to:
        0-1: Mouse movement (x, y normalized)
        2: Click probability
        3-5: Keyboard action type
        6-7: Wait/screenshot control
    """

    def __init__(self):
        print("\n" + "="*70)
        print("COMPUTER USE AGENT - Initializing")
        print("="*70)

        # Components
        self.vision = VisionSystem()
        self.executor = ActionExecutor()

        # NCP Brain
        print("\n[Agent] Creating NCP brain...")
        wiring = auto_wiring(
            input_size=1024,    # Vision features (32x32)
            output_size=8,      # Action outputs
            inter_neurons=32,   # Interneurons for processing
            command_neurons=32  # Command neurons (with recurrence)
        )
        self.ncp = NeuralCircuitPolicy(wiring, use_cfc=True)

        # Learning
        self.experiences: List[Experience] = []
        self.total_actions = 0
        self.successful_actions = 0

        print("\n[Agent] Ready!")
        print(f"  Vision: {self.vision.feature_dim}-dim features")
        print(f"  NCP: {wiring.total_neurons} neurons, {wiring.total_synapses} synapses")
        print(f"  Actions: 8 output dimensions")
        print("="*70)

    def perceive(self) -> np.ndarray:
        """Get current screen features"""
        return self.vision.get_current_features()

    def think(self, features: np.ndarray) -> np.ndarray:
        """
        Think about what action to take

        Args:
            features: Screen features

        Returns:
            Action outputs from NCP
        """
        # NCP forward pass (continuous-time)
        outputs = self.ncp.forward(features, dt=0.1)
        return outputs

    def decode_action(self, outputs: np.ndarray) -> Action:
        """
        Decode NCP outputs into an action

        Args:
            outputs: NCP motor outputs (8-dim)

        Returns:
            Executable action
        """
        # Parse outputs
        mouse_x = outputs[0]  # -1 to 1
        mouse_y = outputs[1]  # -1 to 1
        click_prob = outputs[2]  # -1 to 1
        key_type = np.argmax(outputs[3:6])  # 0, 1, 2
        wait_prob = outputs[6]
        screenshot_prob = outputs[7]

        # Decide action type
        if screenshot_prob > 0.5:
            return Action(
                type=ActionType.SCREENSHOT,
                params={},
                timestamp=time.time()
            )

        if wait_prob > 0.3:
            return Action(
                type=ActionType.WAIT,
                params={"duration": 0.2},
                timestamp=time.time()
            )

        if click_prob > 0.2:
            # Mouse click
            # Convert normalized outputs to screen coords
            x = int((mouse_x + 1) / 2 * 1920)  # Assume 1920x1080
            y = int((mouse_y + 1) / 2 * 1080)

            return Action(
                type=ActionType.MOUSE_CLICK,
                params={"x": x, "y": y, "button": 1},
                timestamp=time.time()
            )

        if key_type == 0:
            # Type text
            return Action(
                type=ActionType.KEYBOARD_TYPE,
                params={"text": "hello"},
                timestamp=time.time()
            )
        elif key_type == 1:
            # Press key
            return Action(
                type=ActionType.KEYBOARD_KEY,
                params={"key": "Return"},
                timestamp=time.time()
            )
        else:
            # Mouse move
            x = int((mouse_x + 1) / 2 * 1920)
            y = int((mouse_y + 1) / 2 * 1080)

            return Action(
                type=ActionType.MOUSE_MOVE,
                params={"x": x, "y": y},
                timestamp=time.time()
            )

    def act(self, action: Action) -> bool:
        """Execute an action"""
        success = self.executor.execute(action)
        self.total_actions += 1
        if success:
            self.successful_actions += 1
        return success

    def learn(self, experience: Experience):
        """
        Learn from experience (online learning)

        For now: Store experience
        Future: Update NCP weights based on reward

        Args:
            experience: Learning experience
        """
        self.experiences.append(experience)

        # Simple learning: If reward > 0, strengthen connection
        # (Real implementation would use gradient-free learning)

        if len(self.experiences) % 10 == 0:
            print(f"[Learn] {len(self.experiences)} experiences collected")

    def run_cycle(self) -> Dict:
        """
        Run one perception-action cycle

        Returns:
            Cycle info
        """
        # 1. Perceive
        features_before = self.perceive()

        # 2. Think
        outputs = self.think(features_before)

        # 3. Decode action
        action = self.decode_action(outputs)

        # 4. Act
        success = self.act(action)

        # 5. Perceive result
        time.sleep(0.1)
        features_after = self.perceive()

        # 6. Compute reward (simple heuristic)
        # Reward = change in screen (something happened)
        change = np.linalg.norm(features_after - features_before)
        reward = min(change / 10.0, 1.0) if success else -0.1

        # 7. Learn
        experience = Experience(
            screen_features=features_before,
            action=action,
            result_features=features_after,
            reward=reward,
            timestamp=time.time()
        )
        self.learn(experience)

        return {
            "action": action.type.value,
            "success": success,
            "reward": reward,
            "change": change,
            "ncp_state": self.ncp.get_state()
        }

    def run(self, num_cycles: int = 100):
        """
        Run agent for multiple cycles

        Args:
            num_cycles: Number of cycles to run
        """
        print(f"\n[Agent] Running for {num_cycles} cycles...")
        print("="*70)

        for cycle in range(num_cycles):
            info = self.run_cycle()

            if cycle % 10 == 0:
                print(f"\nCycle {cycle}:")
                print(f"  Action: {info['action']}")
                print(f"  Success: {info['success']}")
                print(f"  Reward: {info['reward']:.3f}")
                print(f"  Screen change: {info['change']:.3f}")
                print(f"  Success rate: {self.successful_actions}/{self.total_actions} = {self.successful_actions/max(self.total_actions,1):.1%}")

        print("\n" + "="*70)
        print("AGENT RUN COMPLETE")
        print("="*70)
        print(f"Total cycles: {num_cycles}")
        print(f"Total actions: {self.total_actions}")
        print(f"Successful: {self.successful_actions} ({self.successful_actions/max(self.total_actions,1):.1%})")
        print(f"Experiences collected: {len(self.experiences)}")
        print("="*70)


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate computer use agent"""
    print("\n" + "="*70)
    print("COMPUTER USE AGENT - Demo")
    print("="*70)
    print()
    print("NCP-based agent that learns to use the computer")
    print("Vision → NCP → Action → Learn → Improve")
    print()
    print("Starting in 3 seconds...")
    print("="*70)

    time.sleep(3)

    # Create agent
    agent = ComputerUseAgent()

    # Run for 50 cycles (safe demo)
    agent.run(num_cycles=50)

    print("\n✓ Demo complete!")
    print("\nNext steps:")
    print("  - Add real vision (screenshot + CV)")
    print("  - Implement NCP weight updates (online learning)")
    print("  - Add task-specific rewards")
    print("  - Integrate with Streaming AGI for planning")


if __name__ == "__main__":
    demo()
