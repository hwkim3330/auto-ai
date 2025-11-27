#!/usr/bin/env python3
"""
Environment Adapter - Unified interface for games/simulators
==============================================================

"모든 환경을 하나의 API로"

Supports:
- Screen-based environments (games via screenshot + keyboard/mouse)
- Simulators (CARLA, Isaac Sim, Unity ML-Agents)
- Custom environments

Interface:
- reset(task_spec) → initial observation
- step(action) → observation, reward, done, info
- get_observation() → current state
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
from PIL import Image


# ============================================================================
# Environment Types
# ============================================================================

class EnvType(Enum):
    """Supported environment types"""
    SCREEN_BASED = "screen"      # Games via screen capture
    CARLA = "carla"              # CARLA simulator
    ISAAC_SIM = "isaac"          # NVIDIA Isaac Sim
    UNITY = "unity"              # Unity ML-Agents
    CUSTOM = "custom"            # Custom environment


@dataclass
class Observation:
    """
    Unified observation format

    All environments return this structure
    """
    # Visual
    image: Optional[np.ndarray] = None      # RGB image (H, W, 3)
    depth: Optional[np.ndarray] = None      # Depth map (H, W)

    # State
    position: Optional[np.ndarray] = None   # (x, y, z) or (x, y)
    velocity: Optional[np.ndarray] = None   # Velocity vector
    rotation: Optional[np.ndarray] = None   # Euler angles or quaternion

    # Task-specific
    inventory: Optional[Dict] = None        # Game inventory
    mission_state: Optional[Dict] = None    # Mission progress

    # Metadata
    timestamp: float = 0.0
    frame_id: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for LLM"""
        return {
            'has_image': self.image is not None,
            'image_shape': self.image.shape if self.image is not None else None,
            'position': self.position.tolist() if self.position is not None else None,
            'velocity': self.velocity.tolist() if self.velocity is not None else None,
            'inventory': self.inventory,
            'mission_state': self.mission_state,
            'timestamp': self.timestamp,
        }


@dataclass
class Action:
    """
    Unified action format

    Supports both discrete and continuous actions
    """
    # Discrete actions (keyboard/buttons)
    keys: Optional[List[str]] = None        # ['W', 'A', 'Space', ...]
    buttons: Optional[List[int]] = None     # [0, 1, ...] button indices

    # Continuous actions
    steering: Optional[float] = None        # [-1, 1]
    throttle: Optional[float] = None        # [0, 1]
    brake: Optional[float] = None           # [0, 1]

    # Mouse/camera
    mouse_dx: Optional[float] = None        # Mouse x movement
    mouse_dy: Optional[float] = None        # Mouse y movement
    mouse_click: Optional[str] = None       # 'left', 'right', 'middle'

    # Camera control
    camera_pitch: Optional[float] = None
    camera_yaw: Optional[float] = None

    # Custom
    custom_params: Optional[Dict] = None


# ============================================================================
# Base Environment Adapter
# ============================================================================

class BaseEnvAdapter:
    """
    Base class for all environment adapters

    Subclass this to support new environments
    """

    def __init__(self, env_config: Dict):
        self.env_config = env_config
        self.env_type = EnvType(env_config.get('type', 'screen'))

        self.current_obs = None
        self.episode_step = 0
        self.total_steps = 0

        print(f"[EnvAdapter] Initialized {self.env_type.value} environment")

    def reset(self, task_spec: Optional[Dict] = None) -> Observation:
        """
        Reset environment and start new episode

        Args:
            task_spec: Task specification (goal, constraints, etc.)

        Returns:
            Initial observation
        """
        raise NotImplementedError

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Execute action and get next observation

        Args:
            action: Action to execute

        Returns:
            (observation, reward, done, info)
        """
        raise NotImplementedError

    def get_observation(self) -> Observation:
        """Get current observation without stepping"""
        return self.current_obs

    def close(self):
        """Clean up resources"""
        pass


# ============================================================================
# Screen-based Environment (Games)
# ============================================================================

class ScreenEnvAdapter(BaseEnvAdapter):
    """
    Environment adapter for screen-based games

    Uses:
    - Screenshot capture for vision
    - Keyboard/mouse control for actions
    - Simple heuristic rewards
    """

    def __init__(self, env_config: Dict):
        super().__init__(env_config)

        # Import vision system from computer agent
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / "computer-use-ncp"))

        from computer_agent import VisionSystem, ActionExecutor

        self.vision = VisionSystem(use_real_vision=True)
        self.executor = ActionExecutor()

        print(f"[ScreenEnv] Ready - Vision: {self.vision.feature_dim}D features")

    def reset(self, task_spec: Optional[Dict] = None) -> Observation:
        """Reset by capturing initial screen"""
        self.episode_step = 0

        # Capture initial screen
        img = self.vision.capture_screen()
        features = self.vision.get_current_features()

        self.current_obs = Observation(
            image=np.array(img) if img else None,
            position=None,  # Not available for screen-based
            velocity=None,
            inventory=None,
            mission_state=task_spec if task_spec else {},
            timestamp=time.time(),
            frame_id=0
        )

        print(f"[ScreenEnv] Reset - Task: {task_spec}")
        return self.current_obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Execute action via keyboard/mouse

        Reward is simple heuristic (can be overridden by LLM evaluator)
        """
        self.episode_step += 1
        self.total_steps += 1

        # Execute keyboard actions
        if action.keys:
            for key in action.keys:
                self.executor.press_key(key)
                time.sleep(0.05)

        # Execute mouse actions
        if action.mouse_dx is not None or action.mouse_dy is not None:
            dx = action.mouse_dx or 0
            dy = action.mouse_dy or 0
            self.executor.move_mouse(int(dx), int(dy))

        if action.mouse_click:
            self.executor.click_mouse(action.mouse_click)

        # Wait for action to take effect
        time.sleep(0.1)

        # Capture new screen
        img = self.vision.capture_screen()

        self.current_obs = Observation(
            image=np.array(img) if img else None,
            timestamp=time.time(),
            frame_id=self.episode_step
        )

        # Simple heuristic reward (to be replaced by LLM evaluator)
        reward = 0.0
        done = False
        info = {'step': self.episode_step}

        return self.current_obs, reward, done, info


# ============================================================================
# Simulator Environment (CARLA, Isaac, Unity)
# ============================================================================

class SimulatorEnvAdapter(BaseEnvAdapter):
    """
    Environment adapter for simulators

    Supports:
    - CARLA (autonomous driving)
    - Isaac Sim (robotics)
    - Unity ML-Agents (games)

    These have native Python APIs
    """

    def __init__(self, env_config: Dict):
        super().__init__(env_config)

        # Lazy import based on type
        self.sim = None

        if self.env_type == EnvType.CARLA:
            self._init_carla()
        elif self.env_type == EnvType.ISAAC_SIM:
            self._init_isaac()
        elif self.env_type == EnvType.UNITY:
            self._init_unity()

    def _init_carla(self):
        """Initialize CARLA simulator"""
        try:
            import carla

            host = self.env_config.get('host', 'localhost')
            port = self.env_config.get('port', 2000)

            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()

            print(f"[CARLA] Connected to {host}:{port}")
        except ImportError:
            print("[CARLA] Not available - install with: pip install carla")
            self.sim = None

    def _init_isaac(self):
        """Initialize Isaac Sim"""
        print("[Isaac] Not implemented yet")
        self.sim = None

    def _init_unity(self):
        """Initialize Unity ML-Agents"""
        try:
            from mlagents_envs.environment import UnityEnvironment

            env_path = self.env_config.get('env_path')
            self.sim = UnityEnvironment(file_name=env_path)

            print(f"[Unity] Loaded environment: {env_path}")
        except ImportError:
            print("[Unity] Not available - install with: pip install mlagents")
            self.sim = None

    def reset(self, task_spec: Optional[Dict] = None) -> Observation:
        """Reset simulator"""
        if self.sim is None:
            # Fallback to screen-based
            print("[Simulator] Not available, using screen-based fallback")
            return Observation(timestamp=time.time())

        # Simulator-specific reset
        # TODO: Implement for each simulator
        return Observation(timestamp=time.time())

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """Execute action in simulator"""
        if self.sim is None:
            return Observation(), 0.0, False, {}

        # Simulator-specific step
        # TODO: Implement for each simulator
        return Observation(), 0.0, False, {}


# ============================================================================
# Environment Factory
# ============================================================================

def create_env(env_config: Dict) -> BaseEnvAdapter:
    """
    Factory function to create environment adapter

    Args:
        env_config: Configuration dict with 'type' key

    Returns:
        Environment adapter instance

    Example:
        env = create_env({'type': 'screen'})
        env = create_env({'type': 'carla', 'host': 'localhost', 'port': 2000})
    """
    env_type = EnvType(env_config.get('type', 'screen'))

    if env_type == EnvType.SCREEN_BASED:
        return ScreenEnvAdapter(env_config)
    elif env_type in [EnvType.CARLA, EnvType.ISAAC_SIM, EnvType.UNITY]:
        return SimulatorEnvAdapter(env_config)
    else:
        return ScreenEnvAdapter(env_config)  # Default fallback


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate environment adapter"""
    print("\n" + "="*70)
    print("ENVIRONMENT ADAPTER - Demo")
    print("="*70)

    # Create screen-based environment
    env = create_env({'type': 'screen'})

    # Reset
    obs = env.reset(task_spec={'goal': 'Open text editor'})
    print(f"\n[Demo] Initial observation:")
    print(f"  Image: {obs.image.shape if obs.image is not None else None}")
    print(f"  Timestamp: {obs.timestamp}")

    # Execute simple action
    action = Action(keys=['Super_L'])  # Open app menu
    obs, reward, done, info = env.step(action)

    print(f"\n[Demo] After action:")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    print(f"  Info: {info}")

    env.close()
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    demo()
