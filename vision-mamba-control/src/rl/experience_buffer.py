"""
Experience Replay Buffer for Real-time Reinforcement Learning

Stores transitions (state, action, reward, next_state, done) for training
"""

import numpy as np
from collections import deque
from typing import Tuple, List, Dict
import pickle


class ExperienceBuffer:
    """
    Experience replay buffer for RL

    Features:
    - Fixed-size circular buffer
    - Batch sampling for training
    - Prioritized experience replay (optional)
    - Save/load functionality
    """

    def __init__(self, capacity: int = 100000, prioritized: bool = False):
        """
        Args:
            capacity: Maximum buffer size
            prioritized: Use prioritized experience replay
        """
        self.capacity = capacity
        self.prioritized = prioritized

        # Storage
        self.buffer = deque(maxlen=capacity)

        # Prioritized replay
        if prioritized:
            self.priorities = deque(maxlen=capacity)
            self.priority_alpha = 0.6
            self.priority_beta = 0.4

        # Statistics
        self.total_transitions = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict = None
    ):
        """
        Add transition to buffer

        Args:
            state: Current state (image + camera stats)
            action: Action taken (steering, throttle, brake)
            reward: Reward received
            next_state: Next state
            done: Episode done flag
            info: Additional information
        """
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info or {}
        }

        self.buffer.append(transition)

        if self.prioritized:
            # New experiences get max priority
            max_priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(max_priority)

        self.total_transitions += 1

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch from buffer

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch of transitions (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        if self.prioritized:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities ** self.priority_alpha
            probabilities /= probabilities.sum()

            indices = np.random.choice(
                len(self.buffer),
                batch_size,
                p=probabilities,
                replace=False
            )
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        # Extract transitions
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for idx in indices:
            transition = self.buffer[idx]
            states.append(transition['state'])
            actions.append(transition['action'])
            rewards.append(transition['reward'])
            next_states.append(transition['next_state'])
            dones.append(transition['done'])

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled transitions

        Args:
            indices: Indices of transitions
            priorities: New priorities (TD errors)
        """
        if not self.prioritized:
            return

        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Add small constant

    def __len__(self):
        return len(self.buffer)

    def save(self, filepath: str):
        """Save buffer to disk"""
        data = {
            'buffer': list(self.buffer),
            'priorities': list(self.priorities) if self.prioritized else None,
            'total_transitions': self.total_transitions
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"💾 Buffer saved: {len(self.buffer)} transitions → {filepath}")

    def load(self, filepath: str):
        """Load buffer from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.buffer = deque(data['buffer'], maxlen=self.capacity)

        if self.prioritized and data['priorities']:
            self.priorities = deque(data['priorities'], maxlen=self.capacity)

        self.total_transitions = data['total_transitions']

        print(f"📂 Buffer loaded: {len(self.buffer)} transitions from {filepath}")

    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'total_transitions': self.total_transitions,
                'avg_reward': 0.0,
                'max_reward': 0.0,
                'min_reward': 0.0
            }

        rewards = [t['reward'] for t in self.buffer]

        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'total_transitions': self.total_transitions,
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'fill_rate': len(self.buffer) / self.capacity * 100
        }


class EpisodeMemory:
    """
    Stores complete episodes for on-policy algorithms (PPO)
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add transition to episode"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get_batch(self) -> Dict:
        """Get all transitions as batch"""
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'dones': np.array(self.dones)
        }

    def clear(self):
        """Clear episode memory"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def __len__(self):
        return len(self.states)


if __name__ == "__main__":
    # Test experience buffer
    print("Testing Experience Buffer...")

    buffer = ExperienceBuffer(capacity=1000, prioritized=False)

    # Add some transitions
    for i in range(100):
        state = np.random.randn(3, 64, 64)
        action = np.random.randn(3)
        reward = np.random.randn()
        next_state = np.random.randn(3, 64, 64)
        done = np.random.rand() < 0.1

        buffer.add(state, action, reward, next_state, done)

    print(f"Buffer size: {len(buffer)}")
    print(f"Stats: {buffer.get_stats()}")

    # Sample batch
    batch = buffer.sample(32)
    print(f"Batch shapes:")
    print(f"  States: {batch[0].shape}")
    print(f"  Actions: {batch[1].shape}")
    print(f"  Rewards: {batch[2].shape}")
    print(f"  Next states: {batch[3].shape}")
    print(f"  Dones: {batch[4].shape}")

    print("\n✅ Experience buffer test passed")
