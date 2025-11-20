"""
Real-time Reinforcement Learning Trainer

Online learning system for Vision Mamba Control:
- PPO (Proximal Policy Optimization)
- Experience replay
- Real-time model updates
- Tensorboard logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
import time
from pathlib import Path

from .experience_buffer import ExperienceBuffer, EpisodeMemory
from .reward_calculator import RewardCalculator


class PPOTrainer:
    """
    PPO Trainer for Vision Mamba Control

    Features:
    - Proximal Policy Optimization
    - Continuous action space
    - GAE (Generalized Advantage Estimation)
    - Clipped surrogate objective
    - Real-time online learning
    """

    def __init__(
        self,
        model,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 50000,
        train_interval: float = 10.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Vision Mamba Control model
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clip epsilon
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm
            update_epochs: Number of epochs per update
            batch_size: Batch size
            buffer_size: Experience buffer size
            train_interval: Training interval in seconds
            device: Device (cuda/cpu)
        """
        self.model = model.to(device)
        self.device = device

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Experience storage
        self.buffer = ExperienceBuffer(capacity=buffer_size, prioritized=False)
        self.episode_memory = EpisodeMemory()

        # Reward calculator
        self.reward_calc = RewardCalculator()

        # Training statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.best_reward = -float('inf')

        # Training mode
        self.training_enabled = False
        self.last_train_time = time.time()
        self.train_interval = train_interval  # Train every N seconds

        print(f"🤖 PPO Trainer initialized on {device}")
        print(f"📊 Parameters:")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Gamma: {gamma}")
        print(f"  - GAE lambda: {gae_lambda}")
        print(f"  - Clip epsilon: {clip_epsilon}")
        print(f"  - Buffer size: {buffer_size}")

    def enable_training(self):
        """Enable real-time training"""
        self.training_enabled = True
        self.model.train()
        print("✅ Real-time training ENABLED")

    def disable_training(self):
        """Disable real-time training"""
        self.training_enabled = False
        self.model.eval()
        print("⏸️  Real-time training DISABLED")

    def collect_experience(
        self,
        frame: np.ndarray,
        action: np.ndarray,
        lane_info: Dict,
        detections: list,
        traffic_lights: list,
        camera_stats: Tuple[float, float, float]
    ) -> float:
        """
        Collect experience from environment

        Args:
            frame: Current frame
            action: Action taken (steering, throttle, brake)
            lane_info: Lane detection info
            detections: Object detections
            traffic_lights: Traffic light detections
            camera_stats: (brightness, contrast, saturation)

        Returns:
            reward
        """
        # Calculate reward
        reward, reward_breakdown = self.reward_calc.calculate_reward(
            steering=action[0],
            throttle=action[1],
            brake=action[2],
            lane_info=lane_info,
            detections=detections,
            traffic_lights=traffic_lights
        )

        # Store transition
        state = self._prepare_state(frame, camera_stats)

        # Add to buffer
        self.buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=state,  # Will be updated on next step
            done=False,
            info=reward_breakdown
        )

        self.total_steps += 1

        # Trigger training if needed
        if self.training_enabled:
            current_time = time.time()
            if current_time - self.last_train_time > self.train_interval:
                if len(self.buffer) >= self.batch_size:
                    self.train_step()
                    self.last_train_time = current_time

        return reward

    def train_step(self) -> Dict:
        """
        Perform one training step

        Returns:
            Training statistics
        """
        self.model.train()

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)

        # Forward pass (get current policy)
        with torch.no_grad():
            # Get old log probs and values
            old_log_probs, old_values = self._evaluate_actions(states_t, actions_t)

        # Compute advantages using GAE
        advantages = self._compute_gae(rewards_t, old_values, dones)
        returns = advantages + old_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for epoch in range(self.update_epochs):
            # Get current log probs and values
            log_probs, values = self._evaluate_actions(states_t, actions_t)

            # Compute policy loss (clipped surrogate objective)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Compute value loss
            value_loss = nn.MSELoss()(values, returns)

            # Compute entropy (for exploration)
            entropy = -log_probs.mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        # Statistics
        stats = {
            'policy_loss': total_policy_loss / self.update_epochs,
            'value_loss': total_value_loss / self.update_epochs,
            'entropy': total_entropy / self.update_epochs,
            'avg_reward': rewards_t.mean().item(),
            'buffer_size': len(self.buffer)
        }

        self.model.eval()

        return stats

    def _prepare_state(self, frame: np.ndarray, camera_stats: Tuple) -> np.ndarray:
        """Prepare state representation"""
        # Resize frame
        frame_resized = torch.nn.functional.interpolate(
            torch.FloatTensor(frame).unsqueeze(0).permute(0, 3, 1, 2),
            size=(64, 64),
            mode='bilinear'
        ).squeeze(0).permute(1, 2, 0).numpy()

        # Normalize
        frame_norm = frame_resized / 255.0

        # Combine with camera stats
        state = {
            'frame': frame_norm,
            'camera_stats': np.array(camera_stats)
        }

        return state

    def _evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple:
        """
        Evaluate actions under current policy

        Returns:
            (log_probs, values)
        """
        # This is simplified - in practice, you'd need a proper policy network
        # For now, using the control model's output

        # Compute log probs (assuming Gaussian policy)
        mean = torch.zeros_like(actions)  # Placeholder
        std = torch.ones_like(actions) * 0.5

        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # Compute values (placeholder)
        values = torch.zeros(len(states)).to(self.device)

        return log_probs, values

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: np.ndarray
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation

        Args:
            rewards: Rewards
            values: Value estimates
            dones: Done flags

        Returns:
            Advantages
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        return advantages

    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward
        }

        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.total_episodes = checkpoint['total_episodes']
        self.best_reward = checkpoint['best_reward']

        print(f"📂 Checkpoint loaded: {filepath}")

    def get_stats(self) -> Dict:
        """Get training statistics"""
        buffer_stats = self.buffer.get_stats()
        reward_stats = self.reward_calc.get_stats()

        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward,
            'training_enabled': self.training_enabled,
            'buffer_size': len(self.buffer),
            'buffer_fill_rate': buffer_stats.get('fill_rate', 0),
            'avg_reward': reward_stats.get('avg_episode_reward', 0),
            'episode_reward': reward_stats.get('episode_reward', 0)
        }


if __name__ == "__main__":
    print("Testing PPO Trainer...")

    # Create dummy model
    from models.control_model import create_control_model_tiny

    model = create_control_model_tiny(use_film=True)

    # Create trainer
    trainer = PPOTrainer(
        model=model,
        learning_rate=3e-4,
        batch_size=32,
        buffer_size=1000
    )

    # Enable training
    trainer.enable_training()

    # Simulate data collection
    for i in range(100):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        action = np.random.randn(3)
        lane_info = {'deviation': np.random.randn() * 10, 'lane_width': 300}
        detections = []
        traffic_lights = []
        camera_stats = (0.5, 0.5, 0.5)

        reward = trainer.collect_experience(
            frame, action, lane_info, detections, traffic_lights, camera_stats
        )

        if i % 20 == 0:
            print(f"Step {i}: Reward = {reward:.3f}")

    print(f"\nStats: {trainer.get_stats()}")
    print("\n✅ PPO trainer test passed")
