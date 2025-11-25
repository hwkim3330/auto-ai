#!/usr/bin/env python3
"""
The Sentinel - Self-Learning Surveillance System
================================================

Inspired by Person of Interest's Machine:
- Watches all cameras and learns continuously
- Uses Liquid NN for efficient online learning
- UltraThink for autonomous reasoning
- Code Agent for recursive self-improvement

Architecture:
    CCTV → Vision → Liquid NN → UltraThink → Code Agent → Loop
"""

import sys
import os
import time
import threading
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent / "liquid-nn-ai"))
sys.path.append(str(Path(__file__).parent.parent / "ultrathink-agi"))

try:
    import torch
    import torch.nn as nn
    from liquid_nn import LiquidNeuralNetwork, count_parameters
    from ultrathink import UltraThink
except ImportError as e:
    print(f"Warning: {e}")
    print("Some features may be unavailable. Install dependencies:")
    print("pip install torch numpy")


@dataclass
class CameraStream:
    """Represents a camera feed"""
    id: str
    url: str
    location: str
    active: bool = True
    last_frame_time: Optional[float] = None


@dataclass
class Observation:
    """Single observation from camera"""
    camera_id: str
    timestamp: float
    features: np.ndarray  # Visual features extracted
    anomaly_score: float = 0.0
    metadata: Dict = None


@dataclass
class LearningMetrics:
    """Metrics for the learning system"""
    total_observations: int = 0
    model_updates: int = 0
    average_loss: float = 0.0
    learning_rate: float = 0.001
    last_update: Optional[float] = None


class VisionPerception:
    """
    Vision processing layer
    Extracts features from camera streams
    """

    def __init__(self, feature_dim: int = 128):
        self.feature_dim = feature_dim
        print(f"[Vision] Initialized with {feature_dim}-dim features")

    def process_frame(self, camera_id: str, frame_data: np.ndarray) -> Observation:
        """
        Extract features from camera frame

        For MVP: Use simple synthetic features
        In production: Use Vision Mamba or similar
        """
        # Synthetic feature extraction for MVP
        features = np.random.randn(self.feature_dim).astype(np.float32)

        # Add some structure based on camera_id
        features[0] = hash(camera_id) % 100 / 100.0

        return Observation(
            camera_id=camera_id,
            timestamp=time.time(),
            features=features,
            metadata={"source": "synthetic"}
        )

    def detect_anomaly(self, observation: Observation, baseline: np.ndarray) -> float:
        """Compute anomaly score vs baseline"""
        if baseline is None or len(baseline) == 0:
            return 0.0

        # Simple distance-based anomaly detection
        distance = np.linalg.norm(observation.features - baseline)
        return float(distance)


class OnlineLearningEngine:
    """
    Continuous learning with Liquid Neural Networks
    Updates model in real-time as new data arrives
    """

    def __init__(self, feature_dim: int = 128, hidden_size: int = 64):
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size

        # Initialize Liquid NN for online learning
        try:
            self.model = LiquidNeuralNetwork(
                input_size=feature_dim,
                hidden_size=hidden_size,
                output_size=feature_dim,  # Reconstruction task
                num_layers=2
            )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.MSELoss()
            print(f"[Learning] Liquid NN initialized: {count_parameters(self.model):,} params")
        except:
            self.model = None
            print("[Learning] Running in fallback mode (no PyTorch)")

        self.metrics = LearningMetrics()
        self.observation_buffer = []
        self.max_buffer_size = 100

    def add_observation(self, observation: Observation):
        """Add new observation to learning buffer"""
        self.observation_buffer.append(observation)
        self.metrics.total_observations += 1

        # Maintain buffer size
        if len(self.observation_buffer) > self.max_buffer_size:
            self.observation_buffer.pop(0)

    def update_model(self, batch_size: int = 32) -> Optional[float]:
        """
        Online learning update
        Returns: loss if update successful
        """
        if self.model is None or len(self.observation_buffer) < batch_size:
            return None

        # Sample from buffer
        indices = np.random.choice(
            len(self.observation_buffer),
            size=min(batch_size, len(self.observation_buffer)),
            replace=False
        )

        batch_features = np.stack([
            self.observation_buffer[i].features
            for i in indices
        ])

        # Convert to torch
        x = torch.FloatTensor(batch_features).unsqueeze(1)  # (batch, 1, features)

        # Forward pass
        self.optimizer.zero_grad()
        result = self.model(x)
        if isinstance(result, tuple):
            output = result[0]
        else:
            output = result

        # Self-supervised learning: predict next state
        target = x  # Reconstruction for simplicity
        loss = self.criterion(output, target)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Update metrics
        self.metrics.model_updates += 1
        self.metrics.average_loss = 0.9 * self.metrics.average_loss + 0.1 * loss.item()
        self.metrics.last_update = time.time()

        return loss.item()

    def get_baseline_features(self) -> np.ndarray:
        """Compute baseline from recent observations"""
        if len(self.observation_buffer) == 0:
            return np.zeros(self.feature_dim)

        features = np.stack([obs.features for obs in self.observation_buffer])
        return np.mean(features, axis=0)


class CodeAgent:
    """
    Self-improvement engine
    Analyzes own code and generates improvements
    """

    def __init__(self, code_directory: Path):
        self.code_dir = Path(code_directory)
        self.improvement_history = []
        self.metrics_history = []
        print(f"[CodeAgent] Monitoring: {self.code_dir}")

    def analyze_performance(self, metrics: Dict) -> Dict:
        """Analyze system performance metrics"""
        self.metrics_history.append({
            'timestamp': time.time(),
            'metrics': metrics
        })

        # Keep only recent history
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)

        analysis = {
            'efficiency': self._compute_efficiency(metrics),
            'bottlenecks': self._identify_bottlenecks(metrics),
            'suggestions': []
        }

        # Generate improvement suggestions
        if metrics.get('average_loss', 1.0) > 0.5:
            analysis['suggestions'].append({
                'type': 'learning_rate',
                'action': 'decrease',
                'reason': 'High loss - reduce learning rate'
            })

        if metrics.get('model_updates', 0) < 10:
            analysis['suggestions'].append({
                'type': 'update_frequency',
                'action': 'increase',
                'reason': 'Few updates - increase training frequency'
            })

        return analysis

    def _compute_efficiency(self, metrics: Dict) -> float:
        """Compute system efficiency score (0-1)"""
        # Simple efficiency metric
        updates = metrics.get('model_updates', 0)
        observations = metrics.get('total_observations', 1)
        update_ratio = updates / max(observations, 1)

        # Ideal ratio is around 0.1 (1 update per 10 observations)
        efficiency = 1.0 - abs(update_ratio - 0.1)
        return max(0.0, min(1.0, efficiency))

    def _identify_bottlenecks(self, metrics: Dict) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        if metrics.get('average_loss', 0) > 0.8:
            bottlenecks.append("High loss - model not learning effectively")

        if metrics.get('learning_rate', 0.001) > 0.01:
            bottlenecks.append("Learning rate too high - may cause instability")

        return bottlenecks

    def generate_improvement(self, analysis: Dict) -> Optional[str]:
        """
        Generate code improvement based on analysis
        Returns: Python code to apply improvement
        """
        if not analysis.get('suggestions'):
            return None

        suggestion = analysis['suggestions'][0]

        # Generate improvement code
        if suggestion['type'] == 'learning_rate':
            if suggestion['action'] == 'decrease':
                code = """
# Auto-generated improvement
def apply_improvement(learning_engine):
    for param_group in learning_engine.optimizer.param_groups:
        param_group['lr'] *= 0.5
    print(f"[CodeAgent] Reduced learning rate to {param_group['lr']}")
"""
                return code

        return None

    def apply_improvement(self, improvement_code: str, context: Dict):
        """
        Apply self-generated improvement
        WARNING: Executes generated code - use with caution!
        """
        try:
            # Execute in controlled namespace
            namespace = {'__builtins__': __builtins__}
            namespace.update(context)

            exec(improvement_code, namespace)

            # Call the improvement function
            if 'apply_improvement' in namespace:
                namespace['apply_improvement'](**context)

                self.improvement_history.append({
                    'timestamp': time.time(),
                    'code': improvement_code,
                    'status': 'success'
                })

                return True
        except Exception as e:
            print(f"[CodeAgent] Improvement failed: {e}")
            self.improvement_history.append({
                'timestamp': time.time(),
                'code': improvement_code,
                'status': 'failed',
                'error': str(e)
            })
            return False


class TheSentinel:
    """
    Main surveillance and learning system
    Integrates all components into recursive improvement loop
    """

    def __init__(self, feature_dim: int = 128):
        print("=" * 60)
        print("THE SENTINEL - Initializing")
        print("=" * 60)

        # Initialize components
        self.vision = VisionPerception(feature_dim)
        self.learning = OnlineLearningEngine(feature_dim)
        self.reasoning = None  # UltraThink loaded on-demand
        self.code_agent = CodeAgent(Path(__file__).parent)

        # State
        self.cameras: List[CameraStream] = []
        self.running = False
        self.cycle_count = 0

        # Performance tracking
        self.start_time = time.time()

        print("[Sentinel] Initialization complete")

    def add_camera(self, camera_id: str, url: str, location: str):
        """Register a new camera stream"""
        camera = CameraStream(
            id=camera_id,
            url=url,
            location=location
        )
        self.cameras.append(camera)
        print(f"[Sentinel] Added camera: {camera_id} at {location}")

    def process_cameras(self) -> List[Observation]:
        """Process all active camera streams"""
        observations = []

        for camera in self.cameras:
            if not camera.active:
                continue

            # For MVP: Simulate frame data
            # In production: Actually read from camera stream
            frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Extract features
            observation = self.vision.process_frame(camera.id, frame_data)

            # Detect anomalies
            baseline = self.learning.get_baseline_features()
            observation.anomaly_score = self.vision.detect_anomaly(observation, baseline)

            observations.append(observation)
            camera.last_frame_time = time.time()

        return observations

    def learn_from_observations(self, observations: List[Observation]):
        """Update learning model with new observations"""
        for obs in observations:
            self.learning.add_observation(obs)

        # Periodic model update
        if self.cycle_count % 10 == 0:
            loss = self.learning.update_model()
            if loss is not None:
                print(f"[Learning] Update #{self.learning.metrics.model_updates} | Loss: {loss:.4f}")

    def reason_about_observations(self, observations: List[Observation]) -> Dict:
        """
        Use UltraThink to reason about patterns
        Returns: reasoning results
        """
        # Check for significant anomalies
        high_anomaly_obs = [obs for obs in observations if obs.anomaly_score > 2.0]

        if not high_anomaly_obs:
            return {'anomalies_detected': False}

        # Lazy load UltraThink (heavy operation)
        if self.reasoning is None:
            try:
                self.reasoning = UltraThink(feature_dim=128, hidden_size=64)
                print("[Reasoning] UltraThink loaded")
            except:
                print("[Reasoning] UltraThink unavailable")
                return {'anomalies_detected': True, 'reasoning': 'unavailable'}

        # Construct query
        query = f"Detected {len(high_anomaly_obs)} anomalies across cameras. "
        query += f"Locations: {[obs.camera_id for obs in high_anomaly_obs]}. "
        query += "What pattern might this indicate?"

        # Reason about it (expensive - use sparingly)
        result = self.reasoning.think(query, verbose=False)

        return {
            'anomalies_detected': True,
            'count': len(high_anomaly_obs),
            'reasoning': result.get('conclusion', ''),
            'confidence': result.get('confidence', 0.0)
        }

    def self_improve(self):
        """Recursive self-improvement cycle"""
        # Gather metrics
        metrics = {
            'total_observations': self.learning.metrics.total_observations,
            'model_updates': self.learning.metrics.model_updates,
            'average_loss': self.learning.metrics.average_loss,
            'learning_rate': self.learning.metrics.learning_rate,
            'uptime_hours': (time.time() - self.start_time) / 3600,
            'cycles': self.cycle_count
        }

        # Analyze performance
        analysis = self.code_agent.analyze_performance(metrics)

        # Generate and apply improvements
        if analysis.get('suggestions'):
            improvement = self.code_agent.generate_improvement(analysis)
            if improvement:
                print(f"[CodeAgent] Applying self-improvement...")
                success = self.code_agent.apply_improvement(
                    improvement,
                    {'learning_engine': self.learning}
                )
                if success:
                    print(f"[CodeAgent] Improvement applied successfully")

    def run_cycle(self):
        """Single iteration of the sentinel loop"""
        self.cycle_count += 1

        # 1. Vision: Process camera streams
        observations = self.process_cameras()

        # 2. Learning: Update model
        self.learn_from_observations(observations)

        # 3. Reasoning: Analyze patterns (periodic)
        if self.cycle_count % 50 == 0:
            reasoning_result = self.reason_about_observations(observations)
            if reasoning_result.get('anomalies_detected'):
                print(f"[Reasoning] {reasoning_result}")

        # 4. Self-Improvement: Recursive optimization (periodic)
        if self.cycle_count % 100 == 0:
            self.self_improve()

        # Status update
        if self.cycle_count % 20 == 0:
            print(f"[Sentinel] Cycle {self.cycle_count} | "
                  f"Observations: {self.learning.metrics.total_observations} | "
                  f"Updates: {self.learning.metrics.model_updates} | "
                  f"Avg Loss: {self.learning.metrics.average_loss:.4f}")

    def run(self, cycles: Optional[int] = None):
        """
        Main loop

        Args:
            cycles: Number of cycles to run (None = infinite)
        """
        self.running = True
        print(f"\n[Sentinel] Starting main loop...")
        print(f"[Sentinel] Monitoring {len(self.cameras)} cameras")
        print("=" * 60)

        try:
            cycle = 0
            while self.running and (cycles is None or cycle < cycles):
                self.run_cycle()
                cycle += 1
                time.sleep(0.1)  # 10 Hz update rate

        except KeyboardInterrupt:
            print("\n[Sentinel] Shutdown requested")
        finally:
            self.running = False
            self.save_state()

    def save_state(self):
        """Save system state to disk"""
        state = {
            'timestamp': time.time(),
            'cycles': self.cycle_count,
            'cameras': [asdict(cam) for cam in self.cameras],
            'learning_metrics': asdict(self.learning.metrics),
            'improvements': self.code_agent.improvement_history
        }

        output_path = Path(__file__).parent / "sentinel_state.json"
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"[Sentinel] State saved to {output_path}")


def main():
    """Run The Sentinel"""
    # Initialize system
    sentinel = TheSentinel(feature_dim=128)

    # Add cameras (MVP: simulated)
    sentinel.add_camera("CAM_001", "rtsp://10.0.0.1/stream1", "Main Entrance")
    sentinel.add_camera("CAM_002", "rtsp://10.0.0.2/stream1", "Parking Lot")
    sentinel.add_camera("CAM_003", "rtsp://10.0.0.3/stream1", "Server Room")
    sentinel.add_camera("CAM_004", "rtsp://10.0.0.4/stream1", "Emergency Exit")

    # Run for 200 cycles (demo mode)
    # For production: sentinel.run() for infinite loop
    sentinel.run(cycles=200)

    print("\n" + "=" * 60)
    print("THE SENTINEL - Final Report")
    print("=" * 60)
    print(f"Total Cycles: {sentinel.cycle_count}")
    print(f"Total Observations: {sentinel.learning.metrics.total_observations}")
    print(f"Model Updates: {sentinel.learning.metrics.model_updates}")
    print(f"Final Loss: {sentinel.learning.metrics.average_loss:.4f}")
    print(f"Self-Improvements: {len(sentinel.code_agent.improvement_history)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
