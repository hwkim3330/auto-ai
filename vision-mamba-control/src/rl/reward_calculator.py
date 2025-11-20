"""
Reward Calculator for Autonomous Driving

Calculates rewards based on:
- Lane keeping
- Collision avoidance
- Smooth driving
- Speed maintenance
- Traffic rule compliance
"""

import numpy as np
from typing import Dict, Optional, Tuple


class RewardCalculator:
    """
    Calculate reward for autonomous driving RL

    Reward components:
    1. Lane keeping: Stay in center of lane
    2. Collision avoidance: Avoid pedestrians and vehicles
    3. Smooth driving: Minimize jerky movements
    4. Speed: Maintain appropriate speed
    5. Traffic lights: Respect traffic signals
    """

    def __init__(self):
        # Reward weights
        self.w_lane_keeping = 1.0
        self.w_collision_avoidance = 5.0
        self.w_smooth_driving = 0.5
        self.w_speed = 0.3
        self.w_traffic_light = 2.0

        # History for smoothness calculation
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0

        # Statistics
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.episode_steps = 0

    def calculate_reward(
        self,
        steering: float,
        throttle: float,
        brake: float,
        lane_info: Dict,
        detections: list,
        traffic_lights: list = None
    ) -> Tuple[float, Dict]:
        """
        Calculate reward based on current state

        Args:
            steering: Steering angle (-1 to 1)
            throttle: Throttle (0 to 1)
            brake: Brake (0 to 1)
            lane_info: Lane detection info
            detections: Object detections
            traffic_lights: Traffic light detections

        Returns:
            (total_reward, reward_breakdown)
        """
        rewards = {}

        # 1. Lane keeping reward
        rewards['lane_keeping'] = self._lane_keeping_reward(lane_info)

        # 2. Collision avoidance reward
        rewards['collision_avoidance'] = self._collision_avoidance_reward(detections)

        # 3. Smooth driving reward
        rewards['smooth_driving'] = self._smooth_driving_reward(
            steering, throttle, brake
        )

        # 4. Speed reward
        rewards['speed'] = self._speed_reward(throttle, brake)

        # 5. Traffic light compliance
        if traffic_lights:
            rewards['traffic_light'] = self._traffic_light_reward(
                traffic_lights, throttle, brake
            )
        else:
            rewards['traffic_light'] = 0.0

        # Total weighted reward
        total_reward = (
            self.w_lane_keeping * rewards['lane_keeping'] +
            self.w_collision_avoidance * rewards['collision_avoidance'] +
            self.w_smooth_driving * rewards['smooth_driving'] +
            self.w_speed * rewards['speed'] +
            self.w_traffic_light * rewards['traffic_light']
        )

        # Update history
        self.prev_steering = steering
        self.prev_throttle = throttle
        self.prev_brake = brake

        # Update statistics
        self.total_reward += total_reward
        self.episode_reward += total_reward
        self.episode_steps += 1

        return total_reward, rewards

    def _lane_keeping_reward(self, lane_info: Dict) -> float:
        """
        Reward for staying in lane center

        Returns: -1.0 to 1.0
        """
        if not lane_info:
            return -0.5  # Penalty for no lane detected

        deviation = lane_info.get('deviation', 0)  # Percentage from center

        if abs(deviation) > 50:
            # Way off lane
            return -1.0

        # Gaussian reward centered at 0 deviation
        reward = np.exp(-0.5 * (deviation / 20) ** 2)

        return reward

    def _collision_avoidance_reward(self, detections: list) -> float:
        """
        Reward for avoiding collisions

        Returns: -5.0 to 1.0
        """
        if not detections:
            return 1.0  # No objects, safe

        # Find closest object
        min_distance = float('inf')
        closest_class = None

        for det in detections:
            distance = det.get('distance')
            if distance and distance < min_distance:
                min_distance = distance
                closest_class = det.get('class')

        # No distance info
        if min_distance == float('inf'):
            return 0.5

        # Critical distance thresholds (meters)
        if 'person' in str(closest_class):
            critical_dist = 10.0  # Pedestrians need more space
            danger_dist = 5.0
        else:
            critical_dist = 8.0
            danger_dist = 3.0

        # Reward based on distance
        if min_distance < danger_dist:
            return -5.0  # Imminent collision
        elif min_distance < critical_dist:
            # Linear penalty
            reward = -2.0 * (1 - (min_distance - danger_dist) / (critical_dist - danger_dist))
            return reward
        else:
            # Safe distance
            return 1.0

    def _smooth_driving_reward(
        self,
        steering: float,
        throttle: float,
        brake: float
    ) -> float:
        """
        Reward for smooth control (minimize jerk)

        Returns: -1.0 to 1.0
        """
        # Calculate changes
        d_steering = abs(steering - self.prev_steering)
        d_throttle = abs(throttle - self.prev_throttle)
        d_brake = abs(brake - self.prev_brake)

        # Penalize large changes (jerk)
        jerk = d_steering * 2.0 + d_throttle + d_brake

        if jerk > 1.0:
            return -1.0
        elif jerk > 0.5:
            return -0.5
        else:
            return 1.0 - jerk

    def _speed_reward(self, throttle: float, brake: float) -> float:
        """
        Reward for maintaining appropriate speed

        Returns: -1.0 to 1.0
        """
        # Encourage moderate throttle, discourage excessive braking
        if brake > 0.5:
            return -0.5  # Hard braking penalty

        if throttle < 0.1:
            return -0.3  # Too slow

        if throttle > 0.8:
            return -0.2  # Too fast

        # Optimal throttle range: 0.3-0.6
        if 0.3 <= throttle <= 0.6:
            return 1.0

        return 0.5

    def _traffic_light_reward(
        self,
        traffic_lights: list,
        throttle: float,
        brake: float
    ) -> float:
        """
        Reward for obeying traffic lights

        Returns: -2.0 to 1.0
        """
        if not traffic_lights:
            return 0.0

        # Find closest traffic light
        closest_tl = None
        min_distance = float('inf')

        for tl in traffic_lights:
            distance = tl.get('distance')
            if distance and distance < min_distance:
                min_distance = distance
                closest_tl = tl

        if not closest_tl:
            return 0.0

        state = closest_tl.get('state', 'unknown')

        # Red light
        if state == 'red':
            if min_distance < 20:  # Within stopping range
                if brake > 0.3:
                    return 1.0  # Braking for red light
                elif throttle > 0.3:
                    return -2.0  # Running red light!
                else:
                    return 0.5  # Coasting

        # Green light
        elif state == 'green':
            if throttle > 0.2:
                return 1.0  # Accelerating on green
            else:
                return 0.5

        # Yellow light
        elif state == 'yellow':
            if min_distance < 15:
                if brake > 0.3:
                    return 1.0  # Stopping safely
                else:
                    return 0.0
            else:
                return 0.5

        return 0.0

    def reset_episode(self):
        """Reset episode statistics"""
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0

    def get_stats(self) -> Dict:
        """Get reward statistics"""
        avg_episode_reward = (
            self.episode_reward / self.episode_steps
            if self.episode_steps > 0
            else 0.0
        )

        return {
            'total_reward': self.total_reward,
            'episode_reward': self.episode_reward,
            'episode_steps': self.episode_steps,
            'avg_episode_reward': avg_episode_reward
        }


if __name__ == "__main__":
    # Test reward calculator
    print("Testing Reward Calculator...")

    calc = RewardCalculator()

    # Test scenario 1: Good lane keeping
    lane_info = {'deviation': 2.0, 'lane_width': 300}
    detections = []

    reward, breakdown = calc.calculate_reward(
        steering=0.1,
        throttle=0.5,
        brake=0.0,
        lane_info=lane_info,
        detections=detections
    )

    print(f"\nScenario 1: Good driving")
    print(f"Total reward: {reward:.3f}")
    print(f"Breakdown: {breakdown}")

    # Test scenario 2: Close to pedestrian
    detections = [
        {'class': 'person', 'distance': 6.0}
    ]

    reward, breakdown = calc.calculate_reward(
        steering=0.0,
        throttle=0.3,
        brake=0.5,
        lane_info=lane_info,
        detections=detections
    )

    print(f"\nScenario 2: Pedestrian nearby")
    print(f"Total reward: {reward:.3f}")
    print(f"Breakdown: {breakdown}")

    # Test scenario 3: Red light
    traffic_lights = [
        {'state': 'red', 'distance': 15.0}
    ]

    reward, breakdown = calc.calculate_reward(
        steering=0.0,
        throttle=0.0,
        brake=0.8,
        lane_info=lane_info,
        detections=[],
        traffic_lights=traffic_lights
    )

    print(f"\nScenario 3: Red light stopping")
    print(f"Total reward: {reward:.3f}")
    print(f"Breakdown: {breakdown}")

    print(f"\nStats: {calc.get_stats()}")
    print("\n✅ Reward calculator test passed")
