"""
2D Driving Simulator for Vision Mamba Control

무한 시뮬레이션으로 학습 데이터 생성
- 차량, 보행자, 신호등 시뮬레이션
- 차선, 도로 환경
- 랜덤 시나리오 생성
- Vision Mamba가 무한히 학습할 수 있는 환경
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
import random
import time


class Vehicle:
    """시뮬레이션 차량"""
    def __init__(self, x: float, y: float, velocity: float, lane: int):
        self.x = x
        self.y = y
        self.velocity = velocity  # m/s
        self.lane = lane
        self.width = 4.5  # meters
        self.length = 5.0  # meters

    def update(self, dt: float):
        """위치 업데이트"""
        self.y += self.velocity * dt


class Pedestrian:
    """보행자"""
    def __init__(self, x: float, y: float, velocity: float, direction: int):
        self.x = x
        self.y = y
        self.velocity = velocity
        self.direction = direction  # -1: left, 1: right
        self.width = 0.6
        self.height = 1.8

    def update(self, dt: float):
        """위치 업데이트"""
        self.x += self.velocity * self.direction * dt


class TrafficLight:
    """신호등"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.state = 'green'  # red, yellow, green
        self.timer = 0
        self.green_duration = 10.0
        self.yellow_duration = 3.0
        self.red_duration = 10.0

    def update(self, dt: float):
        """신호 변경"""
        self.timer += dt

        if self.state == 'green' and self.timer > self.green_duration:
            self.state = 'yellow'
            self.timer = 0
        elif self.state == 'yellow' and self.timer > self.yellow_duration:
            self.state = 'red'
            self.timer = 0
        elif self.state == 'red' and self.timer > self.red_duration:
            self.state = 'green'
            self.timer = 0


class DrivingSimulator:
    """
    2D 주행 시뮬레이터

    Features:
    - 3차선 도로
    - 차량, 보행자, 신호등 시뮬레이션
    - 랜덤 시나리오 생성
    - 이미지 렌더링 (640x480)
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height

        # World space (meters)
        self.world_width = 30.0  # meters (3 lanes * 10m each)
        self.world_height = 100.0  # meters

        # Ego vehicle (플레이어 차량)
        self.ego_x = self.world_width / 2  # 중앙 차선
        self.ego_y = 20.0  # 하단에서 20m
        self.ego_velocity = 15.0  # m/s (54 km/h)
        self.ego_lane = 1  # 0, 1, 2 (left, center, right)

        # Lane info
        self.lane_width = 10.0  # meters
        self.num_lanes = 3

        # Objects
        self.vehicles: List[Vehicle] = []
        self.pedestrians: List[Pedestrian] = []
        self.traffic_lights: List[TrafficLight] = []

        # Spawn timers
        self.vehicle_spawn_timer = 0
        self.pedestrian_spawn_timer = 0

        # Camera offset (y coordinate in world space)
        self.camera_y = self.ego_y

        # Statistics
        self.time_elapsed = 0
        self.distance_traveled = 0

    def reset(self):
        """시뮬레이션 리셋"""
        self.vehicles.clear()
        self.pedestrians.clear()
        self.traffic_lights.clear()

        self.ego_x = self.world_width / 2
        self.ego_y = 20.0
        self.ego_velocity = 15.0
        self.ego_lane = 1

        self.camera_y = self.ego_y
        self.time_elapsed = 0
        self.distance_traveled = 0

        # 초기 신호등 배치
        self._spawn_traffic_light()

    def _spawn_vehicle(self):
        """랜덤 차량 생성"""
        # 랜덤 차선
        lane = random.randint(0, self.num_lanes - 1)
        x = (lane + 0.5) * self.lane_width

        # 카메라 앞쪽에 생성
        y = self.camera_y + random.uniform(40, 60)

        # 랜덤 속도
        velocity = random.uniform(10, 20)

        vehicle = Vehicle(x, y, velocity, lane)
        self.vehicles.append(vehicle)

    def _spawn_pedestrian(self):
        """랜덤 보행자 생성"""
        # 도로 가장자리에서 횡단
        side = random.choice(['left', 'right'])

        if side == 'left':
            x = 0
            direction = 1
        else:
            x = self.world_width
            direction = -1

        y = self.camera_y + random.uniform(10, 40)
        velocity = random.uniform(1.0, 2.0)

        pedestrian = Pedestrian(x, y, velocity, direction)
        self.pedestrians.append(pedestrian)

    def _spawn_traffic_light(self):
        """신호등 생성"""
        x = self.world_width / 2
        y = self.camera_y + 50

        traffic_light = TrafficLight(x, y)
        self.traffic_lights.append(traffic_light)

    def step(self, steering: float, throttle: float, brake: float, dt: float = 0.033):
        """
        시뮬레이션 1 스텝

        Args:
            steering: -1 ~ 1 (left ~ right)
            throttle: 0 ~ 1
            brake: 0 ~ 1
            dt: delta time (seconds)

        Returns:
            (frame, detections, lane_info, traffic_lights, reward, done)
        """
        # Update ego vehicle
        # Steering affects lane position
        self.ego_x += steering * 5.0 * dt  # 5 m/s lateral speed
        self.ego_x = np.clip(self.ego_x, 0, self.world_width)

        # Throttle/brake affects velocity
        if brake > 0.1:
            self.ego_velocity = max(0, self.ego_velocity - 10.0 * brake * dt)
        else:
            self.ego_velocity = min(25.0, self.ego_velocity + 5.0 * throttle * dt)

        # Update ego position
        self.ego_y += self.ego_velocity * dt
        self.camera_y = self.ego_y

        # Update objects
        for vehicle in self.vehicles:
            vehicle.update(dt)

        for pedestrian in self.pedestrians:
            pedestrian.update(dt)

        for tl in self.traffic_lights:
            tl.update(dt)

        # Remove off-screen objects
        self.vehicles = [v for v in self.vehicles if abs(v.y - self.camera_y) < 70]
        self.pedestrians = [p for p in self.pedestrians if 0 <= p.x <= self.world_width]
        self.traffic_lights = [tl for tl in self.traffic_lights if abs(tl.y - self.camera_y) < 70]

        # Spawn new objects
        self.vehicle_spawn_timer += dt
        if self.vehicle_spawn_timer > random.uniform(1.0, 3.0):
            self._spawn_vehicle()
            self.vehicle_spawn_timer = 0

        self.pedestrian_spawn_timer += dt
        if self.pedestrian_spawn_timer > random.uniform(5.0, 10.0):
            self._spawn_pedestrian()
            self.pedestrian_spawn_timer = 0

        # Spawn traffic light ahead
        if not self.traffic_lights:
            self._spawn_traffic_light()
        elif self.traffic_lights[-1].y < self.camera_y + 40:
            self._spawn_traffic_light()

        # Statistics
        self.time_elapsed += dt
        self.distance_traveled += self.ego_velocity * dt

        # Render
        frame = self._render()

        # Get detections
        detections = self._get_detections()
        lane_info = self._get_lane_info()
        traffic_lights = self._get_traffic_lights()

        # Calculate reward
        reward, done = self._calculate_reward()

        return frame, detections, lane_info, traffic_lights, reward, done

    def _render(self) -> np.ndarray:
        """렌더링"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Background (dark gray)
        frame[:] = (40, 40, 40)

        # Road (gray)
        road_y_start = 0
        road_y_end = self.height
        frame[road_y_start:road_y_end, :] = (60, 60, 60)

        # Lane markings
        for i in range(1, self.num_lanes):
            x_world = i * self.lane_width
            x_pixel = self._world_to_pixel_x(x_world)

            # Dashed lines
            for y in range(0, self.height, 40):
                cv2.line(frame, (x_pixel, y), (x_pixel, y + 20), (255, 255, 255), 2)

        # Draw vehicles
        for vehicle in self.vehicles:
            self._draw_vehicle(frame, vehicle)

        # Draw pedestrians
        for pedestrian in self.pedestrians:
            self._draw_pedestrian(frame, pedestrian)

        # Draw traffic lights
        for tl in self.traffic_lights:
            self._draw_traffic_light(frame, tl)

        # Draw ego vehicle (bottom center)
        ego_x_pixel = self._world_to_pixel_x(self.ego_x)
        ego_y_pixel = int(self.height * 0.75)

        # Car body
        car_width = int(self.lane_width * 0.6 * self.width / self.world_width)
        car_height = int(5.0 * self.height / 60)

        cv2.rectangle(frame,
                     (ego_x_pixel - car_width // 2, ego_y_pixel - car_height // 2),
                     (ego_x_pixel + car_width // 2, ego_y_pixel + car_height // 2),
                     (0, 200, 255), -1)

        return frame

    def _draw_vehicle(self, frame: np.ndarray, vehicle: Vehicle):
        """차량 그리기"""
        x_pixel = self._world_to_pixel_x(vehicle.x)
        y_pixel = self._world_to_pixel_y(vehicle.y)

        if 0 <= y_pixel < self.height:
            car_width = int(vehicle.width * self.width / self.world_width)
            car_height = int(vehicle.length * self.height / 60)

            cv2.rectangle(frame,
                         (x_pixel - car_width // 2, y_pixel - car_height // 2),
                         (x_pixel + car_width // 2, y_pixel + car_height // 2),
                         (100, 100, 200), -1)

    def _draw_pedestrian(self, frame: np.ndarray, pedestrian: Pedestrian):
        """보행자 그리기"""
        x_pixel = self._world_to_pixel_x(pedestrian.x)
        y_pixel = self._world_to_pixel_y(pedestrian.y)

        if 0 <= y_pixel < self.height:
            size = int(pedestrian.width * self.width / self.world_width)
            cv2.circle(frame, (x_pixel, y_pixel), size * 2, (255, 150, 50), -1)

    def _draw_traffic_light(self, frame: np.ndarray, tl: TrafficLight):
        """신호등 그리기"""
        x_pixel = self._world_to_pixel_x(tl.x)
        y_pixel = self._world_to_pixel_y(tl.y)

        if 0 <= y_pixel < self.height:
            color_map = {
                'red': (0, 0, 255),
                'yellow': (0, 255, 255),
                'green': (0, 255, 0)
            }
            color = color_map[tl.state]
            cv2.circle(frame, (x_pixel, y_pixel), 10, color, -1)
            cv2.circle(frame, (x_pixel, y_pixel), 10, (255, 255, 255), 2)

    def _world_to_pixel_x(self, x_world: float) -> int:
        """World X → Pixel X"""
        return int(x_world * self.width / self.world_width)

    def _world_to_pixel_y(self, y_world: float) -> int:
        """World Y → Pixel Y"""
        # Camera-relative
        y_relative = y_world - self.camera_y
        # Map to screen space (camera_y + 30 at top, camera_y - 30 at bottom)
        y_normalized = (30 - y_relative) / 60
        return int(y_normalized * self.height)

    def _get_detections(self) -> List[Dict]:
        """YOLO 형식 감지 정보"""
        detections = []

        for vehicle in self.vehicles:
            x_pixel = self._world_to_pixel_x(vehicle.x)
            y_pixel = self._world_to_pixel_y(vehicle.y)

            if 0 <= y_pixel < self.height:
                # Distance estimation
                distance = abs(vehicle.y - self.ego_y)

                detections.append({
                    'class': 'car',
                    'bbox': (x_pixel - 20, y_pixel - 15, 40, 30),
                    'confidence': 1.0,
                    'distance': distance,
                    'center': (x_pixel, y_pixel)
                })

        for pedestrian in self.pedestrians:
            x_pixel = self._world_to_pixel_x(pedestrian.x)
            y_pixel = self._world_to_pixel_y(pedestrian.y)

            if 0 <= y_pixel < self.height:
                distance = abs(pedestrian.y - self.ego_y)

                detections.append({
                    'class': 'person',
                    'bbox': (x_pixel - 10, y_pixel - 15, 20, 30),
                    'confidence': 1.0,
                    'distance': distance,
                    'center': (x_pixel, y_pixel)
                })

        return detections

    def _get_lane_info(self) -> Dict:
        """차선 정보"""
        # Determine current lane
        current_lane = int(self.ego_x / self.lane_width)
        current_lane = np.clip(current_lane, 0, self.num_lanes - 1)

        # Lane center
        lane_center = (current_lane + 0.5) * self.lane_width

        # Deviation (percentage)
        deviation = ((self.ego_x - lane_center) / (self.lane_width / 2)) * 100

        return {
            'left_lane': current_lane > 0,
            'right_lane': current_lane < self.num_lanes - 1,
            'lane_width': self.lane_width,
            'deviation': deviation
        }

    def _get_traffic_lights(self) -> List[Dict]:
        """신호등 정보"""
        traffic_lights = []

        for tl in self.traffic_lights:
            if tl.y > self.ego_y:  # 앞쪽만
                distance = tl.y - self.ego_y

                traffic_lights.append({
                    'state': tl.state,
                    'distance': distance
                })

        return traffic_lights

    def _calculate_reward(self) -> Tuple[float, bool]:
        """보상 계산"""
        reward = 0.1  # Base reward for staying alive
        done = False

        # Lane keeping
        lane_info = self._get_lane_info()
        deviation = abs(lane_info['deviation'])

        if deviation < 20:
            reward += 0.5
        elif deviation > 80:
            reward -= 1.0
            done = True  # Off road

        # Collision detection
        for vehicle in self.vehicles:
            dist_x = abs(vehicle.x - self.ego_x)
            dist_y = abs(vehicle.y - self.ego_y)

            if dist_x < 3 and dist_y < 4:
                reward -= 10.0
                done = True

        for pedestrian in self.pedestrians:
            dist_x = abs(pedestrian.x - self.ego_x)
            dist_y = abs(pedestrian.y - self.ego_y)

            if dist_x < 2 and dist_y < 3:
                reward -= 20.0
                done = True

        # Speed reward
        if 10 < self.ego_velocity < 20:
            reward += 0.3

        return reward, done


if __name__ == "__main__":
    # 시뮬레이터 테스트
    sim = DrivingSimulator()
    sim.reset()

    print("=== Driving Simulator Test ===")
    print("Running 100 steps...")

    total_reward = 0

    for i in range(100):
        # Random actions
        steering = random.uniform(-0.2, 0.2)
        throttle = random.uniform(0.3, 0.7)
        brake = 0.0

        frame, detections, lane_info, traffic_lights, reward, done = sim.step(
            steering, throttle, brake
        )

        total_reward += reward

        if i % 20 == 0:
            print(f"\nStep {i}:")
            print(f"  Ego: x={sim.ego_x:.1f}m, y={sim.ego_y:.1f}m, v={sim.ego_velocity:.1f}m/s")
            print(f"  Detections: {len(detections)} objects")
            print(f"  Lane deviation: {lane_info['deviation']:.1f}%")
            print(f"  Reward: {reward:.2f}")

        if done:
            print(f"\n❌ Episode ended at step {i}")
            break

    print(f"\n✅ Total reward: {total_reward:.2f}")
    print(f"Distance traveled: {sim.distance_traveled:.1f}m")
    print(f"Time elapsed: {sim.time_elapsed:.1f}s")
