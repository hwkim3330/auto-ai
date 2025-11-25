#!/usr/bin/env python3
"""
Real-Time CCTV Person & Vehicle Tracker
========================================

Integrates with Korea Public Data Portal CCTV APIs
Tracks people and vehicles on map with movement prediction

ETHICS & LEGAL:
- NO facial recognition (개인정보보호법 준수)
- Anonymous tracking only (익명화된 추적)
- Educational/Research purpose (교육/연구 목적)
- Traffic monitoring only (교통 모니터링 용도)

Architecture:
    Korean CCTV API → Object Detection (YOLO) →
    Multi-Object Tracking (DeepSORT) →
    Movement Prediction (Kalman Filter) →
    Real-time Map Visualization
"""

import cv2
import numpy as np
import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import queue


@dataclass
class Detection:
    """Single detection from object detector"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


@dataclass
class TrackedObject:
    """Tracked object with history"""
    track_id: int
    class_name: str
    positions: deque  # Recent positions (x, y)
    timestamps: deque
    velocity: Tuple[float, float]  # (vx, vy)
    predicted_position: Optional[Tuple[float, float]]
    confidence: float
    camera_id: str
    geo_location: Optional[Tuple[float, float]]  # (lat, lon)


@dataclass
class CCTVCamera:
    """CCTV Camera metadata"""
    id: str
    name: str
    location: str
    latitude: float
    longitude: float
    stream_url: str
    active: bool = True
    last_frame_time: Optional[float] = None


class SimpleObjectDetector:
    """
    Simple object detector using background subtraction

    In production: Use YOLOv8, DETR, or similar
    For MVP: Background subtraction for motion detection
    """

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        print("[Detector] Initialized background subtractor")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect moving objects in frame

        Returns: List of Detection objects
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area (person: 500-5000, vehicle: 1000-20000)
            if area < 500:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Classify based on aspect ratio and size
            aspect_ratio = h / max(w, 1)

            if aspect_ratio > 1.5 and 500 < area < 5000:
                class_name = "person"
                class_id = 0
            elif area > 1000:
                class_name = "vehicle"
                class_id = 1
            else:
                continue

            detections.append(Detection(
                bbox=(x, y, x + w, y + h),
                confidence=min(area / 10000, 1.0),
                class_id=class_id,
                class_name=class_name
            ))

        return detections


class SimpleTracker:
    """
    Simple multi-object tracker using IoU matching

    In production: Use DeepSORT, ByteTrack, or SORT
    For MVP: IoU-based tracking with Kalman filter prediction
    """

    def __init__(self, max_history: int = 30):
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 1
        self.max_history = max_history
        self.max_age = 30  # frames
        self.min_hits = 3
        print("[Tracker] Initialized")

    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update(
        self,
        detections: List[Detection],
        camera_id: str,
        camera_location: Tuple[float, float]
    ) -> List[TrackedObject]:
        """
        Update tracks with new detections

        Returns: List of active TrackedObject instances
        """
        current_time = time.time()

        # Match detections to existing tracks using IoU
        matched_tracks = set()
        matched_detections = set()

        for detection_idx, detection in enumerate(detections):
            best_iou = 0.3  # Minimum IoU threshold
            best_track_id = None

            # Get detection center
            x1, y1, x2, y2 = detection.bbox
            det_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue

                # Get last known position
                if len(track.positions) == 0:
                    continue

                last_pos = track.positions[-1]

                # Predict current position using velocity
                predicted_x = last_pos[0] + track.velocity[0]
                predicted_y = last_pos[1] + track.velocity[1]

                # Compute distance (simple matching)
                distance = np.sqrt(
                    (det_center[0] - predicted_x) ** 2 +
                    (det_center[1] - predicted_y) ** 2
                )

                # Use inverse distance as "IoU"
                iou = 1.0 / (1.0 + distance / 100.0)

                if iou > best_iou and track.class_name == detection.class_name:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                track = self.tracks[best_track_id]
                track.positions.append(det_center)
                track.timestamps.append(current_time)
                track.confidence = 0.9 * track.confidence + 0.1 * detection.confidence

                # Update velocity
                if len(track.positions) >= 2:
                    dt = track.timestamps[-1] - track.timestamps[-2]
                    if dt > 0:
                        vx = (track.positions[-1][0] - track.positions[-2][0]) / dt
                        vy = (track.positions[-1][1] - track.positions[-2][1]) / dt
                        track.velocity = (vx, vy)

                # Predict next position
                track.predicted_position = (
                    det_center[0] + track.velocity[0] * 0.5,  # 0.5s ahead
                    det_center[1] + track.velocity[1] * 0.5
                )

                matched_tracks.add(best_track_id)
                matched_detections.add(detection_idx)
            else:
                # Create new track
                track = TrackedObject(
                    track_id=self.next_track_id,
                    class_name=detection.class_name,
                    positions=deque([det_center], maxlen=self.max_history),
                    timestamps=deque([current_time], maxlen=self.max_history),
                    velocity=(0.0, 0.0),
                    predicted_position=None,
                    confidence=detection.confidence,
                    camera_id=camera_id,
                    geo_location=camera_location
                )
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
                matched_detections.add(detection_idx)

        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                # Check age
                if len(track.timestamps) > 0:
                    age = current_time - track.timestamps[-1]
                    if age > self.max_age / 30.0:  # Assume 30 fps
                        tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        return list(self.tracks.values())


class KoreaCCTVIntegration:
    """
    Integration with Korea Public Data Portal CCTV APIs

    APIs available:
    - 서울시 CCTV 위치 정보
    - 국토교통부 ITS CCTV
    - 도로교통공단 CCTV
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "YOUR_API_KEY_HERE"
        self.cameras: List[CCTVCamera] = []
        print("[CCTV API] Initialized")

    def load_seoul_cctv_locations(self) -> List[CCTVCamera]:
        """
        Load Seoul CCTV locations from public API

        API: 서울 열린데이터 광장 - CCTV 설치 현황
        URL: https://data.seoul.go.kr/
        """
        # For MVP: Use sample locations
        # In production: Call actual API

        sample_cameras = [
            CCTVCamera(
                id="SEOUL_001",
                name="강남역 사거리",
                location="서울특별시 강남구 강남대로 지하396",
                latitude=37.4979,
                longitude=127.0276,
                stream_url="rtsp://sample.com/gangnam",
            ),
            CCTVCamera(
                id="SEOUL_002",
                name="광화문 광장",
                location="서울특별시 종로구 세종대로 172",
                latitude=37.5720,
                longitude=126.9769,
                stream_url="rtsp://sample.com/gwanghwamun",
            ),
            CCTVCamera(
                id="SEOUL_003",
                name="홍대입구역",
                location="서울특별시 마포구 양화로 188",
                latitude=37.5566,
                longitude=126.9236,
                stream_url="rtsp://sample.com/hongdae",
            ),
            CCTVCamera(
                id="SEOUL_004",
                name="서울역",
                location="서울특별시 용산구 한강대로 405",
                latitude=37.5547,
                longitude=126.9707,
                stream_url="rtsp://sample.com/seoul_station",
            ),
        ]

        self.cameras = sample_cameras
        print(f"[CCTV API] Loaded {len(self.cameras)} cameras")
        return self.cameras

    def get_stream(self, camera_id: str) -> Optional[cv2.VideoCapture]:
        """
        Get video stream from camera

        For MVP: Return simulated stream
        In production: Return actual RTSP stream
        """
        camera = next((c for c in self.cameras if c.id == camera_id), None)
        if camera is None:
            return None

        # For MVP: Use webcam or video file for testing
        # In production: Use actual RTSP URL
        # cap = cv2.VideoCapture(camera.stream_url)

        # Simulate with test video or webcam
        cap = cv2.VideoCapture(0)  # Webcam for testing

        if not cap.isOpened():
            print(f"[CCTV] Failed to open stream: {camera_id}")
            return None

        return cap


class RealtimeTrackingSystem:
    """
    Real-time tracking system with map visualization
    """

    def __init__(self):
        self.cctv_api = KoreaCCTVIntegration()
        self.detector = SimpleObjectDetector()
        self.tracker = SimpleTracker()

        self.active_cameras: Dict[str, cv2.VideoCapture] = {}
        self.tracking_data: Dict[str, List[TrackedObject]] = {}

        self.running = False
        self.frame_queue = queue.Queue(maxsize=100)

        print("[Tracking System] Initialized")

    def start_camera(self, camera_id: str):
        """Start processing camera stream"""
        stream = self.cctv_api.get_stream(camera_id)
        if stream:
            self.active_cameras[camera_id] = stream
            print(f"[Tracking] Started camera: {camera_id}")

    def process_frame(
        self,
        frame: np.ndarray,
        camera: CCTVCamera
    ) -> Tuple[np.ndarray, List[TrackedObject]]:
        """
        Process single frame: detect → track → visualize
        """
        # Detect objects
        detections = self.detector.detect(frame)

        # Update tracker
        tracks = self.tracker.update(
            detections,
            camera.id,
            (camera.latitude, camera.longitude)
        )

        # Visualize on frame
        vis_frame = frame.copy()

        for track in tracks:
            if len(track.positions) == 0:
                continue

            # Get current position
            x, y = track.positions[-1]
            x, y = int(x), int(y)

            # Draw trajectory
            if len(track.positions) > 1:
                points = np.array(
                    [(int(p[0]), int(p[1])) for p in track.positions],
                    dtype=np.int32
                )
                cv2.polylines(vis_frame, [points], False, (0, 255, 255), 2)

            # Draw current position
            color = (0, 255, 0) if track.class_name == "person" else (255, 0, 0)
            cv2.circle(vis_frame, (x, y), 5, color, -1)

            # Draw predicted position
            if track.predicted_position:
                pred_x, pred_y = int(track.predicted_position[0]), int(track.predicted_position[1])
                cv2.circle(vis_frame, (pred_x, pred_y), 5, (0, 0, 255), 2)
                cv2.line(vis_frame, (x, y), (pred_x, pred_y), (0, 0, 255), 1)

            # Draw label
            label = f"ID{track.track_id} {track.class_name}"
            cv2.putText(
                vis_frame,
                label,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Draw info
        info = f"Camera: {camera.name} | Tracks: {len(tracks)}"
        cv2.putText(
            vis_frame,
            info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return vis_frame, tracks

    def run(self, camera_ids: Optional[List[str]] = None):
        """
        Run real-time tracking

        Args:
            camera_ids: List of camera IDs to process (None = all)
        """
        # Load cameras
        cameras = self.cctv_api.load_seoul_cctv_locations()

        if camera_ids is None:
            camera_ids = [cam.id for cam in cameras[:1]]  # Process first camera only

        # Start cameras
        for camera_id in camera_ids:
            self.start_camera(camera_id)

        if not self.active_cameras:
            print("[Tracking] No active cameras")
            return

        self.running = True
        print("[Tracking] Starting real-time processing...")
        print("Press 'q' to quit")

        try:
            while self.running:
                all_tracks = []

                for camera_id, stream in self.active_cameras.items():
                    ret, frame = stream.read()
                    if not ret:
                        continue

                    # Find camera metadata
                    camera = next((c for c in cameras if c.id == camera_id), None)
                    if camera is None:
                        continue

                    # Process frame
                    vis_frame, tracks = self.process_frame(frame, camera)
                    all_tracks.extend(tracks)

                    # Display
                    cv2.imshow(f"CCTV: {camera.name}", vis_frame)

                # Store tracking data for map visualization
                self.tracking_data = {
                    'timestamp': time.time(),
                    'tracks': all_tracks
                }

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n[Tracking] Interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Stop tracking and cleanup"""
        self.running = False

        for stream in self.active_cameras.values():
            stream.release()

        cv2.destroyAllWindows()
        print("[Tracking] Stopped")

    def export_tracks_to_json(self, output_path: str):
        """Export tracking data for map visualization"""
        if not self.tracking_data:
            return

        export_data = {
            'timestamp': self.tracking_data.get('timestamp'),
            'tracks': []
        }

        for track in self.tracking_data.get('tracks', []):
            export_data['tracks'].append({
                'track_id': track.track_id,
                'class': track.class_name,
                'position': list(track.positions[-1]) if track.positions else None,
                'trajectory': [list(p) for p in track.positions],
                'velocity': track.velocity,
                'predicted': track.predicted_position,
                'location': track.geo_location,
                'camera': track.camera_id
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"[Export] Saved {len(export_data['tracks'])} tracks to {output_path}")


def main():
    """Run real-time tracking system"""
    print("=" * 70)
    print("REAL-TIME CCTV TRACKING SYSTEM")
    print("한국 공공 CCTV 실시간 추적 시스템")
    print("=" * 70)
    print("\nETHICS & LEGAL NOTICE:")
    print("- 얼굴 인식 없음 (No facial recognition)")
    print("- 익명 추적만 (Anonymous tracking only)")
    print("- 교육/연구 목적 (Educational/Research purpose)")
    print("=" * 70)

    # Initialize system
    system = RealtimeTrackingSystem()

    # Run tracking (first camera only for demo)
    system.run(camera_ids=None)

    # Export final state
    system.export_tracks_to_json("tracking_data.json")

    print("\n" + "=" * 70)
    print("TRACKING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
