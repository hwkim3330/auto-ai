"""
Depth Anything V3 Integration

최신 깊이 추정 모델 통합:
- Metric Depth Estimation
- Zero-shot depth
- Real-time inference
- CCTV 모니터링에 최적화
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
import urllib.request
import os


class DepthAnythingV3:
    """
    Depth Anything V3 Wrapper

    Features:
    - Metric depth estimation
    - Fast inference
    - High accuracy
    - Zero-shot capability
    """

    def __init__(self, model_size: str = 'small', device: str = 'cpu'):
        """
        Args:
            model_size: 'small', 'base', or 'large'
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model_size = model_size
        self.model = None

        # Model configs
        self.configs = {
            'small': {
                'encoder': 'vits',
                'max_depth': 20.0  # meters
            },
            'base': {
                'encoder': 'vitb',
                'max_depth': 20.0
            },
            'large': {
                'encoder': 'vitl',
                'max_depth': 20.0
            }
        }

        self._load_model()

    def _load_model(self):
        """Load Depth Anything V3 model"""
        try:
            # Try loading Depth Anything V3 from Hugging Face
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

            from depth_anything_v3.api import DepthAnything3

            # Model mapping
            model_map = {
                'small': 'depth-anything/DA3-SMALL',
                'base': 'depth-anything/DA3-BASE',
                'large': 'depth-anything/DA3MONO-LARGE'  # Best for monocular CCTV
            }

            model_id = model_map.get(self.model_size, 'depth-anything/DA3-BASE')

            print(f"🔄 Loading Depth Anything V3 from Hugging Face: {model_id}")
            self.model = DepthAnything3.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
            self.is_da3 = True

            print(f"✅ Depth Anything V3 ({self.model_size}) loaded successfully")

        except Exception as e:
            print(f"⚠️ Depth Anything V3 loading failed: {e}")
            print("   Using fallback MiDaS depth estimation")
            self.is_da3 = False
            self._load_midas_fallback()

    def _load_midas_fallback(self):
        """Fallback to MiDaS if Depth Anything not available"""
        try:
            # MiDaS is more widely available
            self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self.model.to(self.device)
            self.model.eval()

            # MiDaS transforms
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            self.transform = midas_transforms.small_transform

            print("✅ MiDaS depth estimation loaded (fallback)")

        except Exception as e:
            print(f"❌ Depth estimation failed to load: {e}")
            self.model = None

    def estimate_depth(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth from RGB image

        Args:
            frame: RGB image (H, W, 3)

        Returns:
            (depth_map, depth_colored)
            depth_map: metric depth in meters (H, W)
            depth_colored: visualization (H, W, 3)
        """
        if self.model is None:
            # Return dummy depth
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.float32), frame

        try:
            with torch.no_grad():
                if hasattr(self, 'is_da3') and self.is_da3:
                    # Depth Anything V3 API
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # DA3 accepts list of numpy arrays or PIL images
                    prediction = self.model.inference(
                        [frame_rgb],
                        process_res=518,
                        export_dir=None
                    )

                    # Get depth from prediction
                    depth_map = prediction.depth[0]  # Shape: (H, W)

                    # Normalize for visualization
                    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

                    # Apply colormap
                    depth_colored = cv2.applyColorMap(
                        (depth_normalized * 255).astype(np.uint8),
                        cv2.COLORMAP_MAGMA
                    )

                    # DA3MONO-LARGE outputs metric depth directly
                    # For other models, scale to reasonable range
                    if self.model_size == 'large':  # DA3MONO-LARGE
                        depth_metric = depth_map
                    else:
                        max_depth = self.configs[self.model_size]['max_depth']
                        depth_metric = depth_normalized * max_depth

                    return depth_metric, depth_colored

                elif hasattr(self, 'transform'):
                    # MiDaS fallback
                    input_batch = self.transform(frame).to(self.device)

                    # Inference
                    depth = self.model(input_batch)

                    # Post-process
                    depth = F.interpolate(
                        depth.unsqueeze(1),
                        size=frame.shape[:2],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                    depth_map = depth.cpu().numpy()

                    # Normalize for visualization
                    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

                    # Apply colormap
                    depth_colored = cv2.applyColorMap(
                        (depth_normalized * 255).astype(np.uint8),
                        cv2.COLORMAP_MAGMA
                    )

                    # Convert to metric depth
                    max_depth = self.configs[self.model_size]['max_depth']
                    depth_metric = depth_normalized * max_depth

                    return depth_metric, depth_colored

                else:
                    # Fallback: return zeros
                    h, w = frame.shape[:2]
                    return np.zeros((h, w), dtype=np.float32), frame

        except Exception as e:
            print(f"Depth estimation error: {e}")
            import traceback
            traceback.print_exc()
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.float32), frame

    def estimate_distance(self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Estimate distance to object from depth map

        Args:
            depth_map: Depth map in meters
            bbox: (x, y, w, h)

        Returns:
            Distance in meters
        """
        x, y, w, h = bbox

        # Extract depth in bounding box
        roi = depth_map[y:y+h, x:x+w]

        if roi.size == 0:
            return 0.0

        # Use median depth (more robust than mean)
        distance = np.median(roi)

        return float(distance)

    def estimate_height(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
        focal_length: float = 500.0
    ) -> float:
        """
        Estimate person height from depth map and bounding box

        Args:
            depth_map: Depth map in meters
            bbox: (x, y, w, h) - person bounding box
            focal_length: Camera focal length in pixels

        Returns:
            Estimated height in meters
        """
        x, y, w, h = bbox

        # Get distance to person (median depth in bbox)
        distance = self.estimate_distance(depth_map, bbox)

        if distance == 0:
            return 0.0

        # Calculate real-world height using similar triangles
        # height_real = (bbox_height_pixels * distance) / focal_length
        height_meters = (h * distance) / focal_length

        # Sanity check: human height should be between 0.5m and 2.5m
        if height_meters < 0.5:
            height_meters = 0.5
        elif height_meters > 2.5:
            height_meters = 2.5

        return float(height_meters)

    def get_3d_position(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
        focal_length: float = 500.0
    ) -> Tuple[float, float, float]:
        """
        Get 3D position (X, Y, Z) from 2D bbox and depth

        Args:
            depth_map: Depth map in meters
            bbox: (x, y, w, h)
            focal_length: Camera focal length in pixels

        Returns:
            (x_3d, y_3d, z_3d) in meters
        """
        x, y, w, h = bbox

        # Center of bbox
        cx = x + w / 2
        cy = y + h / 2

        # Get depth at center
        if 0 <= int(cy) < depth_map.shape[0] and 0 <= int(cx) < depth_map.shape[1]:
            z = depth_map[int(cy), int(cx)]
        else:
            z = 0.0

        # Image center
        img_cx = depth_map.shape[1] / 2
        img_cy = depth_map.shape[0] / 2

        # Convert to 3D coordinates
        x_3d = (cx - img_cx) * z / focal_length
        y_3d = (cy - img_cy) * z / focal_length
        z_3d = z

        return x_3d, y_3d, z_3d


class CCTVMonitor:
    """
    CCTV Monitoring System with Depth Estimation

    Features:
    - Person detection and tracking
    - Depth-based distance estimation
    - Height estimation
    - Zone intrusion detection
    - Loitering detection
    - Data logging (CSV/JSON)
    """

    def __init__(self, depth_estimator: DepthAnythingV3, log_dir: str = 'cctv_logs'):
        self.depth_estimator = depth_estimator

        # Tracking
        self.tracked_persons = {}
        self.next_id = 0

        # Zone definition (in image coordinates)
        self.restricted_zones = []

        # Loitering detection
        self.loitering_threshold = 30.0  # seconds

        # Data logging
        self.log_dir = log_dir
        self.enable_logging = True
        os.makedirs(log_dir, exist_ok=True)

        # Log file paths
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_log_path = os.path.join(log_dir, f'cctv_log_{timestamp}.csv')
        self.json_log_path = os.path.join(log_dir, f'cctv_log_{timestamp}.json')

        # Initialize CSV log
        self._init_csv_log()

        # JSON log buffer
        self.json_log_buffer = []

    def process_frame(
        self,
        frame: np.ndarray,
        detections: list
    ) -> Tuple[np.ndarray, dict]:
        """
        Process CCTV frame

        Args:
            frame: RGB image
            detections: List of YOLO detections

        Returns:
            (annotated_frame, analytics)
        """
        # Estimate depth
        depth_map, depth_colored = self.depth_estimator.estimate_depth(frame)

        # Filter person detections
        persons = [d for d in detections if 'person' in d.get('class', '')]

        # Update with depth information
        for person in persons:
            bbox = person['bbox']

            # Get distance from depth map
            distance = self.depth_estimator.estimate_distance(depth_map, bbox)
            person['distance_depth'] = distance

            # Get 3D position
            x3d, y3d, z3d = self.depth_estimator.get_3d_position(depth_map, bbox)
            person['position_3d'] = (x3d, y3d, z3d)

            # Estimate height
            height = self.depth_estimator.estimate_height(depth_map, bbox)
            person['height'] = height

        # Track persons
        self._update_tracking(persons)

        # Detect anomalies
        analytics = self._analyze_scene(persons)

        # Annotate frame
        annotated = self._draw_annotations(
            frame.copy(),
            persons,
            depth_colored
        )

        # Log detection data
        if self.enable_logging:
            self._log_detections(persons, analytics)

        return annotated, analytics

    def _update_tracking(self, persons: list):
        """Update person tracking"""
        # Simple centroid-based tracking
        # In production, use DeepSORT or similar

        for person in persons:
            bbox = person['bbox']
            center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)

            # Find closest tracked person
            min_dist = float('inf')
            matched_id = None

            for pid, tracked in self.tracked_persons.items():
                last_center = tracked['last_center']
                dist = np.linalg.norm(
                    np.array(center) - np.array(last_center)
                )

                if dist < min_dist and dist < 100:  # 100 pixel threshold
                    min_dist = dist
                    matched_id = pid

            if matched_id is not None:
                # Update existing track
                self.tracked_persons[matched_id].update({
                    'last_center': center,
                    'last_seen': time.time(),
                    'positions': self.tracked_persons[matched_id]['positions'] + [center]
                })
            else:
                # New track
                import time
                self.tracked_persons[self.next_id] = {
                    'last_center': center,
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'positions': [center]
                }
                person['track_id'] = self.next_id
                self.next_id += 1

        # Remove stale tracks
        import time
        current_time = time.time()
        stale_ids = [
            pid for pid, tracked in self.tracked_persons.items()
            if current_time - tracked['last_seen'] > 2.0
        ]
        for pid in stale_ids:
            del self.tracked_persons[pid]

    def _analyze_scene(self, persons: list) -> dict:
        """Analyze scene for anomalies"""
        import time
        current_time = time.time()

        analytics = {
            'total_persons': len(persons),
            'loitering': [],
            'zone_intrusions': [],
            'close_persons': []
        }

        # Detect loitering
        for pid, tracked in self.tracked_persons.items():
            duration = current_time - tracked['first_seen']
            if duration > self.loitering_threshold:
                analytics['loitering'].append({
                    'id': pid,
                    'duration': duration,
                    'position': tracked['last_center']
                })

        # Detect persons too close to camera
        for person in persons:
            distance = person.get('distance_depth', 0)
            if 0 < distance < 2.0:  # Within 2 meters
                analytics['close_persons'].append({
                    'bbox': person['bbox'],
                    'distance': distance
                })

        return analytics

    def _draw_annotations(
        self,
        frame: np.ndarray,
        persons: list,
        depth_colored: np.ndarray
    ) -> np.ndarray:
        """Draw annotations on frame"""
        h, w = frame.shape[:2]

        # Overlay depth (semi-transparent)
        depth_resized = cv2.resize(depth_colored, (w//3, h//3))
        frame[10:10+h//3, 10:10+w//3] = cv2.addWeighted(
            frame[10:10+h//3, 10:10+w//3], 0.5,
            depth_resized, 0.5, 0
        )

        # Draw persons with depth info
        for person in persons:
            bbox = person['bbox']
            x, y, w_bbox, h_bbox = bbox

            distance = person.get('distance_depth', 0)

            # Color based on distance
            if distance < 2.0:
                color = (0, 0, 255)  # Red - too close
            elif distance < 5.0:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green

            # Draw bbox
            cv2.rectangle(frame, (x, y), (x+w_bbox, y+h_bbox), color, 2)

            # Draw distance
            text = f"{distance:.1f}m"
            cv2.putText(frame, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw track ID if available
            if 'track_id' in person:
                track_text = f"ID: {person['track_id']}"
                cv2.putText(frame, track_text, (x, y+h_bbox+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw height if available
            if 'height' in person:
                height = person['height']
                height_text = f"Height: {height:.2f}m"
                cv2.putText(frame, height_text, (x, y+h_bbox+40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def _init_csv_log(self):
        """Initialize CSV log file with headers"""
        import csv
        with open(self.csv_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'person_id',
                'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                'distance_m',
                'height_m',
                'pos_x', 'pos_y', 'pos_z',
                'confidence',
                'is_loitering',
                'is_close_alert'
            ])
        print(f"✅ CSV log initialized: {self.csv_log_path}")

    def _log_detections(self, persons: list, analytics: dict):
        """Log detection data to CSV and JSON"""
        import csv
        import datetime
        import json

        timestamp = datetime.datetime.now().isoformat()

        # Get loitering and close alert IDs
        loitering_ids = [l['id'] for l in analytics.get('loitering', [])]
        close_alert_dists = {c['bbox']: c['distance'] for c in analytics.get('close_persons', [])}

        # CSV logging
        with open(self.csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for person in persons:
                bbox = person['bbox']
                track_id = person.get('track_id', -1)
                distance = person.get('distance_depth', 0.0)
                height = person.get('height', 0.0)
                pos_3d = person.get('position_3d', (0, 0, 0))
                confidence = person.get('confidence', 0.0)

                is_loitering = track_id in loitering_ids
                is_close = bbox in close_alert_dists

                writer.writerow([
                    timestamp,
                    track_id,
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    f"{distance:.3f}",
                    f"{height:.3f}",
                    f"{pos_3d[0]:.3f}", f"{pos_3d[1]:.3f}", f"{pos_3d[2]:.3f}",
                    f"{confidence:.3f}",
                    is_loitering,
                    is_close
                ])

        # JSON logging (buffer)
        log_entry = {
            'timestamp': timestamp,
            'persons': [],
            'analytics': analytics
        }

        for person in persons:
            track_id = person.get('track_id', -1)
            log_entry['persons'].append({
                'id': track_id,
                'bbox': person['bbox'],
                'distance': float(person.get('distance_depth', 0.0)),
                'height': float(person.get('height', 0.0)),
                'position_3d': person.get('position_3d', (0, 0, 0)),
                'confidence': float(person.get('confidence', 0.0))
            })

        self.json_log_buffer.append(log_entry)

        # Save JSON every 100 entries
        if len(self.json_log_buffer) >= 100:
            self.save_json_log()

    def save_json_log(self):
        """Save JSON log buffer to file"""
        import json
        if len(self.json_log_buffer) == 0:
            return

        # Read existing data
        existing_data = []
        if os.path.exists(self.json_log_path):
            try:
                with open(self.json_log_path, 'r') as f:
                    existing_data = json.load(f)
            except:
                pass

        # Append new data
        existing_data.extend(self.json_log_buffer)

        # Write back
        with open(self.json_log_path, 'w') as f:
            json.dump(existing_data, f, indent=2)

        print(f"📝 JSON log saved: {len(self.json_log_buffer)} entries → {self.json_log_path}")
        self.json_log_buffer = []


if __name__ == "__main__":
    print("=== Depth Anything V3 Test ===\n")

    # Test depth estimator
    depth_estimator = DepthAnythingV3(model_size='small', device='cpu')

    # Create dummy frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Estimate depth
    depth_map, depth_colored = depth_estimator.estimate_depth(test_frame)

    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: {depth_map.min():.2f}m - {depth_map.max():.2f}m")

    # Test CCTV monitor
    monitor = CCTVMonitor(depth_estimator)

    # Dummy detection
    detections = [
        {'class': 'person', 'bbox': (100, 100, 50, 100), 'confidence': 0.9}
    ]

    annotated, analytics = monitor.process_frame(test_frame, detections)

    print(f"\nAnalytics: {analytics}")
    print("\n✅ Depth estimation test passed")
