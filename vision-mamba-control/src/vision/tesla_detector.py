"""
Tesla Autopilot-style Detection System

Complete detection pipeline:
- YOLOv8 object detection (cars, pedestrians, traffic lights, etc.)
- Enhanced lane detection with polynomial fitting
- Traffic sign recognition
- Distance estimation
- Multi-object tracking
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from collections import defaultdict

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLOv8 not available - install ultralytics")


class TeslaDetector:
    """
    Tesla Autopilot-style detection system

    Features:
    - YOLOv8n for real-time object detection
    - Enhanced lane detection with curve fitting
    - Traffic light state detection
    - Distance estimation
    - Object tracking
    """

    def __init__(self, model_size: str = 'n', confidence_threshold: float = 0.5):
        """
        Args:
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Detection confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        self.yolo_model = None
        self.tracker = ObjectTracker()

        # COCO classes relevant for driving
        self.driving_classes = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            9: 'traffic light',
            10: 'fire hydrant',
            11: 'stop sign',
            13: 'bench',
        }

        # Distance estimation calibration (rough estimates)
        self.focal_length = 800  # pixels
        self.known_widths = {
            'car': 1.8,  # meters
            'bus': 2.5,
            'truck': 2.5,
            'person': 0.5,
            'bicycle': 0.6,
            'motorcycle': 0.8,
        }

        # Load YOLO model
        if YOLO_AVAILABLE:
            try:
                print(f"🔄 Loading YOLOv8{model_size}...")
                self.yolo_model = YOLO(f'yolov8{model_size}.pt')
                # Force CPU mode to avoid CUDA 6.1 incompatibility (GTX 1050 Ti)
                self.yolo_model.to('cpu')
                print(f"✅ YOLOv8{model_size} loaded successfully (CPU mode)")
            except Exception as e:
                print(f"⚠️ YOLOv8 loading failed: {e}")
                self.yolo_model = None

        # Lane detection parameters
        self.lane_history = []
        self.max_lane_history = 5

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects using YOLOv8

        Args:
            frame: BGR image

        Returns:
            List of detections with bbox, class, confidence, distance
        """
        if self.yolo_model is None:
            return []

        try:
            # Run inference
            results = self.yolo_model(frame, verbose=False)[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Filter by confidence and relevant classes
                if confidence < self.confidence_threshold:
                    continue

                if cls_id not in self.driving_classes:
                    continue

                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                class_name = self.driving_classes[cls_id]

                # Estimate distance
                distance = self._estimate_distance(class_name, w)

                detections.append({
                    'bbox': (x, y, w, h),
                    'class': class_name,
                    'confidence': confidence,
                    'distance': distance,
                    'center': (x + w // 2, y + h // 2),
                })

            # Update tracker
            tracked = self.tracker.update(detections)

            return tracked

        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def _estimate_distance(self, class_name: str, pixel_width: int) -> Optional[float]:
        """
        Estimate distance to object using pinhole camera model

        Distance = (Known_Width * Focal_Length) / Pixel_Width
        """
        if class_name not in self.known_widths:
            return None

        if pixel_width < 10:  # Too small
            return None

        known_width = self.known_widths[class_name]
        distance = (known_width * self.focal_length) / pixel_width

        return round(distance, 1)

    def detect_lanes_advanced(self, frame: np.ndarray) -> Dict:
        """
        Enhanced lane detection with polynomial fitting

        Returns:
            {
                'left_lane': np.array or None,
                'right_lane': np.array or None,
                'center_lane': np.array or None,
                'lane_width': float,
                'deviation': float  # from center
            }
        """
        height, width = frame.shape[:2]

        # Region of interest (trapezoid)
        roi_vertices = np.array([
            [(0, height),
             (width * 0.45, height * 0.6),
             (width * 0.55, height * 0.6),
             (width, height)]
        ], dtype=np.int32)

        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, roi_vertices, 255)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply mask
        masked = cv2.bitwise_and(gray, gray, mask=mask)

        # Edge detection
        blur = cv2.GaussianBlur(masked, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Hough line transform
        lines = cv2.HoughLinesP(
            edges,
            rho=2,
            theta=np.pi/180,
            threshold=50,
            minLineLength=40,
            maxLineGap=100
        )

        if lines is None:
            return {
                'left_lane': None,
                'right_lane': None,
                'center_lane': None,
                'lane_width': 0,
                'deviation': 0
            }

        # Separate left and right lanes
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Skip horizontal lines
            if abs(y2 - y1) < 10:
                continue

            # Calculate slope
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            # Left lane (negative slope)
            if slope < -0.5 and x1 < width * 0.5:
                left_lines.append(line[0])
            # Right lane (positive slope)
            elif slope > 0.5 and x1 > width * 0.5:
                right_lines.append(line[0])

        # Fit polynomials
        left_lane = self._fit_lane_polynomial(left_lines, height) if left_lines else None
        right_lane = self._fit_lane_polynomial(right_lines, height) if right_lines else None

        # Calculate lane width and deviation
        lane_width = 0
        deviation = 0

        if left_lane is not None and right_lane is not None:
            # Lane width at bottom of frame
            left_bottom = left_lane[-1]
            right_bottom = right_lane[-1]
            lane_width = abs(right_bottom[0] - left_bottom[0])

            # Deviation from center
            lane_center = (left_bottom[0] + right_bottom[0]) / 2
            frame_center = width / 2
            deviation = (lane_center - frame_center) / width * 100  # percentage

        # Smoothing with history
        self.lane_history.append({
            'left': left_lane,
            'right': right_lane,
            'width': lane_width
        })

        if len(self.lane_history) > self.max_lane_history:
            self.lane_history.pop(0)

        return {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'center_lane': self._calculate_center_lane(left_lane, right_lane),
            'lane_width': lane_width,
            'deviation': deviation
        }

    def _fit_lane_polynomial(self, lines: List, height: int) -> Optional[np.ndarray]:
        """
        Fit polynomial to lane lines
        """
        if not lines:
            return None

        # Collect all points
        points = []
        for x1, y1, x2, y2 in lines:
            points.extend([(x1, y1), (x2, y2)])

        if len(points) < 2:
            return None

        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]

        try:
            # Fit 2nd degree polynomial
            z = np.polyfit(y, x, 2)
            poly_func = np.poly1d(z)

            # Generate smooth lane line
            y_range = np.linspace(height * 0.6, height, 50)
            x_range = poly_func(y_range)

            lane_points = np.column_stack((x_range, y_range)).astype(np.int32)

            return lane_points

        except:
            return None

    def _calculate_center_lane(
        self,
        left_lane: Optional[np.ndarray],
        right_lane: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Calculate center lane from left and right
        """
        if left_lane is None or right_lane is None:
            return None

        if len(left_lane) != len(right_lane):
            return None

        center = ((left_lane + right_lane) / 2).astype(np.int32)
        return center

    def detect_traffic_lights(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Detect traffic light states (red/yellow/green)

        Args:
            frame: BGR image
            detections: Object detections from YOLO

        Returns:
            Traffic light detections with state
        """
        traffic_lights = []

        for det in detections:
            if det['class'] != 'traffic light':
                continue

            x, y, w, h = det['bbox']

            # Extract traffic light region
            tl_roi = frame[y:y+h, x:x+w]

            if tl_roi.size == 0:
                continue

            # Detect state based on color
            state = self._detect_traffic_light_state(tl_roi)

            traffic_lights.append({
                **det,
                'state': state
            })

        return traffic_lights

    def _detect_traffic_light_state(self, roi: np.ndarray) -> str:
        """
        Detect traffic light state from ROI

        Simple color-based detection
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define color ranges (HSV)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])

        green_lower = np.array([40, 100, 100])
        green_upper = np.array([80, 255, 255])

        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        # Count pixels
        red_count = cv2.countNonZero(red_mask)
        yellow_count = cv2.countNonZero(yellow_mask)
        green_count = cv2.countNonZero(green_mask)

        # Determine state
        max_count = max(red_count, yellow_count, green_count)

        if max_count < 10:
            return 'unknown'

        if red_count == max_count:
            return 'red'
        elif yellow_count == max_count:
            return 'yellow'
        else:
            return 'green'


class ObjectTracker:
    """
    Simple object tracker using centroid tracking
    """

    def __init__(self, max_disappeared: int = 5):
        self.next_object_id = 0
        self.objects = {}  # object_id -> centroid
        self.disappeared = {}  # object_id -> count
        self.max_disappeared = max_disappeared

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracked objects
        """
        if not detections:
            # Mark all objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]

            return []

        # Current centroids
        input_centroids = [det['center'] for det in detections]

        if not self.objects:
            # Register new objects
            for i, det in enumerate(detections):
                self._register(input_centroids[i])
                det['track_id'] = self.next_object_id - 1

            return detections

        # Match existing objects with new detections
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Calculate distances
        distances = []
        for oc in object_centroids:
            row = []
            for ic in input_centroids:
                dist = np.linalg.norm(np.array(oc) - np.array(ic))
                row.append(dist)
            distances.append(row)

        distances = np.array(distances)

        # Find best matches (Hungarian algorithm simplified)
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if distances[row, col] > 100:  # Max distance threshold
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0

            detections[col]['track_id'] = object_id

            used_rows.add(row)
            used_cols.add(col)

        # Handle disappeared objects
        unused_rows = set(range(len(object_centroids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1

            if self.disappeared[object_id] > self.max_disappeared:
                del self.objects[object_id]
                del self.disappeared[object_id]

        # Register new objects
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self._register(input_centroids[col])
            detections[col]['track_id'] = self.next_object_id - 1

        return detections

    def _register(self, centroid):
        """Register new object"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1


def test_tesla_detector():
    """Test the Tesla detector"""
    print("Testing Tesla Detector...")

    detector = TeslaDetector(model_size='n', confidence_threshold=0.5)

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detector.detect_objects(frame)

        # Detect lanes
        lane_info = detector.detect_lanes_advanced(frame)

        # Detect traffic lights
        traffic_lights = detector.detect_traffic_lights(frame, detections)

        # Draw results
        result = frame.copy()

        # Draw lanes
        if lane_info['left_lane'] is not None:
            cv2.polylines(result, [lane_info['left_lane']], False, (0, 255, 0), 3)

        if lane_info['right_lane'] is not None:
            cv2.polylines(result, [lane_info['right_lane']], False, (0, 255, 0), 3)

        if lane_info['center_lane'] is not None:
            cv2.polylines(result, [lane_info['center_lane']], False, (0, 255, 255), 2)

        # Draw object detections
        for det in detections:
            x, y, w, h = det['bbox']
            label = f"{det['class']} {det['confidence']:.2f}"

            if det.get('distance'):
                label += f" {det['distance']}m"

            # Color based on class
            if det['class'] == 'person':
                color = (0, 0, 255)  # Red for pedestrians
            elif det['class'] in ['car', 'bus', 'truck']:
                color = (255, 0, 0)  # Blue for vehicles
            else:
                color = (0, 255, 0)  # Green for others

            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw traffic light states
        for tl in traffic_lights:
            x, y, w, h = tl['bbox']
            state = tl.get('state', 'unknown')

            state_color = {
                'red': (0, 0, 255),
                'yellow': (0, 255, 255),
                'green': (0, 255, 0),
                'unknown': (128, 128, 128)
            }[state]

            cv2.rectangle(result, (x, y), (x+w, y+h), state_color, 3)
            cv2.putText(result, f"TL: {state}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

        # Display info
        info_text = [
            f"Objects: {len(detections)}",
            f"Lane Width: {lane_info['lane_width']:.0f}px",
            f"Deviation: {lane_info['deviation']:.1f}%",
        ]

        for i, text in enumerate(info_text):
            cv2.putText(result, text, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Tesla Detector Test', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")


if __name__ == "__main__":
    test_tesla_detector()
