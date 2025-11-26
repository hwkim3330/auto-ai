#!/usr/bin/env python3
"""
Temporal Smoothing for Vision Detection

프레임간 일관성 유지를 위한 시간적 평활화

Features:
- Detection smoothing: Reduce flickering of detections
- Confidence temporal filtering: Smooth confidence values
- Bounding box smoothing: Reduce jitter in bbox positions
- Object persistence: Keep objects visible across frames
"""

import numpy as np
from typing import List, Dict, Optional
from collections import deque
import time


def convert_to_python_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization

    Args:
        obj: Object to convert

    Returns:
        Object with Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj


class TemporalSmoother:
    """
    Temporal smoother for detection results

    프레임간 변화가 너무 크지 않도록 시간적 평활화 적용
    """

    def __init__(
        self,
        history_size: int = 5,
        confidence_alpha: float = 0.7,
        bbox_alpha: float = 0.6,
        min_persistence_frames: int = 3
    ):
        """
        Initialize temporal smoother

        Args:
            history_size: Number of frames to keep in history
            confidence_alpha: Exponential smoothing factor for confidence (0-1)
                             Higher = more weight to current frame
            bbox_alpha: Exponential smoothing factor for bbox (0-1)
            min_persistence_frames: Minimum frames to keep object visible
        """
        self.history_size = history_size
        self.confidence_alpha = confidence_alpha
        self.bbox_alpha = bbox_alpha
        self.min_persistence_frames = min_persistence_frames

        # Detection history: track_id -> deque of detections
        self.detection_history = {}

        # Last smoothed detections
        self.last_detections = []

        # Object persistence tracking
        self.object_frames = {}  # track_id -> frame count

    def smooth_detections(
        self,
        current_detections: List[Dict]
    ) -> List[Dict]:
        """
        Apply temporal smoothing to detections

        Args:
            current_detections: List of current frame detections

        Returns:
            Smoothed detections
        """
        if not current_detections:
            # Return previously visible objects that should persist
            return self._get_persistent_objects()

        smoothed = []

        for det in current_detections:
            track_id = det.get('track_id')

            if track_id is None:
                # No tracking ID, use as-is
                smoothed.append(det)
                continue

            # Update object frame count
            if track_id not in self.object_frames:
                self.object_frames[track_id] = 1
            else:
                self.object_frames[track_id] += 1

            # Get detection history for this object
            if track_id not in self.detection_history:
                self.detection_history[track_id] = deque(maxlen=self.history_size)

            history = self.detection_history[track_id]

            # Apply smoothing
            smoothed_det = self._smooth_single_detection(det, history)

            # Add to history
            history.append(det.copy())

            smoothed.append(smoothed_det)

        # Add persistent objects that weren't detected this frame
        persistent = self._get_persistent_objects()
        for p_det in persistent:
            p_id = p_det.get('track_id')
            # Only add if not already in current detections
            if p_id not in [d.get('track_id') for d in smoothed]:
                smoothed.append(p_det)

        # Clean up old objects
        self._cleanup_old_objects(current_detections)

        # Convert all numpy types to Python types for JSON serialization
        smoothed = [convert_to_python_types(det) for det in smoothed]

        # Store last detections
        self.last_detections = smoothed.copy()

        return smoothed

    def _smooth_single_detection(
        self,
        current_det: Dict,
        history: deque
    ) -> Dict:
        """
        Smooth a single detection using its history

        Args:
            current_det: Current detection
            history: Detection history for this object

        Returns:
            Smoothed detection
        """
        if not history:
            # No history, return as-is
            return current_det.copy()

        smoothed = current_det.copy()

        # Smooth confidence using exponential moving average
        if 'confidence' in current_det:
            prev_conf = history[-1].get('confidence', current_det['confidence'])
            smoothed['confidence'] = (
                self.confidence_alpha * current_det['confidence'] +
                (1 - self.confidence_alpha) * prev_conf
            )

        # Smooth bounding box using exponential moving average
        if 'bbox' in current_det:
            current_bbox = np.array(current_det['bbox'])
            prev_bbox = np.array(history[-1].get('bbox', current_det['bbox']))

            smoothed_bbox = (
                self.bbox_alpha * current_bbox +
                (1 - self.bbox_alpha) * prev_bbox
            )

            smoothed['bbox'] = tuple(smoothed_bbox.astype(int))

            # Update center based on smoothed bbox
            x, y, w, h = smoothed['bbox']
            smoothed['center'] = (x + w // 2, y + h // 2)

        # Smooth distance if available
        if 'distance' in current_det and 'distance' in history[-1]:
            prev_dist = history[-1]['distance']
            if prev_dist is not None and current_det['distance'] is not None:
                smoothed['distance'] = (
                    self.bbox_alpha * current_det['distance'] +
                    (1 - self.bbox_alpha) * prev_dist
                )

        # Smooth depth distance if available
        if 'distance_depth' in current_det and 'distance_depth' in history[-1]:
            prev_depth = history[-1]['distance_depth']
            if prev_depth is not None and current_det['distance_depth'] is not None:
                smoothed['distance_depth'] = (
                    self.bbox_alpha * current_det['distance_depth'] +
                    (1 - self.bbox_alpha) * prev_depth
                )

        # Smooth 3D position if available
        if 'position_3d' in current_det and 'position_3d' in history[-1]:
            current_pos = np.array(current_det['position_3d'])
            prev_pos = np.array(history[-1]['position_3d'])

            smoothed_pos = (
                self.bbox_alpha * current_pos +
                (1 - self.bbox_alpha) * prev_pos
            )

            smoothed['position_3d'] = tuple(smoothed_pos)

        # Smooth height if available
        if 'height' in current_det and 'height' in history[-1]:
            prev_height = history[-1]['height']
            if prev_height is not None and current_det['height'] is not None:
                smoothed['height'] = (
                    self.bbox_alpha * current_det['height'] +
                    (1 - self.bbox_alpha) * prev_height
                )

        return smoothed

    def _get_persistent_objects(self) -> List[Dict]:
        """
        Get objects that should persist even if not detected

        Returns:
            List of persistent detections
        """
        persistent = []

        for det in self.last_detections:
            track_id = det.get('track_id')

            if track_id is None:
                continue

            # Check if object has been visible long enough
            frame_count = self.object_frames.get(track_id, 0)

            if frame_count >= self.min_persistence_frames:
                # Keep object visible with reduced confidence
                persistent_det = det.copy()

                # Reduce confidence to indicate uncertainty
                if 'confidence' in persistent_det:
                    persistent_det['confidence'] *= 0.8

                # Add flag to indicate this is a persistent detection
                persistent_det['is_persistent'] = True

                # Only persist for a few frames
                if frame_count < self.min_persistence_frames + 5:
                    persistent.append(persistent_det)

        return persistent

    def _cleanup_old_objects(self, current_detections: List[Dict]):
        """
        Clean up objects that are no longer being tracked

        Args:
            current_detections: Current frame detections
        """
        current_ids = set(d.get('track_id') for d in current_detections if d.get('track_id') is not None)

        # Remove objects not seen in current frame
        for track_id in list(self.detection_history.keys()):
            if track_id not in current_ids:
                # Check if we should keep it for persistence
                frame_count = self.object_frames.get(track_id, 0)

                if frame_count < self.min_persistence_frames:
                    # Remove immediately
                    del self.detection_history[track_id]
                    if track_id in self.object_frames:
                        del self.object_frames[track_id]
                elif frame_count >= self.min_persistence_frames + 5:
                    # Remove after persistence period
                    del self.detection_history[track_id]
                    if track_id in self.object_frames:
                        del self.object_frames[track_id]

    def reset(self):
        """Reset all history"""
        self.detection_history.clear()
        self.last_detections.clear()
        self.object_frames.clear()


class ConfidenceFilter:
    """
    Confidence-based detection filter

    더 높은 confidence threshold로 flickering 감소
    """

    def __init__(
        self,
        base_threshold: float = 0.3,
        hysteresis: float = 0.1
    ):
        """
        Initialize confidence filter with hysteresis

        Args:
            base_threshold: Base confidence threshold
            hysteresis: Hysteresis margin to prevent flickering
        """
        self.base_threshold = base_threshold
        self.hysteresis = hysteresis
        self.tracked_objects = set()  # track_ids of objects currently visible

    def filter_detections(
        self,
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Filter detections with hysteresis

        Args:
            detections: Input detections

        Returns:
            Filtered detections
        """
        filtered = []
        current_tracked = set()

        for det in detections:
            track_id = det.get('track_id')
            confidence = det.get('confidence', 0.0)

            # Apply hysteresis
            if track_id in self.tracked_objects:
                # Already tracking - use lower threshold (base - hysteresis)
                threshold = self.base_threshold - self.hysteresis
            else:
                # Not tracking - use higher threshold (base + hysteresis)
                threshold = self.base_threshold + self.hysteresis

            # Filter based on threshold
            if confidence >= threshold:
                filtered.append(det)
                if track_id is not None:
                    current_tracked.add(track_id)

        # Update tracked objects
        self.tracked_objects = current_tracked

        return filtered
