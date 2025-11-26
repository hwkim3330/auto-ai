#!/usr/bin/env python3
"""
Unified Vision System

통합 비전 시스템 - 자율주행, 로봇, 모니터링 등 범용 활용

Features:
- Object detection (YOLO)
- Depth estimation (Depth Anything V3)
- Person tracking & analytics
- BEV (Bird's Eye View) generation
- Data logging
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import os
import time

from .tesla_detector import TeslaDetector
from .depth_estimator import DepthAnythingV3, VisionMonitor
from .temporal_smoother import TemporalSmoother, ConfidenceFilter


class UnifiedVisionSystem:
    """
    Unified Vision System

    통합 비전 시스템 - 모든 모델을 하나로 관리
    """

    def __init__(
        self,
        device: str = 'cpu',
        yolo_size: str = 'n',
        depth_model_size: str = 'small',
        depth_interval: int = 50,
        detection_interval: int = 5,
        log_dir: str = 'logs',
        mode: str = 'monitor'  # monitor, autonomous, robot
    ):
        """
        Initialize Unified Vision System

        Args:
            device: 'cpu' or 'cuda'
            yolo_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            depth_model_size: Depth model size ('small', 'base', 'large')
            depth_interval: Run depth estimation every N frames
            detection_interval: Run object detection every N frames
            log_dir: Directory for logs
            mode: Operation mode (monitor, autonomous, robot)
        """
        self.device = device
        self.mode = mode

        print(f"🚀 Initializing Unified Vision System (mode: {mode})")
        print(f"   Detection: every {detection_interval} frames, Depth: every {depth_interval} frames")

        # Object detector (YOLO) - detect ALL objects
        self.detector = TeslaDetector(
            model_size=yolo_size,
            confidence_threshold=0.25,  # Lower base threshold
            detect_all=True  # Detect all objects, not just persons
        )
        print(f"✅ Object Detector (YOLOv8{yolo_size}) initialized - All objects")

        # Temporal smoothing for frame consistency
        self.temporal_smoother = TemporalSmoother(
            history_size=5,
            confidence_alpha=0.7,  # 70% current, 30% history
            bbox_alpha=0.6,  # Smooth bbox movement
            min_persistence_frames=3  # Keep objects for at least 3 frames
        )
        print(f"✅ Temporal Smoother initialized - Frame consistency enabled")

        # Confidence filter with hysteresis to reduce flickering
        self.confidence_filter = ConfidenceFilter(
            base_threshold=0.35,  # Base threshold
            hysteresis=0.1  # ±0.1 hysteresis margin
        )
        print(f"✅ Confidence Filter initialized - Flickering reduction enabled")

        # Depth estimator
        self.depth_estimator = DepthAnythingV3(
            model_size=depth_model_size,
            device=device
        )
        print(f"✅ Depth Estimator (Depth Anything V3 {depth_model_size}) initialized")

        # Vision monitor (tracking, analytics, BEV)
        self.monitor = VisionMonitor(
            self.depth_estimator,
            log_dir=log_dir,
            depth_interval=depth_interval
        )
        print(f"✅ Vision Monitor initialized")

        # Performance optimization
        self.detection_interval = detection_interval
        self.frame_count = 0
        self.cached_detections = []

        # Performance metrics
        self.fps = 0.0
        self.last_time = time.time()

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Process single frame

        Args:
            frame: Input RGB image

        Returns:
            (annotated_frame, bev_frame, analytics)
        """
        # Update FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time) if current_time > self.last_time else 30.0
        self.last_time = current_time

        # Update frame counter
        self.frame_count += 1

        # Object detection (every N frames)
        if self.frame_count % self.detection_interval == 0 or not self.cached_detections:
            # Detect objects
            raw_detections = self.detector.detect_objects(frame)

            # Apply confidence filtering with hysteresis
            filtered_detections = self.confidence_filter.filter_detections(raw_detections)

            # Apply temporal smoothing for frame consistency
            detections = self.temporal_smoother.smooth_detections(filtered_detections)

            self.cached_detections = detections
        else:
            # Use cached detections
            detections = self.cached_detections

        # Vision monitoring (depth + tracking + analytics + BEV)
        annotated_frame, analytics, bev_frame = self.monitor.process_frame(
            frame, detections
        )

        # Add FPS to analytics
        analytics['fps'] = self.fps

        return annotated_frame, bev_frame, analytics

    def get_summary(self) -> dict:
        """Get system summary"""
        return {
            'mode': self.mode,
            'device': self.device,
            'fps': self.fps,
            'detector': 'YOLOv8',
            'depth_model': 'Depth Anything V3'
        }
