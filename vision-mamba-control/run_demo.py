#!/usr/bin/env python3
"""
Vision Mamba Control - Demo Launcher

웹캠 기반 실시간 비전 제어 시스템 데모
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gui.app import main

if __name__ == "__main__":
    print("=" * 60)
    print("Vision Mamba Control - Real-time Vision Control System")
    print("=" * 60)
    print()
    print("🚗 Starting GUI application...")
    print("📹 Make sure your webcam is connected")
    print()
    print("Controls:")
    print("  - Click 'Start' to begin webcam capture")
    print("  - Click 'Stop' to pause")
    print("  - Close window to exit")
    print()
    print("=" * 60)
    print()

    main()
