#!/usr/bin/env python3
"""Test all imports and structure"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing Vision Mamba Control imports...")
print("=" * 60)

# Test webcam module
try:
    from capture.webcam import WebcamCapture
    print("✅ Webcam module OK")
except Exception as e:
    print(f"❌ Webcam module failed: {e}")

# Test GUI module (tkinter)
try:
    import tkinter as tk
    print("✅ Tkinter available")
except Exception as e:
    print(f"❌ Tkinter not available: {e}")

# Test core dependencies
try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__}")
except Exception as e:
    print(f"❌ OpenCV not available: {e}")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except Exception as e:
    print(f"❌ NumPy not available: {e}")

try:
    from PIL import Image
    print(f"✅ Pillow available")
except Exception as e:
    print(f"❌ Pillow not available: {e}")

# Test optional dependencies
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    torch_available = True
except Exception as e:
    print(f"⚠️  PyTorch not available (demo mode only): {e}")
    torch_available = False

try:
    import einops
    print(f"✅ einops available")
except Exception as e:
    print(f"⚠️  einops not available (needed for AI mode): {e}")

print("=" * 60)

# Test model imports (only if torch available)
if torch_available:
    try:
        from models.mamba import VisionMamba
        print("✅ Mamba model module OK")
    except Exception as e:
        print(f"❌ Mamba model failed: {e}")

    try:
        from models.control_model import VisionMambaControl
        print("✅ Control model module OK")
    except Exception as e:
        print(f"❌ Control model failed: {e}")
else:
    print("⚠️  Skipping model tests (PyTorch not available)")

print("=" * 60)
print("\n📋 Summary:")
print("  - Demo mode (webcam + GUI): Available" if 'cv2' in sys.modules and 'tkinter' in sys.modules else "  - Demo mode: NOT available")
print("  - AI mode (full inference): Available" if torch_available else "  - AI mode: NOT available (install PyTorch)")
print()
print("To run demo mode:")
print("  python3 run_demo.py")
print()
if not torch_available:
    print("To enable AI mode, install PyTorch:")
    print("  pip install torch einops")
print("=" * 60)
