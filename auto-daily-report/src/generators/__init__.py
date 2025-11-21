"""
Content Generators Module

Generates report content in various formats:
- Markdown text
- HTML pages
- Audio (TTS)
"""

from .text_generator import TextGenerator
from .audio_generator import AudioGenerator

__all__ = ['TextGenerator', 'AudioGenerator']
