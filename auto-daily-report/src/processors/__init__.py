"""
Data Processors Module

Processes collected data:
- AI summarization (OpenAI GPT-4, Claude)
- Text formatting
"""

from .summarizer import AISummarizer

__all__ = ['AISummarizer']
