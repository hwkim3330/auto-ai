"""
Publishers Module

Publishes reports to various destinations:
- Local files
- Telegram
- Email
- GitHub Pages
"""

from .telegram_publisher import TelegramPublisher

__all__ = ['TelegramPublisher']
