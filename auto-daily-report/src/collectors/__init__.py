"""
Data Collectors Module

Collects data from various external APIs:
- NewsAPI & Google News RSS
- OpenWeatherMap
- Financial data (optional)
"""

from .news_collector import NewsCollector
from .weather_collector import WeatherCollector

__all__ = ['NewsCollector', 'WeatherCollector']
