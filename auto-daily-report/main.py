#!/usr/bin/env python3
"""
Auto Daily Report - Main Orchestrator

Collects news and weather, generates AI-powered daily reports
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from collectors import NewsCollector, WeatherCollector
from processors import AISummarizer
from generators import TextGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto-daily-report.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / 'config' / 'config.yaml'

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("Starting Auto Daily Report Generation")
    logger.info("=" * 60)

    start_time = datetime.now()

    # Load configuration
    config = load_config()

    # Initialize components
    logger.info("\n[1/5] Initializing components...")
    news_collector = NewsCollector(config)
    weather_collector = WeatherCollector(config)
    summarizer = AISummarizer(config)
    text_generator = TextGenerator(config)

    # Step 1: Collect news
    logger.info("\n[2/5] Collecting news...")
    try:
        news_data = news_collector.collect_all()
        total_articles = sum(len(articles) for articles in news_data.values())
        logger.info(f"✓ Collected {total_articles} articles across {len(news_data)} categories")
    except Exception as e:
        logger.error(f"✗ Failed to collect news: {e}")
        news_data = {}

    # Step 2: Collect weather
    logger.info("\n[3/5] Collecting weather data...")
    try:
        weather_data = weather_collector.collect_all()
        if weather_data.get('current'):
            city = weather_data['current']['city']
            temp = weather_data['current']['temp']
            logger.info(f"✓ Collected weather for {city}: {temp}°C")
        else:
            logger.warning("⚠ Weather data incomplete")
    except Exception as e:
        logger.error(f"✗ Failed to collect weather: {e}")
        weather_data = {}

    # Step 3: AI Summarization
    logger.info("\n[4/5] Generating AI summaries...")
    try:
        if news_data:
            news_summaries = summarizer.summarize_news_by_category(news_data)
            logger.info(f"✓ Generated {len(news_summaries)} category summaries")
        else:
            logger.warning("⚠ No news to summarize")
            news_summaries = {}
    except Exception as e:
        logger.error(f"✗ Failed to generate summaries: {e}")
        # Fallback: use simple summaries
        news_summaries = summarizer._create_fallback_summaries(news_data)

    # Step 4: Generate reports
    logger.info("\n[5/5] Generating reports...")
    try:
        # Generate markdown and HTML
        markdown_report = text_generator.generate_markdown(
            news_summaries,
            weather_data,
            start_time
        )

        html_report = text_generator.generate_html(
            news_summaries,
            weather_data,
            start_time
        )

        # Save to files
        saved_files = text_generator.save_report(
            markdown_report,
            html_report,
            start_time
        )

        logger.info("✓ Reports generated successfully:")
        for format_type, file_path in saved_files.items():
            logger.info(f"  - {format_type}: {file_path}")

    except Exception as e:
        logger.error(f"✗ Failed to generate reports: {e}")
        saved_files = {}

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "=" * 60)
    logger.info("Daily Report Generation Complete")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Articles collected: {total_articles if news_data else 0}")
    logger.info(f"Categories summarized: {len(news_summaries)}")
    logger.info(f"Reports generated: {len(saved_files)}")

    if saved_files:
        logger.info("\n📰 Check your reports at:")
        for format_type, file_path in saved_files.items():
            logger.info(f"  {file_path}")

    logger.info("\n" + "=" * 60)

    return 0 if saved_files else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        sys.exit(1)
