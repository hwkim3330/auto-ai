"""
News Collector

Collects news from:
1. NewsAPI (https://newsapi.org/)
2. Google News RSS feeds
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
import feedparser
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class NewsCollector:
    """뉴스 수집기"""

    def __init__(self, config: dict):
        """
        Initialize News Collector

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config.get('news', {})
        self.api_key = os.getenv('NEWS_API_KEY')
        self.keywords = self.config.get('keywords', [])
        self.exclude = self.config.get('exclude', [])
        self.max_articles = self.config.get('max_articles', 10)
        self.max_age_hours = self.config.get('max_age_hours', 24)

    def collect_all(self) -> Dict[str, List[Dict]]:
        """
        모든 소스에서 뉴스 수집

        Returns:
            카테고리별 뉴스 딕셔너리
        """
        news_by_category = {}

        # NewsAPI에서 수집
        if self.api_key:
            logger.info("Collecting news from NewsAPI...")
            newsapi_articles = self._collect_from_newsapi()
            self._merge_articles(news_by_category, newsapi_articles)
        else:
            logger.warning("NewsAPI key not found, skipping NewsAPI")

        # Google News RSS에서 수집
        logger.info("Collecting news from Google News RSS...")
        rss_articles = self._collect_from_rss()
        self._merge_articles(news_by_category, rss_articles)

        # 각 카테고리에서 최대 개수만큼만 유지
        for category in news_by_category:
            news_by_category[category] = news_by_category[category][:self.max_articles]

        logger.info(f"Collected {sum(len(v) for v in news_by_category.values())} articles across {len(news_by_category)} categories")
        return news_by_category

    def _collect_from_newsapi(self) -> Dict[str, List[Dict]]:
        """NewsAPI에서 뉴스 수집"""
        articles_by_category = {}

        categories = self.config.get('categories', ['technology'])
        sources = self.config.get('sources', [])

        for category in categories:
            try:
                # Build query
                query_parts = []
                if self.keywords:
                    query_parts.extend(self.keywords[:3])  # Top 3 keywords

                query = ' OR '.join(query_parts) if query_parts else None

                # API request
                params = {
                    'apiKey': self.api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': self.max_articles
                }

                if query:
                    params['q'] = query
                if sources:
                    params['sources'] = ','.join(sources)
                else:
                    params['category'] = category

                # Calculate from date
                from_date = datetime.utcnow() - timedelta(hours=self.max_age_hours)
                params['from'] = from_date.isoformat()

                response = requests.get(
                    'https://newsapi.org/v2/top-headlines',
                    params=params,
                    timeout=30
                )
                response.raise_for_status()

                data = response.json()

                if data.get('status') == 'ok':
                    articles = data.get('articles', [])

                    # Filter and format
                    formatted_articles = []
                    for article in articles:
                        if self._should_include(article):
                            formatted_articles.append(self._format_article(article, category, 'newsapi'))

                    articles_by_category[category] = formatted_articles
                    logger.info(f"  - {category}: {len(formatted_articles)} articles from NewsAPI")
                else:
                    logger.error(f"NewsAPI error for {category}: {data.get('message')}")

            except Exception as e:
                logger.error(f"Error collecting NewsAPI {category}: {e}")
                articles_by_category[category] = []

        return articles_by_category

    def _collect_from_rss(self) -> Dict[str, List[Dict]]:
        """Google News RSS에서 뉴스 수집"""
        articles_by_category = {}

        rss_feeds = self.config.get('rss_feeds', [])

        for feed_config in rss_feeds:
            url = feed_config.get('url')
            category = feed_config.get('category', 'general')

            try:
                logger.info(f"  - Fetching RSS feed for {category}...")
                feed = feedparser.parse(url)

                articles = []
                for entry in feed.entries[:self.max_articles]:
                    # Check age
                    if hasattr(entry, 'published_parsed'):
                        pub_date = datetime(*entry.published_parsed[:6])
                        age_hours = (datetime.utcnow() - pub_date).total_seconds() / 3600
                        if age_hours > self.max_age_hours:
                            continue

                    article = {
                        'title': entry.get('title', ''),
                        'description': entry.get('summary', ''),
                        'url': entry.get('link', ''),
                        'publishedAt': entry.get('published', ''),
                        'source': {'name': feed.feed.get('title', 'Google News')}
                    }

                    if self._should_include(article):
                        articles.append(self._format_article(article, category, 'rss'))

                if category not in articles_by_category:
                    articles_by_category[category] = []
                articles_by_category[category].extend(articles)

                logger.info(f"  - {category}: {len(articles)} articles from RSS")

            except Exception as e:
                logger.error(f"Error collecting RSS feed {category}: {e}")

        return articles_by_category

    def _should_include(self, article: Dict) -> bool:
        """기사가 포함되어야 하는지 확인"""
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()

        # Exclude keywords check
        for exclude_word in self.exclude:
            if exclude_word.lower() in title or exclude_word.lower() in description:
                return False

        # Minimum content check
        if not title or len(title) < 10:
            return False

        return True

    def _format_article(self, article: Dict, category: str, source_type: str) -> Dict:
        """기사를 표준 형식으로 변환"""
        return {
            'title': article.get('title', ''),
            'description': article.get('description', ''),
            'url': article.get('url', ''),
            'published_at': article.get('publishedAt', ''),
            'source': article.get('source', {}).get('name', 'Unknown'),
            'category': category,
            'source_type': source_type,
            'collected_at': datetime.utcnow().isoformat()
        }

    def _merge_articles(self, target: Dict, source: Dict):
        """기사 병합 (중복 제거)"""
        for category, articles in source.items():
            if category not in target:
                target[category] = []

            # Add unique articles (by URL)
            existing_urls = {art['url'] for art in target[category]}
            for article in articles:
                if article['url'] not in existing_urls:
                    target[category].append(article)
                    existing_urls.add(article['url'])


if __name__ == '__main__':
    # Test
    import yaml

    logging.basicConfig(level=logging.INFO)

    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    collector = NewsCollector(config)
    news = collector.collect_all()

    print("\n=== News Collection Test ===")
    for category, articles in news.items():
        print(f"\n{category.upper()}: {len(articles)} articles")
        for i, article in enumerate(articles[:3], 1):
            print(f"  {i}. {article['title'][:60]}...")
            print(f"     Source: {article['source']}")
