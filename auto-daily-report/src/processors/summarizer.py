"""
AI Summarizer

Summarizes news articles using:
- OpenAI GPT-4 / GPT-3.5
- Anthropic Claude (optional)
"""

import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class AISummarizer:
    """AI 기반 뉴스 요약기"""

    def __init__(self, config: dict):
        """
        Initialize AI Summarizer

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config.get('ai', {})
        self.model = os.getenv('AI_MODEL', self.config.get('model', 'gpt-4'))
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 500)
        self.style = self.config.get('style', 'concise')
        self.language = self.config.get('summary_language', 'ko')

        # Initialize AI client based on model
        if self.model.startswith('gpt'):
            self._init_openai()
        elif self.model.startswith('claude'):
            self._init_anthropic()
        else:
            logger.warning(f"Unknown model: {self.model}, defaulting to GPT-4")
            self._init_openai()

    def _init_openai(self):
        """OpenAI 클라이언트 초기화"""
        try:
            from openai import OpenAI
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = OpenAI(api_key=api_key)
            self.provider = 'openai'
            logger.info(f"Initialized OpenAI client with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            self.client = None
            self.provider = None

    def _init_anthropic(self):
        """Anthropic 클라이언트 초기화"""
        try:
            from anthropic import Anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.client = Anthropic(api_key=api_key)
            self.provider = 'anthropic'
            logger.info(f"Initialized Anthropic client with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")
            self.client = None
            self.provider = None

    def summarize_news_by_category(self, news_by_category: Dict[str, List[Dict]]) -> Dict[str, str]:
        """
        카테고리별 뉴스 요약

        Args:
            news_by_category: 카테고리별 뉴스 딕셔너리

        Returns:
            카테고리별 요약 텍스트
        """
        if not self.client:
            logger.error("AI client not initialized, cannot summarize")
            return self._create_fallback_summaries(news_by_category)

        summaries = {}

        for category, articles in news_by_category.items():
            if not articles:
                continue

            logger.info(f"Summarizing {len(articles)} articles in category: {category}")

            try:
                summary = self._summarize_articles(articles, category)
                summaries[category] = summary
            except Exception as e:
                logger.error(f"Error summarizing {category}: {e}")
                summaries[category] = self._create_fallback_summary(articles, category)

        return summaries

    def _summarize_articles(self, articles: List[Dict], category: str) -> str:
        """기사 목록 요약"""
        # Prepare prompt
        articles_text = self._format_articles_for_prompt(articles)
        prompt = self._create_summary_prompt(articles_text, category)

        # Call AI
        if self.provider == 'openai':
            return self._summarize_with_openai(prompt)
        elif self.provider == 'anthropic':
            return self._summarize_with_anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _format_articles_for_prompt(self, articles: List[Dict]) -> str:
        """기사들을 프롬프트 형식으로 변환"""
        formatted = []
        for i, article in enumerate(articles[:10], 1):  # Max 10 articles
            formatted.append(
                f"{i}. {article['title']}\n"
                f"   {article.get('description', '')[:200]}..."
            )
        return '\n\n'.join(formatted)

    def _create_summary_prompt(self, articles_text: str, category: str) -> str:
        """요약 프롬프트 생성"""
        style_instructions = {
            'concise': "간결하고 핵심만 담아",
            'detailed': "상세하게",
            'bullet-points': "불릿 포인트 형식으로"
        }

        style_instruction = style_instructions.get(self.style, "간결하게")

        category_kr = {
            'politics': '정치',
            'economy': '경제',
            'technology': 'IT/과학',
            'world': '세계'
        }.get(category, category)

        if self.language == 'ko':
            prompt = f"""다음은 오늘의 {category_kr} 뉴스입니다.

{articles_text}

위 뉴스들을 {style_instruction} 요약해주세요.
- 가장 중요한 3-5개 핵심 내용만 포함
- 각 항목은 2-3문장 이내
- 불필요한 세부사항 제외
- 객관적이고 중립적인 톤 유지

요약:"""
        else:
            prompt = f"""Here are today's {category} news articles:

{articles_text}

Please summarize these news articles {style_instruction}.
- Include only 3-5 most important points
- Each point should be 2-3 sentences
- Exclude unnecessary details
- Maintain objective and neutral tone

Summary:"""

        return prompt

    def _summarize_with_openai(self, prompt: str) -> str:
        """OpenAI로 요약"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional news summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _summarize_with_anthropic(self, prompt: str) -> str:
        """Anthropic Claude로 요약"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def _create_fallback_summaries(self, news_by_category: Dict[str, List[Dict]]) -> Dict[str, str]:
        """AI 없이 간단한 요약 생성 (fallback)"""
        summaries = {}
        for category, articles in news_by_category.items():
            summaries[category] = self._create_fallback_summary(articles, category)
        return summaries

    def _create_fallback_summary(self, articles: List[Dict], category: str) -> str:
        """AI 없이 간단한 요약 생성"""
        if not articles:
            return "뉴스 없음"

        summary_lines = []
        for i, article in enumerate(articles[:5], 1):
            title = article['title']
            summary_lines.append(f"{i}. {title}")

        return '\n'.join(summary_lines)


if __name__ == '__main__':
    # Test
    import yaml

    logging.basicConfig(level=logging.INFO)

    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Sample news data
    sample_news = {
        'technology': [
            {
                'title': 'OpenAI releases GPT-5',
                'description': 'OpenAI announced the release of GPT-5, the latest language model...',
                'url': 'https://example.com/1'
            },
            {
                'title': 'Apple announces new AI features',
                'description': 'Apple unveils new AI-powered features in iOS 18...',
                'url': 'https://example.com/2'
            }
        ]
    }

    summarizer = AISummarizer(config)
    summaries = summarizer.summarize_news_by_category(sample_news)

    print("\n=== AI Summarization Test ===")
    for category, summary in summaries.items():
        print(f"\n{category.upper()}:")
        print(summary)
