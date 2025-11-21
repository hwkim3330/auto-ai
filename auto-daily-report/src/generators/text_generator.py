"""
Text Generator

Generates daily reports in Markdown and HTML formats
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TextGenerator:
    """텍스트 리포트 생성기"""

    def __init__(self, config: dict):
        """
        Initialize Text Generator

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config.get('report', {})
        self.language = self.config.get('language', 'ko')
        self.timezone = self.config.get('timezone', 'Asia/Seoul')

    def generate_markdown(
        self,
        news_summaries: Dict[str, str],
        weather_data: Dict,
        date: Optional[datetime] = None
    ) -> str:
        """
        Markdown 리포트 생성

        Args:
            news_summaries: 카테고리별 뉴스 요약
            weather_data: 날씨 정보
            date: 리포트 날짜 (기본값: 오늘)

        Returns:
            Markdown 텍스트
        """
        if date is None:
            date = datetime.now()

        # Header
        if self.language == 'ko':
            weekday_kr = ['월', '화', '수', '목', '금', '토', '일'][date.weekday()]
            header = f"# 📰 Daily Report - {date.year}년 {date.month}월 {date.day}일 {weekday_kr}요일\n\n"
        else:
            header = f"# 📰 Daily Report - {date.strftime('%B %d, %Y %A')}\n\n"

        markdown = header

        # Weather section
        markdown += self._generate_weather_section(weather_data)

        # News sections
        markdown += self._generate_news_sections(news_summaries)

        # Footer
        markdown += self._generate_footer(date)

        return markdown

    def _generate_weather_section(self, weather_data: Dict) -> str:
        """날씨 섹션 생성"""
        if not weather_data or not weather_data.get('current'):
            return "## 🌤️ 날씨\n\n날씨 정보를 가져올 수 없습니다.\n\n---\n\n"

        current = weather_data['current']
        forecast = weather_data.get('forecast', [])
        air_quality = weather_data.get('air_quality')
        alerts = weather_data.get('alerts', [])
        clothing = weather_data.get('clothing_advice')

        section = "## 🌤️ 날씨\n\n"

        # Current weather
        section += f"### 현재 날씨 ({current['city']})\n\n"
        section += f"- **온도**: {current['temp']:.1f}°C (체감: {current['feels_like']:.1f}°C)\n"
        section += f"- **날씨**: {current['weather_description']}\n"
        section += f"- **습도**: {current['humidity']}%\n"
        section += f"- **풍속**: {current['wind_speed']:.1f} m/s\n"

        # Air quality
        if air_quality:
            section += f"- **대기질**: {air_quality['aqi_label_kr']} (PM2.5: {air_quality['pm2_5']} µg/m³)\n"

        section += "\n"

        # Forecast
        if forecast:
            section += "### 일기예보\n\n"
            for fc in forecast:
                fc_date = datetime.fromisoformat(fc['datetime'])
                section += f"- **{fc_date.strftime('%m/%d')}**: {fc['weather_description']}, "
                section += f"{fc['temp']:.1f}°C, 강수확률 {fc['pop']*100:.0f}%\n"
            section += "\n"

        # Clothing advice
        if clothing:
            section += f"### 옷차림 추천\n\n{clothing}\n\n"

        # Alerts
        if alerts:
            section += "### ⚠️ 경고\n\n"
            for alert in alerts:
                section += f"- {alert}\n"
            section += "\n"

        section += "---\n\n"
        return section

    def _generate_news_sections(self, news_summaries: Dict[str, str]) -> str:
        """뉴스 섹션 생성"""
        section = "## 📰 주요 뉴스\n\n"

        category_names_kr = {
            'politics': '정치',
            'economy': '경제',
            'technology': 'IT & 과학',
            'world': '세계'
        }

        category_icons = {
            'politics': '🏛️',
            'economy': '💼',
            'technology': '💻',
            'world': '🌍'
        }

        for category, summary in news_summaries.items():
            icon = category_icons.get(category, '📌')
            name = category_names_kr.get(category, category.title())

            section += f"### {icon} {name}\n\n"
            section += f"{summary}\n\n"

        section += "---\n\n"
        return section

    def _generate_footer(self, date: datetime) -> str:
        """푸터 생성"""
        footer = f"\n---\n\n"
        footer += f"*생성 시간: {date.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        footer += f"*Powered by Auto Daily Report*\n"
        return footer

    def generate_html(
        self,
        news_summaries: Dict[str, str],
        weather_data: Dict,
        date: Optional[datetime] = None
    ) -> str:
        """
        HTML 리포트 생성

        Args:
            news_summaries: 카테고리별 뉴스 요약
            weather_data: 날씨 정보
            date: 리포트 날짜

        Returns:
            HTML 텍스트
        """
        if date is None:
            date = datetime.now()

        # Generate markdown first
        markdown_content = self.generate_markdown(news_summaries, weather_data, date)

        # Convert to HTML (simple version)
        import markdown as md
        html_content = md.markdown(markdown_content)

        # Wrap in HTML template
        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Report - {date.strftime('%Y-%m-%d')}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #764ba2;
            margin-top: 30px;
            border-left: 5px solid #764ba2;
            padding-left: 15px;
        }}
        h3 {{
            color: #667eea;
            margin-top: 20px;
        }}
        hr {{
            border: none;
            border-top: 2px dashed #ddd;
            margin: 30px 0;
        }}
        ul, ol {{
            line-height: 1.8;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .footer {{
            text-align: center;
            color: #888;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>"""

        return html

    def save_report(
        self,
        markdown: str,
        html: str,
        date: Optional[datetime] = None,
        output_dir: str = 'reports/daily'
    ) -> Dict[str, str]:
        """
        리포트 파일로 저장

        Args:
            markdown: Markdown 텍스트
            html: HTML 텍스트
            date: 리포트 날짜
            output_dir: 출력 디렉토리

        Returns:
            저장된 파일 경로 딕셔너리
        """
        if date is None:
            date = datetime.now()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # File names
        date_str = date.strftime('%Y-%m-%d')
        md_file = output_path / f"report_{date_str}.md"
        html_file = output_path / f"report_{date_str}.html"

        saved_files = {}

        # Save Markdown
        try:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
            saved_files['markdown'] = str(md_file)
            logger.info(f"Saved Markdown report: {md_file}")
        except Exception as e:
            logger.error(f"Error saving Markdown: {e}")

        # Save HTML
        try:
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html)
            saved_files['html'] = str(html_file)
            logger.info(f"Saved HTML report: {html_file}")
        except Exception as e:
            logger.error(f"Error saving HTML: {e}")

        return saved_files


if __name__ == '__main__':
    # Test
    import yaml

    logging.basicConfig(level=logging.INFO)

    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Sample data
    news_summaries = {
        'technology': "OpenAI가 GPT-5를 발표했습니다. 새로운 모델은 이전보다 2배 빠르고 정확합니다.",
        'economy': "코스피가 1.5% 상승하며 2,500선을 회복했습니다."
    }

    weather_data = {
        'current': {
            'city': 'Seoul',
            'temp': 15.5,
            'feels_like': 14.0,
            'humidity': 65,
            'wind_speed': 3.2,
            'weather_description': '맑음'
        },
        'air_quality': {
            'aqi_label_kr': '좋음',
            'pm2_5': 25
        },
        'clothing_advice': '가을 자켓 권장'
    }

    generator = TextGenerator(config)

    # Generate reports
    markdown = generator.generate_markdown(news_summaries, weather_data)
    html = generator.generate_html(news_summaries, weather_data)

    print("\n=== Markdown Report ===")
    print(markdown[:500])

    print("\n=== Saving Reports ===")
    saved = generator.save_report(markdown, html)
    print(f"Saved files: {saved}")
