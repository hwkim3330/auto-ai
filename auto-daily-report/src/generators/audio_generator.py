"""
Audio Generator

Generates audio reports using Edge-TTS (Microsoft Text-to-Speech)
"""

import os
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AudioGenerator:
    """음성 리포트 생성기"""

    def __init__(self, config: dict):
        """
        Initialize Audio Generator

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config.get('tts', {})
        self.enabled = self.config.get('enabled', False)
        self.voice = self.config.get('voice', 'ko-KR-SunHiNeural')
        self.rate = self.config.get('rate', '+0%')
        self.volume = self.config.get('volume', '+0%')
        self.output_format = self.config.get('output_format', 'mp3')

        if self.enabled:
            try:
                import edge_tts
                self.edge_tts = edge_tts
                logger.info(f"Audio generator initialized with voice: {self.voice}")
            except ImportError:
                logger.warning("edge-tts not installed. Audio generation disabled.")
                self.enabled = False

    async def generate_audio(
        self,
        text: str,
        output_path: Optional[str] = None,
        date: Optional[datetime] = None
    ) -> Optional[str]:
        """
        텍스트를 음성으로 변환

        Args:
            text: 변환할 텍스트
            output_path: 출력 파일 경로 (선택)
            date: 리포트 날짜

        Returns:
            저장된 파일 경로 또는 None
        """
        if not self.enabled:
            logger.warning("Audio generation is disabled")
            return None

        if date is None:
            date = datetime.now()

        # Default output path
        if output_path is None:
            date_str = date.strftime('%Y-%m-%d')
            output_dir = Path('reports/daily')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"report_{date_str}.{self.output_format}")

        try:
            # Create TTS
            communicate = self.edge_tts.Communicate(
                text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume
            )

            # Save to file
            await communicate.save(output_path)
            logger.info(f"Audio generated: {output_path}")

            # Get file size
            file_size = os.path.getsize(output_path)
            logger.info(f"Audio file size: {file_size / 1024:.1f} KB")

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate audio: {e}")
            return None

    def prepare_text_for_speech(self, markdown_text: str) -> str:
        """
        마크다운 텍스트를 음성에 적합한 형태로 변환

        Args:
            markdown_text: 마크다운 텍스트

        Returns:
            음성용 텍스트
        """
        # Remove markdown formatting
        text = markdown_text

        # Remove headers (#)
        import re
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

        # Remove bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)

        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

        # Remove horizontal rules
        text = re.sub(r'^---+\s*$', '', text, flags=re.MULTILINE)

        # Remove emojis for cleaner speech (optional)
        # text = re.sub(r'[^\w\s가-힣,.!?\-]', '', text)

        # Clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Add pauses for sections
        text = re.sub(r'\n\n', '\n\n... ', text)

        return text.strip()


def generate_audio_sync(text: str, output_path: str, config: dict) -> Optional[str]:
    """
    Synchronous wrapper for audio generation

    Args:
        text: Text to convert
        output_path: Output file path
        config: Configuration dict

    Returns:
        Output file path or None
    """
    import asyncio

    generator = AudioGenerator(config)
    if not generator.enabled:
        return None

    # Prepare text
    speech_text = generator.prepare_text_for_speech(text)

    # Run async function
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        generator.generate_audio(speech_text, output_path)
    )


if __name__ == '__main__':
    # Test
    import yaml
    import asyncio

    logging.basicConfig(level=logging.INFO)

    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Enable TTS for testing
    config['tts']['enabled'] = True

    sample_text = """
# 데일리 리포트 - 2025년 11월 21일

## 날씨
오늘 서울 날씨는 맑음, 기온은 15도입니다.

## 주요 뉴스
기술 분야에서 OpenAI가 새로운 모델을 발표했습니다.
"""

    async def test():
        generator = AudioGenerator(config)
        if generator.enabled:
            result = await generator.generate_audio(
                generator.prepare_text_for_speech(sample_text),
                "test_audio.mp3"
            )
            print(f"Generated: {result}")
        else:
            print("TTS not enabled")

    asyncio.run(test())
