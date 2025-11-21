"""
Telegram Publisher

Sends reports to Telegram using Bot API
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class TelegramPublisher:
    """텔레그램 발송기"""

    def __init__(self, config: dict):
        """
        Initialize Telegram Publisher

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config.get('publishing', {}).get('telegram', {})
        self.enabled = self.config.get('enabled', False)
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.send_text = self.config.get('send_text', True)
        self.send_audio = self.config.get('send_audio', False)

        if self.enabled and (not self.bot_token or not self.chat_id):
            logger.warning("Telegram credentials not configured. Publishing disabled.")
            self.enabled = False
        elif self.enabled:
            logger.info(f"Telegram publisher initialized for chat: {self.chat_id}")

    def publish_text(self, markdown_text: str) -> bool:
        """
        마크다운 텍스트를 텔레그램으로 전송

        Args:
            markdown_text: 마크다운 형식 텍스트

        Returns:
            성공 여부
        """
        if not self.enabled or not self.send_text:
            return False

        try:
            import requests

            # Telegram has 4096 character limit
            max_length = 4000
            text = markdown_text[:max_length]
            if len(markdown_text) > max_length:
                text += "\n\n... (전체 리포트는 파일로 확인하세요)"

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }

            response = requests.post(url, data=data, timeout=30)
            response.raise_for_status()

            logger.info("Successfully sent text to Telegram")
            return True

        except Exception as e:
            logger.error(f"Failed to send to Telegram: {e}")
            return False

    def publish_document(self, file_path: str, caption: Optional[str] = None) -> bool:
        """
        파일을 텔레그램으로 전송

        Args:
            file_path: 파일 경로
            caption: 캡션 (선택)

        Returns:
            성공 여부
        """
        if not self.enabled:
            return False

        try:
            import requests

            url = f"https://api.telegram.org/bot{self.bot_token}/sendDocument"

            with open(file_path, 'rb') as f:
                files = {'document': f}
                data = {'chat_id': self.chat_id}
                if caption:
                    data['caption'] = caption

                response = requests.post(url, data=data, files=files, timeout=60)
                response.raise_for_status()

            logger.info(f"Successfully sent document to Telegram: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to send document to Telegram: {e}")
            return False

    def publish_audio(self, audio_path: str, caption: Optional[str] = None) -> bool:
        """
        음성 파일을 텔레그램으로 전송

        Args:
            audio_path: 음성 파일 경로
            caption: 캡션 (선택)

        Returns:
            성공 여부
        """
        if not self.enabled or not self.send_audio:
            return False

        try:
            import requests

            url = f"https://api.telegram.org/bot{self.bot_token}/sendAudio"

            with open(audio_path, 'rb') as f:
                files = {'audio': f}
                data = {'chat_id': self.chat_id}
                if caption:
                    data['caption'] = caption

                response = requests.post(url, data=data, files=files, timeout=60)
                response.raise_for_status()

            logger.info(f"Successfully sent audio to Telegram: {audio_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to send audio to Telegram: {e}")
            return False


if __name__ == '__main__':
    # Test
    import yaml

    logging.basicConfig(level=logging.INFO)

    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    publisher = TelegramPublisher(config)

    if publisher.enabled:
        # Test text
        test_text = "# 테스트 리포트\n\n오늘의 뉴스입니다."
        result = publisher.publish_text(test_text)
        print(f"Text sent: {result}")
    else:
        print("Telegram not configured")
