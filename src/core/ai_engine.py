"""
AI Engine - Gemini API를 사용한 핵심 AI 엔진
"""

import google.generativeai as genai
from typing import AsyncGenerator, Optional, Dict, Any
import asyncio
from loguru import logger


class AIEngine:
    """Gemini AI 엔진 - 모든 AI 작업의 핵심"""

    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        """
        Args:
            api_key: Gemini API 키
            model_name: 사용할 모델 이름
        """
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"AI Engine initialized with model: {model_name}")

    async def generate_text(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        텍스트 생성

        Args:
            prompt: 프롬프트
            context: 컨텍스트 (이전 대화 등)
            temperature: 창의성 수준 (0.0 ~ 1.0)
            max_tokens: 최대 토큰 수

        Returns:
            생성된 텍스트
        """
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt

            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=generation_config
            )

            result = response.text
            logger.debug(f"Generated text: {result[:100]}...")
            return result

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 텍스트 생성

        Args:
            prompt: 프롬프트
            context: 컨텍스트
            temperature: 창의성 수준

        Yields:
            생성된 텍스트 청크
        """
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt

            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
            )

            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=generation_config,
                stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise

    async def analyze_document(self, document: str, task: str) -> Dict[str, Any]:
        """
        문서 분석

        Args:
            document: 분석할 문서
            task: 분석 작업 (예: "요약", "키워드 추출", "감정 분석")

        Returns:
            분석 결과
        """
        prompt = f"""다음 문서를 {task}하세요:

문서:
{document}

분석 결과를 JSON 형식으로 제공해주세요."""

        try:
            result = await self.generate_text(prompt, temperature=0.3)
            return {"task": task, "result": result, "status": "success"}
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {"task": task, "error": str(e), "status": "error"}

    async def make_decision(
        self,
        situation: str,
        options: list[str],
        criteria: Optional[list[str]] = None
    ) -> Dict[str, Any]:
        """
        의사결정 지원

        Args:
            situation: 상황 설명
            options: 선택 가능한 옵션들
            criteria: 판단 기준들

        Returns:
            의사결정 결과 및 근거
        """
        criteria_text = "\n".join(f"- {c}" for c in criteria) if criteria else "일반적인 비즈니스 기준"
        options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

        prompt = f"""다음 상황에서 최선의 결정을 내려주세요:

상황:
{situation}

선택 가능한 옵션:
{options_text}

판단 기준:
{criteria_text}

다음 형식으로 답변해주세요:
1. 추천 옵션: [번호와 이름]
2. 이유: [상세한 근거]
3. 장점: [예상되는 이점들]
4. 단점: [고려해야 할 리스크들]
5. 실행 계획: [구체적인 실행 단계]"""

        try:
            result = await self.generate_text(prompt, temperature=0.5)
            return {
                "situation": situation,
                "recommendation": result,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            return {"error": str(e), "status": "error"}

    async def extract_information(
        self,
        text: str,
        fields: list[str]
    ) -> Dict[str, str]:
        """
        정보 추출

        Args:
            text: 대상 텍스트
            fields: 추출할 필드들

        Returns:
            추출된 정보 딕셔너리
        """
        fields_text = ", ".join(fields)
        prompt = f"""다음 텍스트에서 아래 정보를 추출하세요: {fields_text}

텍스트:
{text}

JSON 형식으로 결과를 제공해주세요."""

        try:
            result = await self.generate_text(prompt, temperature=0.1)
            return {"fields": fields, "extracted": result, "status": "success"}
        except Exception as e:
            logger.error(f"Error extracting information: {e}")
            return {"error": str(e), "status": "error"}

    async def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        style: str = "bullet_points"
    ) -> str:
        """
        텍스트 요약

        Args:
            text: 요약할 텍스트
            max_length: 최대 길이 (단어 수)
            style: 요약 스타일 (bullet_points, paragraph, executive)

        Returns:
            요약된 텍스트
        """
        style_guide = {
            "bullet_points": "핵심 내용을 불릿 포인트로",
            "paragraph": "간결한 문단으로",
            "executive": "임원 요약 형식으로"
        }

        length_instruction = f"최대 {max_length}단어로" if max_length else ""
        style_instruction = style_guide.get(style, "적절한 형식으로")

        prompt = f"""{length_instruction} {style_instruction} 다음 텍스트를 요약하세요:

{text}"""

        try:
            return await self.generate_text(prompt, temperature=0.3)
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            raise

    async def translate(self, text: str, target_language: str) -> str:
        """
        텍스트 번역

        Args:
            text: 번역할 텍스트
            target_language: 목표 언어

        Returns:
            번역된 텍스트
        """
        prompt = f"""다음 텍스트를 {target_language}로 번역하세요:

{text}"""

        try:
            return await self.generate_text(prompt, temperature=0.1)
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            raise

    async def generate_code(
        self,
        description: str,
        language: str = "python"
    ) -> str:
        """
        코드 생성

        Args:
            description: 코드 설명
            language: 프로그래밍 언어

        Returns:
            생성된 코드
        """
        prompt = f"""다음 기능을 {language}로 구현하는 코드를 작성하세요:

{description}

코드만 제공하고, 설명은 주석으로 포함해주세요."""

        try:
            return await self.generate_text(prompt, temperature=0.2)
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise
