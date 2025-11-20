"""
Multi-AI Orchestrator

여러 AI 모델을 통합하여 최적의 결과 도출:
- OpenAI Codex (Code generation, reasoning)
- Google Gemini (Multimodal, prediction)
- Anthropic Claude (Analysis, planning)

각 AI의 강점을 활용한 오케스트레이션
"""

import os
from typing import Dict, List, Optional, Any
import json
import asyncio


class MultiAIOrchestrator:
    """
    Multi-AI Orchestration System

    Features:
    - 작업에 따라 최적의 AI 선택
    - 여러 AI의 응답 결합
    - Consensus 기반 의사결정
    - 성능 모니터링
    """

    def __init__(self):
        # API Keys (환경 변수에서 로드)
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

        # AI Clients
        self.clients = {}
        self._init_clients()

        # Performance tracking
        self.stats = {
            'codex': {'calls': 0, 'successes': 0, 'latency': []},
            'gemini': {'calls': 0, 'successes': 0, 'latency': []},
            'claude': {'calls': 0, 'successes': 0, 'latency': []}
        }

        # Task routing rules
        self.routing_rules = {
            'code_generation': ['codex', 'claude'],
            'image_analysis': ['gemini', 'claude'],
            'reasoning': ['claude', 'gemini', 'codex'],
            'prediction': ['gemini', 'claude'],
            'planning': ['claude', 'codex'],
            'optimization': ['codex', 'claude']
        }

    def _init_clients(self):
        """AI 클라이언트 초기화"""
        try:
            if self.openai_api_key:
                import openai
                openai.api_key = self.openai_api_key
                self.clients['codex'] = openai
                print("✅ OpenAI Codex initialized")
        except Exception as e:
            print(f"⚠️ Codex init failed: {e}")

        try:
            if self.google_api_key:
                import google.generativeai as genai
                genai.configure(api_key=self.google_api_key)
                self.clients['gemini'] = genai.GenerativeModel('gemini-pro')
                print("✅ Google Gemini initialized")
        except Exception as e:
            print(f"⚠️ Gemini init failed: {e}")

        try:
            if self.anthropic_api_key:
                import anthropic
                self.clients['claude'] = anthropic.Anthropic(api_key=self.anthropic_api_key)
                print("✅ Anthropic Claude initialized")
        except Exception as e:
            print(f"⚠️ Claude init failed: {e}")

    def select_ai(self, task_type: str, preference: Optional[str] = None) -> str:
        """
        작업에 맞는 AI 선택

        Args:
            task_type: 작업 유형
            preference: 선호하는 AI (optional)

        Returns:
            선택된 AI 이름
        """
        if preference and preference in self.clients:
            return preference

        # Routing rules에서 우선순위 가져오기
        candidates = self.routing_rules.get(task_type, ['claude', 'gemini', 'codex'])

        # 사용 가능한 AI 중에서 선택
        for ai in candidates:
            if ai in self.clients:
                return ai

        # Fallback
        if self.clients:
            return list(self.clients.keys())[0]

        return None

    async def query_codex(self, prompt: str, **kwargs) -> Dict:
        """Codex 쿼리"""
        if 'codex' not in self.clients:
            return {'error': 'Codex not available'}

        try:
            import time
            start = time.time()

            response = self.clients['codex'].ChatCompletion.create(
                model="gpt-4-turbo-preview",  # Latest Codex model
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )

            latency = time.time() - start

            self.stats['codex']['calls'] += 1
            self.stats['codex']['successes'] += 1
            self.stats['codex']['latency'].append(latency)

            return {
                'ai': 'codex',
                'response': response.choices[0].message.content,
                'latency': latency,
                'model': 'gpt-4-turbo-preview'
            }

        except Exception as e:
            self.stats['codex']['calls'] += 1
            return {'error': str(e), 'ai': 'codex'}

    async def query_gemini(self, prompt: str, **kwargs) -> Dict:
        """Gemini 쿼리"""
        if 'gemini' not in self.clients:
            return {'error': 'Gemini not available'}

        try:
            import time
            start = time.time()

            response = self.clients['gemini'].generate_content(prompt)

            latency = time.time() - start

            self.stats['gemini']['calls'] += 1
            self.stats['gemini']['successes'] += 1
            self.stats['gemini']['latency'].append(latency)

            return {
                'ai': 'gemini',
                'response': response.text,
                'latency': latency,
                'model': 'gemini-pro'
            }

        except Exception as e:
            self.stats['gemini']['calls'] += 1
            return {'error': str(e), 'ai': 'gemini'}

    async def query_claude(self, prompt: str, **kwargs) -> Dict:
        """Claude 쿼리"""
        if 'claude' not in self.clients:
            return {'error': 'Claude not available'}

        try:
            import time
            start = time.time()

            message = self.clients['claude'].messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            latency = time.time() - start

            self.stats['claude']['calls'] += 1
            self.stats['claude']['successes'] += 1
            self.stats['claude']['latency'].append(latency)

            return {
                'ai': 'claude',
                'response': message.content[0].text,
                'latency': latency,
                'model': 'claude-3-opus'
            }

        except Exception as e:
            self.stats['claude']['calls'] += 1
            return {'error': str(e), 'ai': 'claude'}

    async def query(self, prompt: str, task_type: str = 'reasoning',
                    preference: Optional[str] = None, **kwargs) -> Dict:
        """
        통합 쿼리 인터페이스

        Args:
            prompt: 프롬프트
            task_type: 작업 유형
            preference: 선호 AI (optional)

        Returns:
            응답 딕셔너리
        """
        ai = self.select_ai(task_type, preference)

        if not ai:
            return {'error': 'No AI available'}

        # Route to appropriate AI
        if ai == 'codex':
            return await self.query_codex(prompt, **kwargs)
        elif ai == 'gemini':
            return await self.query_gemini(prompt, **kwargs)
        elif ai == 'claude':
            return await self.query_claude(prompt, **kwargs)

    async def consensus_query(self, prompt: str, task_type: str = 'reasoning') -> Dict:
        """
        여러 AI에게 동시 쿼리하고 consensus 도출

        Args:
            prompt: 프롬프트
            task_type: 작업 유형

        Returns:
            Consensus 결과
        """
        candidates = self.routing_rules.get(task_type, list(self.clients.keys()))
        available = [ai for ai in candidates if ai in self.clients]

        if not available:
            return {'error': 'No AI available'}

        # 병렬 쿼리
        tasks = []
        for ai in available:
            tasks.append(self.query(prompt, task_type, preference=ai))

        responses = await asyncio.gather(*tasks)

        # 성공한 응답만 필터링
        valid_responses = [r for r in responses if 'error' not in r]

        if not valid_responses:
            return {'error': 'All queries failed', 'responses': responses}

        # Consensus (간단 버전: 가장 빠른 응답 사용)
        # 더 정교한 방법: 응답들을 비교하고 투표
        fastest = min(valid_responses, key=lambda x: x['latency'])

        return {
            'consensus_response': fastest['response'],
            'selected_ai': fastest['ai'],
            'all_responses': valid_responses,
            'num_ais': len(valid_responses)
        }

    def get_stats(self) -> Dict:
        """통계 반환"""
        stats_summary = {}

        for ai, data in self.stats.items():
            avg_latency = (
                sum(data['latency']) / len(data['latency'])
                if data['latency'] else 0
            )

            success_rate = (
                data['successes'] / data['calls'] * 100
                if data['calls'] > 0 else 0
            )

            stats_summary[ai] = {
                'total_calls': data['calls'],
                'successes': data['successes'],
                'success_rate': success_rate,
                'avg_latency': avg_latency
            }

        return stats_summary


# Singleton instance
_orchestrator = None


def get_orchestrator() -> MultiAIOrchestrator:
    """Global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAIOrchestrator()
    return _orchestrator


if __name__ == "__main__":
    print("=== Multi-AI Orchestrator Test ===\n")

    orchestrator = MultiAIOrchestrator()

    print("\nAvailable AIs:")
    for ai in orchestrator.clients.keys():
        print(f"  ✓ {ai}")

    print("\nRouting Rules:")
    for task, ais in orchestrator.routing_rules.items():
        print(f"  {task}: {', '.join(ais)}")

    print("\n✅ Multi-AI Orchestrator ready")
    print("\nUsage example:")
    print("  result = await orchestrator.query('Explain AI', task_type='reasoning')")
    print("  consensus = await orchestrator.consensus_query('Solve problem', task_type='reasoning')")
