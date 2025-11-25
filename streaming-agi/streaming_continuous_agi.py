#!/usr/bin/env python3
"""
Streaming Continuous AGI
========================

"생각하는 도중에 결과가 나오고, 결과가 나오는 중에도 계속 생각한다"

핵심 개념:
1. Streaming Generation - 토큰 단위 실시간 출력
2. Continuous Thinking - 멈추지 않는 연속적 사고
3. Parallel Processing - 생각과 출력 동시 진행
4. Recursive Refinement - 생각을 바탕으로 더 깊은 생각

"진정한 AGI는 멈추지 않고 계속 생각한다"
"""

import requests
import json
import time
import threading
from queue import Queue, Empty
from typing import Generator, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
import sys


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Thought:
    """하나의 생각"""
    id: int
    depth: int  # 생각의 깊이 (0 = 초기, 1 = 1차 반성, 2 = 2차 반성...)
    content: str
    timestamp: float
    parent_id: Optional[int] = None
    children: List[int] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class StreamChunk:
    """스트림 청크"""
    thought_id: int
    token: str
    timestamp: float
    is_thought: bool  # True = 사고 과정, False = 최종 출력


# ============================================================================
# Streaming LLM Client
# ============================================================================

class StreamingLLM:
    """
    Ollama를 사용한 스트리밍 LLM 클라이언트

    토큰 단위로 실시간 생성
    """

    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

        print(f"[StreamingLLM] Using model: {model}")

    def generate_stream(self, prompt: str, system: Optional[str] = None) -> Generator[str, None, None]:
        """
        스트리밍 생성

        Args:
            prompt: 프롬프트
            system: 시스템 프롬프트

        Yields:
            토큰 단위 텍스트
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=60
            )

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]

                    if chunk.get("done", False):
                        break

        except Exception as e:
            print(f"\n[StreamingLLM] Error: {e}")
            yield f"\n[Error: {e}]"


# ============================================================================
# Continuous Thinking Engine
# ============================================================================

class ContinuousThinkingEngine:
    """
    연속적 사고 엔진

    "생각 → 출력 → 생각 → 출력" 동시 진행
    """

    def __init__(self, model: str = "qwen2.5:3b"):
        self.llm = StreamingLLM(model=model)
        self.thoughts: List[Thought] = []
        self.thought_id_counter = 0

        # 병렬 처리를 위한 큐
        self.output_queue = Queue()
        self.thinking_queue = Queue()

        print("[ContinuousThinking] Engine initialized")

    def _generate_thought_id(self) -> int:
        """Thought ID 생성"""
        self.thought_id_counter += 1
        return self.thought_id_counter

    def think_continuous(self, query: str, max_depth: int = 3) -> Generator[StreamChunk, None, None]:
        """
        연속적 사고

        Args:
            query: 질문
            max_depth: 최대 사고 깊이

        Yields:
            StreamChunk (실시간 생각 + 출력)
        """
        print(f"\n{'='*70}")
        print(f"🧠 CONTINUOUS THINKING - Starting")
        print(f"{'='*70}\n")

        # 초기 생각 시작
        yield from self._think_level(query, depth=0, max_depth=max_depth)

    def _think_level(self, query: str, depth: int, max_depth: int,
                     parent_id: Optional[int] = None) -> Generator[StreamChunk, None, None]:
        """
        특정 깊이의 사고

        Args:
            query: 질문/사고 대상
            depth: 현재 깊이
            max_depth: 최대 깊이
            parent_id: 부모 생각 ID
        """
        if depth > max_depth:
            return

        # 현재 생각 ID
        thought_id = self._generate_thought_id()

        # 프롬프트 구성
        if depth == 0:
            # 초기 사고
            system = "You are a continuously thinking AGI. Think step by step and show your reasoning process."
            prompt = f"""Question: {query}

Think deeply about this question. Show your reasoning process step by step.

Your response should include:
1. Initial thoughts
2. Analysis
3. Deeper reflection
4. Conclusion

Begin thinking:"""
        else:
            # 깊은 사고 (이전 생각을 바탕으로)
            parent_thought = next((t for t in self.thoughts if t.id == parent_id), None)
            system = "You are refining your previous thoughts. Think more deeply."
            prompt = f"""Previous thought:
{parent_thought.content if parent_thought else 'N/A'}

Now think MORE DEEPLY about this. What did you miss? What can be refined?

Continue thinking:"""

        # 사고 깊이 표시
        indent = "  " * depth
        depth_marker = ["🌱", "🌿", "🌳", "🌲"][min(depth, 3)]

        print(f"{indent}{depth_marker} [Depth {depth}] Thinking...")

        # 스트리밍 생성
        thought_content = ""
        for token in self.llm.generate_stream(prompt, system=system):
            thought_content += token

            # 실시간 출력
            print(token, end="", flush=True)

            # StreamChunk 생성
            chunk = StreamChunk(
                thought_id=thought_id,
                token=token,
                timestamp=time.time(),
                is_thought=(depth > 0)  # depth > 0이면 사고 과정
            )

            yield chunk

        print()  # 줄바꿈

        # Thought 저장
        thought = Thought(
            id=thought_id,
            depth=depth,
            content=thought_content.strip(),
            timestamp=time.time(),
            parent_id=parent_id
        )
        self.thoughts.append(thought)

        if parent_id is not None:
            parent = next((t for t in self.thoughts if t.id == parent_id), None)
            if parent:
                parent.children.append(thought_id)

        # 다음 깊이로 (연속적 사고)
        if depth < max_depth:
            print(f"\n{indent}💭 Reflecting deeper...\n")
            time.sleep(0.5)  # 잠시 대기 (사고 간격)

            yield from self._think_level(
                query=query,
                depth=depth + 1,
                max_depth=max_depth,
                parent_id=thought_id
            )

    def get_thought_tree(self) -> str:
        """사고 트리 시각화"""
        lines = []
        lines.append("\n" + "="*70)
        lines.append("🌳 THOUGHT TREE")
        lines.append("="*70)

        def print_thought(thought_id: int, indent: int = 0):
            thought = next((t for t in self.thoughts if t.id == thought_id), None)
            if not thought:
                return

            prefix = "  " * indent
            marker = ["🌱", "🌿", "🌳", "🌲"][min(thought.depth, 3)]

            lines.append(f"{prefix}{marker} [ID:{thought.id}] Depth {thought.depth}")
            lines.append(f"{prefix}   {thought.content[:100]}...")

            for child_id in thought.children:
                print_thought(child_id, indent + 1)

        # 루트 thoughts 찾기
        roots = [t for t in self.thoughts if t.parent_id is None]
        for root in roots:
            print_thought(root.id)

        lines.append("="*70)
        return "\n".join(lines)


# ============================================================================
# Parallel Thinking-Output System
# ============================================================================

class ParallelThinkingAGI:
    """
    병렬 사고-출력 AGI

    "생각하면서 출력하고, 출력하면서 생각한다"
    """

    def __init__(self, model: str = "qwen2.5:3b"):
        self.engine = ContinuousThinkingEngine(model=model)
        self.output_buffer = []

        print("[ParallelThinkingAGI] Initialized")

    def think(self, query: str, max_depth: int = 3, verbose: bool = True):
        """
        질문에 대해 생각 (병렬 방식)

        Args:
            query: 질문
            max_depth: 사고 깊이
            verbose: 상세 출력
        """
        print(f"\n{'='*70}")
        print(f"💡 Question: {query}")
        print(f"{'='*70}\n")

        # 연속적 사고 시작
        for chunk in self.engine.think_continuous(query, max_depth=max_depth):
            # 최종 출력 (depth 0)만 버퍼에 저장
            if not chunk.is_thought:
                self.output_buffer.append(chunk.token)

        # 사고 트리 출력
        if verbose:
            print(self.engine.get_thought_tree())

        # 최종 답변
        final_answer = "".join(self.output_buffer)

        print(f"\n{'='*70}")
        print(f"✨ FINAL ANSWER")
        print(f"{'='*70}")
        print(final_answer)
        print(f"{'='*70}\n")

        return {
            "query": query,
            "answer": final_answer,
            "thoughts": self.engine.thoughts,
            "total_thoughts": len(self.engine.thoughts)
        }


# ============================================================================
# Interactive Demo
# ============================================================================

def demo_streaming_agi():
    """스트리밍 AGI 데모"""
    print("\n" + "="*70)
    print("🧠 STREAMING CONTINUOUS AGI - Demo")
    print("="*70)
    print()
    print("핵심 개념:")
    print("  1. 생각하는 도중에 결과가 나온다")
    print("  2. 결과가 나오는 중에도 계속 생각한다")
    print("  3. 생각 → 출력 → 생각 → 출력 (병렬 진행)")
    print("="*70)
    print()

    # AGI 생성
    agi = ParallelThinkingAGI(model="qwen2.5:3b")

    # 테스트 질문들
    questions = [
        "What is consciousness?",
        "How can AI become truly intelligent?",
        "What is the meaning of life?"
    ]

    print("Select a question (or press Enter for custom):")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
    print()

    choice = input("Choice (1-3 or Enter): ").strip()

    if choice.isdigit() and 1 <= int(choice) <= 3:
        query = questions[int(choice) - 1]
    else:
        query = input("Your question: ").strip() or questions[0]

    # 사고 시작
    start_time = time.time()

    result = agi.think(
        query=query,
        max_depth=2,  # 3단계 깊이 (0, 1, 2)
        verbose=True
    )

    elapsed = time.time() - start_time

    # 통계
    print(f"\n{'='*70}")
    print(f"📊 STATISTICS")
    print(f"{'='*70}")
    print(f"  Query: {query}")
    print(f"  Total thinking time: {elapsed:.2f}s")
    print(f"  Total thoughts: {result['total_thoughts']}")
    print(f"  Thinking depth: {max(t.depth for t in result['thoughts'])}")
    print(f"  Answer length: {len(result['answer'])} chars")
    print(f"{'='*70}\n")


def interactive_mode():
    """대화형 모드"""
    print("\n" + "="*70)
    print("🧠 STREAMING CONTINUOUS AGI - Interactive Mode")
    print("="*70)
    print()
    print("Commands:")
    print("  /quit - Exit")
    print("  /depth N - Set thinking depth (default: 2)")
    print("  /model NAME - Change model")
    print("  /tree - Show thought tree")
    print()

    agi = ParallelThinkingAGI(model="qwen2.5:3b")
    depth = 2

    while True:
        try:
            query = input("\n💭 You: ").strip()

            if not query:
                continue

            if query == "/quit":
                print("\n👋 Goodbye!")
                break

            if query.startswith("/depth "):
                try:
                    depth = int(query.split()[1])
                    print(f"✓ Thinking depth set to {depth}")
                except:
                    print("✗ Invalid depth")
                continue

            if query.startswith("/model "):
                model = query.split()[1]
                agi = ParallelThinkingAGI(model=model)
                print(f"✓ Model changed to {model}")
                continue

            if query == "/tree":
                print(agi.engine.get_thought_tree())
                continue

            # 생각 시작
            print(f"\n🧠 AGI: ", end="", flush=True)

            result = agi.think(query, max_depth=depth, verbose=False)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    """메인 함수"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        demo_streaming_agi()


if __name__ == "__main__":
    main()
