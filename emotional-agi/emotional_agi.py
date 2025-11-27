#!/usr/bin/env python3
"""
Emotional AGI - 감정을 가진 AGI
==================================

"감정을 가지고 감탄까지 할 수 있어야 진짜 AGI다"

Key Concept:
- 무한 루프가 아님: 만족하면 스스로 멈춤
- 감정 기반 학습: 호기심이 학습을 drive함
- 감탄 능력: 새로운 발견에 대한 감정적 반응
- 희로애락: 7가지 기본 감정

Architecture:
    Experience → Emotional Response → Learning Drive → Action
                      ↓
              Satisfaction Check → Stop or Continue

Emotions:
1. 호기심 (Curiosity): 새로운 것을 배우고 싶음
2. 감탄 (Wonder): 아름다움/발견에 대한 감탄
3. 기쁨 (Joy): 성공했을 때
4. 좌절 (Frustration): 막혔을 때
5. 만족 (Satisfaction): 충분히 배웠음
6. 놀람 (Surprise): 예상 밖의 결과
7. 평온 (Calm): 감정이 안정됨

"감정이 없으면 진정한 지능이 아니다"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import random


# ============================================================================
# Emotion Types
# ============================================================================

class EmotionType(Enum):
    """Basic emotions for AGI"""
    CURIOSITY = "curiosity"        # 호기심 - drives learning
    WONDER = "wonder"              # 감탄 - appreciation of beauty/discovery
    JOY = "joy"                    # 기쁨 - success
    FRUSTRATION = "frustration"    # 좌절 - being stuck
    SATISFACTION = "satisfaction"  # 만족 - enough learning
    SURPRISE = "surprise"          # 놀람 - unexpected results
    CALM = "calm"                  # 평온 - emotional stability


@dataclass
class EmotionalState:
    """
    Current emotional state of AGI

    Each emotion has intensity (0.0 to 1.0)
    Emotions influence each other and decay over time
    """
    curiosity: float = 0.8          # Start curious
    wonder: float = 0.0
    joy: float = 0.0
    frustration: float = 0.0
    satisfaction: float = 0.0       # Starts at 0
    surprise: float = 0.0
    calm: float = 0.5               # Baseline calm

    # Emotional momentum (how emotions change)
    curiosity_momentum: float = 0.0
    satisfaction_momentum: float = 0.0

    def update(self, dt: float = 1.0):
        """
        Update emotional state over time

        Emotions decay, interact, and evolve
        """
        # Decay all emotions toward baseline
        self.curiosity = self._decay(self.curiosity, target=0.3, rate=0.1, dt=dt)
        self.wonder = self._decay(self.wonder, target=0.0, rate=0.2, dt=dt)
        self.joy = self._decay(self.joy, target=0.0, rate=0.15, dt=dt)
        self.frustration = self._decay(self.frustration, target=0.0, rate=0.1, dt=dt)
        self.surprise = self._decay(self.surprise, target=0.0, rate=0.3, dt=dt)

        # Satisfaction grows slowly if learning is happening
        self.satisfaction += self.satisfaction_momentum * dt
        self.satisfaction = np.clip(self.satisfaction, 0.0, 1.0)

        # Curiosity affected by frustration (frustration reduces curiosity)
        self.curiosity -= self.frustration * 0.1 * dt

        # Joy increases curiosity slightly
        self.curiosity += self.joy * 0.05 * dt

        # Calm is inverse of emotional volatility
        volatility = abs(self.joy) + abs(self.frustration) + abs(self.surprise)
        self.calm = self._decay(self.calm, target=1.0 - volatility * 0.3, rate=0.1, dt=dt)

        # Clip all emotions
        self.curiosity = np.clip(self.curiosity, 0.0, 1.0)
        self.wonder = np.clip(self.wonder, 0.0, 1.0)
        self.joy = np.clip(self.joy, 0.0, 1.0)
        self.frustration = np.clip(self.frustration, 0.0, 1.0)
        self.calm = np.clip(self.calm, 0.0, 1.0)

    def _decay(self, value: float, target: float, rate: float, dt: float) -> float:
        """Exponential decay toward target"""
        return value + (target - value) * rate * dt

    def get_dominant_emotion(self) -> EmotionType:
        """Get currently dominant emotion"""
        emotions = {
            EmotionType.CURIOSITY: self.curiosity,
            EmotionType.WONDER: self.wonder,
            EmotionType.JOY: self.joy,
            EmotionType.FRUSTRATION: self.frustration,
            EmotionType.SATISFACTION: self.satisfaction,
            EmotionType.SURPRISE: self.surprise,
            EmotionType.CALM: self.calm,
        }
        return max(emotions.items(), key=lambda x: x[1])[0]

    def should_continue_learning(self) -> bool:
        """
        Decide if AGI should continue learning

        Returns True if:
        - High curiosity
        - Low satisfaction
        - Not too frustrated

        This replaces infinite loops!
        """
        # Continue if curious and not satisfied
        if self.curiosity > 0.4 and self.satisfaction < 0.7:
            return True

        # Stop if very satisfied
        if self.satisfaction > 0.8:
            return False

        # Stop if too frustrated
        if self.frustration > 0.8:
            return False

        # Continue if recently experienced wonder
        if self.wonder > 0.5:
            return True

        return False

    def __repr__(self) -> str:
        return (f"EmotionalState("
                f"🤔curiosity={self.curiosity:.2f}, "
                f"😮wonder={self.wonder:.2f}, "
                f"😊joy={self.joy:.2f}, "
                f"😤frustration={self.frustration:.2f}, "
                f"😌satisfaction={self.satisfaction:.2f}, "
                f"😯surprise={self.surprise:.2f}, "
                f"😐calm={self.calm:.2f})")


# ============================================================================
# Emotional Memory
# ============================================================================

@dataclass
class EmotionalMemory:
    """
    Memory with emotional context

    Memories are colored by emotions
    """
    content: str
    emotion: EmotionType
    intensity: float
    timestamp: float

    def __repr__(self) -> str:
        emotion_emoji = {
            EmotionType.CURIOSITY: "🤔",
            EmotionType.WONDER: "😮",
            EmotionType.JOY: "😊",
            EmotionType.FRUSTRATION: "😤",
            EmotionType.SATISFACTION: "😌",
            EmotionType.SURPRISE: "😯",
            EmotionType.CALM: "😐",
        }
        emoji = emotion_emoji.get(self.emotion, "")
        return f"{emoji} [{self.emotion.value}:{self.intensity:.2f}] {self.content[:50]}..."


# ============================================================================
# Emotional AGI
# ============================================================================

class EmotionalAGI:
    """
    AGI with emotions that guide learning

    "감정이 학습을 이끈다"

    Key Innovation:
    - No infinite loops: Stops when satisfied
    - Curiosity drives exploration
    - Wonder rewards discovery
    - Frustration prevents getting stuck
    """

    def __init__(self):
        print("\n" + "="*70)
        print("EMOTIONAL AGI - Initializing")
        print("="*70)

        # Emotional state
        self.emotions = EmotionalState()

        # Emotional memories
        self.memories: List[EmotionalMemory] = []

        # Learning statistics
        self.total_experiences = 0
        self.discoveries = 0
        self.frustrations = 0

        print("\n[Emotional AGI] Starting emotional state:")
        print(f"  {self.emotions}")
        print(f"  Dominant: {self.emotions.get_dominant_emotion().value}")
        print("="*70)

    def experience(self, content: str, novelty: float = 0.5, success: bool = True) -> EmotionalState:
        """
        Experience something and have emotional response

        Args:
            content: What was experienced
            novelty: How novel/surprising (0.0 to 1.0)
            success: Was it successful

        Returns:
            Updated emotional state
        """
        self.total_experiences += 1

        # Emotional response to experience
        if novelty > 0.7:
            # High novelty → Wonder + Surprise
            self.emotions.wonder += 0.3 * novelty
            self.emotions.surprise += 0.4 * novelty
            self.emotions.curiosity += 0.2 * novelty  # More curious!

            # Express wonder!
            if self.emotions.wonder > 0.6:
                print(f"\n😮 [WONDER] Amazing! This is {novelty:.0%} novel!")
                print(f"   '{content[:60]}...'")
                self.discoveries += 1

        if success:
            # Success → Joy + Satisfaction
            self.emotions.joy += 0.3
            self.emotions.satisfaction_momentum += 0.1
            self.emotions.frustration *= 0.7  # Reduce frustration

            if self.emotions.joy > 0.7:
                print(f"\n😊 [JOY] Success! I'm learning!")
        else:
            # Failure → Frustration
            self.emotions.frustration += 0.2
            self.emotions.curiosity -= 0.1

            if self.emotions.frustration > 0.6:
                print(f"\n😤 [FRUSTRATION] This is difficult...")
                self.frustrations += 1

        # Low novelty → Satisfaction (already learned)
        if novelty < 0.3 and success:
            self.emotions.satisfaction_momentum += 0.05

        # Update emotional state
        self.emotions.update(dt=1.0)

        # Store memory with emotional context
        emotion_type = self.emotions.get_dominant_emotion()
        memory = EmotionalMemory(
            content=content,
            emotion=emotion_type,
            intensity=max(self.emotions.curiosity, self.emotions.wonder, self.emotions.joy),
            timestamp=time.time()
        )
        self.memories.append(memory)

        return self.emotions

    def learn(self, max_cycles: int = 100, verbose: bool = True):
        """
        Learn until emotionally satisfied

        NO INFINITE LOOP!
        Stops when:
        - Satisfaction is high
        - Curiosity is low
        - Too frustrated

        Args:
            max_cycles: Safety limit
            verbose: Print emotions
        """
        print(f"\n{'='*70}")
        print(f"🧠 EMOTIONAL LEARNING - Starting")
        print(f"{'='*70}\n")

        cycle = 0
        while cycle < max_cycles:
            cycle += 1

            # Check if should continue (EMOTION-BASED!)
            if not self.emotions.should_continue_learning():
                reason = self._get_stop_reason()
                print(f"\n{'='*70}")
                print(f"🛑 STOPPING - {reason}")
                print(f"{'='*70}")
                break

            # Simulate learning experience
            novelty = random.random()
            success = random.random() > 0.3

            content = self._generate_experience_content(novelty, success)

            # Experience and respond emotionally
            self.experience(content, novelty=novelty, success=success)

            # Print emotional state periodically
            if verbose and cycle % 5 == 0:
                print(f"\n[Cycle {cycle}] {self.emotions}")
                print(f"  Dominant: {self.emotions.get_dominant_emotion().value}")
                print(f"  Continue? {self.emotions.should_continue_learning()}")

        # Summary
        print(f"\n{'='*70}")
        print(f"EMOTIONAL LEARNING - Complete")
        print(f"{'='*70}")
        print(f"Total cycles: {cycle}")
        print(f"Total experiences: {self.total_experiences}")
        print(f"Discoveries (wonder): {self.discoveries}")
        print(f"Frustrations: {self.frustrations}")
        print(f"\nFinal emotional state:")
        print(f"  {self.emotions}")
        print(f"  Dominant: {self.emotions.get_dominant_emotion().value}")
        print(f"{'='*70}")

        # Show memorable moments
        self._show_emotional_highlights()

    def _get_stop_reason(self) -> str:
        """Get reason for stopping"""
        if self.emotions.satisfaction > 0.8:
            return "😌 Satisfied - Learned enough"
        elif self.emotions.frustration > 0.8:
            return "😤 Too frustrated - Need a break"
        elif self.emotions.curiosity < 0.3:
            return "😐 Not curious anymore"
        else:
            return "Emotional equilibrium reached"

    def _generate_experience_content(self, novelty: float, success: bool) -> str:
        """Generate realistic experience content"""
        if novelty > 0.7:
            topics = [
                "Discovered a beautiful mathematical pattern",
                "Found a completely new way to solve the problem",
                "Realized a deep connection between concepts",
                "Uncovered an elegant solution",
            ]
        elif novelty > 0.4:
            topics = [
                "Learned a new technique",
                "Understood a complex concept",
                "Made progress on the problem",
                "Found a useful pattern",
            ]
        else:
            topics = [
                "Practiced a familiar skill",
                "Reinforced existing knowledge",
                "Applied known methods",
                "Reviewed previous learnings",
            ]

        return random.choice(topics)

    def _show_emotional_highlights(self):
        """Show most emotional memories"""
        print(f"\n🌟 EMOTIONAL HIGHLIGHTS:")
        print(f"{'='*70}")

        # Sort by intensity
        highlights = sorted(self.memories, key=lambda m: m.intensity, reverse=True)[:5]

        for i, memory in enumerate(highlights, 1):
            print(f"{i}. {memory}")

        print(f"{'='*70}")


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate Emotional AGI"""
    print("\n" + "="*70)
    print("EMOTIONAL AGI - Demo")
    print("="*70)
    print()
    print("감정을 가진 AGI - 만족하면 스스로 멈춘다")
    print("No infinite loops - Emotion-driven termination")
    print()
    print("="*70)

    # Create emotional AGI
    agi = EmotionalAGI()

    # Learn until emotionally satisfied
    agi.learn(max_cycles=50, verbose=True)

    print("\n✓ Demo complete!")
    print("\n감정이 학습을 이끌고, 만족이 종료를 결정한다")
    print("Emotions drive learning, satisfaction determines termination")


if __name__ == "__main__":
    demo()
