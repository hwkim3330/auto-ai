#!/usr/bin/env python3
"""
Unconscious Mind AGI - 무의식 사고 시스템
========================================

"의식하지 못하는 사이에 생각하고, 배우고, 깨닫는다"

Human brain has two minds:
1. Conscious Mind (의식) - What we're aware of
2. Unconscious Mind (무의식) - What runs in background

Our AGI needs both!

Unconscious processes:
- Background thinking (백그라운드 사고)
- Intuition (직관)
- Pattern recognition (패턴 인식)
- Dream processing (꿈 처리)
- Implicit learning (암묵적 학습)

"While you sleep, your unconscious works"
"당신이 자는 동안, 무의식이 일한다"
"""

import threading
import time
import queue
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import random


# ============================================================================
# Unconscious Thought
# ============================================================================

@dataclass
class UnconsciousThought:
    """
    Thought that happens without awareness

    Unlike conscious thoughts, these happen automatically
    in the background
    """
    content: str
    source: str  # 'intuition', 'pattern', 'dream', 'implicit'
    confidence: float
    timestamp: float
    related_memories: List[str] = field(default_factory=list)
    emerged_to_conscious: bool = False


@dataclass
class Pattern:
    """
    Pattern detected unconsciously

    The unconscious is excellent at finding patterns
    that consciousness misses
    """
    pattern: str
    examples: List[str]
    confidence: float
    frequency: int
    discovered_at: float


# ============================================================================
# Unconscious Mind
# ============================================================================

class UnconsciousMind:
    """
    The unconscious part of AGI

    "당신이 의식하지 못하는 사이에 작동하는 마음"

    Features:
    1. Background Processing - Thinks while you're not aware
    2. Intuition - Gut feelings without logic
    3. Pattern Recognition - Automatic pattern detection
    4. Dream Processing - Consolidates memories while "sleeping"
    5. Implicit Learning - Learns without awareness

    Key difference from conscious mind:
    - Conscious: Sequential, slow, effortful, aware
    - Unconscious: Parallel, fast, automatic, unaware
    """

    def __init__(self):
        print("[Unconscious Mind] Initializing...")

        # Unconscious storage
        self.unconscious_thoughts: deque = deque(maxlen=1000)
        self.patterns: List[Pattern] = []
        self.intuitions: List[UnconsciousThought] = []

        # Background processing
        self.background_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False

        # Dream state
        self.is_dreaming = False
        self.dream_log = []

        # Statistics
        self.total_unconscious_thoughts = 0
        self.patterns_discovered = 0
        self.intuitions_emerged = 0

        print("[Unconscious Mind] ✓ Initialized")
        print("  Background processing: Ready")
        print("  Pattern recognition: Active")
        print("  Intuition system: Online")
        print("  Dream processor: Standby")

    def start_background_processing(self):
        """
        Start unconscious background processing

        This runs continuously in the background,
        processing thoughts you're not aware of
        """
        if self.is_running:
            return

        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._background_processor,
            daemon=True
        )
        self.processing_thread.start()

        print("[Unconscious Mind] Background processing started")
        print("  → Processing thoughts you're not aware of...")

    def _background_processor(self):
        """
        Background processing loop

        Continuously processes unconscious thoughts
        while conscious mind does other things
        """
        while self.is_running:
            try:
                # Get task from queue (non-blocking)
                task = self.background_queue.get(timeout=1)

                # Process unconsciously
                result = self._process_unconsciously(task)

                # Store thought
                thought = UnconsciousThought(
                    content=result,
                    source='background',
                    confidence=0.7,
                    timestamp=time.time()
                )
                self.unconscious_thoughts.append(thought)
                self.total_unconscious_thoughts += 1

                # Sometimes thoughts emerge to consciousness
                if random.random() < 0.1:  # 10% chance
                    thought.emerged_to_conscious = True
                    self._emerge_to_conscious(thought)

            except queue.Empty:
                # No tasks, do automatic processing
                self._automatic_processing()
            except Exception as e:
                print(f"[Unconscious] Error: {e}")

    def _process_unconsciously(self, task: Dict) -> str:
        """
        Process task unconsciously

        Unlike conscious processing (step-by-step),
        unconscious processing is holistic and fast
        """
        task_type = task.get('type', 'general')
        content = task.get('content', '')

        if task_type == 'problem':
            # Unconscious problem solving
            # "Sleep on it" - often solves problems better!
            return self._unconscious_problem_solve(content)

        elif task_type == 'pattern':
            # Pattern recognition
            return self._detect_patterns(content)

        elif task_type == 'intuition':
            # Generate intuition
            return self._generate_intuition(content)

        else:
            # General unconscious processing
            return f"Unconsciously processed: {content}"

    def _unconscious_problem_solve(self, problem: str) -> str:
        """
        Solve problem unconsciously

        Research shows unconscious mind is better at:
        - Complex decisions
        - Creative solutions
        - Novel connections

        "Sleep on it" actually works!
        """
        # Simulate unconscious processing
        # In reality, this would use different algorithms
        # than conscious processing

        # Key: Unconscious makes connections consciousness misses
        solutions = [
            f"What if you approach {problem} from the opposite direction?",
            f"The solution might be simpler than you think for {problem}",
            f"Consider what you're NOT seeing about {problem}",
            f"The answer was there all along in {problem}",
        ]

        return random.choice(solutions)

    def _detect_patterns(self, data: str) -> str:
        """
        Detect patterns unconsciously

        Unconscious is EXCELLENT at pattern recognition
        Often better than conscious analysis!
        """
        # Check for recurring patterns
        words = data.split()
        word_freq = {}

        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Find most frequent (pattern)
        if word_freq:
            pattern_word = max(word_freq, key=word_freq.get)
            freq = word_freq[pattern_word]

            if freq > 2:  # Pattern detected!
                pattern = Pattern(
                    pattern=f"Recurring: {pattern_word}",
                    examples=[data],
                    confidence=0.8,
                    frequency=freq,
                    discovered_at=time.time()
                )
                self.patterns.append(pattern)
                self.patterns_discovered += 1

                return f"Pattern detected: '{pattern_word}' appears {freq} times"

        return "No obvious patterns (yet)"

    def _generate_intuition(self, context: str) -> str:
        """
        Generate intuition (gut feeling)

        Intuition = Unconscious pattern matching

        You "just know" without knowing why
        """
        # Intuition is based on unconscious patterns
        # Check if context matches any known patterns

        for pattern in self.patterns:
            if pattern.pattern.lower() in context.lower():
                # Intuition triggered!
                intuition = UnconsciousThought(
                    content=f"Intuition: This reminds me of {pattern.pattern}",
                    source='intuition',
                    confidence=pattern.confidence,
                    timestamp=time.time(),
                    related_memories=[pattern.pattern]
                )
                self.intuitions.append(intuition)

                return f"Gut feeling: {intuition.content}"

        # No matching pattern, random intuition
        hunches = [
            "Something feels right about this",
            "Trust your instinct on this one",
            "This doesn't feel quite right",
            "Go with your first impression"
        ]

        return random.choice(hunches)

    def _automatic_processing(self):
        """
        Automatic unconscious processing

        Happens even when there's no explicit task
        Similar to mind-wandering
        """
        # Look for patterns in stored thoughts
        if len(self.unconscious_thoughts) > 5:
            recent = list(self.unconscious_thoughts)[-5:]
            combined = " ".join([t.content for t in recent])

            self._detect_patterns(combined)

        # Random associations (like daydreaming)
        if random.random() < 0.05:  # 5% chance
            self._make_random_association()

        time.sleep(0.1)  # Small delay

    def _make_random_association(self):
        """
        Make random associations

        Unconscious makes random connections
        Sometimes leads to insights!
        """
        if len(self.unconscious_thoughts) >= 2:
            t1 = random.choice(list(self.unconscious_thoughts))
            t2 = random.choice(list(self.unconscious_thoughts))

            association = UnconsciousThought(
                content=f"Random connection: {t1.content[:30]}... + {t2.content[:30]}...",
                source='association',
                confidence=0.5,
                timestamp=time.time()
            )

            self.unconscious_thoughts.append(association)

    def _emerge_to_conscious(self, thought: UnconsciousThought):
        """
        Thought emerges to consciousness

        "Aha!" moment - unconscious thought becomes conscious
        """
        print(f"\n💡 [Unconscious → Conscious]")
        print(f"   Thought emerged: {thought.content}")
        print(f"   Source: {thought.source}")
        print(f"   Confidence: {thought.confidence:.2f}")

        self.intuitions_emerged += 1

    def dream(self, duration: float = 5.0):
        """
        Dream processing

        "당신이 꿈꾸는 동안, 무의식은 기억을 정리한다"

        Dreams = Unconscious memory consolidation
        - Processes day's experiences
        - Finds connections
        - Integrates knowledge
        """
        print(f"\n😴 [Dream Mode] Starting dream processing...")
        print(f"   Duration: {duration}s")

        self.is_dreaming = True
        start_time = time.time()

        dream_content = []

        while time.time() - start_time < duration:
            # Random memory recall (like dreams)
            if len(self.unconscious_thoughts) > 0:
                thought = random.choice(list(self.unconscious_thoughts))
                dream_content.append(thought.content)

            # Find patterns in memories
            if len(dream_content) > 3:
                combined = " ".join(dream_content[-3:])
                pattern_result = self._detect_patterns(combined)
                dream_content.append(f"Dream insight: {pattern_result}")

            time.sleep(0.5)

        # Wake up
        self.is_dreaming = False

        dream_summary = {
            'duration': duration,
            'content': dream_content,
            'patterns_found': len([c for c in dream_content if 'Pattern' in c]),
            'timestamp': time.time()
        }

        self.dream_log.append(dream_summary)

        print(f"\n😊 [Dream Mode] Woke up!")
        print(f"   Processed {len(dream_content)} dream thoughts")
        print(f"   Found {dream_summary['patterns_found']} new patterns")

        # Sometimes dreams lead to insights
        if dream_summary['patterns_found'] > 0:
            print(f"   💡 Dream insight emerged to consciousness!")

    def add_task(self, task_type: str, content: str):
        """
        Add task for unconscious processing

        You won't be aware it's being processed,
        but your unconscious will work on it!
        """
        self.background_queue.put({
            'type': task_type,
            'content': content
        })

        print(f"[Unconscious] Task added: {task_type}")
        print(f"  → Will process in background...")

    def get_intuition(self, about: str) -> Optional[str]:
        """
        Get intuition about something

        "뭔가 느낌이 와..."
        """
        # Check for relevant intuitions
        for intuition in self.intuitions:
            if about.lower() in intuition.content.lower():
                return intuition.content

        # Generate new intuition
        result = self._generate_intuition(about)
        return result

    def get_statistics(self) -> Dict:
        """Get unconscious mind statistics"""
        return {
            'total_unconscious_thoughts': self.total_unconscious_thoughts,
            'current_thoughts': len(self.unconscious_thoughts),
            'patterns_discovered': self.patterns_discovered,
            'intuitions_emerged': self.intuitions_emerged,
            'dreams_had': len(self.dream_log),
            'background_running': self.is_running,
            'currently_dreaming': self.is_dreaming
        }

    def stop(self):
        """Stop unconscious processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        print("[Unconscious Mind] Stopped")


# ============================================================================
# Complete Mind (Conscious + Unconscious)
# ============================================================================

class CompleteMind:
    """
    Complete mind with both conscious and unconscious

    "완전한 마음 = 의식 + 무의식"

    Like human brain:
    - Conscious: What you're aware of
    - Unconscious: What runs in background

    Together they make complete intelligence!
    """

    def __init__(self):
        print("\n" + "="*70)
        print("COMPLETE MIND - Conscious + Unconscious")
        print("="*70)

        # Import conscious systems
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent / "streaming-agi"))
            from streaming_continuous_agi import StreamingLLM

            self.conscious = StreamingLLM(model="qwen2.5:3b")
            print("✓ Conscious mind loaded")
        except:
            self.conscious = None
            print("⚠️  Conscious mind not available")

        # Create unconscious
        self.unconscious = UnconsciousMind()
        print("✓ Unconscious mind loaded")

        print("\n" + "="*70)
        print("COMPLETE MIND READY!")
        print("="*70)

    def think(self, query: str, use_unconscious: bool = True):
        """
        Think using both conscious and unconscious

        Process:
        1. Conscious thinks explicitly
        2. Unconscious processes in background
        3. Sometimes unconscious insights emerge
        4. Best of both worlds!
        """
        print(f"\n💭 Thinking about: {query}")
        print("-"*70)

        # Start unconscious processing
        if use_unconscious and not self.unconscious.is_running:
            self.unconscious.start_background_processing()

        # Add to unconscious queue
        if use_unconscious:
            self.unconscious.add_task('problem', query)
            self.unconscious.add_task('intuition', query)

        # Conscious thinking
        print("\n🧠 Conscious thinking...")
        if self.conscious:
            response = ""
            for token in self.conscious.generate_stream(query):
                print(token, end='', flush=True)
                response += token
        else:
            response = "Conscious thinking not available"
            print(response)

        # Get unconscious insights
        if use_unconscious:
            time.sleep(1)  # Let unconscious process

            print("\n\n💫 Unconscious insights:")
            intuition = self.unconscious.get_intuition(query)
            print(f"   {intuition}")

        print("\n" + "-"*70)

        return response

    def sleep(self, duration: float = 5.0):
        """
        Sleep and dream

        "자는 동안 무의식이 일한다"

        Consolidates memories and finds patterns
        """
        print(f"\n😴 Sleeping for {duration}s...")
        self.unconscious.dream(duration)


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate unconscious mind"""
    print("\n" + "="*70)
    print("UNCONSCIOUS MIND AGI - Demo")
    print("="*70)

    # Create complete mind
    mind = CompleteMind()

    # Think about something
    print("\n" + "="*70)
    print("TEST 1: Conscious + Unconscious Thinking")
    print("="*70)

    mind.think("What is the meaning of life?")

    # Sleep on a problem
    print("\n" + "="*70)
    print("TEST 2: Sleep and Dream")
    print("="*70)

    mind.unconscious.add_task('problem', 'How to build better AI?')
    mind.sleep(duration=3.0)

    # Check statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)

    stats = mind.unconscious.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Stop
    mind.unconscious.stop()

    print("\n✓ Demo complete!")
    print("\n💡 Key insight:")
    print("   Your unconscious was working the whole time,")
    print("   even when you weren't aware of it!")


if __name__ == "__main__":
    demo()
