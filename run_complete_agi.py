#!/usr/bin/env python3
"""
Complete AGI System - Master Launcher
======================================

"처음부터 끝까지, 생각하고 느끼고 행동하고 학습하는 완전한 AGI"

Integrates all components:
1. Perception (Real Vision)
2. Cognition (LLM Reasoning)
3. Emotion (7 Emotions + Natural Termination)
4. Action (Thinking + Acting)
5. Memory (Episodic + Semantic)
6. Learning (Self-Supervised)
7. Embodiment (SIMA-style)

Author: Kim Hyunwoo
Date: November 2025
"""

import sys
import os
from pathlib import Path
import time

# Add all component paths
sys.path.append(str(Path(__file__).parent / "streaming-agi"))
sys.path.append(str(Path(__file__).parent / "emotional-agi"))
sys.path.append(str(Path(__file__).parent / "computer-use-ncp"))
sys.path.append(str(Path(__file__).parent / "thinking-actor-agi"))
sys.path.append(str(Path(__file__).parent / "embodied-sima-agent"))
sys.path.append(str(Path(__file__).parent / "neural-circuit-policies"))


def print_banner():
    """Print AGI system banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║                     COMPLETE AGI SYSTEM                               ║
║                                                                       ║
║         "생각하고, 느끼고, 행동하고, 학습하는 완전한 AGI"              ║
║         "Think, Feel, Act, and Learn - Complete AGI"                  ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

System Components:
  1. 👁️  Perception      - Real computer vision (PIL ImageGrab)
  2. 🧠 Cognition       - LLM reasoning (Ollama qwen2.5:3b)
  3. 💖 Emotion         - 7 emotions + natural termination
  4. 🎯 Action          - Parallel thinking + acting
  5. 💾 Memory          - Episodic + semantic
  6. 📊 Evaluation      - LLM self-assessment
  7. 🎮 Embodiment      - SIMA-style multi-environment

Architecture: ~5,200 lines of modular code
Innovation: First open-source SIMA2 with emotion-based control
    """
    print(banner)


def check_dependencies():
    """Check if all required components are available"""
    print("\n" + "="*70)
    print("DEPENDENCY CHECK")
    print("="*70)

    required = {
        'Streaming AGI': 'streaming-agi/streaming_continuous_agi.py',
        'Emotional AGI': 'emotional-agi/emotional_agi.py',
        'Computer Agent': 'computer-use-ncp/computer_agent.py',
        'Thinking Actor': 'thinking-actor-agi/thinking_actor_agi.py',
        'Embodied Agent': 'embodied-sima-agent/embodied_agent.py',
        'NCP': 'neural-circuit-policies/ncp_core.py',
    }

    all_present = True
    for name, path in required.items():
        full_path = Path(__file__).parent / path
        if full_path.exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} NOT FOUND")
            all_present = False

    if not all_present:
        print("\n⚠️  Some components are missing. Please check installation.")
        sys.exit(1)

    print("\n✓ All components present!\n")


def demo_streaming_agi():
    """Demo: Streaming Continuous AGI"""
    print("\n" + "="*70)
    print("DEMO 1: STREAMING CONTINUOUS AGI")
    print("="*70)
    print("\nToken-by-token thinking with parallel reasoning paths")
    print()

    try:
        from streaming_continuous_agi import ParallelThinkingAGI

        agi = ParallelThinkingAGI(model='qwen2.5:3b')
        query = "What are the key components of AGI?"

        print(f"💡 Question: {query}\n")
        print("-"*70)

        start = time.time()
        result = agi.think(query, max_depth=1, verbose=True)
        elapsed = time.time() - start

        print("-"*70)
        print(f"\n✅ Completed in {elapsed:.2f}s")
        print(f"Total thoughts: {result['total_thoughts']}")

    except Exception as e:
        print(f"❌ Error: {e}")


def demo_emotional_agi():
    """Demo: Emotional AGI with Natural Termination"""
    print("\n" + "="*70)
    print("DEMO 2: EMOTIONAL AGI")
    print("="*70)
    print("\n7 emotions with natural termination (no infinite loops!)")
    print()

    try:
        from emotional_agi import EmotionalAGI

        print("Creating Emotional AGI...")
        agi = EmotionalAGI()

        print("\n📊 Initial Emotional State:")
        agi.emotions.display()

        print("\n🎓 Learning cycle (will stop automatically when satisfied)...")
        print("-"*70)

        # Short demo (max 3 cycles)
        agi.learn(max_cycles=3, verbose=True)

        print("-"*70)
        print("\n📊 Final Emotional State:")
        agi.emotions.display()

        stats = agi.get_statistics()
        print(f"\n✅ Statistics:")
        print(f"  Learning cycles: {stats['learning_cycles']}")
        print(f"  Discoveries: {stats['discoveries']}")
        print(f"  Reflections: {stats['reflections']}")

    except Exception as e:
        print(f"❌ Error: {e}")


def demo_computer_agent():
    """Demo: Computer Use Agent with Real Vision"""
    print("\n" + "="*70)
    print("DEMO 3: COMPUTER USE AGENT")
    print("="*70)
    print("\nReal computer vision + NCP neural brain")
    print()

    try:
        from computer_agent import ComputerUseAgent

        print("Creating Computer Use Agent...")
        agent = ComputerUseAgent(use_real_vision=True)

        print(f"\n👁️  Vision System:")
        print(f"  Real vision: {agent.vision.use_real_vision}")
        print(f"  Feature dim: {agent.vision.feature_dim}")
        print(f"  Target size: {agent.vision.target_size}")

        print(f"\n🧠 NCP Brain:")
        print(f"  Total neurons: {agent.brain.total_neurons}")
        print(f"  Total synapses: {agent.brain.total_synapses}")
        print(f"  Sparsity: {agent.brain.sparsity:.1%}")

        print("\n📸 Capturing screenshot...")
        screenshot = agent.vision.capture_screen()
        if screenshot:
            print(f"  ✓ Screenshot captured: {screenshot.size}")
            features = agent.vision.extract_features(screenshot)
            print(f"  ✓ Features extracted: {features.shape}")
        else:
            print("  ⚠️  Screenshot capture not available (headless mode)")

        print("\n✅ Computer Agent ready!")

    except Exception as e:
        print(f"❌ Error: {e}")


def demo_thinking_actor():
    """Demo: Thinking Actor AGI (Parallel Think + Act)"""
    print("\n" + "="*70)
    print("DEMO 4: THINKING ACTOR AGI")
    print("="*70)
    print("\nThinking and acting in parallel")
    print()

    try:
        from thinking_actor_agi import ThinkingActorAGI

        print("Creating Thinking Actor AGI...")
        agi = ThinkingActorAGI(model='qwen2.5:3b')

        print("\n💡 Query: 'How would you open a text editor?'")
        print("-"*70)

        query = """How would you open a text editor?
Describe the steps and include action commands like [ACTION: click(x, y)]."""

        result = agi.think_and_act(query, max_depth=1, verbose=True)

        print("-"*70)
        print(f"\n✅ Result:")
        print(f"  Actions detected: {result['actions_detected']}")
        print(f"  Actions executed: {result['actions_executed']}")

    except Exception as e:
        print(f"❌ Error: {e}")


def demo_embodied_agent():
    """Demo: Complete SIMA-style Embodied Agent"""
    print("\n" + "="*70)
    print("DEMO 5: EMBODIED SIMA AGENT")
    print("="*70)
    print("\nComplete integration: Environment + Memory + Evaluation + Learning")
    print()

    try:
        from embodied_agent import EmbodiedAgent

        print("Creating Embodied Agent...")
        agent = EmbodiedAgent(
            env_config={'type': 'screen'},
            agent_config={
                'llm_model': 'qwen2.5:3b',
                'use_emotions': True
            }
        )

        print("\n🎯 Task: Observe the current screen")
        print("-"*70)

        # Simple task for demo
        result = agent.execute_task(
            task_description="Observe the current screen and describe what you see",
            max_steps=3,
            verbose=True
        )

        print("-"*70)
        print(f"\n✅ Task Result:")
        print(f"  Success: {result['success']}")
        print(f"  Score: {result['score']:.2f}")
        print(f"  Steps: {result['steps']}")
        print(f"  Duration: {result['duration']:.2f}s")

        # Show statistics
        stats = agent.get_statistics()
        print(f"\n📊 Agent Statistics:")
        print(f"  Total episodes: {stats['total_episodes']}")
        print(f"  Total steps: {stats['total_steps']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Avg score: {stats['avg_score']:.2f}")

    except Exception as e:
        print(f"❌ Error: {e}")


def show_architecture():
    """Show system architecture"""
    print("\n" + "="*70)
    print("SYSTEM ARCHITECTURE")
    print("="*70)

    architecture = """
Complete AGI System (5 layers):

Layer 1: PERCEPTION (Computer Use Agent)
  ↓
  • PIL ImageGrab - Real screenshot capture
  • 1920x1080 → 32x32 grayscale → 1024-dim features
  • File: computer-use-ncp/computer_agent.py (450 lines)

Layer 2: COGNITION (Streaming AGI)
  ↓
  • Token-by-token thinking with Ollama
  • Parallel reasoning paths
  • Local inference (qwen2.5:3b)
  • File: streaming-agi/streaming_continuous_agi.py (380 lines)

Layer 3: EMOTION (Emotional AGI)
  ↓
  • 7 emotions: curiosity, wonder, joy, frustration, satisfaction, surprise, calm
  • Natural termination: while not satisfied (NO INFINITE LOOPS!)
  • Emotion-based learning
  • File: emotional-agi/emotional_agi.py (812 lines)

Layer 4: ACTION (Thinking Actor AGI)
  ↓
  • Parallel thinking + acting
  • Action commands in thinking tokens: [ACTION: click(x, y)]
  • Real-time parsing and execution
  • File: thinking-actor-agi/thinking_actor_agi.py (615 lines)

Layer 5: EMBODIMENT (SIMA-style Agent)
  ↓
  • Environment adapter (multi-environment support)
  • Skill library (natural language → actions)
  • Memory system (episodic + semantic)
  • LLM evaluator (self-assessment)
  • Files: embodied-sima-agent/*.py (2,662 lines)

Layer 6: NEURAL SUBSTRATE (NCP)
  ↓
  • 1096 neurons, 10620 synapses
  • C. elegans-inspired sparse wiring
  • Liquid time-constant dynamics
  • File: neural-circuit-policies/ncp_core.py (320 lines)

═══════════════════════════════════════════════════════════════════
TOTAL: ~5,200 lines of modular AGI code
KEY INNOVATION: Emotion-based control + Self-supervised learning
═══════════════════════════════════════════════════════════════════
"""
    print(architecture)


def main():
    """Main launcher"""
    print_banner()

    # Check dependencies
    check_dependencies()

    # Show architecture
    show_architecture()

    print("\n" + "="*70)
    print("RUNNING INTEGRATED DEMOS")
    print("="*70)

    demos = [
        ("Streaming AGI", demo_streaming_agi),
        ("Emotional AGI", demo_emotional_agi),
        ("Computer Agent", demo_computer_agent),
        ("Thinking Actor", demo_thinking_actor),
        ("Embodied Agent", demo_embodied_agent),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

        if i < len(demos):
            print("\n" + "-"*70)
            input("\nPress Enter to continue to next demo...")

    # Final summary
    print("\n" + "="*70)
    print("ALL DEMOS COMPLETE!")
    print("="*70)

    summary = """
✅ Demonstrated:
  1. Streaming AGI - Token-by-token reasoning
  2. Emotional AGI - 7 emotions with natural termination
  3. Computer Agent - Real vision + NCP brain
  4. Thinking Actor - Parallel thinking + acting
  5. Embodied Agent - Complete SIMA-style integration

🎯 Key Features:
  • Real computer vision (not simulated)
  • LLM-based reasoning (local, no cloud)
  • Emotion-based control (no infinite loops)
  • Self-supervised learning (no human labels)
  • Multi-environment support (games + simulators)
  • Fully modular (easy to extend)

📚 Documentation:
  • Architecture: /home/kim/auto-ai/AGI_ARCHITECTURE.md
  • Individual READMEs in each component directory

🚀 Next Steps:
  • Run individual components for detailed exploration
  • Integrate into your own projects
  • Extend with new capabilities
  • Share and contribute!

💡 Philosophy:
  "AGI는 단순히 똑똑한 것이 아니라,
   생각하고 느끼고 행동하고 학습하는 완전한 시스템이다"

  "AGI is not just smart,
   it's a complete system that thinks, feels, acts, and learns"
"""
    print(summary)

    print("\n✓ Complete AGI System - Ready for exploration!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Launcher interrupted by user")
        print("✓ Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
