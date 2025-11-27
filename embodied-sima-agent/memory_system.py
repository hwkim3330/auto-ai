#!/usr/bin/env python3
"""
Memory System - Short-term and long-term memory
================================================

"경험을 기억하고 활용한다"

Two-tier memory:
1. Episodic Memory (short-term): Recent observations and actions
2. Semantic Memory (long-term): Learned knowledge and strategies

Architecture:
    Experience → Episodic Memory → Consolidation → Semantic Memory
                      ↓                                  ↓
                  Recent context                   General knowledge
                      ↓                                  ↓
                      └────────────→ LLM Planner ←───────┘
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from collections import deque


# ============================================================================
# Memory Entry Types
# ============================================================================

@dataclass
class EpisodicMemory:
    """
    Single episodic memory entry

    Stores what happened at a specific moment
    """
    # Core data
    observation: Any            # Environment observation
    action: Optional[Any]       # Action taken
    reward: float              # Reward received
    next_observation: Any      # Next observation

    # Context
    timestamp: float
    step_id: int
    episode_id: int

    # Annotations
    skill_used: Optional[str] = None
    plan_step: Optional[str] = None
    emotion_state: Optional[Dict] = None
    success: bool = True
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dict for LLM"""
        return {
            'step': self.step_id,
            'timestamp': self.timestamp,
            'skill': self.skill_used,
            'plan': self.plan_step,
            'reward': self.reward,
            'success': self.success,
            'notes': self.notes,
        }


@dataclass
class SemanticMemory:
    """
    Semantic memory entry (learned knowledge)

    Stores general facts, strategies, and patterns
    """
    # Content
    content: str               # Text description
    category: str              # Type of knowledge
    confidence: float          # How confident (0-1)

    # Evidence
    supporting_episodes: List[int] = field(default_factory=list)
    counter_episodes: List[int] = field(default_factory=list)

    # Metadata
    created_at: float = 0.0
    last_used: float = 0.0
    use_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dict for LLM"""
        return {
            'knowledge': self.content,
            'category': self.category,
            'confidence': self.confidence,
            'evidence_count': len(self.supporting_episodes),
            'use_count': self.use_count,
        }


# ============================================================================
# Memory Manager
# ============================================================================

class MemoryManager:
    """
    Manages both episodic and semantic memory

    Features:
    - Store recent experiences (episodic)
    - Extract patterns → semantic knowledge
    - Retrieve relevant memories for planning
    - Memory consolidation (episodic → semantic)
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}

        # Episodic memory (recent experiences)
        self.episodic_capacity = config.get('episodic_capacity', 1000)
        self.episodic_memories: deque = deque(maxlen=self.episodic_capacity)

        # Semantic memory (learned knowledge)
        self.semantic_memories: List[SemanticMemory] = []

        # Current episode tracking
        self.current_episode_id = 0
        self.current_step_id = 0

        # Statistics
        self.total_memories = 0
        self.consolidation_count = 0

        print(f"[MemoryManager] Initialized")
        print(f"  Episodic capacity: {self.episodic_capacity}")

    def store_experience(
        self,
        observation: Any,
        action: Optional[Any],
        reward: float,
        next_observation: Any,
        skill_used: Optional[str] = None,
        plan_step: Optional[str] = None,
        emotion_state: Optional[Dict] = None,
        success: bool = True,
        notes: str = ""
    ):
        """
        Store new experience in episodic memory

        Called after each environment step
        """
        memory = EpisodicMemory(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            timestamp=time.time(),
            step_id=self.current_step_id,
            episode_id=self.current_episode_id,
            skill_used=skill_used,
            plan_step=plan_step,
            emotion_state=emotion_state,
            success=success,
            notes=notes
        )

        self.episodic_memories.append(memory)
        self.current_step_id += 1
        self.total_memories += 1

    def get_recent_context(self, k: int = 10) -> List[EpisodicMemory]:
        """
        Get k most recent episodic memories

        Used to provide context to LLM planner
        """
        return list(self.episodic_memories)[-k:]

    def get_episode_summary(self, episode_id: Optional[int] = None) -> Dict:
        """
        Get summary of episode

        Args:
            episode_id: Episode to summarize (None = current)

        Returns:
            Summary statistics
        """
        if episode_id is None:
            episode_id = self.current_episode_id

        # Filter memories for this episode
        episode_memories = [
            m for m in self.episodic_memories
            if m.episode_id == episode_id
        ]

        if not episode_memories:
            return {}

        total_reward = sum(m.reward for m in episode_memories)
        success_rate = sum(1 for m in episode_memories if m.success) / len(episode_memories)

        # Skill distribution
        skills_used = [m.skill_used for m in episode_memories if m.skill_used]
        unique_skills = set(skills_used)

        return {
            'episode_id': episode_id,
            'total_steps': len(episode_memories),
            'total_reward': total_reward,
            'success_rate': success_rate,
            'skills_used': len(skills_used),
            'unique_skills': len(unique_skills),
            'duration': episode_memories[-1].timestamp - episode_memories[0].timestamp,
        }

    def start_new_episode(self):
        """Start tracking new episode"""
        self.current_episode_id += 1
        self.current_step_id = 0
        print(f"[MemoryManager] Started episode {self.current_episode_id}")

    def consolidate_knowledge(self, llm = None):
        """
        Consolidate episodic memories into semantic knowledge

        Extracts patterns and stores as general knowledge

        Args:
            llm: Language model for analysis (optional)
        """
        print(f"\n[MemoryManager] Consolidating knowledge...")

        # Get recent episode
        episode_id = self.current_episode_id - 1  # Previous episode
        episode_memories = [
            m for m in self.episodic_memories
            if m.episode_id == episode_id
        ]

        if not episode_memories:
            print("  No memories to consolidate")
            return

        # Extract successful patterns
        successful_sequences = []
        current_seq = []

        for mem in episode_memories:
            if mem.success:
                current_seq.append(mem)
            else:
                if len(current_seq) >= 3:  # Min 3 steps
                    successful_sequences.append(current_seq)
                current_seq = []

        if len(current_seq) >= 3:
            successful_sequences.append(current_seq)

        # Create semantic memories from patterns
        for seq in successful_sequences:
            # Extract pattern
            skills = [m.skill_used for m in seq if m.skill_used]
            if len(set(skills)) >= 2:  # At least 2 different skills
                pattern = " → ".join(skills)
                avg_reward = sum(m.reward for m in seq) / len(seq)

                # Create semantic memory
                knowledge = SemanticMemory(
                    content=f"Successful pattern: {pattern}",
                    category="strategy",
                    confidence=min(avg_reward, 1.0),
                    supporting_episodes=[episode_id],
                    created_at=time.time()
                )

                self.semantic_memories.append(knowledge)
                self.consolidation_count += 1

        print(f"  Extracted {len(successful_sequences)} patterns")
        print(f"  Total semantic memories: {len(self.semantic_memories)}")

    def retrieve_relevant_knowledge(self, query: str, k: int = 5) -> List[SemanticMemory]:
        """
        Retrieve relevant semantic knowledge

        Args:
            query: Query string (e.g., current goal)
            k: How many to retrieve

        Returns:
            Most relevant semantic memories
        """
        if not self.semantic_memories:
            return []

        # Simple relevance: keyword matching + confidence
        # TODO: Use proper embeddings + vector search
        scores = []

        query_words = set(query.lower().split())

        for mem in self.semantic_memories:
            content_words = set(mem.content.lower().split())
            overlap = len(query_words & content_words)
            score = overlap * mem.confidence * (1 + mem.use_count * 0.1)
            scores.append((score, mem))

        # Sort by score
        scores.sort(reverse=True, key=lambda x: x[0])

        # Update use count
        relevant = [mem for _, mem in scores[:k]]
        for mem in relevant:
            mem.use_count += 1
            mem.last_used = time.time()

        return relevant

    def get_memory_summary(self) -> Dict:
        """Get overall memory statistics"""
        return {
            'total_memories': self.total_memories,
            'episodic_count': len(self.episodic_memories),
            'semantic_count': len(self.semantic_memories),
            'current_episode': self.current_episode_id,
            'consolidations': self.consolidation_count,
        }

    def format_for_llm(self, include_recent: int = 5, include_knowledge: int = 3) -> str:
        """
        Format memory for LLM prompt

        Returns formatted string with:
        - Recent episodic memories
        - Relevant semantic knowledge
        """
        lines = []

        # Recent context
        if include_recent > 0:
            recent = self.get_recent_context(k=include_recent)
            if recent:
                lines.append("[RECENT CONTEXT]")
                for mem in recent:
                    lines.append(f"  Step {mem.step_id}: {mem.skill_used or 'unknown'} "
                               f"(reward: {mem.reward:.2f}, {'success' if mem.success else 'failed'})")
                lines.append("")

        # Learned knowledge
        if include_knowledge > 0 and self.semantic_memories:
            # Get most confident knowledge
            top_knowledge = sorted(
                self.semantic_memories,
                key=lambda m: m.confidence * (1 + m.use_count * 0.1),
                reverse=True
            )[:include_knowledge]

            if top_knowledge:
                lines.append("[LEARNED KNOWLEDGE]")
                for mem in top_knowledge:
                    lines.append(f"  - {mem.content} (confidence: {mem.confidence:.2f})")
                lines.append("")

        return "\n".join(lines)


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate memory system"""
    print("\n" + "="*70)
    print("MEMORY SYSTEM - Demo")
    print("="*70)

    memory = MemoryManager()

    # Simulate episode
    print("\n[Demo] Simulating episode...")

    memory.start_new_episode()

    # Store some experiences
    for i in range(10):
        memory.store_experience(
            observation=f"obs_{i}",
            action=f"action_{i}",
            reward=np.random.random(),
            next_observation=f"obs_{i+1}",
            skill_used=f"skill_{i % 3}",
            plan_step=f"step_{i}",
            success=i % 4 != 0  # Fail every 4th step
        )

    # Get recent context
    print("\n[Demo] Recent context:")
    recent = memory.get_recent_context(k=3)
    for mem in recent:
        print(f"  {mem.to_dict()}")

    # Episode summary
    print("\n[Demo] Episode summary:")
    summary = memory.get_episode_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Consolidate
    memory.consolidate_knowledge()

    # Memory summary
    print("\n[Demo] Memory summary:")
    mem_summary = memory.get_memory_summary()
    for key, value in mem_summary.items():
        print(f"  {key}: {value}")

    # Format for LLM
    print("\n[Demo] LLM format:")
    print(memory.format_for_llm())

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    demo()
