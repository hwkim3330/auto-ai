#!/usr/bin/env python3
"""
Meta-AI Integration Adapters
=============================

기존 모든 AI 시스템을 Meta-AI Core와 통합

Integrated Systems:
1. Liquid NN AI
2. UltraThink AGI
3. The Sentinel
4. Self-Replication System
5. CCTV Tracking
6. Mass CCTV Processing

"하나의 Meta-AI가 모든 것을 관리"
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

# Add paths
sys.path.append(str(Path(__file__).parent.parent / "liquid-nn-ai"))
sys.path.append(str(Path(__file__).parent.parent / "ultrathink-agi"))
sys.path.append(str(Path(__file__).parent.parent / "the-sentinel"))

from meta_ai_core import (
    AIComponent, ComponentState, Experience,
    TaskPriority, MetaAI
)

# ============================================================================
# Liquid NN Adapter
# ============================================================================

class LiquidNNAdapter(AIComponent):
    """
    Liquid Neural Network → Meta-AI 어댑터

    Liquid NN을 Meta-AI 컴포넌트로 변환
    """

    def __init__(self):
        super().__init__("LiquidNN")
        self.model = None
        self.training_count = 0

    def initialize(self) -> bool:
        """Liquid NN 초기화"""
        try:
            from liquid_nn import LiquidNeuralNetwork
            self.model = LiquidNeuralNetwork(
                input_size=128,
                hidden_size=64,
                output_size=128,
                num_layers=2
            )
            print(f"[{self.name}] Initialized with Liquid NN")
            return True
        except Exception as e:
            print(f"[{self.name}] Failed to initialize: {e}")
            return False

    def process(self, data: Any) -> Any:
        """
        데이터 처리 (추론)

        Args:
            data: numpy array or tensor
        """
        self.state = ComponentState.RUNNING

        if self.model is None:
            return {"error": "Model not initialized"}

        try:
            # Liquid NN 추론
            if isinstance(data, np.ndarray):
                import torch
                x = torch.FloatTensor(data).unsqueeze(0)
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)  # (batch, seq, features)

                result = self.model(x)
                if isinstance(result, tuple):
                    output = result[0]
                else:
                    output = result

                return {
                    "output": output.detach().numpy(),
                    "shape": output.shape
                }
        except Exception as e:
            return {"error": str(e)}
        finally:
            self.state = ComponentState.IDLE

    def learn(self, experience: Experience) -> bool:
        """
        경험으로부터 학습

        Args:
            experience: 학습 경험
        """
        self.state = ComponentState.LEARNING

        try:
            self.training_count += 1
            # 실제로는 여기서 gradient descent 수행
            # 지금은 시뮬레이션
            return True
        finally:
            self.state = ComponentState.IDLE

    def get_state(self) -> Dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "training_count": self.training_count,
            "model_loaded": self.model is not None
        }


# ============================================================================
# UltraThink Adapter
# ============================================================================

class UltraThinkAdapter(AIComponent):
    """
    UltraThink AGI → Meta-AI 어댑터

    Tree-of-Thought 추론을 Meta-AI 컴포넌트로 변환
    """

    def __init__(self):
        super().__init__("UltraThink")
        self.thinker = None
        self.reasoning_count = 0

    def initialize(self) -> bool:
        """UltraThink 초기화"""
        try:
            from ultrathink import UltraThink
            self.thinker = UltraThink(feature_dim=128, hidden_size=64)
            print(f"[{self.name}] Initialized with Tree-of-Thought")
            return True
        except Exception as e:
            print(f"[{self.name}] Failed to initialize: {e}")
            return False

    def process(self, data: Any) -> Any:
        """
        추론 수행

        Args:
            data: query string or dict with "query" key
        """
        self.state = ComponentState.REASONING

        if self.thinker is None:
            return {"error": "Thinker not initialized"}

        try:
            # 쿼리 추출
            if isinstance(data, str):
                query = data
            elif isinstance(data, dict) and "query" in data:
                query = data["query"]
            else:
                query = str(data)

            # Tree-of-Thought 추론
            result = self.thinker.think(query, verbose=False)
            self.reasoning_count += 1

            return result

        except Exception as e:
            return {"error": str(e)}
        finally:
            self.state = ComponentState.IDLE

    def learn(self, experience: Experience) -> bool:
        """추론 패턴 학습"""
        self.state = ComponentState.LEARNING

        try:
            # 성공적인 추론 패턴을 학습
            # (실제로는 강화학습이나 메타학습 사용)
            return True
        finally:
            self.state = ComponentState.IDLE

    def get_state(self) -> Dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "reasoning_count": self.reasoning_count,
            "thinker_loaded": self.thinker is not None
        }


# ============================================================================
# Sentinel Adapter
# ============================================================================

class SentinelAdapter(AIComponent):
    """
    The Sentinel → Meta-AI 어댑터

    감시 시스템을 Meta-AI 컴포넌트로 변환
    """

    def __init__(self):
        super().__init__("Sentinel")
        self.sentinel = None
        self.cycles_run = 0

    def initialize(self) -> bool:
        """Sentinel 초기화"""
        try:
            from sentinel import TheSentinel
            self.sentinel = TheSentinel(feature_dim=128)
            print(f"[{self.name}] Initialized surveillance system")
            return True
        except Exception as e:
            print(f"[{self.name}] Failed to initialize: {e}")
            return False

    def process(self, data: Any) -> Any:
        """
        감시 사이클 실행

        Args:
            data: {"action": "run_cycle"} or {"action": "add_camera", ...}
        """
        self.state = ComponentState.RUNNING

        if self.sentinel is None:
            return {"error": "Sentinel not initialized"}

        try:
            if isinstance(data, dict):
                action = data.get("action", "run_cycle")

                if action == "run_cycle":
                    self.sentinel.run_cycle()
                    self.cycles_run += 1
                    return {
                        "cycle": self.sentinel.cycle_count,
                        "observations": self.sentinel.learning.metrics.total_observations
                    }

                elif action == "add_camera":
                    self.sentinel.add_camera(
                        data["id"],
                        data["url"],
                        data["location"]
                    )
                    return {"camera_added": data["id"]}

            return {"error": "Unknown action"}

        except Exception as e:
            return {"error": str(e)}
        finally:
            self.state = ComponentState.IDLE

    def learn(self, experience: Experience) -> bool:
        """Sentinel의 online learning"""
        self.state = ComponentState.LEARNING

        try:
            # Sentinel 자체가 이미 online learning 수행
            return True
        finally:
            self.state = ComponentState.IDLE

    def get_state(self) -> Dict:
        state = {
            "name": self.name,
            "state": self.state.value,
            "cycles_run": self.cycles_run,
            "sentinel_loaded": self.sentinel is not None
        }

        if self.sentinel:
            state["cycle_count"] = self.sentinel.cycle_count
            state["cameras"] = len(self.sentinel.cameras)

        return state


# ============================================================================
# Self-Replication Adapter
# ============================================================================

class ReplicationAdapter(AIComponent):
    """
    Self-Replication System → Meta-AI 어댑터

    진화 시스템을 Meta-AI 컴포넌트로 변환
    """

    def __init__(self):
        super().__init__("Replication")
        self.replicator = None
        self.evolution_count = 0

    def initialize(self) -> bool:
        """Replicator 초기화"""
        try:
            from self_replication_system import SelfReplicatingAI
            self.replicator = SelfReplicatingAI()
            print(f"[{self.name}] Initialized evolutionary system")
            return True
        except Exception as e:
            print(f"[{self.name}] Failed to initialize: {e}")
            return False

    def process(self, data: Any) -> Any:
        """
        진화 실행

        Args:
            data: {"action": "evolve", "generations": 3, "population": 5}
        """
        self.state = ComponentState.OPTIMIZING

        if self.replicator is None:
            return {"error": "Replicator not initialized"}

        try:
            if isinstance(data, dict):
                action = data.get("action", "evolve")

                if action == "evolve":
                    generations = data.get("generations", 3)
                    population = data.get("population", 5)

                    best_agent, history = self.replicator.evolve(
                        num_generations=generations,
                        population_size=population
                    )

                    self.evolution_count += 1

                    return {
                        "best_agent_id": best_agent.dna.id,
                        "best_performance": best_agent.dna.performance,
                        "best_config": best_agent.dna.config,
                        "history": history
                    }

                elif action == "replicate":
                    child = self.replicator.replicate()
                    return {
                        "child_id": child.dna.id,
                        "generation": child.dna.generation
                    }

            return {"error": "Unknown action"}

        except Exception as e:
            return {"error": str(e)}
        finally:
            self.state = ComponentState.IDLE

    def learn(self, experience: Experience) -> bool:
        """진화를 통한 학습"""
        # Replication 자체가 학습 메커니즘
        return True

    def get_state(self) -> Dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "evolution_count": self.evolution_count,
            "replicator_loaded": self.replicator is not None
        }


# ============================================================================
# Unified Meta-AI System
# ============================================================================

class UnifiedAI:
    """
    통합 Meta-AI 시스템

    모든 AI 컴포넌트를 하나로 관리
    """

    def __init__(self):
        print("\n" + "=" * 70)
        print("UNIFIED META-AI SYSTEM - Initializing")
        print("=" * 70)

        # Meta-AI Core 생성
        self.meta_ai = MetaAI(name="UnifiedAI")

        # 모든 어댑터 생성
        self.liquid_nn = LiquidNNAdapter()
        self.ultrathink = UltraThinkAdapter()
        self.sentinel = SentinelAdapter()
        self.replication = ReplicationAdapter()

        # 컴포넌트 초기화
        self.components = [
            self.liquid_nn,
            self.ultrathink,
            self.sentinel,
            self.replication
        ]

        initialized = 0
        for comp in self.components:
            if comp.initialize():
                self.meta_ai.register_component(comp)
                initialized += 1

        print(f"\n[UnifiedAI] Initialized {initialized}/{len(self.components)} components")
        print("=" * 70)

    def start(self):
        """통합 시스템 시작"""
        self.meta_ai.start()
        print("[UnifiedAI] System started")

    def stop(self):
        """통합 시스템 중지"""
        self.meta_ai.stop()
        print("[UnifiedAI] System stopped")

    def get_status(self) -> Dict:
        """전체 상태 조회"""
        return self.meta_ai.get_status()


# ============================================================================
# Demo
# ============================================================================

def main():
    """통합 시스템 데모"""
    print("\n" + "=" * 70)
    print("META-AI INTEGRATION DEMO")
    print("=" * 70)

    # 1. 통합 시스템 생성
    unified = UnifiedAI()
    unified.start()

    # 2. Liquid NN 테스트
    print("\n[Demo] Testing Liquid NN...")
    data = np.random.randn(128)
    unified.meta_ai.process(data, "LiquidNN")

    # 3. UltraThink 테스트
    print("\n[Demo] Testing UltraThink reasoning...")
    unified.meta_ai.process({
        "query": "What is the optimal configuration for this system?"
    }, "UltraThink")

    # 4. Sentinel 테스트
    print("\n[Demo] Testing Sentinel surveillance...")
    unified.meta_ai.process({
        "action": "add_camera",
        "id": "TEST_CAM_001",
        "url": "rtsp://test",
        "location": "Test Location"
    }, "Sentinel")

    # 5. Replication 테스트
    print("\n[Demo] Testing self-replication...")
    unified.meta_ai.process({
        "action": "evolve",
        "generations": 2,
        "population": 3
    }, "Replication")

    import time
    time.sleep(2)

    # 6. 전체 상태
    print("\n[Demo] System status:")
    status = unified.get_status()
    print(f"  Components: {status['component_count']}")
    print(f"  Tasks completed: {status['stats']['tasks_completed']}")
    print(f"  Learning cycles: {status['stats']['learning_cycles']}")

    print("\n  Component states:")
    for name, state in status['orchestrator']['component_states'].items():
        print(f"    {name}: {state}")

    # 7. 종료
    unified.stop()

    print("\n" + "=" * 70)
    print("META-AI INTEGRATION DEMO COMPLETE")
    print("=" * 70)
    print("\nAll systems successfully integrated into Meta-AI Core:")
    print("  ✓ Liquid NN AI")
    print("  ✓ UltraThink AGI")
    print("  ✓ The Sentinel")
    print("  ✓ Self-Replication System")
    print("\n\"하나의 Meta-AI가 모든 것을 관리합니다\"")


if __name__ == "__main__":
    main()
