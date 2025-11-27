# SIMA-style Embodied Agent - 오픈소스 체화 AGI

> **"생각하고, 기억하고, 행동하고, 평가하고, 학습한다"**
>
> **"Open-source SIMA2 - Embodied AGI that learns games and simulations"**

Complete SIMA2-style embodied agent with modular architecture

---

## 🎯 Core Innovation

**SIMA2 스타일 체화 에이전트를 완전 오픈소스로 구현**

```
환경과 상호작용 → 경험 기억 → 패턴 학습 → 자기평가 → 지속적 개선
```

**Key Features:**
- 🌐 **Multi-Environment**: Games + Simulators (CARLA, Isaac, Unity)
- 🧠 **LLM Planning**: High-level reasoning with Streaming AGI
- 🎯 **Skill Library**: Natural language → Executable actions
- 💾 **Memory System**: Episodic + Semantic memory
- 📊 **LLM Evaluator**: Self-assessment without human labels
- 💖 **Emotional System**: Emotion-driven learning and termination
- 🔄 **Learning Loop**: Continuous improvement from experience

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMA-STYLE EMBODIED AGENT                    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  1. ENVIRONMENT ADAPTER                                │    │
│  │     • Screen-based (games via keyboard/mouse)          │    │
│  │     • CARLA (autonomous driving)                       │    │
│  │     • Isaac Sim (robotics)                             │    │
│  │     • Unity ML-Agents                                  │    │
│  └──────────────────┬─────────────────────────────────────┘    │
│                     ↓                                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  2. HIGH-LEVEL PLANNER (Streaming AGI)                 │    │
│  │     • LLM-based reasoning (Qwen2.5)                    │    │
│  │     • Context from memory                              │    │
│  │     • Natural language plans                           │    │
│  └──────────────────┬─────────────────────────────────────┘    │
│                     ↓                                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  3. SKILL LIBRARY                                      │    │
│  │     • move_to_target                                   │    │
│  │     • interact_with_object                             │    │
│  │     • craft_item                                       │    │
│  │     • navigate_to_location                             │    │
│  │     • use_tool                                         │    │
│  └──────────────────┬─────────────────────────────────────┘    │
│                     ↓                                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  4. LOW-LEVEL CONTROLLER (NCP)                         │    │
│  │     • 1096 neurons, 10620 synapses                     │    │
│  │     • Real vision (PIL screenshot)                     │    │
│  │     • Keyboard/mouse control                           │    │
│  └──────────────────┬─────────────────────────────────────┘    │
│                     ↓                                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  5. MEMORY SYSTEM                                      │    │
│  │     • Episodic: Recent experiences                     │    │
│  │     • Semantic: Learned knowledge                      │    │
│  │     • Consolidation: Episodes → Patterns               │    │
│  └──────────────────┬─────────────────────────────────────┘    │
│                     ↓                                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  6. REWARD EVALUATOR (LLM + Emotions)                  │    │
│  │     • LLM-based episode assessment                     │    │
│  │     • Emotion-based self-evaluation                    │    │
│  │     • Success/score/suggestions                        │    │
│  └──────────────────┬─────────────────────────────────────┘    │
│                     ↓                                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  7. LEARNING LOOP                                      │    │
│  │     • Consolidate memories                             │    │
│  │     • Extract patterns                                 │    │
│  │     • Curriculum generation                            │    │
│  │     • Continuous improvement                           │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💻 Usage

### Quick Start

```python
from embodied_agent import EmbodiedAgent

# Create agent
agent = EmbodiedAgent(
    env_config={'type': 'screen'},  # or 'carla', 'isaac', 'unity'
    agent_config={
        'llm_model': 'qwen2.5:3b',
        'use_emotions': True
    }
)

# Execute task
result = agent.execute_task(
    task_description="Open text editor and type 'Hello World'",
    max_steps=20,
    verbose=True
)

# Check results
print(f"Success: {result['success']}")
print(f"Score: {result['score']:.2f}")
print(f"Steps: {result['steps']}")
```

### Environment Types

**1. Screen-based (Default)**
```python
env_config = {'type': 'screen'}
```
- Works with any game/application
- Uses screenshot capture + keyboard/mouse
- Great for PoC and testing

**2. CARLA Simulator**
```python
env_config = {
    'type': 'carla',
    'host': 'localhost',
    'port': 2000
}
```
- Autonomous driving
- Full sensor suite
- Physics simulation

**3. Isaac Sim / Unity ML-Agents**
```python
env_config = {
    'type': 'unity',
    'env_path': '/path/to/unity/env'
}
```
- Robotics simulation
- Custom environments
- Native Python API

---

## 🔧 Components

### 1. Environment Adapter (`env_adapter.py`)

**Unified interface for all environments**

```python
from env_adapter import create_env, Observation, Action

env = create_env({'type': 'screen'})

# Reset
obs = env.reset(task_spec={'goal': 'Reach the checkpoint'})

# Step
action = Action(keys=['w', 'a'])
next_obs, reward, done, info = env.step(action)
```

**Features:**
- Multi-environment support
- Unified Observation/Action format
- Easy to extend (subclass BaseEnvAdapter)

**File:** `env_adapter.py` (390 lines)

---

### 2. Skill Library (`skill_library.py`)

**High-level behaviors mapped to low-level actions**

```python
from skill_library import SkillLibrary

skills = SkillLibrary()

# Parse natural language
skill_name, params = skills.parse_instruction("move to the workbench")
# → ('move_to_target', {'target': 'workbench'})

# Execute
result = skills.execute_instruction(agent, "craft an axe")
```

**Built-in Skills:**
- `move_to_target` - Navigate to object/location
- `interact_with_object` - Use/open/pickup
- `craft_item` - Build/craft items
- `navigate_to_location` - GPS/coordinate navigation
- `wait` - Wait for duration
- `observe` - Careful observation

**File:** `skill_library.py` (410 lines)

---

### 3. Memory System (`memory_system.py`)

**Two-tier memory: Episodic + Semantic**

```python
from memory_system import MemoryManager

memory = MemoryManager()

# Store experience
memory.store_experience(
    observation=obs,
    action=action,
    reward=reward,
    skill_used="move_to_target",
    success=True
)

# Get recent context
recent = memory.get_recent_context(k=5)

# Consolidate knowledge
memory.consolidate_knowledge()

# Retrieve relevant knowledge
knowledge = memory.retrieve_relevant_knowledge("how to craft axe", k=3)
```

**Features:**
- **Episodic Memory**: Recent step-by-step experiences
- **Semantic Memory**: Extracted patterns and strategies
- **Consolidation**: Automatic pattern extraction
- **Retrieval**: Context-aware knowledge lookup

**File:** `memory_system.py` (380 lines)

---

### 4. Reward Evaluator (`reward_evaluator.py`)

**LLM-based self-assessment**

```python
from reward_evaluator import RewardEvaluator

evaluator = RewardEvaluator(llm_model='qwen2.5:3b')

# Evaluate episode
result = evaluator.evaluate_episode(
    task_goal="Craft an axe",
    episode_log=episode_log
)

# Check results
print(f"Success: {result.success}")
print(f"Score: {result.score:.2f}")
print(f"Strengths: {result.strengths}")
print(f"Weaknesses: {result.weaknesses}")
print(f"Suggestions: {result.suggestions}")
```

**Instead of hand-crafted rewards:**
- LLM evaluates success/quality
- Identifies what went well/wrong
- Provides actionable suggestions
- Enables self-supervised learning

**File:** `reward_evaluator.py` (360 lines)

---

### 5. Embodied Agent (`embodied_agent.py`)

**Main integration - combines all components**

**Complete execution loop:**
1. **Plan** using LLM with memory context
2. **Execute** skills via skill library
3. **Store** experiences in memory
4. **Evaluate** episode with LLM + emotions
5. **Learn** by consolidating knowledge
6. **Improve** through curriculum generation

**File:** `embodied_agent.py` (400 lines)

---

## 📊 Integration with Existing Systems

**This agent integrates all our previous work:**

| Component | Source | Role |
|-----------|--------|------|
| **Streaming AGI** | `streaming-agi/` | High-level planning |
| **Emotional AGI** | `emotional-agi/` | Self-evaluation + termination |
| **Computer Agent** | `computer-use-ncp/` | Low-level control + vision |
| **NCP** | `neural-circuit-policies/` | Biologically-inspired brain |

**New components:**
- Environment Adapter (multi-env support)
- Skill Library (NL → actions)
- Memory System (episodic + semantic)
- Reward Evaluator (LLM-based)

---

## 🎮 Example Tasks

**Screen-based games:**
```python
agent.execute_task("Open settings and change resolution to 1920x1080")
agent.execute_task("Navigate to inventory and equip the sword")
agent.execute_task("Talk to NPC and accept the quest")
```

**Autonomous driving (CARLA):**
```python
agent = EmbodiedAgent(env_config={'type': 'carla'})
agent.execute_task("Drive to the parking lot and park in spot #5")
agent.execute_task("Follow the car ahead while maintaining safe distance")
```

**Robotics (Isaac Sim):**
```python
agent = EmbodiedAgent(env_config={'type': 'isaac'})
agent.execute_task("Pick up the red cube and place it in the bin")
agent.execute_task("Navigate to the charging station")
```

---

## 🔬 Technical Details

### Memory Consolidation

Automatic pattern extraction from episodes:

```python
# Episode: move → interact → craft → success
# Pattern extracted: "move to workbench → interact → craft"

semantic_memory = SemanticMemory(
    content="Successful pattern: move_to_target → interact_with_object → craft_item",
    category="strategy",
    confidence=0.9,
    supporting_episodes=[12, 15, 18]
)
```

### LLM Evaluation Prompt

```
You are an expert evaluator of AI agent performance.

TASK GOAL: Craft an axe

EPISODE LOG:
  Step 1: move_to_target ✓ (reward: 0.10)
  Step 2: interact_with_object ✓ (reward: 0.20)
  Step 3: craft_item ✓ (reward: 0.50)

Please evaluate:
1. SUCCESS: YES/NO
2. SCORE: 0.0 to 1.0
3. STRENGTHS: What went well
4. WEAKNESSES: What went wrong
5. SUGGESTIONS: How to improve
```

### Emotion-based Termination

```python
# Traditional: for i in range(max_steps)
# Our approach:
while not agent.emotions.should_continue_learning():
    # Continue learning
    ...
# Stops when satisfied, frustrated, or not curious
```

---

## 🆚 Comparison

| Aspect | Traditional RL | Imitation Learning | **SIMA-style Agent** |
|--------|---------------|-------------------|---------------------|
| Reward function | Hand-crafted | Human demos | **LLM-based self-eval** |
| Planning | Policy network | Behavior cloning | **LLM reasoning** |
| Memory | Replay buffer | Demo dataset | **Episodic + Semantic** |
| Termination | Max steps/episodes | Fixed curriculum | **Emotion-based** |
| Generalization | Same environment | Similar tasks | **Multi-environment** |
| Learning signal | Reward | Imitation loss | **Self-assessment** |

---

## 🚀 Future Enhancements

### Planned Features

1. **Better Vision**
   - Object detection (YOLO, SAM)
   - OCR for text recognition
   - Depth estimation

2. **Advanced Planning**
   - Multi-step task decomposition
   - Goal-oriented behavior trees
   - Hierarchical planning

3. **Online Learning**
   - Policy gradient updates
   - Meta-learning
   - Transfer learning

4. **Curriculum Generation**
   - Auto-generate harder tasks
   - Focus on weaknesses
   - Progressive difficulty

5. **Multi-agent Coordination**
   - Shared memory
   - Collaborative planning
   - Competitive scenarios

---

## 📁 File Structure

```
/home/kim/auto-ai/embodied-sima-agent/
├── embodied_agent.py          # Main integration (400 lines)
├── env_adapter.py              # Environment adapter (390 lines)
├── skill_library.py            # Skill library (410 lines)
├── memory_system.py            # Memory system (380 lines)
├── reward_evaluator.py         # LLM evaluator (360 lines)
└── README.md                   # This file
```

**Dependencies:**
- `/auto-ai/streaming-agi/` - High-level planner
- `/auto-ai/emotional-agi/` - Emotion-driven learning
- `/auto-ai/computer-use-ncp/` - Low-level controller
- `/auto-ai/neural-circuit-policies/` - NCP brain

**Total:** ~2000 lines of modular, reusable code

---

## 🎉 Summary

### What We Built

✅ **Complete SIMA2-style embodied agent**
✅ **Multi-environment support** (games + simulators)
✅ **LLM-based planning and evaluation**
✅ **Skill library** (natural language → actions)
✅ **Memory system** (episodic + semantic)
✅ **Self-supervised learning** (no human labels)
✅ **Emotion-driven** (automatic termination)
✅ **Fully modular** (easy to extend)

### Key Innovation

**First open-source implementation of SIMA2-style architecture:**
- Google SIMA2: Closed-source, internal only
- **Our approach**: Fully open, modular, extensible

**Emotion-based control:**
- Traditional: `while True` or `for i in range(max_steps)`
- **Our approach**: `while not satisfied` (自己 停止)

**LLM-based evaluation:**
- Traditional: Hand-crafted reward functions
- **Our approach**: LLM self-assessment + suggestions

---

**"생각하고, 기억하고, 행동하고, 평가하고, 학습한다"**

**"Think, Remember, Act, Evaluate, and Learn"**

**GitHub**: https://github.com/hwkim3330/auto-ai
**Location**: `/home/kim/auto-ai/embodied-sima-agent/`

**"레고 블록처럼 조립하는 체화 AGI"**
**"Embodied AGI built like LEGO blocks"**
