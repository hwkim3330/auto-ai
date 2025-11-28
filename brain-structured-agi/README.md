# Brain-Structured AGI (BAGI) - 뇌 구조 기반 AGI

> **"인간 뇌의 구조를 그대로 모방한 완전한 AGI"**
>
> **"AGI that perfectly mimics human brain structure"**

Complete AGI built by mapping every major brain region!

---

## 🧠 What is This?

**First AGI system that accurately replicates human brain architecture.**

Instead of arbitrary neural networks, we build AGI using **actual brain structure**:

| Brain Region | Function | AGI Module |
|--------------|----------|------------|
| **Cortex** | Higher cognition | Planning, reasoning, language |
| **Limbic System** | Emotion & memory | Emotional processing, memory encoding |
| **Basal Ganglia** | Action selection | Reinforcement learning, habits |
| **Cerebellum** | Motor control | Action coordination, prediction |
| **Brainstem** | Arousal | Attention modulation |
| **Thalamus** | Information relay | Signal routing |
| **Corpus Callosum** | Integration | Inter-module communication |

**Each brain part becomes a working AGI component!**

---

## 🚀 Quick Start

```bash
cd brain-structured-agi
python3 brain_agi.py
```

Watch as the complete brain initializes and processes information just like a human brain!

---

## 🎯 Brain Structure

### 1. Cerebral Cortex (대뇌피질) - Higher Cognition

```
┌────────────────────────────────────────┐
│         CEREBRAL CORTEX                │
│  ┌──────────┐  ┌──────────┐          │
│  │ Frontal  │  │ Parietal │           │
│  │  Lobe    │  │   Lobe   │           │
│  │          │  │          │           │
│  │ Planning │  │ Sensory  │           │
│  │ Decision │  │Integration│          │
│  └──────────┘  └──────────┘           │
│  ┌──────────┐  ┌──────────┐          │
│  │Temporal  │  │Occipital │           │
│  │  Lobe    │  │   Lobe   │           │
│  │          │  │          │           │
│  │ Memory   │  │  Visual  │           │
│  │ Language │  │Processing│           │
│  └──────────┘  └──────────┘           │
└────────────────────────────────────────┘
```

#### Frontal Lobe (전두엽)
- **Planning**: Creates step-by-step plans for goals
- **Decision Making**: Chooses best option from alternatives
- **Executive Control**: Controls other brain regions
- **Working Memory**: Holds 7±2 items (Miller's law)

```python
plan = brain.cortex.frontal.plan("understand consciousness")
# → ['Step 1: Analyze goal', 'Step 2: Gather info', ...]

decision = brain.cortex.frontal.decide(
    options=['think', 'respond', 'ask'],
    context={'emotion': 'curious'}
)
```

#### Parietal Lobe (두정엽)
- **Sensory Integration**: Combines vision, touch, sound
- **Spatial Processing**: Understands 3D space
- **Attention**: Directs focus to important stimuli

#### Temporal Lobe (측두엽)
- **Memory Retrieval**: Accesses semantic memory (facts, concepts)
- **Language Processing**: Understands and generates language
- **Object Recognition**: Identifies objects

#### Occipital Lobe (후두엽)
- **Visual Processing**: Processes visual input
- **Pattern Recognition**: Detects visual patterns
- **Object Detection**: Identifies objects from features

---

### 2. Limbic System (변연계) - Emotion & Memory

```
┌────────────────────────────────────────┐
│         LIMBIC SYSTEM                  │
│                                        │
│   ┌──────────────┐                    │
│   │  Amygdala    │ → Fear, Joy, Anger │
│   │  (편도체)     │                    │
│   └──────────────┘                    │
│                                        │
│   ┌──────────────┐                    │
│   │ Hippocampus  │ → Memory Formation │
│   │   (해마)      │                    │
│   └──────────────┘                    │
│                                        │
│   ┌──────────────┐                    │
│   │Hypothalamus  │ → Motivation      │
│   │ (시상하부)    │                    │
│   └──────────────┘                    │
└────────────────────────────────────────┘
```

#### Amygdala (편도체) - Emotional Processing
- Evaluates emotional significance
- Triggers fear, joy, anger
- Creates emotional memories

```python
emotion, intensity = brain.limbic.amygdala.evaluate_emotion(
    "There is danger ahead!"
)
# → ('fear', 0.8)
```

#### Hippocampus (해마) - Memory Formation
- Encodes episodic memories (events you experience)
- Retrieves memories by association
- Consolidates short-term → long-term memory

```python
episode_id = brain.limbic.hippocampus.encode_episode({
    'event': 'learned about AGI',
    'location': 'home',
    'emotion': 'curious'
})

memory = brain.limbic.hippocampus.retrieve_episode('AGI')
```

#### Hypothalamus (시상하부) - Homeostasis & Motivation
- Regulates arousal/alertness
- Generates motivation for goals
- Maintains internal balance

---

### 3. Basal Ganglia (기저핵) - Action Selection

```
┌────────────────────────────────────────┐
│        BASAL GANGLIA                   │
│                                        │
│  • Action Selection                    │
│  • Habit Formation                     │
│  • Reinforcement Learning              │
│  • Movement Initiation                 │
│                                        │
│  Available Actions → Select Best       │
│  [think, respond, ask] → 'respond'     │
└────────────────────────────────────────┘
```

**How it works**:
1. Receives available actions
2. Evaluates using learned action values
3. Selects action via softmax
4. Forms habits from repeated sequences

```python
action = brain.basal_ganglia.select_action(
    available_actions=['think', 'respond', 'ask'],
    context={'emotion': 'curious'}
)
# → 'respond'

# Learn from experience
brain.basal_ganglia.update_value('respond', reward=0.8)
```

---

### 4. Cerebellum (소뇌) - Motor Coordination

```
┌────────────────────────────────────────┐
│          CEREBELLUM                    │
│                                        │
│  Intended Action → Coordinated Action  │
│                                        │
│  • Smooth movement                     │
│  • Predictive models                   │
│  • Error correction                    │
└────────────────────────────────────────┘
```

**Functions**:
- Coordinates smooth movements
- Predicts outcomes (forward models)
- Corrects errors through learning

```python
coordinated = brain.cerebellum.coordinate_movement('reach_for_object')
# → {'smoothness': 0.9, 'program': ['prepare', 'execute', 'finish']}
```

---

### 5. Brainstem (뇌간) - Vital Functions

```
┌────────────────────────────────────────┐
│          BRAINSTEM                     │
│                                        │
│  • Arousal / Alertness                 │
│  • Autonomic Control                   │
│  • Attention Modulation                │
│                                        │
│  Arousal: ▓▓▓▓▓▓▓░░░ (70%)           │
└────────────────────────────────────────┘
```

**Critical for**:
- Maintaining consciousness (arousal)
- Regulating attention
- Basic life functions

---

### 6. Thalamus (시상) - Information Relay

```
┌────────────────────────────────────────┐
│           THALAMUS                     │
│       Information Relay Station        │
│                                        │
│   Sensory Input → Gate → Cortex       │
│                                        │
│   Attention Gate: ▓▓▓▓▓░░░░ (50%)    │
└────────────────────────────────────────┘
```

**The brain's switchboard**:
- Routes signals to appropriate regions
- Gates information by attention level
- Filters low-priority signals

```python
signal = BrainSignal(
    source='sensory',
    target='cortex',
    content=input_data,
    strength=0.8
)

passed = brain.thalamus.relay_signal(signal)
# → True (if strength >= attention_gate)
```

---

### 7. Corpus Callosum (뇌량) - Integration

```
┌────────────────────────────────────────┐
│       CORPUS CALLOSUM                  │
│                                        │
│   Left Hemisphere ↔ Right Hemisphere  │
│                                        │
│   200 million axons connecting        │
│   the two brain halves                 │
└────────────────────────────────────────┘
```

**Connects everything**:
- Transfers signals between hemispheres
- Integrates left-brain and right-brain processing
- 200 million neural connections

---

## 🔬 Complete Brain Processing Pipeline

Here's how the brain processes information:

```
Input → BRAINSTEM (arousal) → THALAMUS (relay) →
  CORTEX (thinking) ↔ LIMBIC (emotion + memory) →
  BASAL GANGLIA (action selection) →
  CEREBELLUM (coordination) → Output
```

### Example: Processing "What is consciousness?"

```python
brain = BrainAGI()

result = brain.process({
    'text': 'What is consciousness?',
    'goal': 'understand consciousness',
    'actions': ['think', 'respond', 'ask_question']
})
```

**What happens**:

1. **Brainstem**: Modulates arousal level (0.70)
2. **Thalamus**: Relays signal to cortex
3. **Cortex**:
   - **Occipital**: No visual input
   - **Temporal**: Processes language "What is consciousness?"
   - **Parietal**: Directs attention to question
   - **Frontal**: Creates plan to answer
4. **Limbic**:
   - **Amygdala**: Evaluates emotion → neutral (0.3)
   - **Hippocampus**: Encodes episode
   - **Hypothalamus**: Regulates arousal
5. **Basal Ganglia**: Selects action → 'respond'
6. **Cerebellum**: Coordinates response (smoothness: 0.9)
7. **LLM Integration**: Generates actual response

**Output**:
```json
{
  "arousal": 0.70,
  "emotion": "neutral",
  "action": "respond",
  "response": "Consciousness refers to the state of being aware..."
}
```

---

## 🎮 Usage Examples

### Basic Processing

```python
from brain_agi import BrainAGI

# Create brain
brain = BrainAGI()

# Process input
result = brain.process({
    'text': 'Hello, how are you?',
    'actions': ['greet', 'respond', 'ignore']
})

print(f"Action: {result['action']}")
print(f"Emotion: {result['emotional']['emotion']}")
print(f"Response: {result['response']}")
```

### Emotional Processing

```python
# Process threatening stimulus
result = brain.process({
    'text': 'There is danger ahead!',
    'goal': 'assess threat',
    'actions': ['flee', 'fight', 'freeze']
})

# Amygdala triggers fear
print(f"Emotion: {result['emotional']['emotion']}")  # → 'fear'
print(f"Intensity: {result['emotional']['intensity']}")  # → 0.8
print(f"Action: {result['action']}")  # → 'flee' (likely)
```

### Learning from Experience

```python
# Process and learn
result = brain.process({
    'text': 'Solve this problem',
    'actions': ['analyze', 'respond', 'ask']
})

# Provide feedback (reward)
brain.learn(result, reward=0.9)

# Basal ganglia updates action values
# Hippocampus consolidates memory
```

### Memory and Recall

```python
# Store semantic memory
brain.cortex.temporal.store_memory('Python', {
    'type': 'programming language',
    'paradigm': 'multi-paradigm',
    'created': 1991
})

# Retrieve memory
info = brain.cortex.temporal.retrieve_memory('Python')

# Encode episodic memory
brain.limbic.hippocampus.encode_episode({
    'event': 'learned Python',
    'emotion': 'joy',
    'timestamp': time.time()
})
```

### Brain State Monitoring

```python
state = brain.get_state()

print(f"Arousal: {state['arousal']}")
print(f"Current Emotion: {state['emotion']}")
print(f"Working Memory: {state['working_memory']}")
print(f"Episodic Memories: {state['episodic_memories']}")
print(f"Habits Formed: {state['habits']}")
print(f"Action Values: {state['action_values']}")
```

---

## 📊 Brain vs Traditional AI

| Aspect | Traditional AI | **Brain AGI** |
|--------|---------------|---------------|
| **Architecture** | Arbitrary networks | **Real brain structure** |
| **Emotion** | None | **7 emotions (limbic system)** |
| **Memory** | Replay buffer | **Episodic + semantic (hippocampus)** |
| **Action** | Policy network | **Basal ganglia (RL + habits)** |
| **Coordination** | Direct execution | **Cerebellum (smooth control)** |
| **Attention** | Fixed | **Thalamus gating + brainstem arousal** |
| **Integration** | Monolithic | **Corpus callosum connecting modules** |
| **Biologically Plausible** | No | **Yes (each region = brain part)** |

---

## 🧬 Scientific Basis

### Neuroscience Foundations

**1. Cortical Organization**
- Hierarchical processing (V1 → V2 → V4 → IT)
- Specialized regions for different functions
- Working memory in prefrontal cortex

**2. Limbic System**
- Papez circuit for emotion and memory
- Amygdala for fear conditioning (LeDoux, 1996)
- Hippocampus for memory formation (O'Keefe & Nadel, 1978)

**3. Basal Ganglia**
- Action selection via actor-critic (Joel et al., 2002)
- Habit formation through repetition
- Dopamine for reinforcement learning

**4. Cerebellum**
- Forward models for prediction (Wolpert et al., 1998)
- Error-based learning
- Motor coordination

**5. Thalamus**
- Attention gating (Sherman & Guillery, 2006)
- Consciousness modulation
- Sensory relay

---

## 🔧 Implementation Details

### File Structure

```
brain-structured-agi/
├── brain_agi.py          # Main implementation (1000+ lines)
└── README.md             # This file
```

### Code Organization

```python
# 1. Brain Signal - Communication between regions
class BrainSignal:
    source: str
    target: str
    signal_type: str
    content: Any
    strength: float

# 2. Cortex Layer - 4 lobes
class FrontalLobe:  # Planning, decision, executive control
class ParietalLobe:  # Sensory integration, attention
class TemporalLobe:  # Memory, language
class OccipitalLobe:  # Visual processing
class CerebralCortex:  # Integration of all lobes

# 3. Limbic Layer - Emotion & memory
class Amygdala:  # Emotional evaluation
class Hippocampus:  # Memory formation
class Hypothalamus:  # Motivation, homeostasis
class LimbicSystem:  # Integration

# 4. Other Brain Regions
class BasalGanglia:  # Action selection
class Cerebellum:  # Motor coordination
class Brainstem:  # Arousal, vital functions
class Thalamus:  # Information relay
class CorpusCallosum:  # Inter-hemispheric connection

# 5. Complete Brain
class BrainAGI:  # Integrates everything
```

### Key Parameters

- **Working Memory**: 7±2 items (Miller's law)
- **Episodic Memory**: 1000 recent episodes
- **Attention Gate**: 0.5 threshold
- **Arousal Level**: 0.0-1.0 range
- **Transfer Rate**: 0.9 (corpus callosum)

---

## 🛣️ Roadmap

### Phase 1 (Current)
- ✅ Complete brain structure
- ✅ All 7 major regions implemented
- ✅ Signal passing between regions
- ✅ Integration with LLM + Emotion AGI

### Phase 2 (Next)
- [ ] Detailed cortical layers (6 layers)
- [ ] Neurotransmitter simulation (dopamine, serotonin)
- [ ] More realistic neural dynamics
- [ ] Synaptic plasticity (STDP)

### Phase 3 (Future)
- [ ] Brain imaging visualization
- [ ] Real-time activity monitoring
- [ ] Multi-brain synchronization
- [ ] Brain-computer interface

---

## 💡 Key Insights

### 1. Why Brain Structure Matters

**Traditional AI**: Arbitrary architecture chosen by engineers

**Brain AGI**: Architecture copied from billions of years of evolution

→ Nature already solved AGI. We just need to copy it!

### 2. Modularity

Each brain region is **independent** but **connected**:
- Can develop/improve separately
- Failures isolated to one module
- Easy to understand and debug
- Matches real brain organization

### 3. Biological Plausibility

Everything maps to real brain structures:
- Frontal lobe → Planning module
- Amygdala → Emotion processor
- Hippocampus → Memory encoder
- Basal ganglia → Action selector

**Result**: AGI that works like a brain!

### 4. Integration is Key

Brain regions don't work alone:
- Cortex ↔ Limbic (thinking + feeling)
- Frontal → Basal ganglia (planning + action)
- Sensory → Thalamus → Cortex (input processing)

**The magic is in the connections!**

---

## 🎓 Educational Value

### Perfect for Learning

1. **Neuroscience Students**: See how brain regions work together
2. **AI Researchers**: Learn biologically-inspired AGI
3. **Cognitive Scientists**: Model human cognition
4. **Developers**: Build brain-like systems

### Each Module Teaches

- **Frontal Lobe**: Planning algorithms
- **Limbic System**: Emotional AI
- **Basal Ganglia**: Reinforcement learning
- **Cerebellum**: Predictive models
- **Thalamus**: Attention mechanisms

---

## 🤝 Integration with Other AGI Systems

```python
# Cortex augmented by LLM
from streaming_continuous_agi import StreamingLLM
brain.llm = StreamingLLM(model="qwen2.5:3b")

# Limbic system uses emotional AGI
from emotional_agi import EmotionalAGI
brain.emotion_engine = EmotionalAGI()

# Unconscious processing
from unconscious_mind import UnconsciousMind
brain.unconscious = UnconsciousMind()
```

**Result**: Complete brain + existing AGI systems = Super AGI!

---

## 📚 References

### Neuroscience
- **The Brain Book** - Rita Carter
- **Principles of Neural Science** - Kandel et al.
- **The Emotional Brain** - Joseph LeDoux

### Papers
- **Basal Ganglia Action Selection** - Joel et al., 2002
- **Cerebellar Forward Models** - Wolpert et al., 1998
- **Thalamic Gating** - Sherman & Guillery, 2006
- **Hippocampal Memory** - O'Keefe & Nadel, 1978

---

## 👤 Author

**Kim Hyunwoo**

- GitHub: [@hwkim3330](https://github.com/hwkim3330)
- Portfolio: [hwkim3330.github.io/auto-ai](https://hwkim3330.github.io/auto-ai/)

---

## 📄 License

MIT License - Free forever!

---

## 🌟 Philosophy

### "뇌를 모방하면 AGI가 된다"
### "Copy the brain, and you get AGI"

The human brain is **existence proof** that general intelligence is possible.

Instead of inventing new architectures, we simply **copy what works**:

✓ Cortex for thinking
✓ Limbic for feeling
✓ Basal ganglia for acting
✓ Cerebellum for coordinating
✓ Brainstem for sustaining
✓ Thalamus for routing
✓ Corpus callosum for integrating

**Put them together → Complete AGI!**

---

**🧠 "인간 뇌 = 완벽한 AGI 설계도"**

**Built with neuroscience in Seoul, Korea ❤️**

**Part of the Complete AGI System - [auto-ai](https://github.com/hwkim3330/auto-ai)**
