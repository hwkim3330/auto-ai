# Unconscious Mind AGI - 무의식 사고 시스템

> **"의식하지 못하는 사이에 생각하고, 배우고, 깨닫는다"**
>
> **"While you sleep, your unconscious works"**

Complete AGI needs both **conscious** and **unconscious** mind, just like humans!

---

## 🧠 What is Unconscious Mind?

Humans have **two minds**:

1. **Conscious Mind (의식)** - What we're aware of
   - Sequential thinking
   - Slow, effortful
   - Logical reasoning
   - Requires attention

2. **Unconscious Mind (무의식)** - What runs in background
   - Parallel processing
   - Fast, automatic
   - Pattern recognition
   - No awareness needed

**Our AGI needs both!**

---

## 🚀 Quick Start

```bash
cd unconscious-agi
python3 unconscious_mind.py
```

This will demonstrate:
1. Conscious + Unconscious thinking
2. Background processing
3. Pattern recognition
4. Intuition generation
5. Dream processing

---

## 💡 Key Features

### 1. Background Processing (백그라운드 사고)

Unconscious mind processes thoughts **while you're not aware**:

```python
from unconscious_mind import UnconsciousMind

unconscious = UnconsciousMind()

# Start background processing
unconscious.start_background_processing()

# Add tasks - they process in background!
unconscious.add_task('problem', 'How to build better AI?')
unconscious.add_task('pattern', 'AI AI machine learning deep learning AI')

# Unconscious works while you do other things!
# Sometimes insights "pop up" to consciousness
```

**How it works**:
- Separate daemon thread
- Continuous processing loop
- Queue-based task management
- Automatic pattern detection
- Random associations (creativity!)

---

### 2. Intuition (직관)

"뭔가 느낌이 와..." - Gut feelings without logic:

```python
# Get intuition about something
intuition = unconscious.get_intuition("Should I use this approach?")
print(intuition)
# → "Trust your instinct on this one"
# → "Something feels right about this"
```

**How intuition works**:
1. Unconscious detects patterns automatically
2. When you ask about something, it matches patterns
3. If pattern matches → strong intuition
4. If no pattern → vague gut feeling

This is exactly how human intuition works!

---

### 3. Pattern Recognition (패턴 인식)

Unconscious is **excellent** at finding patterns:

```python
# Unconscious automatically detects patterns
unconscious.add_task('pattern', 'The cat sat on the cat mat with the cat')

# Pattern detected: "cat" appears 3 times
# Stored in unconscious.patterns
```

**Why this matters**:
- Conscious mind often misses patterns
- Unconscious works automatically
- Finds connections you didn't know existed
- Foundation for intuition

---

### 4. Dream Processing (꿈 처리)

"당신이 꿈꾸는 동안, 무의식은 기억을 정리한다"

Dreams = unconscious memory consolidation:

```python
# Add some tasks
unconscious.add_task('problem', 'Complex problem 1')
unconscious.add_task('problem', 'Complex problem 2')

# Sleep and dream (consolidates memories)
unconscious.dream(duration=5.0)

# After waking:
# - Memories processed
# - Patterns found
# - New insights emerged
```

**Dream process**:
1. Random memory recall (like real dreams)
2. Find patterns in memories
3. Make new connections
4. Consolidate knowledge

**Research shows**: "Sleeping on a problem" actually works!

---

### 5. Emergence to Consciousness (의식으로의 부상)

Sometimes unconscious thoughts "pop up" to consciousness:

```
💡 [Unconscious → Conscious]
   Thought emerged: What if you approach the problem from the opposite direction?
   Source: background
   Confidence: 0.70
```

**Aha! moments** are unconscious thoughts becoming conscious.

---

## 🎯 Complete Mind (Conscious + Unconscious)

The real power is **combining both**:

```python
from unconscious_mind import CompleteMind

# Create complete mind
mind = CompleteMind()

# Think with both conscious AND unconscious
mind.think("What is the meaning of life?")

# Output:
# 🧠 Conscious thinking... [explicit reasoning from LLM]
# 💫 Unconscious insights: [intuition and patterns]
```

**Process**:
1. Conscious thinks explicitly (LLM reasoning)
2. Unconscious processes in background (patterns, intuition)
3. Sometimes unconscious insights emerge
4. **Best of both worlds!**

---

## 📊 System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  COMPLETE MIND                            │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │  CONSCIOUS MIND                                    │  │
│  │  • StreamingLLM (Ollama qwen2.5:3b)               │  │
│  │  • Sequential reasoning                            │  │
│  │  • Explicit, aware thinking                        │  │
│  │  • Slow, effortful                                 │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │  UNCONSCIOUS MIND                                  │  │
│  │  • Background thread (daemon)                      │  │
│  │  • Queue-based task processing                     │  │
│  │  • Automatic pattern recognition                   │  │
│  │  • Intuition generation                            │  │
│  │  • Dream processing                                │  │
│  │  • Parallel, fast, automatic                       │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │  EMERGENCE                                         │  │
│  │  • 10% of unconscious thoughts emerge              │  │
│  │  • "Aha!" moments                                  │  │
│  │  • Sudden insights                                 │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## 🔬 Scientific Basis

### Dual-Process Theory

**System 1 (Unconscious)**:
- Fast, automatic, parallel
- Pattern recognition
- Intuition
- Emotional responses
- **Example**: Recognizing a face instantly

**System 2 (Conscious)**:
- Slow, effortful, sequential
- Logical reasoning
- Deliberate thinking
- **Example**: Solving a math problem

### Research Findings

1. **"Sleep on it" works**: Unconscious problem-solving is real
   - Complex decisions better after sleep
   - Creative solutions emerge unconsciously

2. **Intuition is pattern matching**: Unconscious recognizes patterns
   - Expert intuition = unconscious expertise
   - "Gut feelings" are real pattern detection

3. **Automatic processing**: Most thinking is unconscious
   - 95% of cognitive activity is unconscious
   - Consciousness is the "tip of the iceberg"

---

## 💻 Implementation Details

### Threading Model

```python
# Background processing in separate thread
self.processing_thread = threading.Thread(
    target=self._background_processor,
    daemon=True  # Automatically cleans up
)
self.processing_thread.start()

# Queue for thread-safe communication
self.background_queue = queue.Queue()
```

### Processing Loop

```python
while self.is_running:
    try:
        # Get task from queue (non-blocking)
        task = self.background_queue.get(timeout=1)

        # Process unconsciously
        result = self._process_unconsciously(task)

        # Store thought
        self.unconscious_thoughts.append(result)

        # Sometimes emerge to consciousness (10%)
        if random.random() < 0.1:
            self._emerge_to_conscious(thought)

    except queue.Empty:
        # No tasks, do automatic processing
        self._automatic_processing()
```

### Pattern Detection

```python
def _detect_patterns(self, data: str) -> str:
    words = data.split()
    word_freq = {}

    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Find most frequent pattern
    pattern_word = max(word_freq, key=word_freq.get)
    freq = word_freq[pattern_word]

    if freq > 2:  # Pattern!
        pattern = Pattern(
            pattern=f"Recurring: {pattern_word}",
            frequency=freq,
            confidence=0.8
        )
        self.patterns.append(pattern)
```

---

## 🆚 Comparison

| Aspect | Conscious Mind | Unconscious Mind |
|--------|---------------|------------------|
| **Speed** | Slow | Fast |
| **Effort** | Effortful | Automatic |
| **Mode** | Sequential | Parallel |
| **Awareness** | Aware | Unaware |
| **Best For** | Logic, reasoning | Patterns, intuition |
| **Example** | Solving math | Recognizing faces |
| **Implementation** | LLM (Ollama) | Threading + Queue |

**Complete AGI needs both!**

---

## 🎮 Usage Examples

### Basic Unconscious Processing

```python
from unconscious_mind import UnconsciousMind

unconscious = UnconsciousMind()
unconscious.start_background_processing()

# Add tasks
unconscious.add_task('problem', 'How to optimize this code?')
unconscious.add_task('pattern', 'error error bug error bug error')

# Check statistics
stats = unconscious.get_statistics()
print(stats)
# {
#   'total_unconscious_thoughts': 42,
#   'patterns_discovered': 5,
#   'intuitions_emerged': 3,
#   'background_running': True
# }
```

### Complete Mind (Conscious + Unconscious)

```python
from unconscious_mind import CompleteMind

mind = CompleteMind()

# Think about something
mind.think("What is AGI?")

# Conscious: [Explicit LLM reasoning...]
# Unconscious: [Intuition and patterns...]

# Sleep on a problem
mind.unconscious.add_task('problem', 'Complex challenge')
mind.sleep(duration=5.0)

# Dreams consolidate memories and find patterns!
```

### Dream Processing

```python
# Add experiences during the day
for problem in daily_problems:
    unconscious.add_task('problem', problem)

# Sleep and dream (memory consolidation)
unconscious.dream(duration=10.0)

# Check dream log
for dream in unconscious.dream_log:
    print(f"Patterns found: {dream['patterns_found']}")
    print(f"Dream insights: {dream['content']}")
```

---

## 📈 Statistics and Monitoring

```python
stats = unconscious.get_statistics()

print(stats)
# {
#     'total_unconscious_thoughts': 156,
#     'current_thoughts': 1000,  # maxlen
#     'patterns_discovered': 23,
#     'intuitions_emerged': 8,
#     'dreams_had': 3,
#     'background_running': True,
#     'currently_dreaming': False
# }
```

---

## 🛣️ Roadmap

### Phase 1 (Current)
- ✅ Background processing
- ✅ Pattern recognition
- ✅ Intuition generation
- ✅ Dream processing
- ✅ Conscious + Unconscious integration

### Phase 2 (Next)
- [ ] Long-term memory consolidation
- [ ] Hierarchical pattern learning
- [ ] Emotional integration
- [ ] Priming and subliminal effects

### Phase 3 (Future)
- [ ] Implicit learning (without awareness)
- [ ] Procedural memory
- [ ] Habit formation
- [ ] Creativity enhancement

---

## 🔧 Technical Requirements

```bash
# Core dependencies
pip3 install numpy

# LLM (for conscious mind)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:3b

# Run demo
python3 unconscious_mind.py
```

---

## 🎓 Key Insights

### 1. Why Unconscious Mind Matters

**Traditional AI**: Only conscious reasoning
- Slow, sequential processing
- Limited to what it's aware of
- No intuition or "gut feelings"

**Our Approach**: Conscious + Unconscious
- Fast parallel processing
- Automatic pattern recognition
- Human-like intuition
- More complete intelligence

### 2. Natural Termination

Unconscious mind helps AGI know **when to stop**:
- Satisfaction emerges from unconscious
- Frustration signals to give up
- Curiosity drives continued exploration

Combined with **Emotional AGI**, creates natural stopping conditions!

### 3. Creativity and Insight

**Random associations** in unconscious lead to:
- Novel connections
- Creative solutions
- "Aha!" moments
- Breakthrough insights

This is how human creativity works!

---

## 🤝 Integration with Other Systems

### With Emotional AGI

```python
from emotional_agi import EmotionalAGI
from unconscious_mind import UnconsciousMind

emotions = EmotionalAGI()
unconscious = UnconsciousMind()

# Emotions influence unconscious processing
while not emotions.is_satisfied():
    if emotions.current_emotion == 'curiosity':
        # Unconscious explores more
        unconscious.add_task('exploration', 'new area')

    if emotions.current_emotion == 'frustration':
        # Unconscious tries different approach
        unconscious.add_task('problem', 'alternative solution')
```

### With Complete AGI API

```python
# Add unconscious processing to API server
from unconscious_mind import CompleteMind

class CompleteAGIEngine:
    def __init__(self):
        self.mind = CompleteMind()

    def generate(self, messages):
        # Use both conscious and unconscious!
        query = messages[-1]['content']
        return self.mind.think(query, use_unconscious=True)
```

---

## 📚 Further Reading

### Papers
- **Dual-Process Theory** - Kahneman, 2011 (Thinking, Fast and Slow)
- **Unconscious Thought Theory** - Dijksterhuis, 2006
- **Implicit Learning** - Reber, 1989

### Books
- **Thinking, Fast and Slow** - Daniel Kahneman
- **The Power of Intuition** - Gary Klein
- **How We Decide** - Jonah Lehrer

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

### "완전한 AGI = 의식 + 무의식"
### "Complete AGI = Conscious + Unconscious"

Just like humans, AGI needs:
- **Conscious mind** for deliberate reasoning
- **Unconscious mind** for automatic processing
- **Both working together** for complete intelligence

**"당신이 자는 동안, 무의식이 일한다"**

**"While you sleep, your unconscious works"**

---

**🧠 Built with love in Seoul, Korea**

**Part of the Complete AGI System - [auto-ai](https://github.com/hwkim3330/auto-ai)**
