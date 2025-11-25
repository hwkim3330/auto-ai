# The Sentinel

> *"You are being watched. The government has a secret system, a machine that spies on you every hour of every day."*

**The Sentinel** is a self-learning surveillance system inspired by Person of Interest's Machine. It continuously watches camera feeds, learns from observations, reasons about patterns, and recursively improves its own code.

## Architecture

```
┌─────────────────────────────────────────┐
│         THE SENTINEL (Machine)          │
├─────────────────────────────────────────┤
│  ┌─────────┐      ┌──────────┐         │
│  │  CCTV   │─────>│Perception│         │
│  │ Streams │      │ (Vision) │         │
│  └─────────┘      └──────┬────┘         │
│                          │              │
│                          ▼              │
│              ┌─────────────────┐        │
│              │  Liquid NN      │◄───┐   │
│              │  (Learning)     │    │   │
│              └────────┬─────────┘   │   │
│                       │             │   │
│                       ▼             │   │
│              ┌─────────────────┐   │   │
│              │  UltraThink     │   │   │
│              │  (Reasoning)    │   │   │
│              └────────┬─────────┘   │   │
│                       │             │   │
│                       ▼             │   │
│              ┌─────────────────┐   │   │
│              │  Code Agent     │───┘   │
│              │  (Self-Improve) │        │
│              └─────────────────┘        │
│         Recursive Improvement Loop      │
└─────────────────────────────────────────┘
```

## Core Components

### 1. Vision Perception
- Processes multiple camera streams simultaneously
- Extracts visual features from each frame
- Detects anomalies vs learned baseline
- **Current**: Synthetic features (MVP)
- **Future**: Vision Mamba integration for real camera feeds

### 2. Online Learning Engine (Liquid NN)
- Continuous learning from streaming observations
- **62,304 parameters** - 1.7x more efficient than Transformer
- O(n) complexity vs O(n²) for attention
- Adaptive time constants for temporal modeling
- Self-supervised learning (reconstruction task)

### 3. Reasoning System (UltraThink)
- Multi-agent collaboration (Analyst, Critic, Synthesizer, Innovator)
- Tree-of-Thought exploration
- Meta-cognitive self-reflection
- Triggered on high anomaly detection

### 4. Code Agent (Self-Improvement)
- Analyzes system performance metrics
- Identifies bottlenecks and inefficiencies
- Generates improvement code
- **Recursively applies improvements to itself**
- Maintains improvement history

## Features

- **Multi-Camera Monitoring**: Simultaneously process 4+ camera streams
- **Continuous Learning**: Real-time model updates as new data arrives
- **Anomaly Detection**: Detect unusual patterns vs learned baseline
- **Autonomous Reasoning**: UltraThink analyzes high-confidence anomalies
- **Self-Improvement**: Code Agent recursively optimizes performance
- **Live Dashboard**: Web-based real-time monitoring interface
- **State Persistence**: Saves system state to disk

## Installation

```bash
cd /home/kim/auto-ai/the-sentinel

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch numpy
```

## Usage

### Run The Sentinel

```bash
python3 sentinel.py
```

This will:
1. Initialize 4 simulated cameras (CAM_001 - CAM_004)
2. Run for 200 cycles (demo mode)
3. Save state to `sentinel_state.json`

### View Live Dashboard

```bash
# Open in browser
firefox dashboard.html
# or
chromium-browser dashboard.html
```

The dashboard shows:
- System metrics (uptime, cycles, observations)
- Camera feed status
- Learning engine performance
- Loss curve visualization
- Self-improvement log

### Infinite Mode

For production deployment:

```python
# In sentinel.py main():
sentinel.run()  # No cycle limit - runs forever
```

## Performance Metrics

**Demo Run (200 cycles)**:
- Total Observations: 800 (4 cameras × 200 cycles)
- Model Updates: ~20 (1 update per 10 cycles)
- Self-Improvements: ~2 (1 per 100 cycles)
- Final Loss: ~0.4-0.6 (depends on random init)

## System Loop

Each cycle (100ms / 10 Hz):

1. **Vision**: Process all active camera streams
2. **Learning**: Add observations to buffer, periodic model update
3. **Reasoning**: Analyze patterns every 50 cycles if anomalies detected
4. **Self-Improvement**: Optimize system every 100 cycles

## Code Agent Capabilities

The Code Agent can:
- Monitor learning rate and adjust if loss is high
- Increase update frequency if model updates are too sparse
- Optimize buffer size based on observation rate
- Generate and apply Python code improvements
- Track improvement history (success/failure)

**Example Auto-Generated Improvement**:
```python
# Auto-generated improvement
def apply_improvement(learning_engine):
    for param_group in learning_engine.optimizer.param_groups:
        param_group['lr'] *= 0.5
    print(f"[CodeAgent] Reduced learning rate to {param_group['lr']}")
```

## Integration with Other Projects

### Liquid NN AI
```python
from liquid_nn import LiquidNeuralNetwork, count_parameters

model = LiquidNeuralNetwork(
    input_size=128,
    hidden_size=64,
    output_size=128,
    num_layers=2
)
```

### UltraThink AGI
```python
from ultrathink import UltraThink

reasoning = UltraThink(feature_dim=128, hidden_size=64)
result = reasoning.think("What pattern might this indicate?")
```

## Future Enhancements

- [ ] Real RTSP camera stream integration
- [ ] Vision Mamba for feature extraction
- [ ] Multi-GPU distributed learning
- [ ] Advanced anomaly detection (Isolation Forest, One-Class SVM)
- [ ] Natural language event descriptions
- [ ] Video clip extraction for anomalies
- [ ] Multi-agent reasoning (debate before conclusion)
- [ ] More sophisticated code generation (using LLM API)
- [ ] Automatic hyperparameter optimization
- [ ] Cloud deployment with edge inference

## Technical Details

**Liquid Neural Network Dynamics**:
```
dx/dt = -x/tau(x,I) + f(x,I,theta)

where:
  x: hidden state
  tau: adaptive time constant (0.1 to 10.0)
  f: backbone network
  I: sensory input
```

**Code Agent Improvement Loop**:
```
Metrics → Analysis → Suggestions → Code Generation → Execution → Metrics
    ↑                                                              ↓
    └──────────────────────── Feedback ─────────────────────────────┘
```

## Files

- `sentinel.py` - Main system implementation
- `dashboard.html` - Live monitoring web interface
- `sentinel_state.json` - Saved system state (generated at runtime)
- `.gitignore` - Git ignore rules
- `README.md` - This file

## Security Note

**WARNING**: The Code Agent executes self-generated Python code using `exec()`. This is powerful but potentially dangerous. In production:

1. Run in sandboxed environment (Docker container, VM)
2. Add code validation before execution
3. Restrict allowed operations (whitelist approach)
4. Log all generated code for audit
5. Implement rollback mechanism for failed improvements

Current implementation includes basic error handling but should not be exposed to untrusted inputs.

## License

MIT License - Part of the auto-ai project suite

## Related Projects

- `liquid-nn-ai/` - Liquid Neural Network implementation and training
- `ultrathink-agi/` - AGI-inspired reasoning system
- `vision-mamba-control/` - Vision Mamba for camera processing

---

**"You are being watched. And the Machine is learning."** 🔍
