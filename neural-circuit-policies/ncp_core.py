#!/usr/bin/env python3
"""
Neural Circuit Policies (NCP) - Core Implementation
====================================================

Biologically-inspired neural networks based on C. elegans nervous system

Key Concepts:
1. LTC (Liquid Time-Constant) - Continuous-time neurons with adaptive dynamics
2. NCP Wiring - Sensory → Inter → Command → Motor hierarchical structure
3. CfC (Closed-form Continuous-time) - Efficient approximation of LTC
4. Sparse connectivity - Inspired by biological neural circuits

References:
- Lechner et al. "Neural Circuit Policies Enabling Auditable Autonomy" (2020)
- Hasani et al. "Liquid Time-constant Networks" (2021)
- GitHub: https://github.com/mlech26l/ncps

"302 neurons control C. elegans. We can do better with structure."
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# Neuron Types (following C. elegans structure)
# ============================================================================

class NeuronType(Enum):
    """Neuron types in NCP architecture"""
    SENSORY = "sensory"      # Input neurons (like C. elegans sensory neurons)
    INTER = "inter"          # Interneurons (processing)
    COMMAND = "command"      # Command neurons (decision-making)
    MOTOR = "motor"          # Output neurons (action)


# ============================================================================
# NCP Wiring Structure
# ============================================================================

@dataclass
class NCPWiring:
    """
    Neural Circuit Wiring inspired by C. elegans

    Architecture:
        Sensory → Inter → Command → Motor
                   ↑         ↓
                   └─────────┘ (recurrent in command layer)
    """
    sensory_neurons: int
    inter_neurons: int
    command_neurons: int
    motor_neurons: int

    # Connection matrices (sparse)
    sensory_to_inter: np.ndarray = field(default=None, repr=False)
    inter_to_command: np.ndarray = field(default=None, repr=False)
    command_to_command: np.ndarray = field(default=None, repr=False)  # Recurrent
    command_to_motor: np.ndarray = field(default=None, repr=False)
    inter_to_motor: np.ndarray = field(default=None, repr=False)      # Skip connection

    sparsity: float = 0.3  # 30% connectivity (inspired by biological sparsity)

    def __post_init__(self):
        """Initialize connection matrices"""
        if self.sensory_to_inter is None:
            self.sensory_to_inter = self._create_sparse_matrix(
                self.sensory_neurons, self.inter_neurons
            )

        if self.inter_to_command is None:
            self.inter_to_command = self._create_sparse_matrix(
                self.inter_neurons, self.command_neurons
            )

        if self.command_to_command is None:
            self.command_to_command = self._create_sparse_matrix(
                self.command_neurons, self.command_neurons, recurrent=True
            )

        if self.command_to_motor is None:
            self.command_to_motor = self._create_sparse_matrix(
                self.command_neurons, self.motor_neurons
            )

        if self.inter_to_motor is None:
            self.inter_to_motor = self._create_sparse_matrix(
                self.inter_neurons, self.motor_neurons, sparsity=0.1  # Very sparse skip
            )

    def _create_sparse_matrix(self, rows: int, cols: int,
                             recurrent: bool = False,
                             sparsity: Optional[float] = None) -> np.ndarray:
        """Create sparse connection matrix"""
        if sparsity is None:
            sparsity = self.sparsity

        # Initialize with small random weights
        matrix = np.random.randn(rows, cols) * 0.1

        # Apply sparsity mask
        mask = np.random.rand(rows, cols) < sparsity
        matrix = matrix * mask

        # For recurrent, zero out diagonal (no self-connections)
        if recurrent:
            np.fill_diagonal(matrix, 0)

        return matrix

    @property
    def total_neurons(self) -> int:
        """Total number of neurons"""
        return (self.sensory_neurons + self.inter_neurons +
                self.command_neurons + self.motor_neurons)

    @property
    def total_synapses(self) -> int:
        """Total number of synapses (non-zero connections)"""
        return int(
            np.count_nonzero(self.sensory_to_inter) +
            np.count_nonzero(self.inter_to_command) +
            np.count_nonzero(self.command_to_command) +
            np.count_nonzero(self.command_to_motor) +
            np.count_nonzero(self.inter_to_motor)
        )


def auto_wiring(input_size: int, output_size: int,
                inter_neurons: int = 12, command_neurons: int = 12) -> NCPWiring:
    """
    Automatically create NCP wiring

    Args:
        input_size: Number of input features (sensory neurons)
        output_size: Number of outputs (motor neurons)
        inter_neurons: Number of interneurons
        command_neurons: Number of command neurons (recurrent)

    Returns:
        NCPWiring with appropriate structure

    Example:
        wiring = auto_wiring(32, 8)  # 32 inputs → 8 outputs
        # Creates: 32 sensory → 12 inter → 12 command → 8 motor
    """
    return NCPWiring(
        sensory_neurons=input_size,
        inter_neurons=inter_neurons,
        command_neurons=command_neurons,
        motor_neurons=output_size
    )


# ============================================================================
# Liquid Time-Constant (LTC) Neuron
# ============================================================================

class LTCNeuron:
    """
    Liquid Time-Constant Neuron

    Continuous-time dynamics:
        dx/dt = -x/tau + f(W·input + b)

    Where:
        x: neuron state
        tau: time constant (learnable)
        f: activation function (tanh)
        W: weights
        b: bias
    """

    def __init__(self, input_size: int, num_neurons: int):
        self.input_size = input_size
        self.num_neurons = num_neurons

        # Parameters
        self.W = np.random.randn(num_neurons, input_size) * 0.1
        self.b = np.zeros(num_neurons)
        self.tau = np.ones(num_neurons) * 1.0  # Time constants (learnable)

        # State
        self.x = np.zeros(num_neurons)

    def forward(self, inputs: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        Forward pass with ODE integration (Euler method)

        Args:
            inputs: Input vector
            dt: Time step for integration

        Returns:
            Neuron activations
        """
        # Compute input current
        I = np.dot(self.W, inputs) + self.b

        # Apply activation
        activation = np.tanh(I)

        # ODE: dx/dt = -x/tau + activation
        dx_dt = (-self.x / self.tau) + activation

        # Euler integration: x(t+dt) = x(t) + dx/dt * dt
        self.x = self.x + dx_dt * dt

        # Clip to prevent explosion
        self.x = np.clip(self.x, -10, 10)

        return self.x

    def reset(self):
        """Reset neuron state"""
        self.x = np.zeros(self.num_neurons)


# ============================================================================
# Closed-form Continuous (CfC) Neuron
# ============================================================================

class CfCNeuron:
    """
    Closed-form Continuous-time Neuron

    Efficient approximation of LTC using closed-form solution:
        x(t+dt) = x(t) * exp(-dt/tau) + f(W·input + b) * (1 - exp(-dt/tau))

    Faster than ODE integration while maintaining continuous-time behavior
    """

    def __init__(self, input_size: int, num_neurons: int):
        self.input_size = input_size
        self.num_neurons = num_neurons

        # Parameters
        self.W = np.random.randn(num_neurons, input_size) * 0.1
        self.b = np.zeros(num_neurons)
        self.tau = np.ones(num_neurons) * 1.0

        # State
        self.x = np.zeros(num_neurons)

    def forward(self, inputs: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        Forward pass with closed-form solution

        Args:
            inputs: Input vector
            dt: Time step

        Returns:
            Neuron activations
        """
        # Compute input current
        I = np.dot(self.W, inputs) + self.b
        activation = np.tanh(I)

        # Closed-form solution
        decay = np.exp(-dt / self.tau)
        self.x = self.x * decay + activation * (1 - decay)

        # Clip to prevent explosion
        self.x = np.clip(self.x, -10, 10)

        return self.x

    def reset(self):
        """Reset neuron state"""
        self.x = np.zeros(self.num_neurons)


# ============================================================================
# Neural Circuit Policy (NCP) Network
# ============================================================================

class NeuralCircuitPolicy:
    """
    Complete Neural Circuit Policy

    Combines:
    - NCP wiring (hierarchical structure)
    - CfC neurons (efficient continuous-time dynamics)
    - Biological inspiration (C. elegans)

    Architecture:
        Input (sensory) → Inter → Command (recurrent) → Motor (output)
    """

    def __init__(self, wiring: NCPWiring, use_cfc: bool = True):
        """
        Args:
            wiring: NCP wiring structure
            use_cfc: Use CfC neurons (True) or LTC neurons (False)
        """
        self.wiring = wiring
        self.use_cfc = use_cfc

        NeuronClass = CfCNeuron if use_cfc else LTCNeuron

        # Create neuron layers
        self.sensory = None  # Passthrough (no computation)

        self.inter = NeuronClass(
            wiring.sensory_neurons,
            wiring.inter_neurons
        )

        self.command = NeuronClass(
            wiring.inter_neurons + wiring.command_neurons,  # Inter + recurrent
            wiring.command_neurons
        )

        self.motor = NeuronClass(
            wiring.command_neurons + wiring.inter_neurons,  # Command + skip
            wiring.motor_neurons
        )

        # Apply wiring weights
        self._apply_wiring()

        print(f"[NCP] Created network:")
        print(f"  Neurons: {wiring.total_neurons} ({wiring.sensory_neurons}→{wiring.inter_neurons}→{wiring.command_neurons}→{wiring.motor_neurons})")
        print(f"  Synapses: {wiring.total_synapses}")
        print(f"  Neuron type: {'CfC' if use_cfc else 'LTC'}")
        print(f"  Sparsity: {wiring.sparsity:.1%}")

    def _apply_wiring(self):
        """Apply wiring matrices to neuron layers"""
        # Inter neurons receive from sensory
        self.inter.W = self.wiring.sensory_to_inter.T

        # Command neurons receive from inter + command (recurrent)
        self.command.W = np.column_stack([
            self.wiring.inter_to_command.T,
            self.wiring.command_to_command.T
        ])

        # Motor neurons receive from command + inter (skip connection)
        self.motor.W = np.column_stack([
            self.wiring.command_to_motor.T,
            self.wiring.inter_to_motor.T
        ])

    def forward(self, sensory_input: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        Forward pass through the circuit

        Args:
            sensory_input: Input vector (size = sensory_neurons)
            dt: Time step for continuous-time integration

        Returns:
            Motor output (size = motor_neurons)
        """
        # Sensory → Inter
        inter_out = self.inter.forward(sensory_input, dt)

        # Inter + Command(recurrent) → Command
        command_input = np.concatenate([inter_out, self.command.x])
        command_out = self.command.forward(command_input, dt)

        # Command + Inter(skip) → Motor
        motor_input = np.concatenate([command_out, inter_out])
        motor_out = self.motor.forward(motor_input, dt)

        return motor_out

    def reset(self):
        """Reset all neuron states"""
        self.inter.reset()
        self.command.reset()
        self.motor.reset()

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current state of all neurons"""
        return {
            "inter": self.inter.x.copy(),
            "command": self.command.x.copy(),
            "motor": self.motor.x.copy()
        }


# ============================================================================
# Demo & Testing
# ============================================================================

def demo_ncp():
    """Demonstrate NCP capabilities"""
    print("\n" + "=" * 70)
    print("NEURAL CIRCUIT POLICIES - Demo")
    print("=" * 70)
    print()
    print("Inspired by C. elegans nervous system (302 neurons)")
    print("Hierarchical: Sensory → Inter → Command → Motor")
    print("Sparse connectivity (~30%)")
    print("Continuous-time dynamics (LTC/CfC)")
    print("=" * 70)
    print()

    # Create wiring
    print("[1] Creating NCP wiring...")
    wiring = auto_wiring(
        input_size=32,      # 32 sensory inputs
        output_size=8,      # 8 motor outputs
        inter_neurons=12,   # 12 interneurons
        command_neurons=12  # 12 command neurons (recurrent)
    )

    # Create NCP
    print("\n[2] Creating Neural Circuit Policy...")
    ncp = NeuralCircuitPolicy(wiring, use_cfc=True)

    # Test forward pass
    print("\n[3] Testing forward pass...")
    sensory_input = np.random.randn(32)

    print(f"  Input shape: {sensory_input.shape}")

    # Run for 10 time steps
    outputs = []
    for t in range(10):
        output = ncp.forward(sensory_input, dt=0.1)
        outputs.append(output)

        if t == 0:
            print(f"  t=0: Motor output = {output[:4]}... (showing first 4)")

    print(f"  t=9: Motor output = {output[:4]}...")

    # Show neuron states
    print("\n[4] Neuron states:")
    state = ncp.get_state()
    for layer, values in state.items():
        print(f"  {layer:8s}: mean={values.mean():.3f}, std={values.std():.3f}, active={np.sum(np.abs(values) > 0.1)}/{len(values)}")

    # Compare to C. elegans
    print("\n[5] Comparison to C. elegans:")
    print(f"  C. elegans: 302 neurons, ~7000 synapses")
    print(f"  Our NCP:    {wiring.total_neurons} neurons, {wiring.total_synapses} synapses")
    print(f"  Efficiency: {wiring.total_neurons / 302:.2f}x fewer neurons!")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  - Integrate with Meta-AI Core")
    print("  - Use in Streaming Continuous AGI for time-series reasoning")
    print("  - Replace existing Liquid NN with real LTC/CfC")
    print()


if __name__ == "__main__":
    demo_ncp()
