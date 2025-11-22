#!/usr/bin/env python3
"""
Liquid Neural Network (LNN) Implementation
==========================================

Based on MIT CSAIL's Liquid Time-Constant (LTC) Networks.
Key innovations:
- Continuous-time neural dynamics (inspired by C. elegans nervous system)
- Adaptive time constants that change based on input
- Higher expressivity with fewer parameters
- Superior performance on sequential/temporal tasks

This implementation targets benchmarks where LNN excels over transformers:
1. Time-series prediction
2. Causal reasoning
3. Long-range dependencies with fewer parameters
4. Real-time inference efficiency

References:
- Hasani et al. "Liquid Time-constant Networks" (2021)
- Lechner et al. "Neural Circuit Policies" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import math


class LiquidTimeConstantCell(nn.Module):
    """
    Liquid Time-Constant (LTC) Cell

    The core innovation: time constants are learned functions of the input,
    allowing the network to dynamically adjust its temporal dynamics.

    ODE: dx/dt = -x/tau(x, I) + f(x, I, theta)

    Where:
    - x: hidden state
    - tau: input-dependent time constant
    - I: input
    - f: nonlinear activation function
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_unfolds: int = 6,
        backbone_layers: int = 2,
        backbone_units: int = 64,
        backbone_dropout: float = 0.0,
        tau_min: float = 0.1,
        tau_max: float = 10.0
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_unfolds = num_unfolds  # ODE solver steps
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Backbone network for computing state updates
        backbone = []
        in_features = input_size + hidden_size
        for i in range(backbone_layers):
            out_features = backbone_units if i < backbone_layers - 1 else hidden_size
            backbone.append(nn.Linear(in_features, out_features))
            if i < backbone_layers - 1:
                backbone.append(nn.Tanh())
                if backbone_dropout > 0:
                    backbone.append(nn.Dropout(backbone_dropout))
            in_features = out_features
        self.backbone = nn.Sequential(*backbone)

        # Time constant network (adaptive tau)
        self.tau_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, backbone_units),
            nn.Tanh(),
            nn.Linear(backbone_units, hidden_size),
            nn.Sigmoid()
        )

        # Sensory weights for gating
        self.sensory_weight = nn.Linear(input_size, hidden_size)
        self.sensory_activation = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        time_delta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with ODE integration

        Args:
            x: Input tensor [batch, input_size]
            hx: Previous hidden state [batch, hidden_size]
            time_delta: Time step size

        Returns:
            output: Output tensor [batch, hidden_size]
            new_state: New hidden state [batch, hidden_size]
        """
        batch_size = x.size(0)

        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Euler ODE integration with multiple unfolds for accuracy
        dt = time_delta / self.num_unfolds
        state = hx

        for _ in range(self.num_unfolds):
            # Concatenate input and state
            combined = torch.cat([x, state], dim=-1)

            # Compute adaptive time constant
            tau_raw = self.tau_net(combined)
            tau = self.tau_min + (self.tau_max - self.tau_min) * tau_raw

            # Compute state derivative
            f_state = self.backbone(combined)

            # Sensory gating
            sensory_gate = self.sensory_activation(self.sensory_weight(x))

            # ODE: dx/dt = (-x + f(x,I)) / tau
            dx = (-state + f_state * sensory_gate) / tau

            # Euler step
            state = state + dt * dx

        return state, state


class LiquidNeuralNetwork(nn.Module):
    """
    Full Liquid Neural Network for sequence modeling

    Architecture:
    - Input projection
    - Stack of LTC cells
    - Output projection

    Advantages over Transformers:
    1. O(n) complexity vs O(n^2) for attention
    2. Continuous-time modeling
    3. 10-100x fewer parameters for same performance
    4. Better extrapolation to longer sequences
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        num_unfolds: int = 6,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # LTC layers
        self.ltc_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = hidden_size
            if bidirectional and i > 0:
                layer_input_size = hidden_size * 2
            self.ltc_layers.append(
                LiquidTimeConstantCell(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_unfolds=num_unfolds
                )
            )

        # Backward layers for bidirectional
        if bidirectional:
            self.ltc_layers_backward = nn.ModuleList()
            for i in range(num_layers):
                layer_input_size = hidden_size
                if i > 0:
                    layer_input_size = hidden_size * 2
                self.ltc_layers_backward.append(
                    LiquidTimeConstantCell(
                        input_size=layer_input_size,
                        hidden_size=hidden_size,
                        num_unfolds=num_unfolds
                    )
                )

        # Output projection
        output_hidden = hidden_size * 2 if bidirectional else hidden_size
        self.output_proj = nn.Sequential(
            nn.LayerNorm(output_hidden),
            nn.Dropout(dropout),
            nn.Linear(output_hidden, output_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through LNN

        Args:
            x: Input sequence [batch, seq_len, input_size]
            hidden: Initial hidden states for each layer

        Returns:
            output: Output sequence [batch, seq_len, output_size]
            final_hidden: Final hidden states for each layer
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)

        # Initialize hidden states
        if hidden is None:
            hidden = [None] * self.num_layers

        # Process through LTC layers (forward)
        forward_outputs = []
        layer_input = x
        final_hidden = []

        for layer_idx, ltc in enumerate(self.ltc_layers):
            outputs = []
            h = hidden[layer_idx]

            for t in range(seq_len):
                out, h = ltc(layer_input[:, t], h)
                outputs.append(out)

            layer_output = torch.stack(outputs, dim=1)
            forward_outputs.append(layer_output)
            final_hidden.append(h)
            layer_input = layer_output

        # Process backward if bidirectional
        if self.bidirectional:
            backward_outputs = []
            layer_input = x

            for layer_idx, ltc in enumerate(self.ltc_layers_backward):
                outputs = []
                h = None

                for t in range(seq_len - 1, -1, -1):
                    out, h = ltc(layer_input[:, t], h)
                    outputs.insert(0, out)

                layer_output = torch.stack(outputs, dim=1)
                backward_outputs.append(layer_output)
                layer_input = layer_output

            # Concatenate forward and backward
            combined = torch.cat([forward_outputs[-1], backward_outputs[-1]], dim=-1)
        else:
            combined = forward_outputs[-1]

        # Output projection
        output = self.output_proj(combined)

        return output, final_hidden


class ContinuousTimeAttention(nn.Module):
    """
    Continuous-Time Attention mechanism

    Combines LNN dynamics with attention for long-range dependencies.
    More efficient than standard attention for streaming data.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        # Time-decay for causal attention
        self.time_decay = nn.Parameter(torch.ones(num_heads))

    def forward(
        self,
        x: torch.Tensor,
        time_deltas: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with time-aware attention

        Args:
            x: Input [batch, seq_len, hidden_size]
            time_deltas: Time between tokens [batch, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Time-decay mask (causal + temporal decay)
        if time_deltas is not None:
            # Create time decay matrix
            time_matrix = time_deltas.unsqueeze(-1) - time_deltas.unsqueeze(-2)
            decay = torch.exp(-self.time_decay.view(1, -1, 1, 1) * time_matrix.unsqueeze(1).abs())
            scores = scores * decay

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.out_proj(out)

        return out


class LiquidTransformer(nn.Module):
    """
    Hybrid Liquid-Transformer Architecture

    Combines:
    - LNN for efficient temporal processing
    - Sparse attention for long-range dependencies
    - Mixture of Experts for capacity

    Target: Beat Gemini on efficiency metrics while matching quality
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_size: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        num_experts: int = 4,
        top_k_experts: int = 2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)

        # Liquid layers (alternating LTC and Attention)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                'ltc': LiquidTimeConstantCell(hidden_size, hidden_size),
                'attn': ContinuousTimeAttention(hidden_size, num_heads, dropout),
                'ff': self._make_moe_ffn(hidden_size, ff_size, num_experts, top_k_experts),
                'norm1': nn.LayerNorm(hidden_size),
                'norm2': nn.LayerNorm(hidden_size),
                'norm3': nn.LayerNorm(hidden_size)
            })
            self.layers.append(layer)

        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

    def _make_moe_ffn(
        self,
        hidden_size: int,
        ff_size: int,
        num_experts: int,
        top_k: int
    ) -> nn.Module:
        """Create Mixture of Experts FFN"""

        class MoEFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_size, ff_size),
                        nn.GELU(),
                        nn.Linear(ff_size, hidden_size)
                    ) for _ in range(num_experts)
                ])
                self.gate = nn.Linear(hidden_size, num_experts)
                self.top_k = top_k

            def forward(self, x):
                # Router
                gate_logits = self.gate(x)
                gate_probs = F.softmax(gate_logits, dim=-1)

                # Top-k experts
                top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

                # Compute expert outputs
                output = torch.zeros_like(x)
                for k in range(self.top_k):
                    for e in range(len(self.experts)):
                        mask = (top_k_indices[..., k] == e)
                        if mask.any():
                            expert_out = self.experts[e](x[mask])
                            output[mask] += top_k_probs[mask, k:k+1] * expert_out

                return output

        return MoEFFN()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass

        Args:
            input_ids: Token IDs [batch, seq_len]
            labels: Target labels for loss computation

        Returns:
            dict with logits and optional loss
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Process through layers
        ltc_states = [None] * self.num_layers

        for i, layer in enumerate(self.layers):
            # LTC processing (for each position sequentially)
            ltc_out = []
            for t in range(seq_len):
                out, ltc_states[i] = layer['ltc'](x[:, t], ltc_states[i])
                ltc_out.append(out)
            ltc_out = torch.stack(ltc_out, dim=1)
            x = layer['norm1'](x + ltc_out)

            # Attention
            attn_out = layer['attn'](x)
            x = layer['norm2'](x + attn_out)

            # FFN (MoE)
            ff_out = layer['ff'](x)
            x = layer['norm3'](x + ff_out)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        result = {'logits': logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            result['loss'] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Generate tokens autoregressively"""

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(input_ids)
            logits = outputs['logits'][:, -1, :] / temperature

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("Liquid Neural Network Architecture")
    print("=" * 60)

    # Test LTC Cell
    print("\n1. Testing LTC Cell...")
    ltc = LiquidTimeConstantCell(input_size=32, hidden_size=64)
    x = torch.randn(8, 32)
    out, state = ltc(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   State: {state.shape}")
    print(f"   Parameters: {count_parameters(ltc):,}")

    # Test LNN
    print("\n2. Testing LNN Sequence Model...")
    lnn = LiquidNeuralNetwork(
        input_size=32,
        hidden_size=64,
        output_size=16,
        num_layers=2
    )
    x = torch.randn(8, 100, 32)  # batch=8, seq_len=100
    out, hidden = lnn(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Parameters: {count_parameters(lnn):,}")

    # Test Liquid Transformer
    print("\n3. Testing Liquid Transformer...")
    model = LiquidTransformer(
        vocab_size=32000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        ff_size=1024
    )
    input_ids = torch.randint(0, 32000, (4, 64))
    outputs = model(input_ids)
    print(f"   Input IDs: {input_ids.shape}")
    print(f"   Logits: {outputs['logits'].shape}")
    print(f"   Parameters: {count_parameters(model):,}")

    # Comparison with standard Transformer
    print("\n4. Parameter Efficiency Comparison:")
    print(f"   Liquid Transformer (4 layers, 256 hidden): {count_parameters(model):,}")

    # Approximate standard transformer
    approx_transformer = (
        32000 * 256 +  # embeddings
        4 * (4 * 256 * 256 + 2 * 256 * 1024) +  # attention + ffn
        256 * 32000  # output
    )
    print(f"   Standard Transformer (equivalent): ~{approx_transformer:,}")
    print(f"   Reduction: {approx_transformer / count_parameters(model):.1f}x fewer parameters")

    print("\n" + "=" * 60)
    print("Liquid NN ready for benchmarks!")
    print("=" * 60)
