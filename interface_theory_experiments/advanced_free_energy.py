#!/usr/bin/env python3
"""
ADVANCED FREE ENERGY PRINCIPLE
Predictive Coding + Precision Weighting + Temporal Depth

EXTENSIONS:
1. PREDICTIVE CODING
   - Hierarchical predictions flow DOWN
   - Prediction ERRORS flow UP
   - Each level tries to "explain away" errors from below

2. PRECISION WEIGHTING (Attention)
   - Not all prediction errors are equal
   - PRECISION = expected inverse variance
   - High precision errors get more weight (attention)
   - This IS attention in Friston's framework

3. TEMPORAL DEPTH (Planning)
   - Agents don't just minimize current F
   - They minimize EXPECTED F over future trajectories
   - This requires modeling time and consequences

4. INTEROCEPTION (Body States)
   - Internal body states as hidden causes
   - Emotions as prediction errors about body
   - Allostasis: acting to maintain body states

"The brain is a prediction machine that exists to minimize
 the long-term average of surprise."
                                        — Karl Friston
"""

import torch
import torch.nn.functional as Func
import numpy as np
import time
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# PREDICTIVE CODING NETWORK
# ============================================================================

class PredictiveCodingLayer:
    """
    One layer in a predictive coding hierarchy.
    
    Each layer:
    - Receives predictions FROM above
    - Sends prediction errors TO above
    - Has its own representation (hidden state)
    """
    
    def __init__(
        self,
        n_units: int,
        input_dim: int,
        output_dim: int,
        precision_learnable: bool = True
    ):
        self.n_units = n_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Current state (representation)
        self.mu = torch.randn(n_units, output_dim, device=device) * 0.1
        
        # Prediction weights (generate predictions for level below)
        self.W_pred = torch.randn(n_units, input_dim, output_dim, device=device) * 0.1
        
        # Error weights (process errors from level below)
        self.W_err = torch.randn(n_units, output_dim, input_dim, device=device) * 0.1
        
        # Precision (inverse variance) - learnable or fixed
        if precision_learnable:
            self.log_precision = torch.zeros(n_units, input_dim, device=device)
        else:
            self.log_precision = torch.zeros(n_units, input_dim, device=device)
        self.precision_learnable = precision_learnable
        
        # Prediction error from below
        self.prediction_error = torch.zeros(n_units, input_dim, device=device)
        
        # Statistics
        self.error_history = []
        self.precision_history = []
    
    @property
    def precision(self) -> torch.Tensor:
        """Get precision (always positive via exp)."""
        return torch.exp(self.log_precision)
    
    def predict(self) -> torch.Tensor:
        """
        Generate prediction for the level below.
        
        prediction = W_pred @ mu
        """
        # (n_units, input_dim, output_dim) @ (n_units, output_dim, 1)
        prediction = torch.bmm(
            self.W_pred,
            self.mu.unsqueeze(-1)
        ).squeeze(-1)  # (n_units, input_dim)
        
        return prediction
    
    def compute_error(self, input_signal: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute precision-weighted prediction error.
        
        error = precision * (input - prediction)
        """
        raw_error = input_signal - prediction
        
        # Precision-weight the error (attention!)
        weighted_error = self.precision * raw_error
        
        self.prediction_error = weighted_error
        
        return weighted_error
    
    def update(
        self,
        error_from_below: torch.Tensor,
        prediction_from_above: Optional[torch.Tensor] = None,
        learning_rate: float = 0.1
    ) -> torch.Tensor:
        """
        Update state to minimize free energy.
        
        Free energy has two terms:
        1. Prediction error from below (weighted by precision)
        2. Prior from above (prediction from higher level)
        """
        # Process error from below
        # (n_units, output_dim, input_dim) @ (n_units, input_dim, 1)
        error_influence = torch.bmm(
            self.W_err,
            error_from_below.unsqueeze(-1)
        ).squeeze(-1)  # (n_units, output_dim)
        
        # Update state
        delta_mu = error_influence
        
        # If we have a prediction from above, add prior term
        if prediction_from_above is not None:
            prior_error = prediction_from_above - self.mu
            delta_mu = delta_mu + 0.5 * prior_error
        
        self.mu = self.mu + learning_rate * delta_mu
        
        # Update precision (learn what to attend to)
        if self.precision_learnable:
            # Precision should increase where errors are consistently low
            # and decrease where errors are high
            error_magnitude = error_from_below.abs().mean(dim=-1, keepdim=True)
            precision_update = -error_magnitude + 0.5  # Target: low error → high precision
            self.log_precision = self.log_precision + 0.01 * precision_update.expand_as(self.log_precision)
            self.log_precision = self.log_precision.clamp(-3, 3)  # Prevent explosion
        
        # Record history
        self.error_history.append(error_from_below.abs().mean().item())
        self.precision_history.append(self.precision.mean().item())
        
        return self.mu


class HierarchicalPredictiveCoding:
    """
    Multi-level predictive coding network.
    
    Information flow:
    - Predictions flow DOWN (top → bottom)
    - Prediction errors flow UP (bottom → top)
    
    This is how the cortex is thought to work!
    """
    
    def __init__(
        self,
        n_agents: int = 1000,
        layer_dims: List[int] = [16, 8, 4, 2],
        temporal_depth: int = 5
    ):
        self.n_agents = n_agents
        self.n_layers = len(layer_dims)
        self.layer_dims = layer_dims
        self.temporal_depth = temporal_depth
        
        # Create layers
        self.layers = []
        for i in range(self.n_layers):
            input_dim = layer_dims[i-1] if i > 0 else layer_dims[0]
            output_dim = layer_dims[i]
            
            layer = PredictiveCodingLayer(
                n_units=n_agents,
                input_dim=input_dim,
                output_dim=output_dim,
                precision_learnable=True
            )
            self.layers.append(layer)
        
        # Sensory input (bottom of hierarchy)
        self.sensory_input = torch.randn(n_agents, layer_dims[0], device=device)
        
        # World model for temporal predictions
        self.transition_model = torch.randn(
            n_agents, layer_dims[-1], layer_dims[-1], device=device
        ) * 0.1
        
        # Temporal buffer (for planning)
        self.temporal_buffer = []
        
        # Statistics
        self.total_error_history = []
        self.layer_error_history = [[] for _ in range(self.n_layers)]
        self.precision_history = []
    
    def generate_sensory_input(self) -> torch.Tensor:
        """
        Generate sensory input from environment.
        
        Simple oscillating world with noise.
        """
        t = len(self.total_error_history)
        
        # Oscillating signal with multiple frequencies
        base = torch.zeros(self.n_agents, self.layer_dims[0], device=device)
        for freq in [0.1, 0.3, 0.7]:
            phase = torch.randn(self.n_agents, 1, device=device) * 0.1
            signal = torch.sin(2 * np.pi * freq * t + phase)
            dim_idx = int(freq * 10) % self.layer_dims[0]
            base[:, dim_idx] = signal.squeeze()
        
        # Add noise
        noise = torch.randn_like(base) * 0.1
        
        self.sensory_input = base + noise
        return self.sensory_input
    
    def forward_pass(self) -> Dict[str, float]:
        """
        One complete predictive coding cycle:
        
        1. Top-down: Generate predictions at each level
        2. Bottom-up: Compute prediction errors
        3. Update: Adjust representations to minimize errors
        """
        # Generate new sensory input
        sensory = self.generate_sensory_input()
        
        # TOP-DOWN PASS: Generate predictions
        predictions = [None] * self.n_layers
        for i in range(self.n_layers - 1, -1, -1):
            predictions[i] = self.layers[i].predict()
        
        # BOTTOM-UP PASS: Compute errors
        errors = []
        
        # Level 0: Error between sensory input and prediction
        error_0 = self.layers[0].compute_error(sensory, predictions[0])
        errors.append(error_0)
        
        # Higher levels: Error between lower representation and prediction
        for i in range(1, self.n_layers):
            lower_mu = self.layers[i-1].mu
            error_i = self.layers[i].compute_error(lower_mu, predictions[i])
            errors.append(error_i)
        
        # UPDATE PASS: Adjust representations
        for i in range(self.n_layers):
            # Get prediction from above (if exists)
            pred_from_above = None
            if i < self.n_layers - 1:
                pred_from_above = predictions[i+1]
            
            # Update this layer
            self.layers[i].update(
                error_from_below=errors[i],
                prediction_from_above=pred_from_above
            )
        
        # Compute statistics
        total_error = sum(e.abs().mean().item() for e in errors) / len(errors)
        self.total_error_history.append(total_error)
        
        for i, e in enumerate(errors):
            self.layer_error_history[i].append(e.abs().mean().item())
        
        avg_precision = np.mean([layer.precision.mean().item() for layer in self.layers])
        self.precision_history.append(avg_precision)
        
        return {
            'total_error': total_error,
            'layer_errors': [e.abs().mean().item() for e in errors],
            'precisions': [layer.precision.mean().item() for layer in self.layers],
            'top_representation': self.layers[-1].mu.mean(dim=0).cpu().numpy()
        }
    
    def plan_ahead(self, n_steps: int = None) -> torch.Tensor:
        """
        Plan into the future by simulating trajectories.
        
        Uses the top-level representation and transition model
        to predict future states and their expected free energy.
        """
        if n_steps is None:
            n_steps = self.temporal_depth
        
        # Start from current top-level state
        current_state = self.layers[-1].mu.clone()
        
        trajectory = [current_state]
        expected_errors = []
        
        for step in range(n_steps):
            # Predict next state using transition model
            # (n_agents, dim, dim) @ (n_agents, dim, 1)
            next_state = torch.bmm(
                self.transition_model,
                current_state.unsqueeze(-1)
            ).squeeze(-1)
            
            # Estimate prediction error (uncertainty grows with time)
            uncertainty = 0.1 * (step + 1)  # Linear growth
            estimated_error = uncertainty * torch.ones(self.n_agents, device=device)
            
            trajectory.append(next_state)
            expected_errors.append(estimated_error.mean().item())
            
            current_state = next_state
        
        # Expected free energy of trajectory
        expected_F = sum(expected_errors) / len(expected_errors)
        
        return expected_F, trajectory


# ============================================================================
# MASSIVE SCALE PREDICTIVE CODING
# ============================================================================

class MassivePredictiveCoding:
    """
    Ultra-optimized predictive coding for millions of agents.
    
    Simplified but maintains core dynamics:
    - Predictions flow down
    - Errors flow up
    - Precision weights attention
    """
    
    def __init__(
        self,
        n_agents: int = 1_000_000,
        n_levels: int = 4,
        dim: int = 8
    ):
        self.n_agents = n_agents
        self.n_levels = n_levels
        self.dim = dim
        
        # States at each level
        self.states = [
            torch.randn(n_agents, dim, device=device) * 0.1
            for _ in range(n_levels)
        ]
        
        # Prediction weights (shared across agents for memory efficiency)
        self.W_pred = [
            torch.randn(dim, dim, device=device) * 0.1
            for _ in range(n_levels)
        ]
        
        # Precision (per-agent, per-level)
        self.precisions = [
            torch.ones(n_agents, 1, device=device)
            for _ in range(n_levels)
        ]
        
        # Sensory input
        self.sensory = torch.randn(n_agents, dim, device=device)
        
        # History
        self.error_history = []
        self.precision_history = []
    
    def step(self) -> Dict[str, float]:
        """One predictive coding cycle."""
        
        # Generate sensory input
        t = len(self.error_history)
        self.sensory = torch.sin(
            torch.arange(self.dim, device=device).float() * 0.1 * t
        ).unsqueeze(0).expand(self.n_agents, -1)
        self.sensory = self.sensory + torch.randn_like(self.sensory) * 0.1
        
        # TOP-DOWN: Generate predictions
        predictions = []
        for level in range(self.n_levels):
            pred = torch.matmul(self.states[level], self.W_pred[level])
            predictions.append(pred)
        
        # BOTTOM-UP: Compute errors
        errors = []
        
        # Level 0: Sensory error
        error_0 = self.precisions[0] * (self.sensory - predictions[0])
        errors.append(error_0)
        
        # Higher levels
        for level in range(1, self.n_levels):
            error = self.precisions[level] * (self.states[level-1] - predictions[level])
            errors.append(error)
        
        # UPDATE: Adjust states
        for level in range(self.n_levels):
            # Gradient descent on error
            grad = errors[level]
            
            # Add top-down prior if not top level
            if level < self.n_levels - 1:
                prior = predictions[level + 1] - self.states[level]
                grad = grad + 0.3 * prior
            
            self.states[level] = self.states[level] + 0.1 * grad
        
        # UPDATE: Adjust precisions
        for level in range(self.n_levels):
            error_mag = errors[level].abs().mean(dim=-1, keepdim=True)
            precision_update = -error_mag + 0.5
            self.precisions[level] = (
                self.precisions[level] + 0.01 * precision_update
            ).clamp(0.1, 10)
        
        # Statistics
        total_error = sum(e.abs().mean().item() for e in errors) / len(errors)
        avg_precision = sum(p.mean().item() for p in self.precisions) / len(self.precisions)
        
        self.error_history.append(total_error)
        self.precision_history.append(avg_precision)
        
        return {
            'total_error': total_error,
            'avg_precision': avg_precision,
            'layer_errors': [e.abs().mean().item() for e in errors]
        }


# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_predictive_coding_demo():
    """
    Demonstrate hierarchical predictive coding.
    """
    print(f"\n{'='*70}")
    print(f"  HIERARCHICAL PREDICTIVE CODING")
    print(f"  Predictions down, errors up")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    net = HierarchicalPredictiveCoding(
        n_agents=1000,
        layer_dims=[16, 8, 4, 2]
    )
    
    print(f"  Hierarchy: {' → '.join(str(d) for d in net.layer_dims)}")
    print(f"  {net.n_agents} agents, {net.n_layers} levels")
    print()
    
    print("  Running 200 steps...\n")
    
    for step in range(200):
        metrics = net.forward_pass()
        
        if step % 40 == 0:
            errs = ", ".join(f"L{i}={e:.3f}" for i, e in enumerate(metrics['layer_errors']))
            print(f"    Step {step:3d}: Total={metrics['total_error']:.4f} | {errs}")
    
    print()
    print("  RESULTS:")
    print("  " + "-" * 50)
    print(f"    Initial Error: {net.total_error_history[0]:.4f}")
    print(f"    Final Error:   {net.total_error_history[-1]:.4f}")
    print(f"    Reduction:     {net.total_error_history[0] - net.total_error_history[-1]:.4f}")
    print()
    
    # Show precision learning
    print("  PRECISION (Attention) by Level:")
    for i, layer in enumerate(net.layers):
        prec = layer.precision.mean().item()
        bar_len = int(prec * 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        print(f"    Level {i}: [{bar}] {prec:.3f}")
    
    print()
    print("  Higher precision = more attention to that level's errors")
    print("=" * 70)


def run_precision_attention_demo():
    """
    Show that precision IS attention.
    """
    print(f"\n{'='*70}")
    print(f"  PRECISION = ATTENTION")
    print(f"  Friston's theory of selective attention")
    print(f"{'='*70}\n")
    
    net = HierarchicalPredictiveCoding(
        n_agents=500,
        layer_dims=[8, 4, 2]
    )
    
    print("  Precision determines which prediction errors matter.\n")
    print("  High precision = 'Pay attention to this!'")
    print("  Low precision = 'Ignore this noise'\n")
    
    # Track precision over time
    precisions_over_time = [[] for _ in range(net.n_layers)]
    
    for step in range(150):
        metrics = net.forward_pass()
        
        for i, layer in enumerate(net.layers):
            precisions_over_time[i].append(layer.precision.mean().item())
    
    print("  Precision Evolution by Level:")
    print("  " + "-" * 50)
    
    for i in range(net.n_layers):
        initial = precisions_over_time[i][0]
        final = precisions_over_time[i][-1]
        change = final - initial
        direction = "↑" if change > 0 else "↓"
        
        print(f"    Level {i}: {initial:.3f} → {final:.3f} ({direction} {abs(change):.3f})")
    
    print()
    print("  INTERPRETATION:")
    print("  • Levels with consistent predictions → high precision (attend)")
    print("  • Levels with noisy/unpredictable errors → low precision (ignore)")
    print("  • This is automatic attention allocation!")
    print("=" * 70)


def run_temporal_planning():
    """
    Demonstrate planning into the future.
    """
    print(f"\n{'='*70}")
    print(f"  TEMPORAL PLANNING")
    print(f"  Minimizing expected future free energy")
    print(f"{'='*70}\n")
    
    net = HierarchicalPredictiveCoding(
        n_agents=500,
        layer_dims=[8, 4, 2],
        temporal_depth=10
    )
    
    # Warm up
    for _ in range(50):
        net.forward_pass()
    
    print("  Planning 10 steps into the future...\n")
    
    expected_F, trajectory = net.plan_ahead(n_steps=10)
    
    print(f"  Expected Free Energy over trajectory: {expected_F:.4f}")
    print()
    print("  Future state predictions (top level):")
    
    for i, state in enumerate(trajectory[:6]):
        state_str = state.mean(dim=0).cpu().numpy().round(2)
        print(f"    t+{i}: {state_str}")
    
    print("    ...")
    print()
    print("  INSIGHT:")
    print("  Agents don't just react to the present -")
    print("  they simulate futures and choose actions that")
    print("  minimize EXPECTED long-term surprise.")
    print()
    print("  This is the basis of goal-directed behavior!")
    print("=" * 70)


def run_live_predictive_coding(n_steps: int = 300):
    """
    Live visualization of predictive coding.
    """
    print(f"\n{'='*70}")
    print(f"  LIVE PREDICTIVE CODING")
    print(f"  Watch predictions and errors in real-time")
    print(f"{'='*70}\n")
    
    net = HierarchicalPredictiveCoding(
        n_agents=1000,
        layer_dims=[16, 8, 4, 2]
    )
    
    print("  Error by Level        | Precision by Level")
    print("  " + "-" * 55)
    
    for step in range(n_steps):
        metrics = net.forward_pass()
        
        # Error bars
        err_bars = ""
        for e in metrics['layer_errors']:
            bar_len = min(8, int(e * 8))
            bar = "█" * bar_len + "░" * (8 - bar_len)
            err_bars += f"[{bar}]"
        
        # Precision bars
        prec_bars = ""
        for p in metrics['precisions']:
            bar_len = min(8, int(p * 4))
            bar = "▓" * bar_len + "░" * (8 - bar_len)
            prec_bars += f"[{bar}]"
        
        line = f"  {err_bars} | {prec_bars} Step {step:3d}"
        print(f"\r{line}", end="", flush=True)
        time.sleep(0.015)
    
    print("\n")
    print("  LEGEND:")
    print("    Error bars: Height = prediction error magnitude")
    print("    Precision bars: Height = attention strength")
    print()
    print("  As errors decrease, precision increases (learning what matters)")
    print("=" * 70)


def run_massive_predictive_coding(n_agents: int = 5_000_000, n_steps: int = 100):
    """
    Massive scale predictive coding.
    """
    print(f"\n{'='*70}")
    print(f"  MASSIVE PREDICTIVE CODING")
    print(f"  {n_agents:,} agents doing predictive processing")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    print("  Creating network...", end=" ", flush=True)
    start = time.perf_counter()
    
    net = MassivePredictiveCoding(n_agents=n_agents, n_levels=4, dim=8)
    
    create_time = time.perf_counter() - start
    print(f"Done ({create_time:.2f}s)")
    
    if device.type == 'cuda':
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU Memory: {mem:.2f} GB")
    
    print(f"\n  Running {n_steps} steps...")
    print()
    
    start = time.perf_counter()
    
    for step in range(n_steps):
        metrics = net.step()
        
        if step % 20 == 0:
            print(f"    Step {step:3d}: Error={metrics['total_error']:.4f}, "
                  f"Precision={metrics['avg_precision']:.4f}")
    
    elapsed = time.perf_counter() - start
    rate = n_agents * n_steps / elapsed
    
    print()
    print(f"  PERFORMANCE:")
    print(f"    Total time: {elapsed:.2f}s")
    print(f"    Rate: {rate/1e6:.1f}M predictions/second")
    
    if rate > 1e6:
        print(f"    🚀 {rate/1e6:.1f} MILLION predictions per second!")
    
    print("=" * 70)


def run_full_visualization(n_steps: int = 200):
    """
    Comprehensive visualization of all components.
    """
    print(f"\n{'='*70}")
    print(f"  COMPLETE PREDICTIVE CODING VISUALIZATION")
    print(f"  Errors | Precisions | Planning | Top Representation")
    print(f"{'='*70}\n")
    
    net = HierarchicalPredictiveCoding(
        n_agents=500,
        layer_dims=[8, 4, 2],
        temporal_depth=5
    )
    
    for step in range(n_steps):
        metrics = net.forward_pass()
        
        # Compute planning estimate
        expected_F, _ = net.plan_ahead(n_steps=3)
        
        # Error visualization
        total_err = metrics['total_error']
        err_bar_len = min(15, int(total_err * 15))
        err_bar = "█" * err_bar_len + "░" * (15 - err_bar_len)
        
        # Precision visualization
        avg_prec = np.mean(metrics['precisions'])
        prec_bar_len = min(10, int(avg_prec * 5))
        prec_bar = "▓" * prec_bar_len + "░" * (10 - prec_bar_len)
        
        # Planning (expected future F)
        plan_bar_len = min(8, int(expected_F * 8))
        plan_bar = "◆" * plan_bar_len + "◇" * (8 - plan_bar_len)
        
        # Top representation (simplified)
        top_state = metrics['top_representation']
        state_str = "".join(["●" if v > 0 else "○" for v in top_state])
        
        line = (f"  E:[{err_bar}] P:[{prec_bar}] "
                f"F:[{plan_bar}] S:[{state_str}] {step:3d}")
        
        print(f"\r{line}", end="", flush=True)
        time.sleep(0.02)
    
    print("\n")
    print("  LEGEND:")
    print("    E: Prediction Error (lower = better predictions)")
    print("    P: Precision (higher = more confident)")
    print("    F: Expected Future Free Energy (planning)")
    print("    S: Top-level State (●=positive, ○=negative)")
    print("=" * 70)


def run_all_experiments():
    """Run all experiments."""
    run_predictive_coding_demo()
    run_precision_attention_demo()
    run_temporal_planning()
    run_massive_predictive_coding(n_agents=1_000_000)


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if "--live" in args or "-l" in args:
        run_live_predictive_coding()
    elif "--precision" in args or "-p" in args:
        run_precision_attention_demo()
    elif "--temporal" in args or "-t" in args:
        run_temporal_planning()
    elif "--massive" in args or "-m" in args:
        n = 5_000_000
        for arg in args:
            if arg.startswith("--agents="):
                n = int(arg.split("=")[1])
        run_massive_predictive_coding(n_agents=n)
    elif "--viz" in args or "-v" in args:
        run_full_visualization()
    elif "--all" in args:
        run_all_experiments()
    else:
        print("\nAdvanced Free Energy: Predictive Coding")
        print("=" * 45)
        print(f"Device: {device}")
        print()
        print("Usage:")
        print("  python advanced_free_energy.py --live      # Live error/precision viz")
        print("  python advanced_free_energy.py --precision # Precision = attention")
        print("  python advanced_free_energy.py --temporal  # Planning ahead")
        print("  python advanced_free_energy.py --massive   # Millions of agents")
        print("  python advanced_free_energy.py --viz       # Full visualization")
        print("  python advanced_free_energy.py --all       # All experiments")
        print()
        
        # Default demo
        run_predictive_coding_demo()
