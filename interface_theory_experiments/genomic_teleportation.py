#!/usr/bin/env python3
"""
Genomic Teleportation Experiment

Tests the "Admin Access" concept from the Hoffman/Arkani-Hamed theory:
- In EVOLUTION space: Agents can "jump" without traversing distance
- In PHASE space: Agents must integrate the path (actual time)

This measures the "latency" difference between:
1. Coordinate Update (instant) - "Change the file path"
2. Path Integration (slow) - "Move the file across the drive"

THE INSIGHT:
Evolution operates in the "mathematical world" where distance is similarity,
not physical space. A mutation can teleport to a distant peak instantly.

Physics operates in "phase space" where you must integrate time to move.

If Hoffman is right, the "integration" is just rendering latency.
"""

import numpy as np
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Agent:
    """An agent that can exist in either evolution or phase space."""
    position: np.ndarray  # Current position in search space
    fitness: float = 0.0
    
    def distance_to(self, target: np.ndarray) -> float:
        return np.linalg.norm(self.position - target)


def fitness_landscape(pos: np.ndarray) -> float:
    """
    Multi-modal fitness landscape with peaks and valleys.
    """
    # Multiple peaks
    peak1 = np.exp(-np.sum((pos - np.array([10, 10]))**2) / 20)
    peak2 = np.exp(-np.sum((pos - np.array([-10, -10]))**2) / 20) * 0.8
    peak3 = np.exp(-np.sum((pos - np.array([10, -10]))**2) / 20) * 0.6
    
    return peak1 + peak2 + peak3


def evolution_teleport(agent: Agent, target: np.ndarray) -> Tuple[float, int]:
    """
    GENOMIC/EVOLUTION SPACE: Teleportation via mutation.
    
    In evolution, we don't "move" - offspring just APPEAR at new location.
    This is O(1) - instant coordinate update.
    
    Returns: (time_taken, steps)
    """
    start_time = time.perf_counter()
    
    # Instant teleport - just update coordinates
    agent.position = target.copy()
    agent.fitness = fitness_landscape(agent.position)
    
    elapsed = time.perf_counter() - start_time
    return elapsed, 1  # 1 step - instant


def phase_space_traverse(agent: Agent, target: np.ndarray, 
                         step_size: float = 0.5) -> Tuple[float, int]:
    """
    PHASE SPACE: Must integrate the path.
    
    In physics, you can't teleport. You must accelerate, travel, decelerate.
    This is O(distance) - linear in distance traveled.
    
    Returns: (time_taken, steps)
    """
    start_time = time.perf_counter()
    steps = 0
    
    while agent.distance_to(target) > step_size:
        # Calculate direction
        direction = target - agent.position
        direction = direction / np.linalg.norm(direction)
        
        # Move one step
        agent.position = agent.position + direction * step_size
        agent.fitness = fitness_landscape(agent.position)
        steps += 1
        
        # Simulate "computation cost" of integration
        time.sleep(0.001)  # 1ms per step
    
    # Final step
    agent.position = target.copy()
    agent.fitness = fitness_landscape(agent.position)
    steps += 1
    
    elapsed = time.perf_counter() - start_time
    return elapsed, steps


def run_teleportation_experiment(n_trials: int = 10):
    """
    Compare evolution teleportation vs phase space traversal.
    """
    print("\n" + "=" * 70)
    print("  GENOMIC TELEPORTATION EXPERIMENT")
    print("  Evolution Space vs Phase Space")
    print("=" * 70 + "\n")
    
    distances = [5, 10, 20, 50, 100]
    results = []
    
    print("  Testing different distances...\n")
    print(f"  {'Distance':>10} │ {'Evo Time':>12} │ {'Phase Time':>12} │ {'Speedup':>10}")
    print("  " + "─" * 52)
    
    for dist in distances:
        evo_times = []
        phase_times = []
        evo_steps = []
        phase_steps = []
        
        for _ in range(n_trials):
            # Random start and target
            start = np.array([0.0, 0.0])
            angle = np.random.uniform(0, 2 * np.pi)
            target = np.array([dist * np.cos(angle), dist * np.sin(angle)])
            
            # Evolution teleport
            agent_evo = Agent(position=start.copy())
            t_evo, s_evo = evolution_teleport(agent_evo, target)
            evo_times.append(t_evo)
            evo_steps.append(s_evo)
            
            # Phase space traverse
            agent_phase = Agent(position=start.copy())
            t_phase, s_phase = phase_space_traverse(agent_phase, target)
            phase_times.append(t_phase)
            phase_steps.append(s_phase)
        
        avg_evo = np.mean(evo_times) * 1000  # Convert to ms
        avg_phase = np.mean(phase_times) * 1000
        speedup = avg_phase / max(avg_evo, 0.001)
        
        print(f"  {dist:>10.0f} │ {avg_evo:>10.3f}ms │ {avg_phase:>10.1f}ms │ {speedup:>8.0f}x")
        
        results.append({
            'distance': dist,
            'evolution_time_ms': avg_evo,
            'phase_time_ms': avg_phase,
            'evolution_steps': np.mean(evo_steps),
            'phase_steps': np.mean(phase_steps),
            'speedup': speedup
        })
    
    print()
    print("=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    print()
    print("  Evolution (Genomic Layer):")
    print("    • O(1) - Constant time regardless of distance")
    print("    • Just updates coordinates (changes file path)")
    print()
    print("  Phase Space (Space-Time Layer):")
    print("    • O(distance) - Linear in distance traveled")
    print("    • Must integrate the path (move file across drive)")
    print()
    print("  THE INSIGHT:")
    print("    If evolution can 'teleport' but physics cannot,")
    print("    then space-time is a RENDERING CONSTRAINT,")
    print("    not a fundamental property of reality.")
    print()
    print('    "I didn\'t move the file. I changed the file path."')
    print("                              — Admin Access")
    
    return results


def run_live_teleportation(distance: float = 50.0):
    """
    Visual demonstration of teleportation vs traversal.
    """
    print("\n" + "=" * 70)
    print("  LIVE TELEPORTATION DEMO")
    print("=" * 70 + "\n")
    
    start = np.array([0.0, 0.0])
    target = np.array([distance, 0.0])
    
    bar_width = 50
    
    print(f"  Distance: {distance} units\n")
    
    # Phase space traversal (slow)
    print("  PHASE SPACE (Must traverse):")
    agent = Agent(position=start.copy())
    step = 0
    
    while agent.distance_to(target) > 1.0:
        progress = 1 - (agent.distance_to(target) / distance)
        bar_len = int(progress * bar_width)
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        
        print(f"\r    [{bar}] {progress*100:5.1f}% ", end="", flush=True)
        
        direction = target - agent.position
        direction = direction / np.linalg.norm(direction)
        agent.position = agent.position + direction * 2.0
        step += 1
        time.sleep(0.05)
    
    print(f"\r    [{'█' * bar_width}] 100.0% ✓ ({step} steps)")
    
    # Evolution teleport (instant)
    print("\n  EVOLUTION SPACE (Teleport):")
    agent = Agent(position=start.copy())
    
    print(f"    [{'░' * bar_width}]   0.0%", end="", flush=True)
    time.sleep(0.5)
    
    agent.position = target.copy()  # INSTANT
    
    print(f"\r    [{'█' * bar_width}] 100.0% ⚡ (1 step - INSTANT)")
    
    print()
    print("  The difference:")
    print(f"    Phase space: {step} steps (integration required)")
    print(f"    Evolution:   1 step (coordinate update only)")
    print()
    print('    "Space-time is just a data structure."')


if __name__ == "__main__":
    import sys
    
    if "--live" in sys.argv or "-l" in sys.argv:
        run_live_teleportation(distance=50)
    else:
        run_teleportation_experiment(n_trials=20)
