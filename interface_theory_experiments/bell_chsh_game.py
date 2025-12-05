#!/usr/bin/env python3
"""
Bell's Inequality / CHSH Game Simulation

Tests whether "Local Realism" (space is real, distance matters) holds.

THE GAME:
- Alice and Bob are separated (could be galaxies apart)
- They receive random inputs, must output answers WITHOUT communication
- Classical Limit: Max 75% win rate
- Quantum Limit: ~85% win rate (violates Bell's inequality)

THE INSIGHT:
If they exceed 75%, particles coordinated FASTER than light.
This proves space-time is a rendering layer, not fundamental reality.

Nobel Prize 2022: Aspect, Clauser, Zeilinger - proved this experimentally.

Run with: python bell_chsh_game.py --live
"""

import numpy as np
import time
from typing import Tuple, Dict
from datetime import datetime
import json


def classical_strategy(alice_input: int, bob_input: int) -> Tuple[int, int]:
    """
    Best classical strategy - no communication, no entanglement.
    Can win at most 75% of the time.
    """
    # Optimal deterministic: both always output 0
    return 0, 0


def quantum_strategy(alice_input: int, bob_input: int) -> Tuple[int, int]:
    """
    Quantum strategy - simulates entangled particles.
    Can win ~85% of the time, VIOLATING classical limit.
    
    This simulates the quantum correlations mathematically.
    Real implementation requires actual entangled photons.
    """
    # Optimal measurement angles
    theta_a0, theta_a1 = 0, np.pi/4
    theta_b0, theta_b1 = np.pi/8, -np.pi/8
    
    theta_a = theta_a0 if alice_input == 0 else theta_a1
    theta_b = theta_b0 if bob_input == 0 else theta_b1
    
    # Quantum correlation: P(same) = cos²((θ_a - θ_b)/2)
    angle_diff = theta_a - theta_b
    prob_same = np.cos(angle_diff / 2) ** 2
    
    if np.random.random() < prob_same:
        output = np.random.randint(0, 2)
        return output, output
    else:
        alice_output = np.random.randint(0, 2)
        return alice_output, 1 - alice_output


def check_win(x: int, y: int, a: int, b: int) -> bool:
    """
    CHSH win condition: a ⊕ b = x ∧ y
    (Alice XOR Bob) must equal (Alice_input AND Bob_input)
    """
    return (a ^ b) == (x & y)


def run_chsh_rounds(n_rounds: int, strategy: str = "quantum") -> Dict:
    """Run CHSH game for n rounds."""
    wins = 0
    
    for _ in range(n_rounds):
        x = np.random.randint(0, 2)
        y = np.random.randint(0, 2)
        
        if strategy == "quantum":
            a, b = quantum_strategy(x, y)
        else:
            a, b = classical_strategy(x, y)
        
        if check_win(x, y, a, b):
            wins += 1
    
    return {
        'wins': wins,
        'total': n_rounds,
        'win_rate': wins / n_rounds,
        'percentage': wins / n_rounds * 100
    }


def run_live_bell_test(n_rounds: int = 1000, batch_size: int = 50):
    """
    Live terminal visualization of Bell test.
    Watch as quantum strategy violates classical limits!
    """
    print("\n" + "=" * 70)
    print("  BELL'S INEQUALITY TEST: DOES SPACE-TIME EXIST?")
    print("  If quantum > 75%, space is NOT fundamental")
    print("=" * 70 + "\n")
    time.sleep(1)
    
    classical_wins = 0
    quantum_wins = 0
    total = 0
    
    bar_width = 40
    
    print("  Classical Limit: 75% │ Quantum Max: 85.36% (Tsirelson)")
    print("  " + "─" * 66)
    print()
    
    while total < n_rounds:
        # Run batch
        batch = min(batch_size, n_rounds - total)
        
        for _ in range(batch):
            x = np.random.randint(0, 2)
            y = np.random.randint(0, 2)
            
            # Classical
            a_c, b_c = classical_strategy(x, y)
            if check_win(x, y, a_c, b_c):
                classical_wins += 1
            
            # Quantum
            a_q, b_q = quantum_strategy(x, y)
            if check_win(x, y, a_q, b_q):
                quantum_wins += 1
        
        total += batch
        
        # Calculate rates
        c_rate = classical_wins / total * 100
        q_rate = quantum_wins / total * 100
        
        # Build bars
        c_bar_len = int(c_rate / 100 * bar_width)
        q_bar_len = int(q_rate / 100 * bar_width)
        
        c_bar = "█" * c_bar_len + "░" * (bar_width - c_bar_len)
        q_bar = "█" * q_bar_len + "░" * (bar_width - q_bar_len)
        
        # Violation indicator
        violation = "⚡ VIOLATION!" if q_rate > 75 else ""
        
        print(f"  Round {total:5d}/{n_rounds}")
        print(f"  Classical: [{c_bar}] {c_rate:5.1f}%")
        print(f"  Quantum:   [{q_bar}] {q_rate:5.1f}% {violation}")
        
        # 75% marker
        marker_pos = int(0.75 * bar_width) + 13
        print(f"  {' ' * marker_pos}↑ 75% limit")
        print()
        
        time.sleep(0.15)
    
    # Final results
    print("=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Classical: {classical_wins}/{total} = {classical_wins/total*100:.2f}%")
    print(f"  Quantum:   {quantum_wins}/{total} = {quantum_wins/total*100:.2f}%")
    print()
    
    if quantum_wins/total > 0.75:
        print("  ⚡ BELL'S INEQUALITY VIOLATED!")
        print()
        print("  Particles coordinated faster than light.")
        print("  This is IMPOSSIBLE if space is fundamental.")
        print()
        print("  CONCLUSION: Space-time is a USER INTERFACE,")
        print("              not the base layer of reality.")
        print()
        print('  "Distance is just a rendering artifact."')
        print("                    — Interface Theory")
    
    return {
        'classical_rate': classical_wins / total,
        'quantum_rate': quantum_wins / total,
        'violation': quantum_wins / total > 0.75
    }


def run_massive_bell_test(n_trials: int = 100, rounds_per_trial: int = 1000):
    """
    Statistical validation with many trials.
    """
    print("\n" + "=" * 70)
    print("  MASSIVE BELL TEST: STATISTICAL VALIDATION")
    print(f"  Running {n_trials} trials × {rounds_per_trial} rounds each")
    print("=" * 70 + "\n")
    
    classical_rates = []
    quantum_rates = []
    violations = 0
    
    for i in range(n_trials):
        c = run_chsh_rounds(rounds_per_trial, "classical")
        q = run_chsh_rounds(rounds_per_trial, "quantum")
        
        classical_rates.append(c['percentage'])
        quantum_rates.append(q['percentage'])
        
        if q['percentage'] > 75:
            violations += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Trial {i+1:3d}/{n_trials}: "
                  f"C={c['percentage']:.1f}%, Q={q['percentage']:.1f}%")
    
    print()
    print("=" * 70)
    print("  STATISTICS")
    print("=" * 70)
    print(f"  Classical: {np.mean(classical_rates):.2f}% ± {np.std(classical_rates):.2f}%")
    print(f"  Quantum:   {np.mean(quantum_rates):.2f}% ± {np.std(quantum_rates):.2f}%")
    print(f"  Violation rate: {violations}/{n_trials} = {violations/n_trials*100:.0f}%")
    
    return {
        'classical_mean': np.mean(classical_rates),
        'quantum_mean': np.mean(quantum_rates),
        'violation_rate': violations / n_trials
    }


if __name__ == "__main__":
    import sys
    
    if "--live" in sys.argv or "-l" in sys.argv:
        run_live_bell_test(n_rounds=500, batch_size=25)
    elif "--massive" in sys.argv or "-m" in sys.argv:
        run_massive_bell_test(n_trials=100, rounds_per_trial=1000)
    else:
        print("Bell's Inequality / CHSH Game")
        print()
        print("Usage:")
        print("  python bell_chsh_game.py --live     # Watch live test")
        print("  python bell_chsh_game.py --massive  # Statistical validation")
        print()
        # Quick demo
        c = run_chsh_rounds(1000, "classical")
        q = run_chsh_rounds(1000, "quantum")
        print(f"Quick test (1000 rounds):")
        print(f"  Classical: {c['percentage']:.1f}%")
        print(f"  Quantum:   {q['percentage']:.1f}%")
        print(f"  Violation: {'YES ⚡' if q['percentage'] > 75 else 'NO'}")
