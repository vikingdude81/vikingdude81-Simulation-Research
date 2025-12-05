#!/usr/bin/env python3
"""
HOFFMAN FITNESS VS TRUTH - PyTorch GPU-Accelerated Version (v2)

The CORRECT formulation: Truth agents see continuous values and can
make nuanced decisions, but pay a compute cost. Interface agents 
see binary icons (good/bad) and sometimes make wrong decisions,
but pay zero compute cost.

Key insight from Hoffman:
- Truth = Accurate perception with HIGH compute cost
- Interface = Cheap icons that are SOMETIMES wrong but efficient

In environments where:
1. Resources are scarce
2. Speed matters
3. "Good enough" beats "perfect"

...the Interface agents DOMINATE.
"""

import torch
import sys
import time
from typing import Tuple, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_hoffman_v2(
    num_agents: int = 100_000,
    generations: int = 500,
    truth_cost: float = 0.3,
    icon_error_rate: float = 0.1,  # Icons are wrong 10% of the time
    verbose: bool = True
) -> Tuple[List[int], List[int], float]:
    """
    Improved Hoffman simulation with realistic tradeoffs:
    
    Truth Agents:
    - See actual resource values (0-100)
    - Can optimize perfectly (eat only when value > threshold)
    - Pay compute cost EVERY decision
    
    Interface Agents:
    - See binary icons: "Looks Good" or "Looks Bad"
    - Icons are wrong {icon_error_rate}% of the time
    - Pay ZERO compute cost
    
    The key: Interface errors are rare, but Truth costs are constant.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  PYTORCH HOFFMAN v2: REALISTIC TRADEOFFS")
        print(f"  Device: {device} | Agents: {num_agents:,}")
        print(f"  Truth Cost: {truth_cost} | Icon Error Rate: {icon_error_rate*100:.0f}%")
        print(f"{'='*70}\n")
    
    start_time = time.perf_counter()
    
    # Population: 0 = Interface, 1 = Truth
    agent_types = torch.randint(0, 2, (num_agents,), device=device).float()
    
    # Energy levels
    energy = torch.ones(num_agents, device=device) * 20
    
    # Base metabolism (everyone pays this)
    base_metabolism = 0.2
    
    # History tracking
    history_truth = []
    history_simple = []
    
    for gen in range(generations):
        # 1. THE WORLD - Resource with actual value (some are traps!)
        # Resource value 0-100, but some "look good" but are bad
        true_value = torch.rand(1, device=device).item() * 100
        
        # The ICON (what Interface agents see)
        # Usually correct, but sometimes wrong
        if torch.rand(1).item() < icon_error_rate:
            # Icon is WRONG - shows opposite of reality
            icon_good = true_value < 50  # Bad resource looks good
        else:
            # Icon is correct
            icon_good = true_value > 50
        
        # 2. DECISIONS
        # Truth agents: eat if true_value > 50 (always correct)
        truth_decision = true_value > 50
        
        # Interface agents: eat if icon says good (sometimes wrong)
        interface_decision = icon_good
        
        # 3. PAYOFFS
        # Reality: eating good food = +10, eating bad food = -15 (poison!)
        actual_good = true_value > 50
        
        # Truth agent payoffs (always correct decision, but pay cost)
        truth_gain = torch.zeros(num_agents, device=device)
        truth_mask = agent_types == 1
        
        if truth_decision and actual_good:
            truth_gain[truth_mask] = 10  # Correct eat
        elif not truth_decision and not actual_good:
            truth_gain[truth_mask] = 0   # Correct avoid
        # Truth agents never make mistakes, but they always pay the cost
        
        # Interface agent payoffs (cheap but sometimes wrong)
        interface_gain = torch.zeros(num_agents, device=device)
        interface_mask = agent_types == 0
        
        if interface_decision:
            if actual_good:
                interface_gain[interface_mask] = 10  # Correct eat
            else:
                interface_gain[interface_mask] = -15  # ATE POISON!
        else:
            if not actual_good:
                interface_gain[interface_mask] = 0   # Correct avoid
            else:
                interface_gain[interface_mask] = 0   # Missed food, but survived
        
        # Combine payoffs
        total_gain = truth_gain + interface_gain
        
        # Subtract costs
        # Truth agents pay compute cost EVERY generation
        cost = agent_types * truth_cost
        
        # Everyone pays metabolism
        energy += total_gain - cost - base_metabolism
        
        # 4. EVOLUTION
        dead_mask = energy <= 0
        num_dead = dead_mask.sum().item()
        
        if num_dead > 0:
            living_mask = energy > 0
            if living_mask.sum() == 0:
                if verbose:
                    print("  ☠️  EXTINCTION!")
                break
            
            living_indices = torch.nonzero(living_mask).squeeze()
            if living_indices.dim() == 0:
                living_indices = living_indices.unsqueeze(0)
            
            living_energy = energy[living_indices]
            probs = living_energy / living_energy.sum()
            
            parent_indices = living_indices[
                torch.multinomial(probs, num_dead, replacement=True)
            ]
            
            agent_types[dead_mask] = agent_types[parent_indices]
            energy[dead_mask] = 10.0
        
        # Track
        n_truth = (agent_types == 1).sum().item()
        n_simple = (agent_types == 0).sum().item()
        history_truth.append(n_truth)
        history_simple.append(n_simple)
        
        if verbose and (gen % 50 == 0 or gen == generations - 1):
            truth_pct = n_truth / num_agents * 100
            simple_pct = n_simple / num_agents * 100
            print(f"  Gen {gen:4d} | Truth: {truth_pct:5.1f}% | Interface: {simple_pct:5.1f}%")
    
    elapsed = time.perf_counter() - start_time
    
    if verbose:
        print(f"\n  Completed in {elapsed:.2f}s")
        final_truth = history_truth[-1] if history_truth else 0
        final_simple = history_simple[-1] if history_simple else 0
        print(f"  Final: Truth={final_truth:,} | Interface={final_simple:,}")
    
    return history_truth, history_simple, elapsed


def run_live_v2(num_agents: int = 50_000, generations: int = 300):
    """
    Live terminal visualization with the correct tradeoffs.
    """
    print(f"\n{'='*70}")
    print(f"  LIVE HOFFMAN v2: TRUTH vs INTERFACE")
    print(f"  Device: {device} | Agents: {num_agents:,}")
    print(f"{'='*70}")
    print(f"  Truth: Always correct but pays 0.3 energy/gen")
    print(f"  Interface: Cheap but icons wrong 10% of time (poison!)")
    print(f"{'='*70}\n")
    
    bar_width = 50
    truth_cost = 0.3
    icon_error_rate = 0.10
    base_metabolism = 0.2
    
    agent_types = torch.randint(0, 2, (num_agents,), device=device).float()
    energy = torch.ones(num_agents, device=device) * 20
    
    poison_events = 0
    correct_avoids = 0
    
    for gen in range(generations):
        # World
        true_value = torch.rand(1, device=device).item() * 100
        actual_good = true_value > 50
        
        # Icon (sometimes wrong)
        icon_wrong = torch.rand(1).item() < icon_error_rate
        if icon_wrong:
            icon_good = not actual_good  # Inverted!
        else:
            icon_good = actual_good
        
        # Decisions
        truth_decision = actual_good  # Truth always correct
        interface_decision = icon_good
        
        # Payoffs
        truth_mask = agent_types == 1
        interface_mask = agent_types == 0
        
        gain = torch.zeros(num_agents, device=device)
        
        # Truth agents
        if truth_decision:
            gain[truth_mask] = 10
        
        # Interface agents
        if interface_decision:
            if actual_good:
                gain[interface_mask] = 10
            else:
                gain[interface_mask] = -15  # POISON!
                poison_events += 1
        else:
            if actual_good:
                # Missed opportunity but survived
                pass
            else:
                correct_avoids += 1
        
        # Costs
        cost = agent_types * truth_cost
        energy += gain - cost - base_metabolism
        
        # Evolution
        dead_mask = energy <= 0
        num_dead = dead_mask.sum().item()
        
        if num_dead > 0:
            living_mask = energy > 0
            if living_mask.sum() == 0:
                print("\n  ☠️  TOTAL EXTINCTION!")
                break
            
            living_indices = torch.nonzero(living_mask).squeeze()
            if living_indices.dim() == 0:
                living_indices = living_indices.unsqueeze(0)
            
            living_energy = energy[living_indices]
            probs = living_energy / living_energy.sum()
            parent_indices = living_indices[
                torch.multinomial(probs, num_dead, replacement=True)
            ]
            
            agent_types[dead_mask] = agent_types[parent_indices]
            energy[dead_mask] = 10.0
        
        # Visualization
        n_truth = (agent_types == 1).sum().item()
        n_simple = (agent_types == 0).sum().item()
        
        truth_pct = n_truth / num_agents
        simple_pct = n_simple / num_agents
        
        simple_bar = int(simple_pct * bar_width)
        truth_bar = int(truth_pct * bar_width)
        
        bar = "█" * simple_bar + "▓" * truth_bar + "░" * (bar_width - simple_bar - truth_bar)
        
        status = "☠️ POISON!" if (interface_decision and not actual_good) else ""
        
        print(f"\r  Gen {gen:3d} [{bar}] I:{simple_pct*100:5.1f}% T:{truth_pct*100:5.1f}% {status}    ", 
              end="", flush=True)
        
        time.sleep(0.02)
        
        if n_truth == 0:
            print(f"\n\n  ⚡ TRUTH EXTINCT at Generation {gen}")
            break
        if n_simple == 0:
            print(f"\n\n  🎯 INTERFACE EXTINCT at Generation {gen} (too many poison events)")
            break
    
    print("\n")
    print("=" * 70)
    print(f"  FINAL RESULTS:")
    n_truth = (agent_types == 1).sum().item()
    n_simple = (agent_types == 0).sum().item()
    print(f"    Truth Agents:     {n_truth:,} ({n_truth/num_agents*100:.1f}%)")
    print(f"    Interface Agents: {n_simple:,} ({n_simple/num_agents*100:.1f}%)")
    print(f"    Poison Events:    {poison_events}")
    print()
    
    winner = "INTERFACE" if n_simple > n_truth else "TRUTH"
    print(f"  WINNER: {winner}")
    print()
    if winner == "INTERFACE":
        print('  "The cost of processing reality exceeded the cost of occasional mistakes."')
        print("  Interface agents won by being CHEAP, not by being RIGHT.")
    else:
        print('  "In this environment, accuracy paid off."')
        print("  Truth agents survived despite the compute cost.")
    print("=" * 70)


def parameter_sweep():
    """
    Sweep across different truth_cost and error_rate combinations.
    """
    print(f"\n{'='*70}")
    print(f"  PARAMETER SWEEP: Finding the Phase Transition")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    truth_costs = [0.1, 0.2, 0.3, 0.4, 0.5]
    error_rates = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    print(f"  {'':8} │ ", end="")
    for er in error_rates:
        print(f" {er*100:4.0f}%  ", end="")
    print(" │ (Icon Error Rate)")
    print(f"  {'─'*8}─┼─{'─'*7 * len(error_rates)}─┤")
    
    for tc in truth_costs:
        print(f"  TC={tc:.1f}  │ ", end="")
        
        for er in error_rates:
            # Run short simulation
            truth_hist, simple_hist, _ = run_hoffman_v2(
                num_agents=20_000,
                generations=200,
                truth_cost=tc,
                icon_error_rate=er,
                verbose=False
            )
            
            # Who won?
            final_truth = truth_hist[-1] if truth_hist else 0
            final_simple = simple_hist[-1] if simple_hist else 0
            
            if final_truth > final_simple * 1.5:
                symbol = "  T   "  # Truth dominates
            elif final_simple > final_truth * 1.5:
                symbol = "  I   "  # Interface dominates
            else:
                symbol = "  ~   "  # Mixed
            
            print(symbol, end=" ")
        
        print(" │")
    
    print(f"  {'─'*8}─┴─{'─'*7 * len(error_rates)}─┘")
    print()
    print("  Legend: T = Truth wins, I = Interface wins, ~ = Mixed")
    print()
    print("  The phase transition occurs where the cost of truth exceeds")
    print("  the cost of occasional mistakes from using cheap icons.")


def benchmark_scale():
    """
    Benchmark PyTorch scaling on GPU.
    """
    print(f"\n{'='*70}")
    print(f"  SCALE BENCHMARK")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    scales = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000]
    
    print(f"  {'Agents':>12} │ {'Time':>10} │ {'Agents×Gens/sec':>18}")
    print(f"  {'─'*12}─┼─{'─'*10}─┼─{'─'*18}")
    
    for n in scales:
        try:
            _, _, elapsed = run_hoffman_v2(
                num_agents=n,
                generations=100,
                verbose=False
            )
            rate = (n * 100) / elapsed
            print(f"  {n:>12,} │ {elapsed:>8.2f}s │ {rate:>16,.0f}/s")
        except RuntimeError:
            print(f"  {n:>12,} │   OOM    │")
            break
    
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if "--live" in args or "-l" in args:
        n = 50_000
        for arg in args:
            if arg.startswith("--agents="):
                n = int(arg.split("=")[1])
        run_live_v2(num_agents=n)
        
    elif "--sweep" in args or "-s" in args:
        parameter_sweep()
        
    elif "--benchmark" in args or "-b" in args:
        benchmark_scale()
        
    else:
        print("\nHoffman v2: Realistic Truth vs Interface Tradeoffs")
        print("=" * 50)
        print(f"Device: {device}")
        print()
        print("Usage:")
        print("  python hoffman_pytorch_v2.py --live              # Watch evolution")
        print("  python hoffman_pytorch_v2.py --live --agents=100000")
        print("  python hoffman_pytorch_v2.py --sweep             # Parameter sweep")
        print("  python hoffman_pytorch_v2.py --benchmark         # Scale test")
        print()
        print("Key Differences from v1:")
        print("  - Interface agents can EAT POISON (icons sometimes wrong)")
        print("  - Truth agents ALWAYS pay compute cost")
        print("  - Phase transition depends on error_rate vs truth_cost")
        print()
        run_live_v2(num_agents=50_000, generations=300)
