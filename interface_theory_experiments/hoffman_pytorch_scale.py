#!/usr/bin/env python3
"""
HOFFMAN FITNESS VS TRUTH - PyTorch GPU-Accelerated Version

Scales from 100 agents to 10 MILLION agents using tensor operations.

Key Insight:
- Instead of loops, we treat the ENTIRE POPULATION as a single tensor
- GPU parallelizes all agent calculations simultaneously
- We can find the exact "tipping point" where Truth becomes viable

The result: You'll watch Truth agents get DEMOLISHED because they waste
compute cycles processing reality when a simple icon would suffice.

"Spacetime is a desktop. It's there to hide the truth so you can work efficiently."
                                                        — Donald Hoffman
"""

import torch
import sys
import time
from typing import Tuple, List

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_pytorch_evolution(
    num_agents: int = 100_000,
    generations: int = 500,
    truth_cost: float = 0.5,
    resource_max: int = 100,
    verbose: bool = True
) -> Tuple[List[int], List[int], float]:
    """
    Run the Hoffman evolution simulation using PyTorch tensors.
    
    All agents are processed in parallel as a single tensor operation.
    
    Args:
        num_agents: Population size (can scale to millions on GPU)
        generations: Number of evolutionary rounds
        truth_cost: Energy cost per generation for Truth agents
        resource_max: Maximum resource value in the world
        verbose: Print progress
        
    Returns:
        (truth_history, simple_history, elapsed_time)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  PYTORCH HOFFMAN SIMULATION")
        print(f"  Device: {device} | Agents: {num_agents:,} | Generations: {generations}")
        print(f"{'='*70}\n")
    
    start_time = time.perf_counter()
    
    # Initialize population: 50% Truth (1), 50% Simple/Interface (0)
    agent_types = torch.randint(0, 2, (num_agents,), device=device).float()
    
    # Energy levels (all start with 10)
    energy = torch.ones(num_agents, device=device) * 10
    
    # Cost tensor: Truth agents pay truth_cost, Simple agents pay 0
    cost_tensor = agent_types * truth_cost
    
    # History tracking
    history_truth = []
    history_simple = []
    
    for gen in range(generations):
        # 1. THE WORLD - Random resource value (broadcast to all agents)
        resource_val = torch.randint(0, resource_max, (1,), device=device).item()
        
        # 2. PERCEPTION & DECISION (Vectorized)
        # Both types decide to eat if resource > 50
        # The DIFFERENCE is the COST they pay to perceive
        did_eat = (resource_val > 50)
        
        # 3. PAYOFF CALCULATION
        gains = torch.zeros(num_agents, device=device)
        if did_eat:
            gains += 10
        
        # Subtract costs (Truth agents pay more every generation)
        # This is the "tax" of processing reality
        current_payoff = gains - cost_tensor - 0.1  # 0.1 = base metabolism
        
        # Update energy
        energy += current_payoff
        
        # 4. EVOLUTION (The Reaper)
        dead_mask = energy <= 0
        num_dead = dead_mask.sum().item()
        
        if num_dead > 0:
            living_mask = energy > 0
            if living_mask.sum() == 0:
                if verbose:
                    print("  ☠️  EXTINCTION EVENT!")
                break
            
            # Get indices of living agents
            living_indices = torch.nonzero(living_mask).squeeze()
            if living_indices.dim() == 0:
                living_indices = living_indices.unsqueeze(0)
            
            # Weight reproduction by energy (richer = more offspring)
            living_energy = energy[living_indices]
            probs = living_energy / living_energy.sum()
            
            # Sample parents to repopulate dead slots
            parent_indices = living_indices[
                torch.multinomial(probs, num_dead, replacement=True)
            ]
            
            # Clone parent strategy to dead agents
            agent_types[dead_mask] = agent_types[parent_indices]
            
            # Update cost tensor for new agents
            cost_tensor[dead_mask] = agent_types[dead_mask] * truth_cost
            
            # Reset energy of new births
            energy[dead_mask] = 5.0
        
        # Track populations
        n_truth = (agent_types == 1).sum().item()
        n_simple = (agent_types == 0).sum().item()
        history_truth.append(n_truth)
        history_simple.append(n_simple)
        
        # Progress output
        if verbose and (gen % 50 == 0 or gen == generations - 1):
            truth_pct = n_truth / num_agents * 100
            simple_pct = n_simple / num_agents * 100
            print(f"  Gen {gen:4d} | Truth: {truth_pct:5.1f}% | Interface: {simple_pct:5.1f}%")
    
    elapsed = time.perf_counter() - start_time
    
    if verbose:
        print(f"\n  Completed in {elapsed:.2f}s")
        print(f"  Final: Truth={history_truth[-1]:,} | Interface={history_simple[-1]:,}")
    
    return history_truth, history_simple, elapsed


def run_live_terminal(num_agents: int = 50_000, generations: int = 200):
    """
    Live terminal visualization of the evolution.
    """
    print(f"\n{'='*70}")
    print(f"  LIVE PYTORCH EVOLUTION")
    print(f"  Device: {device} | Agents: {num_agents:,}")
    print(f"{'='*70}\n")
    
    bar_width = 50
    truth_cost = 0.5
    
    # Initialize
    agent_types = torch.randint(0, 2, (num_agents,), device=device).float()
    energy = torch.ones(num_agents, device=device) * 10
    cost_tensor = agent_types * truth_cost
    
    for gen in range(generations):
        # World
        resource_val = torch.randint(0, 100, (1,), device=device).item()
        did_eat = (resource_val > 50)
        
        # Payoff
        gains = torch.zeros(num_agents, device=device)
        if did_eat:
            gains += 10
        current_payoff = gains - cost_tensor - 0.1
        energy += current_payoff
        
        # Evolution
        dead_mask = energy <= 0
        num_dead = dead_mask.sum().item()
        
        if num_dead > 0:
            living_mask = energy > 0
            if living_mask.sum() == 0:
                print("\n  ☠️  EXTINCTION!")
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
            cost_tensor[dead_mask] = agent_types[dead_mask] * truth_cost
            energy[dead_mask] = 5.0
        
        # Visualization
        n_truth = (agent_types == 1).sum().item()
        n_simple = (agent_types == 0).sum().item()
        
        truth_pct = n_truth / num_agents
        simple_pct = n_simple / num_agents
        
        truth_bar = int(truth_pct * bar_width)
        simple_bar = int(simple_pct * bar_width)
        
        # Stacked bar visualization
        bar = "█" * simple_bar + "▓" * truth_bar + "░" * (bar_width - simple_bar - truth_bar)
        
        print(f"\r  Gen {gen:3d} [{bar}] Interface:{simple_pct*100:5.1f}% Truth:{truth_pct*100:5.1f}%", 
              end="", flush=True)
        
        time.sleep(0.02)
        
        if n_truth == 0:
            print(f"\n\n  ⚡ TRUTH EXTINCT at Generation {gen}")
            break
    
    print("\n")
    print("=" * 70)
    print("  RESULT: Interface agents dominate!")
    print("  ")
    print('  "The agents that wasted GPU cycles processing reality')
    print('   got out-competed by agents that just looked for the food icon."')
    print("=" * 70)


def find_truth_tipping_point():
    """
    Use binary search to find the EXACT cost where Truth becomes viable.
    
    This is the "gradient descent" on the simulation - we're asking:
    "What is the maximum cost of truth the system can tolerate?"
    """
    print(f"\n{'='*70}")
    print(f"  FINDING THE TRUTH TIPPING POINT")
    print(f"  Binary search for critical cost threshold")
    print(f"{'='*70}\n")
    
    low_cost = 0.0
    high_cost = 2.0
    
    # How many generations of survival = "viable"
    survival_threshold = 300
    num_agents = 20_000
    generations = 400
    
    for iteration in range(10):
        mid_cost = (low_cost + high_cost) / 2
        
        print(f"  Testing cost = {mid_cost:.4f}...", end=" ", flush=True)
        
        # Run simulation silently
        truth_hist, _, _ = run_pytorch_evolution(
            num_agents=num_agents,
            generations=generations,
            truth_cost=mid_cost,
            verbose=False
        )
        
        # Find when Truth goes extinct (if ever)
        extinction_gen = None
        for gen, count in enumerate(truth_hist):
            if count == 0:
                extinction_gen = gen
                break
        
        if extinction_gen is None:
            # Truth survived - cost is too low
            print(f"Truth SURVIVED all {generations} generations")
            low_cost = mid_cost
        else:
            # Truth died - cost is too high
            print(f"Truth EXTINCT at gen {extinction_gen}")
            high_cost = mid_cost
    
    critical_cost = (low_cost + high_cost) / 2
    
    print(f"\n{'='*70}")
    print(f"  CRITICAL THRESHOLD FOUND: {critical_cost:.4f}")
    print(f"{'='*70}")
    print(f"\n  Below {critical_cost:.4f}: Truth can survive")
    print(f"  Above {critical_cost:.4f}: Truth goes extinct")
    print(f"\n  This is the 'price of reality' - the maximum compute cost")
    print(f"  an organism can pay for accurate perception and still survive.")
    
    return critical_cost


def run_scale_benchmark():
    """
    Benchmark how fast PyTorch can scale.
    """
    print(f"\n{'='*70}")
    print(f"  PYTORCH SCALE BENCHMARK")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    scales = [1_000, 10_000, 100_000, 500_000, 1_000_000]
    generations = 100
    
    print(f"  {'Agents':>12} │ {'Time':>10} │ {'Agents/sec':>15}")
    print(f"  {'─'*12}─┼─{'─'*10}─┼─{'─'*15}")
    
    for n in scales:
        try:
            _, _, elapsed = run_pytorch_evolution(
                num_agents=n,
                generations=generations,
                verbose=False
            )
            rate = (n * generations) / elapsed
            print(f"  {n:>12,} │ {elapsed:>8.2f}s │ {rate:>13,.0f}/s")
        except RuntimeError as e:
            print(f"  {n:>12,} │ OUT OF MEMORY")
            break
    
    print(f"\n  GPU acceleration enables massive parallel evolution!")


def run_with_matplotlib(num_agents: int = 100_000, generations: int = 500):
    """
    Run with matplotlib visualization (optional).
    """
    try:
        import matplotlib.pyplot as plt
        
        truth_hist, simple_hist, elapsed = run_pytorch_evolution(
            num_agents=num_agents,
            generations=generations
        )
        
        plt.figure(figsize=(12, 6))
        plt.stackplot(
            range(len(truth_hist)), 
            [simple_hist, truth_hist],
            labels=['Interface (Icons)', 'Truth (Realists)'],
            colors=['#2ecc71', '#3498db'],
            alpha=0.8
        )
        plt.title(f'Evolution at Scale: {num_agents:,} Agents ({elapsed:.1f}s)')
        plt.xlabel('Generations')
        plt.ylabel('Population Count')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("  matplotlib not available, running terminal version...")
        run_live_terminal(num_agents, generations)


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if "--live" in args or "-l" in args:
        # Parse agent count
        n = 50_000
        for arg in args:
            if arg.startswith("--agents="):
                n = int(arg.split("=")[1])
        run_live_terminal(num_agents=n)
        
    elif "--tipping" in args or "-t" in args:
        find_truth_tipping_point()
        
    elif "--benchmark" in args or "-b" in args:
        run_scale_benchmark()
        
    elif "--plot" in args or "-p" in args:
        n = 100_000
        for arg in args:
            if arg.startswith("--agents="):
                n = int(arg.split("=")[1])
        run_with_matplotlib(num_agents=n)
        
    else:
        print("\nPyTorch Hoffman Evolution Simulation")
        print("=" * 40)
        print(f"Device: {device}")
        print()
        print("Usage:")
        print("  python hoffman_pytorch_scale.py --live              # Terminal viz")
        print("  python hoffman_pytorch_scale.py --live --agents=100000")
        print("  python hoffman_pytorch_scale.py --tipping           # Find critical cost")
        print("  python hoffman_pytorch_scale.py --benchmark         # Scale test")
        print("  python hoffman_pytorch_scale.py --plot              # Matplotlib viz")
        print()
        print("Running live demo with 50k agents...")
        print()
        run_live_terminal(num_agents=50_000, generations=200)
