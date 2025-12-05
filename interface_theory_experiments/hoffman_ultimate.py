#!/usr/bin/env python3
"""
HOFFMAN SCALE: The Ultimate Truth vs Interface Showdown

This version implements the EXACT dynamics from the paper:
- Multiple resource types with different "true values"
- Truth agents can discriminate between nuanced values
- Interface agents only see "good/bad" icons

The key insight: In complex environments, SIMPLE wins over ACCURATE
because the cost of accuracy compounds faster than its benefits.

GPU-accelerated to simulate MILLIONS of agents.
"""

import torch
import sys
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_ultimate_showdown(
    num_agents: int = 100_000,
    generations: int = 500,
    num_resources: int = 10,  # Multiple resource types per round
    truth_cost_per_resource: float = 0.1,  # Cost to evaluate EACH resource
    verbose: bool = True
):
    """
    The Ultimate Hoffman Showdown.
    
    Each generation:
    - Multiple resources appear (num_resources)
    - Each has a true value 0-100
    - Truth agents evaluate ALL resources (pay cost per resource)
    - Interface agents see "green/red" icons (cheap, sometimes wrong)
    
    Truth agents pick the BEST resource
    Interface agents pick a random "green" one
    
    As num_resources increases, Truth cost grows linearly
    but Interface accuracy stays roughly constant.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  ULTIMATE HOFFMAN SHOWDOWN")
        print(f"  Device: {device} | Agents: {num_agents:,}")
        print(f"  Resources per round: {num_resources}")
        print(f"  Truth cost per resource: {truth_cost_per_resource}")
        print(f"  Total truth cost/gen: {num_resources * truth_cost_per_resource:.1f}")
        print(f"{'='*70}\n")
    
    start_time = time.perf_counter()
    
    # Population: 0 = Interface, 1 = Truth
    agent_types = torch.randint(0, 2, (num_agents,), device=device).float()
    energy = torch.ones(num_agents, device=device) * 50
    
    base_metabolism = 1.0
    icon_error_rate = 0.15  # 15% of icons are wrong
    
    history_truth = []
    history_simple = []
    
    for gen in range(generations):
        # 1. GENERATE RESOURCES
        resource_values = torch.rand(num_resources, device=device) * 100
        
        # 2. TRUTH AGENT STRATEGY
        # Evaluate ALL resources, pick the best one
        best_idx = torch.argmax(resource_values)
        best_value = resource_values[best_idx]
        truth_reward = best_value / 10  # Max reward = 10
        truth_cost = num_resources * truth_cost_per_resource
        
        # 3. INTERFACE AGENT STRATEGY
        # See icons (green if value > 50, but sometimes wrong)
        true_icons = resource_values > 50
        
        # Apply icon errors (flip some icons)
        icon_noise = torch.rand(num_resources, device=device) < icon_error_rate
        perceived_icons = true_icons ^ icon_noise  # XOR flips when noise is True
        
        # Pick random resource that looks "green"
        green_indices = torch.nonzero(perceived_icons).squeeze()
        if green_indices.numel() == 0:
            # No green resources - pick random
            interface_choice = torch.randint(0, num_resources, (1,), device=device)
        elif green_indices.dim() == 0:
            interface_choice = green_indices.unsqueeze(0)
        else:
            random_idx = torch.randint(0, green_indices.numel(), (1,), device=device)
            interface_choice = green_indices[random_idx]
        
        interface_value = resource_values[interface_choice].item()
        
        # Interface gets value/10 as reward, but no cost
        interface_reward = interface_value / 10
        
        # 4. CALCULATE PAYOFFS
        truth_mask = agent_types == 1
        interface_mask = agent_types == 0
        
        payoffs = torch.zeros(num_agents, device=device)
        payoffs[truth_mask] = truth_reward - truth_cost
        payoffs[interface_mask] = interface_reward
        
        # Subtract metabolism
        energy += payoffs - base_metabolism
        
        # 5. EVOLUTION
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
            
            n_dead = int(num_dead)
            parent_indices = living_indices[
                torch.multinomial(probs, n_dead, replacement=True)
            ]
            
            agent_types[dead_mask] = agent_types[parent_indices]
            energy[dead_mask] = 20.0
        
        # Track
        n_truth = int((agent_types == 1).sum().item())
        n_simple = int((agent_types == 0).sum().item())
        history_truth.append(n_truth)
        history_simple.append(n_simple)
        
        if verbose and (gen % 50 == 0 or gen == generations - 1):
            truth_pct = n_truth / num_agents * 100
            simple_pct = n_simple / num_agents * 100
            print(f"  Gen {gen:4d} | Truth: {truth_pct:5.1f}% | Interface: {simple_pct:5.1f}%")
    
    elapsed = time.perf_counter() - start_time
    
    if verbose:
        print(f"\n  Completed in {elapsed:.2f}s")
    
    return history_truth, history_simple, elapsed


def run_live_ultimate(num_agents: int = 50_000, generations: int = 300, num_resources: int = 20):
    """
    Live visualization of the ultimate showdown.
    """
    print(f"\n{'='*70}")
    print(f"  LIVE ULTIMATE SHOWDOWN")
    print(f"  Device: {device} | Agents: {num_agents:,}")
    print(f"  Resources per round: {num_resources}")
    print(f"{'='*70}")
    print(f"  Truth: Evaluates ALL resources, picks best (cost = {num_resources * 0.1:.1f}/gen)")
    print(f"  Interface: Picks random 'green' icon (cost = 0)")
    print(f"{'='*70}\n")
    
    bar_width = 50
    truth_cost_per = 0.1
    icon_error_rate = 0.15
    base_metabolism = 1.0
    
    agent_types = torch.randint(0, 2, (num_agents,), device=device).float()
    energy = torch.ones(num_agents, device=device) * 50
    
    for gen in range(generations):
        # Resources
        resource_values = torch.rand(num_resources, device=device) * 100
        
        # Truth: pick best
        best_value = resource_values.max().item()
        truth_reward = best_value / 10
        truth_cost = num_resources * truth_cost_per
        
        # Interface: pick random green
        true_icons = resource_values > 50
        icon_noise = torch.rand(num_resources, device=device) < icon_error_rate
        perceived_icons = true_icons ^ icon_noise
        
        green_indices = torch.nonzero(perceived_icons).squeeze()
        if green_indices.numel() == 0:
            interface_choice = torch.randint(0, num_resources, (1,), device=device)
        elif green_indices.dim() == 0:
            interface_choice = green_indices.unsqueeze(0)
        else:
            random_idx = torch.randint(0, green_indices.numel(), (1,), device=device)
            interface_choice = green_indices[random_idx]
        
        interface_reward = resource_values[interface_choice].item() / 10
        
        # Payoffs
        truth_mask = agent_types == 1
        interface_mask = agent_types == 0
        
        payoffs = torch.zeros(num_agents, device=device)
        payoffs[truth_mask] = truth_reward - truth_cost
        payoffs[interface_mask] = interface_reward
        
        energy += payoffs - base_metabolism
        
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
            n_dead = int(num_dead)
            parent_indices = living_indices[
                torch.multinomial(probs, n_dead, replacement=True)
            ]
            
            agent_types[dead_mask] = agent_types[parent_indices]
            energy[dead_mask] = 20.0
        
        # Visualization
        n_truth = int((agent_types == 1).sum().item())
        n_simple = int((agent_types == 0).sum().item())
        
        truth_pct = n_truth / num_agents
        simple_pct = n_simple / num_agents
        
        simple_bar = int(simple_pct * bar_width)
        truth_bar = int(truth_pct * bar_width)
        
        bar = "█" * simple_bar + "▓" * truth_bar + "░" * (bar_width - simple_bar - truth_bar)
        
        print(f"\r  Gen {gen:3d} [{bar}] I:{simple_pct*100:5.1f}% T:{truth_pct*100:5.1f}%", 
              end="", flush=True)
        
        time.sleep(0.015)
        
        if n_truth == 0:
            print(f"\n\n  ⚡ TRUTH EXTINCT at Generation {gen}")
            break
        if n_simple == 0:
            print(f"\n\n  🎯 INTERFACE EXTINCT at Generation {gen}")
            break
    
    print("\n")
    print("=" * 70)
    n_truth = int((agent_types == 1).sum().item())
    n_simple = int((agent_types == 0).sum().item())
    
    winner = "INTERFACE" if n_simple > n_truth else "TRUTH"
    
    print(f"  WINNER: {winner}")
    print()
    print(f"  Truth agents paid {num_resources * 0.1:.1f} energy/gen to evaluate all options.")
    print(f"  Interface agents paid 0 energy and picked random 'green' icons.")
    print()
    if winner == "INTERFACE":
        print('  "Knowing EVERYTHING costs more than knowing ENOUGH."')
        print("  The interface won by being computationally cheap.")
    else:
        print('  "In this environment, exhaustive analysis paid off."')
    print("=" * 70)


def complexity_scaling():
    """
    Show how truth agents fare as environment complexity increases.
    """
    print(f"\n{'='*70}")
    print(f"  COMPLEXITY SCALING EXPERIMENT")
    print(f"  How does Truth fare as #resources increases?")
    print(f"{'='*70}\n")
    
    resource_counts = [5, 10, 20, 50, 100]
    
    print(f"  {'Resources':>10} │ {'Truth Cost':>12} │ {'Winner':>10} │ {'Truth %':>10}")
    print(f"  {'─'*10}─┼─{'─'*12}─┼─{'─'*10}─┼─{'─'*10}")
    
    for n_res in resource_counts:
        truth_hist, simple_hist, _ = run_ultimate_showdown(
            num_agents=30_000,
            generations=300,
            num_resources=n_res,
            truth_cost_per_resource=0.1,
            verbose=False
        )
        
        final_truth = truth_hist[-1] if truth_hist else 0
        final_simple = simple_hist[-1] if simple_hist else 0
        total = final_truth + final_simple
        
        truth_pct = (final_truth / total * 100) if total > 0 else 0
        winner = "Truth" if final_truth > final_simple else "Interface"
        truth_cost = n_res * 0.1
        
        print(f"  {n_res:>10} │ {truth_cost:>10.1f}/g │ {winner:>10} │ {truth_pct:>9.1f}%")
    
    print()
    print("  As complexity increases, Truth agents must evaluate more options,")
    print("  but Interface agents just pick a random 'green' icon.")
    print()
    print('  "The curse of dimensionality: knowing everything becomes impossible."')


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if "--live" in args or "-l" in args:
        n = 50_000
        r = 20
        for arg in args:
            if arg.startswith("--agents="):
                n = int(arg.split("=")[1])
            if arg.startswith("--resources="):
                r = int(arg.split("=")[1])
        run_live_ultimate(num_agents=n, num_resources=r)
        
    elif "--scaling" in args or "-s" in args:
        complexity_scaling()
        
    else:
        print("\nUltimate Hoffman Showdown")
        print("=" * 40)
        print(f"Device: {device}")
        print()
        print("Usage:")
        print("  python hoffman_ultimate.py --live")
        print("  python hoffman_ultimate.py --live --resources=50")
        print("  python hoffman_ultimate.py --scaling")
        print()
        run_live_ultimate(num_agents=50_000, num_resources=20)
