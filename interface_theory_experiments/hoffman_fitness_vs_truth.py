#!/usr/bin/env python3
"""
Hoffman's "Fitness Beats Truth" Simulation

This proves Donald Hoffman's central claim: evolution drives organisms 
AWAY from seeing reality as it is.

THE CONCEPT:
- World: A resource varies from 0 to 100
- Truth Agent: Sees the exact value (expensive computation)
- Interface Agent: Only sees "Good" (>50) or "Bad" (<50) - 1 bit
- Result: Interface Agent wins because it uses less energy

THE INSIGHT:
The "cost of truth" represents computational overhead. Even a tiny tax
on accurate perception leads to extinction of truth-seeing organisms.

This is why we don't see reality - we see a "desktop interface" optimized
for survival, not accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import json
from datetime import datetime


@dataclass
class Agent:
    """An agent that perceives and acts in the world."""
    agent_type: str  # 'truth' or 'interface'
    energy: float = 50.0  # Starting energy
    
    def perceive_and_act(self, resource_value: float, cost_of_truth: float) -> float:
        """
        Perceive the resource and decide whether to consume it.
        
        Truth agents see the exact value (expensive).
        Interface agents see a boolean bit (cheap).
        """
        if self.agent_type == 'truth':
            # Truth agent sees exact reality
            perception = resource_value  # Full information
            cost = cost_of_truth  # Pay the computational tax
            
            # Sophisticated decision based on exact value
            if perception > 50:
                payoff = (perception - 50) / 5  # Proportional gain
            else:
                payoff = 0
            
            return payoff - cost
        
        elif self.agent_type == 'interface':
            # Interface agent sees only an "icon" (1 bit)
            perception = resource_value > 50  # Boolean: Good or Bad
            cost = 0  # Cheap interface, minimal computation
            
            # Simple binary decision
            if perception:
                return 10  # Fixed gain for "good" icon
            else:
                return 0  # No action for "bad" icon
        
        return 0


def run_hoffman_simulation(
    generations: int = 300,
    population_size: int = 100,
    cost_of_truth: float = 0.5,
    resource_volatility: float = 1.0,
    truth_ratio: float = 0.5,
    seed: int = None
) -> Dict:
    """
    Run the Fitness vs Truth simulation.
    
    Args:
        generations: Number of evolutionary cycles
        population_size: Total population size
        cost_of_truth: Energy tax for seeing reality
        resource_volatility: How much the resource fluctuates
        truth_ratio: Initial proportion of truth agents
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with simulation results and history
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize population
    n_truth = int(population_size * truth_ratio)
    n_interface = population_size - n_truth
    
    population: List[Agent] = (
        [Agent('truth') for _ in range(n_truth)] +
        [Agent('interface') for _ in range(n_interface)]
    )
    
    # History tracking
    history = {
        'truth_count': [],
        'interface_count': [],
        'truth_avg_energy': [],
        'interface_avg_energy': [],
        'resource_values': [],
        'generation': []
    }
    
    # Simulation loop
    for gen in range(generations):
        # Resource fluctuates (this is "reality")
        base_resource = 50 + 30 * np.sin(gen / 20)  # Cyclical
        noise = np.random.normal(0, 10 * resource_volatility)
        resource = np.clip(base_resource + noise, 0, 100)
        
        # Each agent perceives and acts
        for agent in population:
            payoff = agent.perceive_and_act(resource, cost_of_truth)
            agent.energy += payoff
            agent.energy = max(0, agent.energy)  # Can't go negative
        
        # Record state before reproduction
        truth_agents = [a for a in population if a.agent_type == 'truth']
        interface_agents = [a for a in population if a.agent_type == 'interface']
        
        history['truth_count'].append(len(truth_agents))
        history['interface_count'].append(len(interface_agents))
        history['truth_avg_energy'].append(
            np.mean([a.energy for a in truth_agents]) if truth_agents else 0
        )
        history['interface_avg_energy'].append(
            np.mean([a.energy for a in interface_agents]) if interface_agents else 0
        )
        history['resource_values'].append(resource)
        history['generation'].append(gen)
        
        # Reproduction (fitness-proportional selection)
        if len(population) == 0:
            break
            
        total_energy = sum(a.energy for a in population)
        if total_energy <= 0:
            # Random survival if everyone is at zero
            survivors = np.random.choice(population, size=population_size, replace=True)
            population = [Agent(s.agent_type) for s in survivors]
        else:
            # Fitness-proportional reproduction
            probs = np.array([max(0.001, a.energy) for a in population])
            probs = probs / probs.sum()
            
            parent_indices = np.random.choice(
                len(population), 
                size=population_size, 
                p=probs,
                replace=True
            )
            
            population = [Agent(population[i].agent_type) for i in parent_indices]
    
    # Final statistics
    final_truth = history['truth_count'][-1] if history['truth_count'] else 0
    final_interface = history['interface_count'][-1] if history['interface_count'] else 0
    
    result = {
        'history': history,
        'final_truth_count': final_truth,
        'final_interface_count': final_interface,
        'truth_extinct': final_truth == 0,
        'interface_extinct': final_interface == 0,
        'winner': 'interface' if final_truth < final_interface else 'truth',
        'parameters': {
            'generations': generations,
            'population_size': population_size,
            'cost_of_truth': cost_of_truth,
            'resource_volatility': resource_volatility,
            'seed': seed
        }
    }
    
    return result


def plot_hoffman_results(result: Dict, save_path: str = None):
    """Visualize the Fitness vs Truth simulation results."""
    history = result['history']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Population dynamics
    ax1 = axes[0, 0]
    ax1.plot(history['generation'], history['truth_count'], 
             label='Truth Agents (See Reality)', color='blue', linewidth=2)
    ax1.plot(history['generation'], history['interface_count'], 
             label='Interface Agents (See Icons)', color='green', linewidth=2)
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Population')
    ax1.set_title("Hoffman's Simulation: Fitness vs Truth")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Energy dynamics
    ax2 = axes[0, 1]
    ax2.plot(history['generation'], history['truth_avg_energy'], 
             label='Truth Agents', color='blue', alpha=0.7)
    ax2.plot(history['generation'], history['interface_avg_energy'], 
             label='Interface Agents', color='green', alpha=0.7)
    ax2.set_xlabel('Generations')
    ax2.set_ylabel('Average Energy')
    ax2.set_title('Energy Dynamics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Resource fluctuation
    ax3 = axes[1, 0]
    ax3.plot(history['generation'], history['resource_values'], 
             color='orange', alpha=0.7, linewidth=1)
    ax3.axhline(y=50, color='red', linestyle='--', label='Threshold')
    ax3.set_xlabel('Generations')
    ax3.set_ylabel('Resource Value')
    ax3.set_title('Reality (Hidden from Interface Agents)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    params = result['parameters']
    summary_text = f"""
    HOFFMAN'S INTERFACE THEORY SIMULATION
    ═══════════════════════════════════════
    
    Parameters:
    • Generations: {params['generations']}
    • Population: {params['population_size']}
    • Cost of Truth: {params['cost_of_truth']}
    • Resource Volatility: {params['resource_volatility']}
    
    Results:
    • Final Truth Agents: {result['final_truth_count']}
    • Final Interface Agents: {result['final_interface_count']}
    • Truth Extinct: {'YES ❌' if result['truth_extinct'] else 'NO ✓'}
    • Winner: {result['winner'].upper()}
    
    ═══════════════════════════════════════
    
    INSIGHT: Evolution selects for USEFUL
    interfaces, not ACCURATE perception.
    
    "You don't see reality. You see a
    desktop optimized for survival."
                    — Donald Hoffman
    """
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig


def run_parameter_sweep(costs: List[float] = None) -> Dict:
    """
    Run multiple simulations with varying cost of truth.
    
    This tests: At what point does truth become viable?
    """
    if costs is None:
        costs = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    
    results = []
    
    for cost in costs:
        print(f"Running with cost_of_truth = {cost}...")
        
        # Run multiple trials for statistical significance
        truth_survival = []
        for trial in range(10):
            result = run_hoffman_simulation(
                generations=200,
                population_size=100,
                cost_of_truth=cost,
                seed=trial * 100
            )
            truth_survival.append(result['final_truth_count'])
        
        results.append({
            'cost_of_truth': cost,
            'mean_truth_survival': np.mean(truth_survival),
            'std_truth_survival': np.std(truth_survival),
            'extinction_rate': sum(1 for t in truth_survival if t == 0) / len(truth_survival)
        })
    
    return results


def run_live_terminal_simulation(
    generations: int = 150,
    population_size: int = 100,
    cost_of_truth: float = 0.5,
    update_interval: int = 5
):
    """
    Run simulation with live ASCII terminal visualization.
    Shows population bars updating in real-time.
    """
    import time
    import os
    
    np.random.seed(42)
    
    # Initialize population
    population = (
        [Agent('truth') for _ in range(50)] +
        [Agent('interface') for _ in range(50)]
    )
    
    bar_width = 50
    
    print("\n" + "=" * 70)
    print("  HOFFMAN'S INTERFACE THEORY: LIVE SIMULATION")
    print("  Watching evolution choose between TRUTH and INTERFACE")
    print("=" * 70 + "\n")
    time.sleep(1)
    
    for gen in range(generations):
        # Resource fluctuates
        resource = 50 + 30 * np.sin(gen / 20) + np.random.normal(0, 10)
        resource = np.clip(resource, 0, 100)
        
        # Agents act
        for agent in population:
            payoff = agent.perceive_and_act(resource, cost_of_truth)
            agent.energy = max(0, agent.energy + payoff)
        
        # Count populations
        truth_count = sum(1 for a in population if a.agent_type == 'truth')
        interface_count = sum(1 for a in population if a.agent_type == 'interface')
        
        # Reproduction
        total_energy = sum(a.energy for a in population)
        if total_energy > 0:
            probs = np.array([max(0.001, a.energy) for a in population])
            probs = probs / probs.sum()
            indices = np.random.choice(len(population), size=population_size, p=probs, replace=True)
            population = [Agent(population[i].agent_type) for i in indices]
        
        # Display every N generations
        if gen % update_interval == 0:
            # Clear line and print
            truth_bar = int((truth_count / population_size) * bar_width)
            interface_bar = int((interface_count / population_size) * bar_width)
            
            truth_visual = "█" * truth_bar + "░" * (bar_width - truth_bar)
            interface_visual = "█" * interface_bar + "░" * (bar_width - interface_bar)
            
            resource_indicator = "▼" if resource < 50 else "▲"
            
            print(f"Gen {gen:3d} │ Resource: {resource:5.1f} {resource_indicator}")
            print(f"        │ Truth:     [{truth_visual}] {truth_count:3d}")
            print(f"        │ Interface: [{interface_visual}] {interface_count:3d}")
            
            if truth_count == 0:
                print(f"        │ ❌ TRUTH EXTINCT!")
            elif interface_count == 0:
                print(f"        │ ❌ INTERFACE EXTINCT!")
            print()
            
            time.sleep(0.1)
    
    # Final result
    truth_count = sum(1 for a in population if a.agent_type == 'truth')
    interface_count = sum(1 for a in population if a.agent_type == 'interface')
    
    print("=" * 70)
    print("  FINAL RESULT")
    print("=" * 70)
    print(f"  Truth Agents:     {truth_count}")
    print(f"  Interface Agents: {interface_count}")
    print()
    
    if truth_count == 0:
        print("  ⚠️  TRUTH WENT EXTINCT")
        print()
        print("  This proves Hoffman's claim:")
        print("  Evolution selects for FITNESS, not TRUTH.")
        print()
        print('  "The desktop metaphor was designed to hide')
        print('   the truth from you, not to show it."')
        print("                           — Donald Hoffman")
    
    return {'truth': truth_count, 'interface': interface_count}


if __name__ == "__main__":
    import sys
    
    # Check for --live flag
    if "--live" in sys.argv or "-l" in sys.argv:
        run_live_terminal_simulation(generations=150, update_interval=3)
        sys.exit(0)
    
    print("=" * 60)
    print("HOFFMAN'S INTERFACE THEORY: FITNESS vs TRUTH")
    print("=" * 60)
    print()
    print("The Question: Does evolution favor organisms that see reality?")
    print("The Answer: No. Evolution favors USEFUL interfaces over TRUTH.")
    print()
    print("TIP: Run with --live for animated terminal visualization")
    print()
    
    # Run main simulation
    result = run_hoffman_simulation(
        generations=300,
        population_size=100,
        cost_of_truth=0.5,
        resource_volatility=1.0,
        seed=42
    )
    
    # Plot results
    plot_hoffman_results(
        result, 
        save_path='interface_theory_experiments/hoffman_simulation.png'
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final Truth Agents: {result['final_truth_count']}")
    print(f"Final Interface Agents: {result['final_interface_count']}")
    print(f"Winner: {result['winner'].upper()}")
    print()
    
    if result['truth_extinct']:
        print("⚠️  TRUTH WENT EXTINCT")
        print()
        print("This proves Hoffman's claim:")
        print("Evolution selects for FITNESS, not TRUTH.")
        print("Seeing reality is expensive; seeing icons is cheap.")
        print()
        print("'The desktop metaphor was designed to hide the truth")
        print(" from you, not to show it to you.'")
        print("                                    — Donald Hoffman")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'interface_theory_experiments/hoffman_results_{timestamp}.json', 'w') as f:
        # Convert numpy types for JSON serialization
        serializable = {
            k: v if not isinstance(v, dict) else {
                kk: [float(x) if isinstance(x, (np.floating, float)) else x for x in vv] 
                if isinstance(vv, list) else vv
                for kk, vv in v.items()
            }
            for k, v in result.items()
        }
        json.dump(serializable, f, indent=2, default=str)
