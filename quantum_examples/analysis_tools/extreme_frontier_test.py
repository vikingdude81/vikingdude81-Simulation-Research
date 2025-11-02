
"""
🚀 EXTREME MUTATION FRONTIER TEST
Push mutation rates beyond 6.27 to find the ceiling
Based on Strategy Comparison results showing μ=6.27 achieves 10²⁸ fitness
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_genetic_agents import QuantumAgent
import os

# Optimal parameters from Strategy Comparison
OPTIMAL_PHASE = 1.05
OPTIMAL_DECOHERENCE = 0.011
OPTIMAL_OSCILLATION = 1.0

def test_extreme_mutations():
    print("\n" + "=" * 80)
    print("  🚀 TESTING EXTREME MUTATION FRONTIER (μ = 6.0 to 10.0)")
    print("=" * 80)
    print(f"\n🎯 Using optimal parameters from Strategy Comparison:")
    print(f"   Phase: {OPTIMAL_PHASE}")
    print(f"   Decoherence: {OPTIMAL_DECOHERENCE} (locked)")
    print(f"   Oscillation: {OPTIMAL_OSCILLATION}")
    
    # Test mutation rates beyond the winning 6.27
    mutation_rates = np.linspace(6.0, 10.0, 25)
    fitness_results = []
    
    print(f"\n🧪 Testing {len(mutation_rates)} extreme mutation rates...")
    
    for i, mut_rate in enumerate(mutation_rates):
        genome = [mut_rate, OPTIMAL_OSCILLATION, OPTIMAL_DECOHERENCE, OPTIMAL_PHASE]
        
        agent = QuantumAgent(0, genome)
        for t in range(1, 50):
            agent.evolve(t)
        
        fitness = agent.get_final_fitness()
        fitness_results.append(fitness)
        
        if (i + 1) % 5 == 0 or fitness > 1e30:
            print(f"   Progress: {i+1}/{len(mutation_rates)} - μ={mut_rate:.2f}, Fitness={fitness:.2e}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Linear scale
    axes[0, 0].plot(mutation_rates, fitness_results, 'ro-', lw=2, markersize=8, alpha=0.7)
    axes[0, 0].axvline(6.27, color='green', linestyle='--', lw=2, label='Strategy Comparison best (μ=6.27)')
    axes[0, 0].set_xlabel('Mutation Rate (μ)', fontsize=11)
    axes[0, 0].set_ylabel('Fitness', fontsize=11)
    axes[0, 0].set_title('Extreme Frontier Exploration', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Log-log to see scaling
    axes[0, 1].loglog(mutation_rates, fitness_results, 'bo-', lw=2, markersize=8, alpha=0.7)
    axes[0, 1].axvline(6.27, color='green', linestyle='--', lw=2, label='Best known')
    axes[0, 1].set_xlabel('Mutation Rate (μ)', fontsize=11)
    axes[0, 1].set_ylabel('Fitness (log scale)', fontsize=11)
    axes[0, 1].set_title('Scaling Analysis (log-log)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Derivative (rate of change)
    axes[1, 0].plot(mutation_rates[1:], np.diff(fitness_results), 'mo-', lw=2, alpha=0.7)
    axes[1, 0].axhline(0, color='red', linestyle='--', lw=1)
    axes[1, 0].set_xlabel('Mutation Rate (μ)', fontsize=11)
    axes[1, 0].set_ylabel('Δ Fitness', fontsize=11)
    axes[1, 0].set_title('Fitness Improvement Rate', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    
    best_idx = np.argmax(fitness_results)
    best_mut = mutation_rates[best_idx]
    best_fit = fitness_results[best_idx]
    
    # Find if there's a ceiling (diminishing returns)
    improvements = np.diff(fitness_results)
    declining = np.sum(improvements < 0)
    
    summary_text = f"""
🚀 EXTREME FRONTIER RESULTS

Best μ: {best_mut:.2f}
Max fitness: {best_fit:.2e}

Baseline (μ=6.27): {fitness_results[np.argmin(np.abs(mutation_rates - 6.27))]:.2e}

Improvement: {best_fit / fitness_results[0]:.2e}×

📊 ANALYSIS:
• Declining points: {declining}/{len(improvements)}
• Trend: {'Saturating' if declining > 5 else 'Still growing'}

💡 CONCLUSION:
{'Frontier ceiling found!' if best_mut < 9.0 and declining > 5 else 'Frontier continues beyond μ=10.0'}
"""
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    plt.suptitle('🚀 Extreme Mutation Frontier Exploration (μ > 6.0)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/extreme_frontier.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: visualizations/extreme_frontier.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print(f"🏆 EXTREME FRONTIER RESULTS:")
    print(f"   Optimal μ: {best_mut:.2f}")
    print(f"   Max fitness: {best_fit:.2e}")
    print(f"   vs. μ=6.27: {best_fit / fitness_results[0]:.2e}× improvement")
    print("=" * 80)
    
    return mutation_rates, fitness_results, best_mut, best_fit

if __name__ == "__main__":
    test_extreme_mutations()
