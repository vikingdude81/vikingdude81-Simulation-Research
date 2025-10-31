"""
🎬 Simple Real-Time Evolution Viewer
Watch strategies evolve in real-time!
"""
import matplotlib
import sys
import os

# Try TkAgg, fallback to others if needed
backends = ['TkAgg', 'Qt5Agg', 'WXAgg', 'Agg']
for backend in backends:
    try:
        matplotlib.use(backend, force=True)
        break
    except:
        continue

import matplotlib.pyplot as plt
import numpy as np
from strategy_evolution import EvolutionSimulation, generate_synthetic_market
from trading_agent import CONDITION_NAMES, ACTION_BUY, ACTION_SELL, ACTION_HOLD

print("="*80)
print("🎬 REAL-TIME EVOLUTION VIEWER")
print("="*80)
print("\nWatch strategies evolve generation-by-generation!")
print("Close the window when you've seen enough.")
print("\n" + "="*80)

# Generate market
print("\n📊 Generating market data...")
prices = generate_synthetic_market(periods=100, trend=0.001, volatility=0.02)
print(f"   Price: ${prices[0]:.2f} → ${prices[-1]:.2f} ({((prices[-1]/prices[0])-1)*100:+.1f}%)")

# Create simulation
sim = EvolutionSimulation(
    population_size=30,
    elite_size=3,
    mutation_rate=0.15,
    crossover_rate=0.7
)
sim.initialize_population()

# Setup interactive plot
plt.ion()  # Interactive mode
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('🧬 EVOLUTION IN PROGRESS - LIVE VIEW', fontsize=16, fontweight='bold')

ax_fitness, ax_strategy, ax_diversity, ax_best = axes.flatten()

print("\n🚀 Starting evolution...")
print("   (Window will update every generation)")
print("   (Close window or press Ctrl+C to stop)")
print("\n" + "="*80)

# Initialize plot first
fig.show()
plt.pause(0.1)

try:
    for gen in range(30):  # Run 30 generations
        try:
            # Run one generation
            sim.evolve_generation(prices, verbose=False)
            
            # Get stats from history (evolve_generation updates these)
            best = sim.best_fitness_history[-1]
            avg = sim.avg_fitness_history[-1]
            
            # Get best agent
            best_agent = max(sim.population, key=lambda a: getattr(a, 'fitness', 0))
            
            print(f"Gen {gen+1:2d} | Best: ${best:.2f} | Avg: ${avg:.2f} | Strategy: {best_agent.chromosome}")
        except Exception as e:
            print(f"   Error in generation {gen+1}: {e}")
            continue
        
        # Update plots
        ax_fitness.clear()
        ax_strategy.clear()
        ax_diversity.clear()
        ax_best.clear()
        
        # Plot 1: Fitness progression
        gens = range(len(sim.best_fitness_history))
        ax_fitness.plot(gens, sim.best_fitness_history, 'g-', linewidth=3, marker='o', label='Best', markersize=8)
        ax_fitness.plot(gens, sim.avg_fitness_history, 'b-', linewidth=2, marker='s', label='Average', markersize=6)
        ax_fitness.fill_between(gens, sim.best_fitness_history, sim.avg_fitness_history, alpha=0.2, color='green')
        ax_fitness.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax_fitness.set_ylabel('Fitness ($)', fontsize=11, fontweight='bold')
        ax_fitness.set_title(f'FITNESS EVOLUTION (Gen {gen+1})', fontsize=12, fontweight='bold')
        ax_fitness.legend()
        ax_fitness.grid(True, alpha=0.3)
        
        # Plot 2: Strategy diversity heatmap
        gene_counts = np.zeros((5, 3))
        action_map = {ACTION_BUY: 0, ACTION_SELL: 1, ACTION_HOLD: 2}
        
        for agent in sim.population:
            for condition, action in enumerate(agent.chromosome.genes):
                gene_counts[condition, action_map[action]] += 1
        
        gene_counts = gene_counts / len(sim.population) * 100
        
        im = ax_strategy.imshow(gene_counts, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        ax_strategy.set_xticks([0, 1, 2])
        ax_strategy.set_xticklabels(['BUY', 'SELL', 'HOLD'], fontsize=10)
        ax_strategy.set_yticks(range(5))
        ax_strategy.set_yticklabels([CONDITION_NAMES[i][:12] for i in range(5)], fontsize=9)
        ax_strategy.set_title('STRATEGY DISTRIBUTION (%)', fontsize=12, fontweight='bold')
        
        for i in range(5):
            for j in range(3):
                ax_strategy.text(j, i, f'{gene_counts[i, j]:.0f}',
                       ha="center", va="center", 
                       color="black" if gene_counts[i, j] < 50 else "white",
                       fontsize=10, fontweight='bold')
        
        # Plot 3: Best strategy chromosome
        genes = best_agent.chromosome.genes
        action_colors = {ACTION_BUY: 'green', ACTION_SELL: 'red', ACTION_HOLD: 'gray'}
        colors = [action_colors[g] for g in genes]
        
        y_pos = np.arange(5)
        ax_diversity.barh(y_pos, [1]*5, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax_diversity.set_yticks(y_pos)
        ax_diversity.set_yticklabels([CONDITION_NAMES[i][:12] for i in range(5)], fontsize=10)
        ax_diversity.set_xlim(0, 1)
        ax_diversity.set_xticks([])
        
        for i, action in enumerate(genes):
            ax_diversity.text(0.5, i, action, ha='center', va='center', 
                   fontsize=16, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        ax_diversity.set_title(f'BEST STRATEGY\n{best_agent.chromosome}', fontsize=12, fontweight='bold')
        ax_diversity.spines['top'].set_visible(False)
        ax_diversity.spines['right'].set_visible(False)
        ax_diversity.spines['bottom'].set_visible(False)
        
        # Plot 4: Improvement stats
        ax_best.axis('off')
        improvement = ((sim.best_fitness_history[-1] / sim.best_fitness_history[0]) - 1) * 100
        
        stats_text = f"""
        EVOLUTION STATS
        
        Generation: {gen+1}/30
        
        Best Fitness: ${best:.2f}
        Avg Fitness: ${avg:.2f}
        
        Improvement: {improvement:+.2f}%
        
        Best Strategy:
        {best_agent.chromosome}
        
        Trades: {len(best_agent.trades)}
        W/L: {best_agent.wins}/{best_agent.losses}
        """
        
        ax_best.text(0.1, 0.5, stats_text, fontsize=12, 
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        try:
            plt.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.5)  # Update every 0.5 seconds
        except Exception as e:
            print(f"   Plot update error: {e}")
        
except KeyboardInterrupt:
    print("\n\n⏸️  Evolution interrupted by user")
except Exception as e:
    print(f"\n\n❌ Error during evolution: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("✅ EVOLUTION COMPLETE!")
print("="*80)

# Final summary
if len(sim.best_fitness_history) > 0:
    best_final = max(sim.population, key=lambda a: getattr(a, 'fitness', 0))
    improvement = ((sim.best_fitness_history[-1] / sim.best_fitness_history[0]) - 1) * 100
    final_fitness = getattr(best_final, 'fitness', 0)

    print(f"\n🏆 BEST EVOLVED STRATEGY: {best_final.chromosome}")
    print(f"   Fitness: ${final_fitness:.2f}")
    print(f"   Improvement: {improvement:+.2f}%")
    print(f"   Trades: {len(best_final.trades)} | W/L: {best_final.wins}/{best_final.losses}")

print("\n💡 Keeping window open - close window to exit, or press Ctrl+C")
print("="*80)

plt.ioff()
try:
    plt.show(block=True)  # Block until window closed
except:
    input("\nPress Enter to exit...")
