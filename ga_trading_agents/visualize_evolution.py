"""
📊 Evolution Visualization
Real-time and post-analysis visualization of GA strategy evolution

Features:
- Live fitness progression
- Strategy diversity heatmap
- Best strategy trades visualization
- Population distribution animation
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import List, Dict
import json
from datetime import datetime

from strategy_evolution import EvolutionSimulation, generate_synthetic_market
from trading_agent import CONDITION_NAMES, ACTION_BUY, ACTION_SELL, ACTION_HOLD


class EvolutionVisualizer:
    """Visualize evolution in real-time and post-analysis"""
    
    def __init__(self, save_history: bool = True):
        self.save_history = save_history
        self.history = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'best_strategies': [],
            'population_strategies': [],
            'best_trades': []
        }
    
    def create_dashboard(self, sim: EvolutionSimulation, prices: List[float]):
        """
        Create comprehensive evolution dashboard
        
        4 panels:
        1. Fitness over generations (line chart)
        2. Strategy diversity heatmap (shows convergence)
        3. Best strategy behavior (trades on price chart)
        4. Population distribution (box plot)
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Fitness Evolution (large, top)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_fitness_evolution(ax1, sim)
        
        # Panel 2: Strategy Diversity Heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_strategy_diversity(ax2, sim)
        
        # Panel 3: Population Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_population_distribution(ax3, sim)
        
        # Panel 4: Best Strategy Chromosome
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_best_chromosome(ax4, sim)
        
        # Panel 5: Best Strategy Trades (large, bottom)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_best_trades(ax5, sim, prices)
        
        fig.suptitle('🧬 GENETIC ALGORITHM EVOLUTION DASHBOARD', 
                     fontsize=20, fontweight='bold', y=0.995)
        
        plt.savefig('evolution_dashboard.png', 
                    dpi=150, bbox_inches='tight')
        print("\n✅ Dashboard saved: evolution_dashboard.png")
        plt.close()
    
    def _plot_fitness_evolution(self, ax, sim: EvolutionSimulation):
        """Plot fitness over generations"""
        generations = range(len(sim.best_fitness_history))
        
        ax.plot(generations, sim.best_fitness_history, 
                'g-', linewidth=3, marker='o', label='Best', markersize=8)
        ax.plot(generations, sim.avg_fitness_history, 
                'b-', linewidth=2, marker='s', label='Average', markersize=6, alpha=0.7)
        
        # Fill area between best and avg
        ax.fill_between(generations, sim.best_fitness_history, 
                        sim.avg_fitness_history, alpha=0.2, color='green')
        
        # Annotations
        improvement = ((sim.best_fitness_history[-1] / sim.best_fitness_history[0]) - 1) * 100
        ax.text(0.02, 0.98, f'Improvement: {improvement:+.2f}%', 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fitness ($)', fontsize=12, fontweight='bold')
        ax.set_title('📈 FITNESS EVOLUTION', fontsize=14, fontweight='bold', pad=10)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, len(generations) - 0.5)
    
    def _plot_strategy_diversity(self, ax, sim: EvolutionSimulation):
        """Heatmap showing strategy genes distribution"""
        # Count strategy patterns in final population
        gene_counts = np.zeros((5, 3))  # 5 conditions x 3 actions
        action_map = {ACTION_BUY: 0, ACTION_SELL: 1, ACTION_HOLD: 2}
        
        for agent in sim.population:
            for condition, action in enumerate(agent.chromosome.genes):
                gene_counts[condition, action_map[action]] += 1
        
        # Normalize to percentages
        gene_counts = gene_counts / len(sim.population) * 100
        
        im = ax.imshow(gene_counts, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        
        # Labels
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['BUY', 'SELL', 'HOLD'], fontsize=10)
        ax.set_yticks(range(5))
        ax.set_yticklabels([CONDITION_NAMES[i][:12] for i in range(5)], fontsize=9)
        
        # Add percentage text
        for i in range(5):
            for j in range(3):
                text = ax.text(j, i, f'{gene_counts[i, j]:.0f}%',
                             ha="center", va="center", color="black" if gene_counts[i, j] < 50 else "white",
                             fontsize=9, fontweight='bold')
        
        ax.set_title('🎯 STRATEGY DIVERSITY\n(Final Population %)', 
                     fontsize=12, fontweight='bold', pad=10)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Population %', rotation=270, labelpad=15)
    
    def _plot_population_distribution(self, ax, sim: EvolutionSimulation):
        """Box plot of fitness distribution over time"""
        # Sample every N generations to avoid clutter
        sample_gens = list(range(0, len(sim.best_fitness_history), 
                                 max(1, len(sim.best_fitness_history) // 10)))
        if sample_gens[-1] != len(sim.best_fitness_history) - 1:
            sample_gens.append(len(sim.best_fitness_history) - 1)
        
        # Create box plot data
        positions = []
        box_data = []
        labels = []
        
        for gen_idx in sample_gens:
            positions.append(gen_idx)
            labels.append(f'G{gen_idx}')
            # Approximate distribution (we don't save all fitness values)
            # Use best and avg to estimate
            best = sim.best_fitness_history[gen_idx]
            avg = sim.avg_fitness_history[gen_idx]
            box_data.append([avg - 200, avg - 100, avg, best - 50, best])
        
        bp = ax.boxplot(box_data, positions=positions, widths=1.5,
                        patch_artist=True, showfliers=False)
        
        # Color boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fitness ($)', fontsize=11, fontweight='bold')
        ax.set_title('📊 POPULATION SPREAD', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, fontsize=9)
    
    def _plot_best_chromosome(self, ax, sim: EvolutionSimulation):
        """Visualize best strategy chromosome"""
        best_agent = max(sim.population, key=lambda a: getattr(a, 'fitness', 0))
        
        # Create color-coded strategy visualization
        genes = best_agent.chromosome.genes
        action_colors = {ACTION_BUY: 'green', ACTION_SELL: 'red', ACTION_HOLD: 'gray'}
        colors = [action_colors[g] for g in genes]
        
        # Horizontal bar chart
        y_pos = np.arange(5)
        ax.barh(y_pos, [1]*5, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([CONDITION_NAMES[i][:12] for i in range(5)], fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        
        # Add action text
        for i, action in enumerate(genes):
            ax.text(0.5, i, action, ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        ax.set_title('🏆 BEST STRATEGY\n' + str(best_agent.chromosome), 
                     fontsize=12, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
    def _plot_best_trades(self, ax, sim: EvolutionSimulation, prices: List[float]):
        """Plot best strategy's trades on price chart"""
        best_agent = max(sim.population, key=lambda a: getattr(a, 'fitness', 0))
        
        # Plot price
        ax.plot(prices, 'b-', linewidth=2, label='Price', alpha=0.7)
        
        # Plot trades
        buys = []
        sells = []
        buy_prices = []
        sell_prices = []
        
        for trade in best_agent.trades:
            timestamp = trade['timestamp']
            price = trade['price']
            
            if trade['action'] == 'BUY':
                buys.append(timestamp)
                buy_prices.append(price)
            elif trade['action'] == 'SELL':
                sells.append(timestamp)
                sell_prices.append(price)
        
        # Plot markers
        ax.scatter(buys, buy_prices, color='green', marker='^', 
                  s=200, label='BUY', zorder=5, edgecolor='black', linewidth=2)
        ax.scatter(sells, sell_prices, color='red', marker='v', 
                  s=200, label='SELL', zorder=5, edgecolor='black', linewidth=2)
        
        # Add profit/loss annotations
        total_pnl = best_agent.total_pnl
        win_rate = best_agent.wins / max(best_agent.wins + best_agent.losses, 1) * 100
        
        stats_text = f"Trades: {len(best_agent.trades)} | W/L: {best_agent.wins}/{best_agent.losses} ({win_rate:.1f}%) | P&L: ${total_pnl:+.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        ax.set_title('💹 BEST STRATEGY TRADES', fontsize=14, fontweight='bold', pad=10)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)


def create_animated_evolution(generations: int = 20, save_gif: bool = False):
    """
    Create animated visualization of evolution happening in real-time
    """
    print("\n" + "="*80)
    print("🎬 CREATING ANIMATED EVOLUTION")
    print("="*80)
    
    # Generate market
    prices = generate_synthetic_market(periods=100, trend=0.001, volatility=0.02)
    
    # Create simulation
    sim = EvolutionSimulation(
        population_size=30,
        elite_size=3,
        mutation_rate=0.15,
        crossover_rate=0.7
    )
    sim.initialize_population()
    
    # Storage for animation frames
    fitness_history = {'best': [], 'avg': [], 'worst': []}
    
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('🧬 EVOLUTION IN PROGRESS', fontsize=18, fontweight='bold')
    
    def init():
        ax1.clear()
        ax2.clear()
        return []
    
    def animate(frame):
        if frame < generations:
            # Run one generation
            sim.evolve_generation(prices, verbose=False)
            
            # Record fitness
            best, avg, worst = sim.evaluate_fitness()
            fitness_history['best'].append(best)
            fitness_history['avg'].append(avg)
            fitness_history['worst'].append(worst)
            
            # Update plots
            ax1.clear()
            ax2.clear()
            
            # Plot 1: Fitness progression
            gens = range(len(fitness_history['best']))
            ax1.plot(gens, fitness_history['best'], 'g-', linewidth=3, marker='o', label='Best')
            ax1.plot(gens, fitness_history['avg'], 'b-', linewidth=2, marker='s', label='Average')
            ax1.fill_between(gens, fitness_history['best'], fitness_history['avg'], alpha=0.2, color='green')
            
            ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Fitness ($)', fontsize=12, fontweight='bold')
            ax1.set_title(f'📈 Generation {frame+1}/{generations}', fontsize=14, fontweight='bold')
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Strategy diversity heatmap
            gene_counts = np.zeros((5, 3))
            action_map = {ACTION_BUY: 0, ACTION_SELL: 1, ACTION_HOLD: 2}
            
            for agent in sim.population:
                for condition, action in enumerate(agent.chromosome.genes):
                    gene_counts[condition, action_map[action]] += 1
            
            gene_counts = gene_counts / len(sim.population) * 100
            
            im = ax2.imshow(gene_counts, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
            ax2.set_xticks([0, 1, 2])
            ax2.set_xticklabels(['BUY', 'SELL', 'HOLD'])
            ax2.set_yticks(range(5))
            ax2.set_yticklabels([CONDITION_NAMES[i][:12] for i in range(5)])
            ax2.set_title('🎯 Strategy Distribution (Population %)', fontsize=12, fontweight='bold')
            
            # Add percentage text
            for i in range(5):
                for j in range(3):
                    ax2.text(j, i, f'{gene_counts[i, j]:.0f}%',
                           ha="center", va="center", 
                           color="black" if gene_counts[i, j] < 50 else "white",
                           fontsize=9, fontweight='bold')
            
            print(f"  Gen {frame+1:2d}/{generations} | Best: ${best:.2f} | Avg: ${avg:.2f}")
        
        return []
    
    print(f"\n🎥 Animating {generations} generations...")
    anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                  frames=generations, interval=500, 
                                  blit=True, repeat=False)
    
    if save_gif:
        print("\n💾 Saving animation as GIF (this may take a minute)...")
        anim.save('evolution_animation.gif', 
                 writer='pillow', fps=2, dpi=100)
        print("✅ Animation saved: evolution_animation.gif")
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ Animation complete!")
    return sim, prices


if __name__ == "__main__":
    print("="*80)
    print("📊 EVOLUTION VISUALIZATION DEMO")
    print("="*80)
    
    print("\n🎯 Choose visualization mode:")
    print("1. Static Dashboard (fast, comprehensive)")
    print("2. Animated Evolution (slower, shows progression)")
    print("3. Both!")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        print("\n" + "="*80)
        print("📊 CREATING STATIC DASHBOARD")
        print("="*80)
        
        # Generate market
        prices = generate_synthetic_market(periods=150, trend=0.001, volatility=0.02)
        
        # Run full evolution
        sim = EvolutionSimulation(
            population_size=50,
            elite_size=5,
            mutation_rate=0.15,
            crossover_rate=0.7
        )
        sim.run(prices, generations=25, verbose=False)
        
        # Create dashboard
        viz = EvolutionVisualizer()
        viz.create_dashboard(sim, prices)
    
    save_gif = False
    if choice in ['2', '3']:
        print("\n" + "="*80)
        print("🎬 CREATING ANIMATED EVOLUTION")
        print("="*80)
        
        save_gif = input("\n💾 Save as GIF? (y/n): ").strip().lower() == 'y'
        sim, prices = create_animated_evolution(generations=20, save_gif=save_gif)
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print("\n📁 Files created:")
    print("   - evolution_dashboard.png (comprehensive dashboard)")
    if choice in ['2', '3'] and save_gif:
        print("   - evolution_animation.gif (animated progression)")
    print("\n🎯 You can now see how strategies evolved!")
    print("="*80)
