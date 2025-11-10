"""
🎨 Visualization for Simple Prisoner's Dilemma Evolution

Creates comprehensive dashboard showing:
1. Fitness evolution over generations
2. Strategy diversity and convergence
3. Population distribution
4. Strategy frequency heatmap
"""

import matplotlib.pyplot as plt
import numpy as np
from prisoner_evolution import PrisonerEvolution

class PrisonerVisualizer:
    """Creates visualizations for Prisoner's Dilemma evolution."""
    
    def __init__(self, simulation: PrisonerEvolution):
        self.sim = simulation
    
    def create_dashboard(self, filename: str = "prisoner_evolution_dashboard.png"):
        """Creates a comprehensive 4-panel dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🧬 Prisoner\'s Dilemma Evolution Dashboard', 
                     fontsize=16, fontweight='bold')
        
        # Panel 1: Fitness Evolution
        self._plot_fitness_evolution(axes[0, 0])
        
        # Panel 2: Strategy Diversity Over Time
        self._plot_strategy_diversity(axes[0, 1])
        
        # Panel 3: Final Population Distribution
        self._plot_final_distribution(axes[1, 0])
        
        # Panel 4: Strategy Interpretation
        self._plot_strategy_interpretation(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Dashboard saved: {filename}")
        
        return fig
    
    def _plot_fitness_evolution(self, ax):
        """Plot best and average fitness over generations."""
        generations = range(len(self.sim.best_fitness_history))
        
        ax.plot(generations, self.sim.best_fitness_history, 
                label='Best Fitness', color='green', linewidth=2)
        ax.plot(generations, self.sim.avg_fitness_history, 
                label='Average Fitness', color='blue', linewidth=2, alpha=0.7)
        
        ax.fill_between(generations, 
                        self.sim.avg_fitness_history, 
                        self.sim.best_fitness_history,
                        alpha=0.2, color='green')
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness (Total Score)', fontsize=12)
        ax.set_title('Fitness Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def _plot_strategy_diversity(self, ax):
        """Plot how strategies change over time as a stacked area chart."""
        # Get all unique strategies
        all_strategies = set()
        for counts in self.sim.strategy_counts_history:
            all_strategies.update(counts.keys())
        
        all_strategies = sorted(all_strategies)
        
        # Build matrix: generations x strategies
        generations = len(self.sim.strategy_counts_history)
        strategy_matrix = np.zeros((generations, len(all_strategies)))
        
        for gen_idx, counts in enumerate(self.sim.strategy_counts_history):
            for strat_idx, strategy in enumerate(all_strategies):
                strategy_matrix[gen_idx, strat_idx] = counts.get(strategy, 0)
        
        # Create stacked area chart
        gen_range = range(generations)
        ax.stackplot(gen_range, *strategy_matrix.T, 
                    labels=all_strategies, alpha=0.8)
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Agent Count', fontsize=12)
        ax.set_title('Strategy Diversity Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_final_distribution(self, ax):
        """Plot final generation's strategy distribution as bar chart."""
        final_counts = self.sim.strategy_counts_history[-1]
        strategies = list(final_counts.keys())
        counts = list(final_counts.values())
        
        # Sort by count
        sorted_data = sorted(zip(strategies, counts), key=lambda x: x[1], reverse=True)
        strategies, counts = zip(*sorted_data)
        
        colors = ['green' if s == 'CDC' else 'red' if s == 'DDD' else 'blue' if s == 'CCC' else 'gray' 
                  for s in strategies]
        
        bars = ax.bar(range(len(strategies)), counts, color=colors, alpha=0.7)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45)
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('Agent Count', fontsize=12)
        ax.set_title(f'Final Population Distribution (Gen {self.sim.generation})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    def _plot_strategy_interpretation(self, ax):
        """Display text interpretation of dominant strategies."""
        ax.axis('off')
        
        # Get top 5 strategies
        final_counts = self.sim.strategy_counts_history[-1]
        top_strategies = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        text_content = "🎯 Dominant Strategies\n" + "="*40 + "\n\n"
        
        for i, (strategy, count) in enumerate(top_strategies, 1):
            pct = (count / self.sim.population_size) * 100
            text_content += f"{i}. {strategy} ({count} agents, {pct:.1f}%)\n"
            text_content += self._interpret_strategy_short(strategy) + "\n\n"
        
        # Add famous strategy legend
        text_content += "\n" + "="*40 + "\n"
        text_content += "📚 Famous Strategies:\n"
        text_content += "• CDC = Tit-for-Tat (copy opponent)\n"
        text_content += "• DDD = Always Defect (betray all)\n"
        text_content += "• CCC = Always Cooperate (trust all)\n"
        text_content += "• DDC = Suspicious TFT (start mean)\n"
        text_content += "• CCD = Grudger (never forgive)\n"
        
        ax.text(0.05, 0.95, text_content, 
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    def _interpret_strategy_short(self, chromosome: str) -> str:
        """Short interpretation of a strategy."""
        interpretations = {
            "CDC": "   ✨ Tit-for-Tat: Start nice, copy opponent",
            "DDD": "   💀 Always Defect: Never cooperate",
            "CCC": "   😇 Always Cooperate: Always trust",
            "DDC": "   🤨 Suspicious TFT: Start mean, then copy",
            "CCD": "   😤 Grudger: Nice until betrayed"
        }
        
        return interpretations.get(chromosome, f"   Gene0={chromosome[0]}, vs_C={chromosome[1]}, vs_D={chromosome[2]}")

def run_and_visualize():
    """Run evolution and create visualization."""
    print("🧬 Running Prisoner's Dilemma Evolution...")
    
    sim = PrisonerEvolution(
        population_size=50,
        elite_size=5,
        mutation_rate=0.15,
        crossover_rate=0.7,
        rounds_per_matchup=50
    )
    
    sim.run(generations=50, print_every=10)
    
    print("\n🎨 Creating visualization...")
    visualizer = PrisonerVisualizer(sim)
    visualizer.create_dashboard()
    
    print("\n✅ Complete! Check prisoner_evolution_dashboard.png")

if __name__ == "__main__":
    run_and_visualize()
