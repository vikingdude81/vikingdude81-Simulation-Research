"""
🎨 Visualization for 64-Gene Prisoner's Dilemma Evolution

Creates comprehensive dashboard showing:
1. Fitness evolution over generations
2. Chromosome heatmap (64 genes visualized)
3. TFT similarity tracking
4. Gene distribution analysis
"""

import matplotlib.pyplot as plt
import numpy as np
from prisoner_64gene import AdvancedPrisonerEvolution, create_tit_for_tat

class Advanced64Visualizer:
    """Creates visualizations for 64-gene Prisoner's Dilemma evolution."""
    
    def __init__(self, simulation: AdvancedPrisonerEvolution):
        self.sim = simulation
        self.tft = create_tit_for_tat()
    
    def create_dashboard(self, filename: str = "prisoner_64gene_dashboard.png"):
        """Creates a comprehensive 4-panel dashboard."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        fig.suptitle('🧬 64-Gene Prisoner\'s Dilemma Evolution Dashboard', 
                     fontsize=16, fontweight='bold')
        
        # Panel 1: Fitness Evolution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_fitness_evolution(ax1)
        
        # Panel 2: TFT Similarity (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_tft_similarity(ax2)
        
        # Panel 3: Best Chromosome Heatmap (middle, spanning both columns)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_chromosome_heatmap(ax3)
        
        # Panel 4: Gene Distribution (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_gene_distribution(ax4)
        
        # Panel 5: Strategy Analysis (bottom right)
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_strategy_analysis(ax5)
        
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
    
    def _plot_tft_similarity(self, ax):
        """Plot similarity to Tit-for-Tat over time."""
        similarities = []
        
        for chrom in self.sim.best_chromosome_history:
            similarity = sum(1 for a, b in zip(chrom, self.tft) if a == b)
            similarity_pct = (similarity / 64) * 100
            similarities.append(similarity_pct)
        
        generations = range(len(similarities))
        
        ax.plot(generations, similarities, color='purple', linewidth=2)
        ax.axhline(y=100, color='red', linestyle='--', linewidth=1, 
                   label='Perfect TFT', alpha=0.5)
        ax.axhline(y=50, color='gray', linestyle=':', linewidth=1, 
                   label='Random (50%)', alpha=0.5)
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Similarity to TFT (%)', fontsize=12)
        ax.set_title('Convergence to Tit-for-Tat Strategy', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
    
    def _plot_chromosome_heatmap(self, ax):
        """
        Visualize the best chromosome as an 8x8 heatmap.
        Each cell represents one gene (C=1, D=0).
        """
        best_chrom = self.sim.best_chromosome_history[-1]
        
        # Convert to numeric: C=1, D=0
        chrom_numeric = [1 if gene == 'C' else 0 for gene in best_chrom]
        
        # Reshape to 8x8 grid
        heatmap_data = np.array(chrom_numeric).reshape(8, 8)
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', 
                      vmin=0, vmax=1, interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, fraction=0.05)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Defect (D)', 'Cooperate (C)'])
        
        # Add grid
        ax.set_xticks(np.arange(8))
        ax.set_yticks(np.arange(8))
        ax.set_xticklabels(range(8))
        ax.set_yticklabels(range(8))
        ax.grid(which='both', color='white', linewidth=1)
        
        # Labels
        ax.set_xlabel('Gene Column (Low 3 bits of index)', fontsize=12)
        ax.set_ylabel('Gene Row (High 3 bits of index)', fontsize=12)
        ax.set_title('Best Chromosome (64-Gene Lookup Table)', 
                    fontsize=14, fontweight='bold')
        
        # Add text annotations for a few cells
        for i in range(8):
            for j in range(8):
                index = i * 8 + j
                if index % 8 == 0:  # Label every 8th gene
                    text = ax.text(j, i, f'{index}',
                                 ha="center", va="center", 
                                 color="black", fontsize=6, alpha=0.5)
    
    def _plot_gene_distribution(self, ax):
        """Plot distribution of C vs D genes in best chromosome over time."""
        c_percentages = []
        
        for chrom in self.sim.best_chromosome_history:
            c_count = chrom.count('C')
            c_pct = (c_count / 64) * 100
            c_percentages.append(c_pct)
        
        generations = range(len(c_percentages))
        
        ax.plot(generations, c_percentages, color='green', linewidth=2, 
               label='Cooperate (C)')
        ax.plot(generations, [100 - c for c in c_percentages], 
               color='red', linewidth=2, label='Defect (D)')
        
        ax.axhline(y=50, color='gray', linestyle=':', linewidth=1, 
                  label='50/50 Split', alpha=0.5)
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Percentage of Genes (%)', fontsize=12)
        ax.set_title('Gene Distribution in Best Strategy', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
    
    def _plot_strategy_analysis(self, ax):
        """Display text analysis of the final strategy."""
        ax.axis('off')
        
        best_chrom = self.sim.best_chromosome_history[-1]
        best_fitness = self.sim.best_fitness_history[-1]
        
        # Calculate statistics
        c_count = best_chrom.count('C')
        d_count = best_chrom.count('D')
        c_pct = (c_count / 64) * 100
        d_pct = (d_count / 64) * 100
        
        # TFT similarity
        tft_similarity = sum(1 for a, b in zip(best_chrom, self.tft) if a == b)
        tft_pct = (tft_similarity / 64) * 100
        
        text_content = f"🎯 Final Best Strategy Analysis\n"
        text_content += "="*50 + "\n\n"
        text_content += f"Generation: {self.sim.generation}\n"
        text_content += f"Best Fitness: {best_fitness:.1f}\n\n"
        
        text_content += f"Gene Distribution:\n"
        text_content += f"  Cooperate (C): {c_count}/64 ({c_pct:.1f}%)\n"
        text_content += f"  Defect (D):    {d_count}/64 ({d_pct:.1f}%)\n\n"
        
        text_content += f"Similarity to Tit-for-Tat:\n"
        text_content += f"  {tft_similarity}/64 genes match ({tft_pct:.1f}%)\n\n"
        
        if best_chrom == self.tft:
            text_content += "✨ PERFECT TIT-FOR-TAT EVOLVED!\n\n"
        elif tft_pct > 90:
            text_content += "⭐ Very close to Tit-for-Tat!\n\n"
        elif tft_pct > 70:
            text_content += "🎯 TFT-like strategy evolved\n\n"
        
        text_content += "Chromosome Sample:\n"
        text_content += f"  First 16: {best_chrom[:16]}\n"
        text_content += f"  Last 16:  {best_chrom[-16:]}\n\n"
        
        text_content += "Interpretation:\n"
        text_content += "• First 16 = Action after (C,C),(C,C),[X,X]\n"
        text_content += "• Last 16  = Action after (D,D),(D,D),[X,X]\n"
        text_content += "• Index formula: (h0*16)+(h1*4)+h2\n"
        
        ax.text(0.05, 0.95, text_content, 
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

def run_and_visualize():
    """Run evolution and create visualization."""
    print("🧬 Running 64-Gene Prisoner's Dilemma Evolution...")
    
    sim = AdvancedPrisonerEvolution(
        population_size=50,
        elite_size=5,
        mutation_rate=0.01,
        crossover_rate=0.7,
        rounds_per_matchup=100
    )
    
    sim.run(generations=100, print_every=20)
    
    print("\n🎨 Creating visualization...")
    visualizer = Advanced64Visualizer(sim)
    visualizer.create_dashboard()
    
    print("\n✅ Complete! Check prisoner_64gene_dashboard.png")

if __name__ == "__main__":
    run_and_visualize()
