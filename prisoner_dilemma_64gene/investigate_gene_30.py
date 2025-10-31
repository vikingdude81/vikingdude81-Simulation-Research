"""
🔬 DEEP DIVE: GENE 30 INVESTIGATION
===================================

Gene 30 is the #1 most important gene (4.56% importance) for predicting convergence speed.

Game State Decoded:
- Binary: 011110 (30 in decimal)
- Your last 3 moves: DCC (011)
- Opponent's last 3 moves: CCD (110)

Context: "Mutual Recovery State"
- You started defecting, then cooperated twice
- Opponent cooperated twice, then defected once
- This is a critical decision point: forgive and continue cooperation, or return to defection?

This script analyzes:
1. What does Gene 30 encode? (Cooperate or Defect in this state?)
2. How does Gene 30 frequency correlate with convergence speed?
3. How does Gene 30 evolve over generations?
4. Do successful runs have different Gene 30 patterns than the outlier?
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from datetime import datetime

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (16, 12)

def load_dataset():
    """Load the 1,000-run dataset"""
    filepath = r"c:\Users\akbon\OneDrive\Documents\PRICE-DETECTION-TEST-1\PRICE-DETECTION-TEST-1\prisoner_dilemma_64gene\chaos_unified_GPU_1000runs_20251031_000616.json"
    print("📂 Loading dataset...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data['runs'])} runs\n")
    return data

def decode_gene_30():
    """Decode what Gene 30 represents in game theory terms"""
    print("="*70)
    print("🧬 GENE 30 DECODER")
    print("="*70)
    
    gene_idx = 30
    binary = format(gene_idx, '06b')
    
    print(f"\n📊 Gene Index: {gene_idx}")
    print(f"   Binary: {binary}")
    print(f"   Split: {binary[:3]} | {binary[3:]}")
    
    # Decode your moves
    your_moves = binary[:3]
    your_str = ''.join(['D' if b == '0' else 'C' for b in your_moves])
    
    # Decode opponent moves
    opp_moves = binary[3:]
    opp_str = ''.join(['D' if b == '0' else 'C' for b in opp_moves])
    
    print(f"\n🎮 Game State:")
    print(f"   Your last 3 moves:       {your_str} (most recent on right)")
    print(f"   Opponent's last 3 moves: {opp_str} (most recent on right)")
    
    print(f"\n📖 Interpretation:")
    print(f"   YOU: Started with D, then C, then C → 'Recovery from defection'")
    print(f"   OPP: Started with C, then C, then D → 'Testing or retaliation'")
    
    print(f"\n❓ The Critical Question Gene 30 Answers:")
    print(f"   'After I tried to cooperate (CC) following my defection,")
    print(f"    but opponent just defected once (maybe testing me),")
    print(f"    should I:")
    print(f"    - COOPERATE (forgive the test, rebuild trust) → C")
    print(f"    - DEFECT (punish the defection, be cautious) → D'")
    
    print(f"\n🎯 Strategic Importance:")
    print(f"   This is a 'FORGIVENESS TEST' moment:")
    print(f"   - If Gene 30 = C: Resilient cooperation (mutual recovery)")
    print(f"   - If Gene 30 = D: Fragile cooperation (spiral back to conflict)")
    
    return gene_idx

def analyze_gene30_correlation(data):
    """Analyze correlation between Gene 30 and convergence outcomes"""
    print("\n" + "="*70)
    print("📊 GENE 30 CORRELATION ANALYSIS")
    print("="*70)
    
    gene_30_initial = []  # Gene 30 frequency at generation 0
    gene_30_final = []    # Gene 30 frequency at generation 99
    gene_30_mean = []     # Average Gene 30 frequency across all gens
    convergence_times = []
    final_fitness = []
    behaviors = []
    
    for run in data['runs']:
        gene_freq_matrix = np.array(run['gene_frequency_matrix'])
        
        gene_30_initial.append(gene_freq_matrix[0, 30])
        gene_30_final.append(gene_freq_matrix[-1, 30])
        gene_30_mean.append(gene_freq_matrix[:, 30].mean())
        
        convergence_times.append(run['chaos_analysis']['convergence_time'])
        final_fitness.append(run['chaos_analysis']['final_fitness'])
        behaviors.append(run['chaos_analysis']['behavior'])
    
    gene_30_initial = np.array(gene_30_initial)
    gene_30_final = np.array(gene_30_final)
    gene_30_mean = np.array(gene_30_mean)
    convergence_times = np.array(convergence_times)
    final_fitness = np.array(final_fitness)
    
    print(f"\n📈 Gene 30 Frequency Statistics:")
    print(f"   Initial (Gen 0):")
    print(f"      Mean: {gene_30_initial.mean():.3f} ± {gene_30_initial.std():.3f}")
    print(f"      Range: [{gene_30_initial.min():.3f}, {gene_30_initial.max():.3f}]")
    
    print(f"\n   Final (Gen 99):")
    print(f"      Mean: {gene_30_final.mean():.3f} ± {gene_30_final.std():.3f}")
    print(f"      Range: [{gene_30_final.min():.3f}, {gene_30_final.max():.3f}]")
    
    print(f"\n   Evolution: {gene_30_final.mean() - gene_30_initial.mean():+.3f} change")
    
    # Correlation with convergence time
    corr_initial, p_initial = pearsonr(gene_30_initial, convergence_times)
    corr_final, p_final = pearsonr(gene_30_final, convergence_times)
    corr_mean, p_mean = pearsonr(gene_30_mean, convergence_times)
    
    print(f"\n🔗 Correlation with Convergence Time:")
    print(f"   Gene 30 (Initial):  r={corr_initial:+.4f}, p={p_initial:.4f}")
    print(f"   Gene 30 (Final):    r={corr_final:+.4f}, p={p_final:.4f}")
    print(f"   Gene 30 (Mean):     r={corr_mean:+.4f}, p={p_mean:.4f}")
    
    # Correlation with final fitness
    corr_fitness_initial, p_fit_init = pearsonr(gene_30_initial, final_fitness)
    corr_fitness_final, p_fit_final = pearsonr(gene_30_final, final_fitness)
    
    print(f"\n🔗 Correlation with Final Fitness:")
    print(f"   Gene 30 (Initial):  r={corr_fitness_initial:+.4f}, p={p_fit_init:.4f}")
    print(f"   Gene 30 (Final):    r={corr_fitness_final:+.4f}, p={p_fit_final:.4f}")
    
    # Compare convergent vs periodic
    periodic_indices = [i for i, b in enumerate(behaviors) if b == 'periodic']
    convergent_indices = [i for i, b in enumerate(behaviors) if b == 'convergent']
    
    if len(periodic_indices) > 0:
        print(f"\n🔍 Convergent vs Periodic Comparison:")
        print(f"   Convergent runs (n={len(convergent_indices)}):")
        print(f"      Gene 30 Initial: {gene_30_initial[convergent_indices].mean():.3f}")
        print(f"      Gene 30 Final:   {gene_30_final[convergent_indices].mean():.3f}")
        print(f"\n   Periodic runs (n={len(periodic_indices)}):")
        print(f"      Gene 30 Initial: {gene_30_initial[periodic_indices].mean():.3f}")
        print(f"      Gene 30 Final:   {gene_30_final[periodic_indices].mean():.3f}")
        
        diff_initial = gene_30_initial[convergent_indices].mean() - gene_30_initial[periodic_indices].mean()
        diff_final = gene_30_final[convergent_indices].mean() - gene_30_final[periodic_indices].mean()
        
        print(f"\n   Difference (Convergent - Periodic):")
        print(f"      Initial: {diff_initial:+.3f}")
        print(f"      Final:   {diff_final:+.3f}")
    
    return {
        'gene_30_initial': gene_30_initial,
        'gene_30_final': gene_30_final,
        'gene_30_mean': gene_30_mean,
        'convergence_times': convergence_times,
        'final_fitness': final_fitness,
        'behaviors': behaviors
    }

def analyze_gene30_evolution(data):
    """Analyze how Gene 30 evolves across generations"""
    print("\n" + "="*70)
    print("🔄 GENE 30 EVOLUTIONARY TRAJECTORY")
    print("="*70)
    
    # Sample 10 runs for detailed analysis
    sample_runs = [0, 10, 30, 50, 100, 200, 300, 500, 700, 999]  # Include run 30 (the periodic one)
    
    print(f"\n📊 Sampling {len(sample_runs)} runs for detailed evolution:\n")
    
    evolution_data = {}
    
    for run_idx in sample_runs:
        run = data['runs'][run_idx]
        gene_freq_matrix = np.array(run['gene_frequency_matrix'])
        gene_30_traj = gene_freq_matrix[:, 30]
        
        behavior = run['chaos_analysis']['behavior']
        conv_time = run['chaos_analysis']['convergence_time']
        final_fit = run['chaos_analysis']['final_fitness']
        
        print(f"Run {run_idx:4d} ({behavior:12s}): ", end="")
        print(f"Conv={conv_time:2d}, Fit={final_fit:7.1f}, ", end="")
        print(f"Gene30: {gene_30_traj[0]:.3f} → {gene_30_traj[-1]:.3f} ", end="")
        print(f"(Δ={gene_30_traj[-1] - gene_30_traj[0]:+.3f})")
        
        evolution_data[run_idx] = {
            'trajectory': gene_30_traj,
            'behavior': behavior,
            'convergence_time': conv_time,
            'final_fitness': final_fit
        }
    
    # Statistics on Gene 30 evolution
    all_changes = []
    for run in data['runs']:
        gene_freq_matrix = np.array(run['gene_frequency_matrix'])
        change = gene_freq_matrix[-1, 30] - gene_freq_matrix[0, 30]
        all_changes.append(change)
    
    all_changes = np.array(all_changes)
    
    print(f"\n📊 Gene 30 Evolution Statistics (all 1000 runs):")
    print(f"   Mean change: {all_changes.mean():+.3f}")
    print(f"   Std change:  {all_changes.std():.3f}")
    print(f"   Increased:   {np.sum(all_changes > 0)} runs ({np.sum(all_changes > 0)/10:.1f}%)")
    print(f"   Decreased:   {np.sum(all_changes < 0)} runs ({np.sum(all_changes < 0)/10:.1f}%)")
    print(f"   Unchanged:   {np.sum(all_changes == 0)} runs ({np.sum(all_changes == 0)/10:.1f}%)")
    
    return evolution_data

def create_visualizations(data, analysis_results):
    """Create comprehensive visualizations"""
    print("\n" + "="*70)
    print("📊 CREATING VISUALIZATIONS")
    print("="*70)
    
    fig = plt.figure(figsize=(20, 12))
    
    # Extract data
    gene_30_initial = analysis_results['gene_30_initial']
    gene_30_final = analysis_results['gene_30_final']
    gene_30_mean = analysis_results['gene_30_mean']
    convergence_times = analysis_results['convergence_times']
    final_fitness = analysis_results['final_fitness']
    behaviors = analysis_results['behaviors']
    
    # 1. Distribution of Gene 30 (Initial vs Final)
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(gene_30_initial, bins=30, alpha=0.6, label='Generation 0', color='blue')
    ax1.hist(gene_30_final, bins=30, alpha=0.6, label='Generation 99', color='red')
    ax1.set_xlabel('Gene 30 Frequency (Cooperation Rate)')
    ax1.set_ylabel('Number of Runs')
    ax1.set_title('Gene 30 Distribution: Initial vs Final')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Gene 30 vs Convergence Time
    ax2 = plt.subplot(2, 3, 2)
    scatter = ax2.scatter(gene_30_mean, convergence_times, 
                         c=convergence_times, cmap='viridis', alpha=0.6, s=30)
    ax2.set_xlabel('Gene 30 Mean Frequency')
    ax2.set_ylabel('Convergence Time (generations)')
    ax2.set_title('Gene 30 vs Convergence Speed')
    plt.colorbar(scatter, ax=ax2, label='Conv Time')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation
    corr, p = pearsonr(gene_30_mean, convergence_times)
    ax2.text(0.05, 0.95, f'r={corr:.3f}, p={p:.4f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Gene 30 vs Final Fitness
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(gene_30_final, final_fitness, 
                         c=final_fitness, cmap='plasma', alpha=0.6, s=30)
    ax3.set_xlabel('Gene 30 Final Frequency')
    ax3.set_ylabel('Final Fitness')
    ax3.set_title('Gene 30 vs Final Fitness')
    plt.colorbar(scatter, ax=ax3, label='Fitness')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation
    corr, p = pearsonr(gene_30_final, final_fitness)
    ax3.text(0.05, 0.95, f'r={corr:.3f}, p={p:.4f}', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Gene 30 Evolution Trajectories (sample runs)
    ax4 = plt.subplot(2, 3, 4)
    
    # Sample diverse runs
    sample_indices = [0, 30, 50, 100, 500, 999]  # Include run 30 (periodic)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, run_idx in enumerate(sample_indices):
        run = data['runs'][run_idx]
        gene_freq_matrix = np.array(run['gene_frequency_matrix'])
        gene_30_traj = gene_freq_matrix[:, 30]
        behavior = run['chaos_analysis']['behavior']
        
        label = f"Run {run_idx} ({behavior})"
        if behavior == 'periodic':
            ax4.plot(gene_30_traj, color=colors[idx], linewidth=3, 
                    label=label, linestyle='--', alpha=0.9)
        else:
            ax4.plot(gene_30_traj, color=colors[idx], linewidth=1.5, 
                    label=label, alpha=0.7)
    
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Gene 30 Frequency')
    ax4.set_title('Gene 30 Evolution (Sample Runs)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Change in Gene 30 distribution
    ax5 = plt.subplot(2, 3, 5)
    changes = gene_30_final - gene_30_initial
    ax5.hist(changes, bins=40, color='teal', alpha=0.7, edgecolor='black')
    ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax5.set_xlabel('Change in Gene 30 Frequency (Final - Initial)')
    ax5.set_ylabel('Number of Runs')
    ax5.set_title('Distribution of Gene 30 Evolution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Convergent vs Periodic comparison
    ax6 = plt.subplot(2, 3, 6)
    
    periodic_indices = [i for i, b in enumerate(behaviors) if b == 'periodic']
    convergent_indices = [i for i, b in enumerate(behaviors) if b == 'convergent']
    
    data_to_plot = [
        gene_30_initial[convergent_indices],
        gene_30_final[convergent_indices],
        gene_30_initial[periodic_indices] if len(periodic_indices) > 0 else [0],
        gene_30_final[periodic_indices] if len(periodic_indices) > 0 else [0]
    ]
    
    positions = [1, 2, 4, 5]
    labels = ['Conv\nInitial', 'Conv\nFinal', 'Period\nInitial', 'Period\nFinal']
    colors_box = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    
    bp = ax6.boxplot(data_to_plot, positions=positions, labels=labels,
                     patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    
    ax6.set_ylabel('Gene 30 Frequency')
    ax6.set_title('Gene 30: Convergent vs Periodic Runs')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'gene30_investigation_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {filename}")
    
    return filename

def main():
    print("\n" + "="*70)
    print("🔬 DEEP DIVE: GENE 30 INVESTIGATION")
    print("="*70)
    print("\nGene 30: #1 Most Important Gene (4.56% importance)")
    print("Game State: You=DCC, Opponent=CCD")
    print("Context: 'Mutual Recovery State' - forgiveness test")
    print("="*70)
    
    # Load data
    data = load_dataset()
    
    # Decode Gene 30
    gene_idx = decode_gene_30()
    
    # Correlation analysis
    analysis_results = analyze_gene30_correlation(data)
    
    # Evolution analysis
    evolution_data = analyze_gene30_evolution(data)
    
    # Visualizations
    viz_file = create_visualizations(data, analysis_results)
    
    print("\n" + "="*70)
    print("✅ INVESTIGATION COMPLETE")
    print("="*70)
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   • Gene 30 encodes a 'forgiveness test' decision")
    print(f"   • Critical for mutual recovery after conflict")
    print(f"   • Shows {analysis_results['gene_30_final'].mean() - analysis_results['gene_30_initial'].mean():+.3f} average evolution")
    print(f"   • Weak direct correlation but high ML importance")
    print(f"   • Suggests INTERACTION EFFECTS with other genes")
    print("="*70)

if __name__ == "__main__":
    main()
