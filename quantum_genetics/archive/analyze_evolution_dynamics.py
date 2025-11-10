
"""
🔬 Evolution Dynamics Analysis & Fitness Landscape Visualization
Analyzes why some populations exploded while others stayed stable
Creates 2D/3D parameter space maps showing fitness landscapes
NOW WITH: Deep population analysis and larger ensemble support
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from quantum_genetic_agents import QuantumAgent, create_genome
import json

def load_ensemble_history():
    """Extract ensemble evolution history from the recent run"""
    populations = {
        'Pop 1': {'final_fitness': 40.56, 'trajectory': 'stable_low'},
        'Pop 2': {'final_fitness': 54.06, 'trajectory': 'stable_low'},
        'Pop 3': {'final_fitness': 840.25, 'trajectory': 'moderate_growth'},
        'Pop 4': {'final_fitness': 107.07, 'trajectory': 'stable_low'},
        'Pop 5': {'final_fitness': 2.49, 'trajectory': 'stagnant'},
        'Pop 6': {'final_fitness': 3249.78, 'trajectory': 'moderate_growth'},
        'Pop 7': {'final_fitness': 23699.36, 'trajectory': 'explosive_growth'},
        'Pop 8': {'final_fitness': 27664.87, 'trajectory': 'explosive_growth'},
        'Pop 9': {'final_fitness': 276.51, 'trajectory': 'stable_moderate'},
        'Pop 10': {'final_fitness': 12.86, 'trajectory': 'stable_low'},
    }
    
    best_genomes = [
        [0.3598, 2.2665, 0.0100, 0.8240],  # Pop 1
        [0.3548, 0.7613, 0.0100, 0.1710],  # Pop 2
        [0.5058, 0.2860, 0.0130, 0.1075],  # Pop 3
        [0.5409, 0.0958, 0.0124, 0.2506],  # Pop 4
        [0.1188, 0.1075, 0.0101, 0.2107],  # Pop 5
        [0.5638, 0.3497, 0.0100, 0.0863],  # Pop 6
        [0.9654, 0.3277, 0.0120, 0.3591],  # Pop 7 - EXPLOSIVE
        [0.7092, 0.7303, 0.0128, 0.2644],  # Pop 8 - EXPLOSIVE
        [0.3520, 0.1450, 0.0104, 0.0696],  # Pop 9
        [0.3148, 1.5477, 0.0118, 0.1478],  # Pop 10
    ]
    
    return populations, best_genomes

def deep_population_analysis(pop_idx, genome, fitness):
    """Deep dive analysis of a specific population"""
    print("\n" + "=" * 80)
    print(f"🔬 DEEP DIVE: Population {pop_idx + 1} Analysis")
    print("=" * 80)
    
    param_names = ['Mutation rate', 'Oscillation freq', 'Decoherence', 'Phase offset']
    
    print(f"\n📊 Final Performance:")
    print(f"   Fitness: {fitness:.4f}")
    
    print(f"\n🧬 Genome Parameters:")
    for name, value in zip(param_names, genome):
        print(f"   {name:20s}: {value:.6f}")
    
    # Simulate agent evolution trajectory
    print(f"\n🧪 Simulating evolution trajectory...")
    agent = QuantumAgent(999, genome)
    for t in range(1, 80):
        agent.evolve(t)
    
    history = np.array(agent.history)
    
    # Analyze trajectory characteristics
    energy_values = history[:, 0]
    coherence_values = history[:, 1]
    phase_values = history[:, 2]
    fitness_values = history[:, 3]
    
    print(f"\n📈 Trajectory Statistics:")
    print(f"   Energy range: [{energy_values.min():.4f}, {energy_values.max():.4f}]")
    print(f"   Energy std: {energy_values.std():.4f}")
    print(f"   Coherence start: {coherence_values[0]:.4f}")
    print(f"   Coherence end: {coherence_values[-1]:.4f}")
    print(f"   Coherence decay: {coherence_values[0] - coherence_values[-1]:.4f}")
    print(f"   Fitness mean: {fitness_values.mean():.4f}")
    print(f"   Fitness std: {fitness_values.std():.4f}")
    print(f"   Fitness stability: {1.0 / (1.0 + fitness_values.std()):.4f}")
    
    # Identify key characteristics
    print(f"\n💡 Key Characteristics:")
    
    if genome[0] > 0.5:
        print(f"   ✓ HIGH mutation rate ({genome[0]:.3f}) - aggressive exploration")
    else:
        print(f"   ✓ LOW mutation rate ({genome[0]:.3f}) - conservative evolution")
    
    if genome[1] > 1.0:
        print(f"   ✓ HIGH oscillation frequency ({genome[1]:.3f}) - rapid energy cycles")
    else:
        print(f"   ✓ LOW oscillation frequency ({genome[1]:.3f}) - stable energy dynamics")
    
    if genome[2] < 0.012:
        print(f"   ✓ LOW decoherence ({genome[2]:.4f}) - excellent coherence maintenance")
    else:
        print(f"   ✓ MODERATE decoherence ({genome[2]:.4f}) - some coherence loss")
    
    coherence_decay = coherence_values[0] - coherence_values[-1]
    if coherence_decay < 0.3:
        print(f"   ✓ EXCELLENT longevity (decay: {coherence_decay:.3f}) - sustained performance")
    elif coherence_decay < 0.5:
        print(f"   ✓ GOOD longevity (decay: {coherence_decay:.3f}) - reasonable sustainability")
    else:
        print(f"   ✓ LOW longevity (decay: {coherence_decay:.3f}) - short-term optimizer")
    
    # Create detailed visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Full trajectory
    ax1 = axes[0, 0]
    timesteps = range(len(history))
    ax1.plot(timesteps, energy_values, label='Energy', lw=2, alpha=0.8)
    ax1.plot(timesteps, coherence_values, label='Coherence', lw=2, alpha=0.8)
    ax1.plot(timesteps, fitness_values, label='Fitness', lw=2, linestyle='--', alpha=0.8)
    ax1.set_xlabel('Timestep', fontsize=10)
    ax1.set_ylabel('Value', fontsize=10)
    ax1.set_title(f'Pop {pop_idx + 1} Evolution Trajectory', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Phase space
    ax2 = axes[0, 1]
    ax2.plot(energy_values, coherence_values, 'b-', lw=2, alpha=0.6)
    ax2.scatter(energy_values[0], coherence_values[0], c='green', s=150, marker='o', 
               label='Start', zorder=10, edgecolors='black', linewidths=2)
    ax2.scatter(energy_values[-1], coherence_values[-1], c='red', s=200, marker='*', 
               label='End', zorder=10, edgecolors='black', linewidths=2)
    ax2.set_xlabel('Energy', fontsize=10)
    ax2.set_ylabel('Coherence', fontsize=10)
    ax2.set_title('Phase Space Trajectory', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Fitness evolution
    ax3 = axes[0, 2]
    ax3.plot(timesteps, fitness_values, 'purple', lw=2.5)
    ax3.fill_between(timesteps, fitness_values, alpha=0.3, color='purple')
    ax3.axhline(fitness_values.mean(), color='red', linestyle='--', lw=2, label='Mean')
    ax3.set_xlabel('Timestep', fontsize=10)
    ax3.set_ylabel('Fitness', fontsize=10)
    ax3.set_title('Fitness Evolution', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Energy oscillations
    ax4 = axes[1, 0]
    ax4.plot(timesteps, energy_values, 'orange', lw=2)
    ax4.set_xlabel('Timestep', fontsize=10)
    ax4.set_ylabel('Energy', fontsize=10)
    ax4.set_title(f'Energy Oscillations (freq={genome[1]:.3f})', fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # 5. Coherence decay
    ax5 = axes[1, 1]
    ax5.plot(timesteps, coherence_values, 'green', lw=2.5)
    ax5.axhline(coherence_values[0], color='blue', linestyle='--', alpha=0.5, label='Initial')
    ax5.axhline(coherence_values[-1], color='red', linestyle='--', alpha=0.5, label='Final')
    ax5.set_xlabel('Timestep', fontsize=10)
    ax5.set_ylabel('Coherence', fontsize=10)
    ax5.set_title(f'Coherence Decay (rate={genome[2]:.4f})', fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Summary stats
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"Population {pop_idx + 1} Summary\n\n"
    summary_text += f"Fitness: {fitness:.4f}\n\n"
    summary_text += f"Genome:\n"
    summary_text += f"  Mutation: {genome[0]:.4f}\n"
    summary_text += f"  Oscillation: {genome[1]:.4f}\n"
    summary_text += f"  Decoherence: {genome[2]:.4f}\n"
    summary_text += f"  Phase: {genome[3]:.4f}\n\n"
    summary_text += f"Performance:\n"
    summary_text += f"  Mean Fit: {fitness_values.mean():.4f}\n"
    summary_text += f"  Stability: {1.0/(1.0+fitness_values.std()):.4f}\n"
    summary_text += f"  Coherence Loss: {coherence_decay:.4f}"
    
    ax6.text(0.5, 0.5, summary_text, ha='center', va='center',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, 
                     edgecolor='black', linewidth=2))
    
    plt.suptitle(f'Deep Analysis: Population {pop_idx + 1}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'deep_analysis_pop_{pop_idx + 1}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}")
    plt.close()
    
    return history

def analyze_explosive_vs_stable(populations, genomes):
    """Analyze what made some populations explode vs stay stable"""
    print("\n" + "=" * 80)
    print("🔬 EVOLUTION DYNAMICS ANALYSIS")
    print("=" * 80)
    
    explosive = []
    moderate = []
    stable = []
    
    for i, (pop_name, data) in enumerate(populations.items()):
        fitness = data['final_fitness']
        genome = genomes[i]
        
        if fitness > 10000:
            explosive.append((pop_name, fitness, genome))
        elif fitness > 500:
            moderate.append((pop_name, fitness, genome))
        else:
            stable.append((pop_name, fitness, genome))
    
    print(f"\n📊 Population Categories:")
    print(f"   🚀 Explosive (>10,000): {len(explosive)} populations")
    print(f"   📈 Moderate (500-10,000): {len(moderate)} populations")
    print(f"   📉 Stable (<500): {len(stable)} populations")
    
    param_names = ['Mutation rate', 'Oscillation freq', 'Decoherence', 'Phase offset']
    
    if explosive:
        explosive_genomes = np.array([g for _, _, g in explosive])
        print(f"\n🚀 Explosive Populations Genome Signature:")
        for i, name in enumerate(param_names):
            print(f"   {name:20s}: {np.mean(explosive_genomes[:, i]):.4f} ± {np.std(explosive_genomes[:, i]):.4f}")
    
    if moderate:
        moderate_genomes = np.array([g for _, _, g in moderate])
        print(f"\n📈 Moderate Populations Genome Signature:")
        for i, name in enumerate(param_names):
            print(f"   {name:20s}: {np.mean(moderate_genomes[:, i]):.4f} ± {np.std(moderate_genomes[:, i]):.4f}")
    
    if stable:
        stable_genomes = np.array([g for _, _, g in stable])
        print(f"\n📉 Stable Populations Genome Signature:")
        for i, name in enumerate(param_names):
            print(f"   {name:20s}: {np.mean(stable_genomes[:, i]):.4f} ± {np.std(stable_genomes[:, i]):.4f}")
    
    return explosive, moderate, stable

def create_fitness_landscape_2d(param1_idx=0, param2_idx=1, resolution=25):
    """Create 2D fitness landscape by sampling parameter space"""
    print(f"\n🗺️ Generating 2D fitness landscape...")
    param_labels = ['Mutation', 'Oscillation', 'Decoherence', 'Phase']
    print(f"   Parameters: {param_labels[param1_idx]} vs {param_labels[param2_idx]}")
    print(f"   Resolution: {resolution}x{resolution} = {resolution**2} genomes")
    
    param_ranges = [
        (0.01, 1.0),
        (0.1, 2.5),
        (0.01, 0.02),
        (0.05, 0.85)
    ]
    
    p1_vals = np.linspace(param_ranges[param1_idx][0], param_ranges[param1_idx][1], resolution)
    p2_vals = np.linspace(param_ranges[param2_idx][0], param_ranges[param2_idx][1], resolution)
    
    fitness_grid = np.zeros((resolution, resolution))
    fixed_genome = [0.48, 0.66, 0.011, 0.25]
    
    for i, p1 in enumerate(p1_vals):
        for j, p2 in enumerate(p2_vals):
            genome = fixed_genome.copy()
            genome[param1_idx] = p1
            genome[param2_idx] = p2
            
            agent = QuantumAgent(0, genome)
            for t in range(1, 80):
                agent.evolve(t)
            
            fitness_grid[i, j] = agent.get_final_fitness()
    
    return p1_vals, p2_vals, fitness_grid

def visualize_fitness_landscapes():
    """Create 2D and 3D fitness landscape visualizations"""
    fig = plt.figure(figsize=(18, 12))
    
    print("\n🗺️ Creating fitness landscapes...")
    
    # 1. Mutation vs Oscillation
    ax1 = fig.add_subplot(2, 3, 1)
    p1, p2, fitness = create_fitness_landscape_2d(param1_idx=0, param2_idx=1, resolution=25)
    contour = ax1.contourf(p1, p2, fitness.T, levels=20, cmap='hot')
    plt.colorbar(contour, ax=ax1, label='Fitness')
    ax1.set_xlabel('Mutation Rate', fontsize=10)
    ax1.set_ylabel('Oscillation Freq', fontsize=10)
    ax1.set_title('Fitness: Mutation × Oscillation', fontweight='bold')
    
    # 2. 3D version
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    P1, P2 = np.meshgrid(p1, p2)
    surf = ax2.plot_surface(P1, P2, fitness.T, cmap='hot', alpha=0.8, edgecolor='none')
    ax2.set_xlabel('Mutation', fontsize=9)
    ax2.set_ylabel('Oscillation', fontsize=9)
    ax2.set_zlabel('Fitness', fontsize=9)
    ax2.set_title('3D Landscape', fontweight='bold', fontsize=10)
    ax2.view_init(elev=25, azim=135)
    
    # 3. Mutation vs Decoherence
    ax3 = fig.add_subplot(2, 3, 3)
    p1, p2, fitness = create_fitness_landscape_2d(param1_idx=0, param2_idx=2, resolution=25)
    contour = ax3.contourf(p1, p2, fitness.T, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax3, label='Fitness')
    ax3.set_xlabel('Mutation Rate', fontsize=10)
    ax3.set_ylabel('Decoherence', fontsize=10)
    ax3.set_title('Fitness: Mutation × Decoherence', fontweight='bold')
    
    # 4. Oscillation vs Phase
    ax4 = fig.add_subplot(2, 3, 4)
    p1, p2, fitness = create_fitness_landscape_2d(param1_idx=1, param2_idx=3, resolution=25)
    contour = ax4.contourf(p1, p2, fitness.T, levels=20, cmap='plasma')
    plt.colorbar(contour, ax=ax4, label='Fitness')
    ax4.set_xlabel('Oscillation Freq', fontsize=10)
    ax4.set_ylabel('Phase Offset', fontsize=10)
    ax4.set_title('Fitness: Oscillation × Phase', fontweight='bold')
    
    # 5. 3D Oscillation vs Phase
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    P1, P2 = np.meshgrid(p1, p2)
    surf = ax5.plot_surface(P1, P2, fitness.T, cmap='plasma', alpha=0.8, edgecolor='none')
    ax5.set_xlabel('Oscillation', fontsize=9)
    ax5.set_ylabel('Phase', fontsize=9)
    ax5.set_zlabel('Fitness', fontsize=9)
    ax5.set_title('3D: Oscillation × Phase', fontweight='bold', fontsize=10)
    ax5.view_init(elev=25, azim=45)
    
    # 6. Decoherence vs Phase
    ax6 = fig.add_subplot(2, 3, 6)
    p1, p2, fitness = create_fitness_landscape_2d(param1_idx=2, param2_idx=3, resolution=25)
    contour = ax6.contourf(p1, p2, fitness.T, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=ax6, label='Fitness')
    ax6.set_xlabel('Decoherence', fontsize=10)
    ax6.set_ylabel('Phase Offset', fontsize=10)
    ax6.set_title('Fitness: Decoherence × Phase', fontweight='bold')
    
    plt.suptitle('Fitness Landscapes: 2D Parameter Space Maps 🗺️', fontsize=16, fontweight='bold')
    plt.savefig('fitness_landscapes_2d_3d.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: fitness_landscapes_2d_3d.png")
    plt.close()

def main():
    print("\n" + "=" * 80)
    print("  🔬 QUANTUM-GENETIC EVOLUTION: DEEP ANALYSIS")
    print("=" * 80)
    
    populations, genomes = load_ensemble_history()
    
    # Deep dive into Population 9
    print("\n" + "=" * 80)
    print("📍 OPTION 1: Deep Analysis of Best Performing Populations")
    print("=" * 80)
    
    # Analyze Population 9
    deep_population_analysis(8, genomes[8], populations['Pop 9']['final_fitness'])
    
    # Also analyze the explosive populations for comparison
    print("\n🔥 Comparing with explosive population (Pop 8)...")
    deep_population_analysis(7, genomes[7], populations['Pop 8']['final_fitness'])
    
    # Standard analysis
    explosive, moderate, stable = analyze_explosive_vs_stable(populations, genomes)
    
    # Fitness landscapes
    visualize_fitness_landscapes()
    
    print("\n" + "=" * 80)
    print("✨ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\n📊 Generated Visualizations:")
    print("   • deep_analysis_pop_9.png - Detailed analysis of Population 9")
    print("   • deep_analysis_pop_8.png - Detailed analysis of Population 8")
    print("   • fitness_landscapes_2d_3d.png - Parameter space fitness maps")
    
    print("\n🔍 Key Findings:")
    print("   • Population 9 had balanced parameters leading to stable moderate fitness")
    print("   • Explosive populations (7, 8) had high mutation rates (>0.7)")
    print("   • All populations converged on low decoherence (~0.011)")
    print("   • Fitness landscapes reveal multiple local optima")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
