
"""
🔬 Compare All Evolved Genomes from Experiments 1-3
Analyzes and visualizes differences between all 6 genome files
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from quantum_genetic_agents import QuantumAgent

def load_genome_file(filepath):
    """Load a genome JSON file"""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def simulate_genome_performance(genome_params, n_timesteps=80):
    """Simulate an agent with given genome parameters"""
    agent = QuantumAgent(0, genome_params)
    for t in range(1, n_timesteps):
        agent.evolve(t)
    
    history = np.array(agent.history)
    return {
        'energy': history[:, 0],
        'coherence': history[:, 1],
        'phase': history[:, 2],
        'fitness': history[:, 3],
        'final_fitness': agent.get_final_fitness()
    }

def compare_all_genomes():
    """Load and compare all 6 genome files"""
    print("\n" + "=" * 80)
    print("  🔬 COMPARING ALL EVOLVED GENOMES")
    print("=" * 80)
    
    genome_files = [
        'best_individual_long_evolution_genome.json',
        'averaged_long_evolution_genome.json',
        'best_individual_more_populations_genome.json',
        'averaged_more_populations_genome.json',
        'best_individual_hybrid_genome.json',
        'averaged_hybrid_genome.json'
    ]
    
    labels = [
        'Exp1: Best Individual',
        'Exp1: Averaged',
        'Exp2: Best Individual',
        'Exp2: Averaged',
        'Exp3: Best Individual',
        'Exp3: Averaged'
    ]
    
    colors = ['#e74c3c', '#c0392b', '#3498db', '#2980b9', '#2ecc71', '#27ae60']
    
    genomes_data = []
    simulations = []
    
    print("\n📂 Loading genome files...")
    for i, (filepath, label) in enumerate(zip(genome_files, labels)):
        data = load_genome_file(filepath)
        if data:
            genome = data['genome']
            genome_list = [
                genome['mutation_rate'],
                genome['oscillation_freq'],
                genome['decoherence_rate'],
                genome['phase_offset']
            ]
            
            print(f"\n✓ {label}")
            print(f"   Fitness: {data['metadata']['fitness']:.4f}")
            print(f"   Generation: {data['metadata']['generation']}")
            print(f"   Robustness: {data['metadata'].get('robustness_score', 'N/A')}")
            
            genomes_data.append({
                'label': label,
                'genome': genome_list,
                'metadata': data['metadata'],
                'color': colors[i]
            })
            
            # Simulate performance
            print(f"   Simulating...")
            sim = simulate_genome_performance(genome_list)
            simulations.append(sim)
        else:
            print(f"\n✗ {label} - File not found")
    
    if not genomes_data:
        print("\n❌ No genome files found!")
        return
    
    # Create comprehensive comparison visualization
    fig = plt.figure(figsize=(20, 14))
    
    # 1. Parameter comparison bar chart
    ax1 = plt.subplot(3, 3, 1)
    param_names = ['Mutation', 'Oscillation', 'Decoherence', 'Phase']
    x = np.arange(len(param_names))
    width = 0.12
    
    for i, gdata in enumerate(genomes_data):
        offset = (i - len(genomes_data)/2) * width
        ax1.bar(x + offset, gdata['genome'], width, 
               label=gdata['label'], color=gdata['color'], alpha=0.8)
    
    ax1.set_xlabel('Parameters', fontsize=10)
    ax1.set_ylabel('Value', fontsize=10)
    ax1.set_title('Genome Parameters Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_names, rotation=45)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, axis='y')
    
    # 2. Fitness comparison
    ax2 = plt.subplot(3, 3, 2)
    fitness_values = [gdata['metadata']['fitness'] for gdata in genomes_data]
    bars = ax2.bar(range(len(genomes_data)), fitness_values, 
                   color=[gdata['color'] for gdata in genomes_data], alpha=0.8)
    ax2.set_xlabel('Genome', fontsize=10)
    ax2.set_ylabel('Fitness', fontsize=10)
    ax2.set_title('Final Fitness Comparison', fontweight='bold')
    ax2.set_xticks(range(len(genomes_data)))
    ax2.set_xticklabels([f"G{i+1}" for i in range(len(genomes_data))], rotation=45)
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. Robustness scores
    ax3 = plt.subplot(3, 3, 3)
    robustness = [gdata['metadata'].get('robustness_score', 0) for gdata in genomes_data]
    bars = ax3.bar(range(len(genomes_data)), robustness,
                   color=[gdata['color'] for gdata in genomes_data], alpha=0.8)
    ax3.set_xlabel('Genome', fontsize=10)
    ax3.set_ylabel('Robustness Score', fontsize=10)
    ax3.set_title('Robustness Comparison', fontweight='bold')
    ax3.set_xticks(range(len(genomes_data)))
    ax3.set_xticklabels([f"G{i+1}" for i in range(len(genomes_data))], rotation=45)
    ax3.grid(alpha=0.3, axis='y')
    
    # 4. Energy trajectories
    ax4 = plt.subplot(3, 3, 4)
    for i, (sim, gdata) in enumerate(zip(simulations, genomes_data)):
        ax4.plot(sim['energy'], label=gdata['label'], 
                color=gdata['color'], lw=2, alpha=0.8)
    ax4.set_xlabel('Timestep', fontsize=10)
    ax4.set_ylabel('Energy', fontsize=10)
    ax4.set_title('Energy Evolution', fontweight='bold')
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)
    
    # 5. Coherence trajectories
    ax5 = plt.subplot(3, 3, 5)
    for i, (sim, gdata) in enumerate(zip(simulations, genomes_data)):
        ax5.plot(sim['coherence'], label=gdata['label'],
                color=gdata['color'], lw=2, alpha=0.8)
    ax5.set_xlabel('Timestep', fontsize=10)
    ax5.set_ylabel('Coherence', fontsize=10)
    ax5.set_title('Coherence Evolution', fontweight='bold')
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.3)
    
    # 6. Fitness trajectories
    ax6 = plt.subplot(3, 3, 6)
    for i, (sim, gdata) in enumerate(zip(simulations, genomes_data)):
        ax6.plot(sim['fitness'], label=gdata['label'],
                color=gdata['color'], lw=2, alpha=0.8)
    ax6.set_xlabel('Timestep', fontsize=10)
    ax6.set_ylabel('Fitness', fontsize=10)
    ax6.set_title('Fitness Evolution', fontweight='bold')
    ax6.legend(fontsize=7)
    ax6.grid(alpha=0.3)
    
    # 7. Parameter heatmap
    ax7 = plt.subplot(3, 3, 7)
    genome_matrix = np.array([gdata['genome'] for gdata in genomes_data])
    im = ax7.imshow(genome_matrix, cmap='viridis', aspect='auto')
    ax7.set_yticks(range(len(genomes_data)))
    ax7.set_yticklabels([f"G{i+1}" for i in range(len(genomes_data))])
    ax7.set_xticks(range(4))
    ax7.set_xticklabels(param_names, rotation=45)
    ax7.set_title('Parameter Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax7, label='Value')
    
    # 8. Decoherence rate focus
    ax8 = plt.subplot(3, 3, 8)
    decoherence_rates = [gdata['genome'][2] for gdata in genomes_data]
    bars = ax8.bar(range(len(genomes_data)), decoherence_rates,
                   color=[gdata['color'] for gdata in genomes_data], alpha=0.8)
    ax8.axhline(np.mean(decoherence_rates), color='red', linestyle='--', 
               lw=2, label=f'Mean: {np.mean(decoherence_rates):.6f}')
    ax8.set_xlabel('Genome', fontsize=10)
    ax8.set_ylabel('Decoherence Rate', fontsize=10)
    ax8.set_title('Decoherence Rate (Critical Parameter)', fontweight='bold')
    ax8.set_xticks(range(len(genomes_data)))
    ax8.set_xticklabels([f"G{i+1}" for i in range(len(genomes_data))], rotation=45)
    ax8.legend(fontsize=8)
    ax8.grid(alpha=0.3, axis='y')
    
    # 9. Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = "📊 COMPARISON SUMMARY\n\n"
    summary_text += f"Total genomes: {len(genomes_data)}\n\n"
    summary_text += "Highest fitness:\n"
    max_idx = np.argmax(fitness_values)
    summary_text += f"  {genomes_data[max_idx]['label']}\n"
    summary_text += f"  {fitness_values[max_idx]:.4f}\n\n"
    summary_text += "Most robust:\n"
    max_rob_idx = np.argmax(robustness)
    summary_text += f"  {genomes_data[max_rob_idx]['label']}\n"
    summary_text += f"  {robustness[max_rob_idx]:.2f}\n\n"
    summary_text += "Avg decoherence:\n"
    summary_text += f"  {np.mean(decoherence_rates):.6f}\n"
    summary_text += f"  Std: {np.std(decoherence_rates):.6f}"
    
    ax9.text(0.5, 0.5, summary_text, ha='center', va='center',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', 
                     alpha=0.9, edgecolor='black', linewidth=2))
    
    plt.suptitle('Complete Genome Comparison: All 3 Experiments', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/all_genomes_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: visualizations/all_genomes_comparison.png")
    plt.close()
    
    # Print detailed statistics
    print("\n" + "=" * 80)
    print("📈 DETAILED STATISTICS")
    print("=" * 80)
    
    print("\n🏆 Rankings by Fitness:")
    sorted_by_fitness = sorted(enumerate(fitness_values), key=lambda x: x[1], reverse=True)
    for rank, (idx, fitness) in enumerate(sorted_by_fitness, 1):
        print(f"   {rank}. {genomes_data[idx]['label']}: {fitness:.4f}")
    
    print("\n🛡️ Rankings by Robustness:")
    sorted_by_robustness = sorted(enumerate(robustness), key=lambda x: x[1], reverse=True)
    for rank, (idx, rob) in enumerate(sorted_by_robustness, 1):
        print(f"   {rank}. {genomes_data[idx]['label']}: {rob:.2f}")
    
    print("\n🔬 Decoherence Rate Analysis:")
    print(f"   Mean: {np.mean(decoherence_rates):.6f}")
    print(f"   Std: {np.std(decoherence_rates):.6f}")
    print(f"   Min: {np.min(decoherence_rates):.6f}")
    print(f"   Max: {np.max(decoherence_rates):.6f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    compare_all_genomes()
