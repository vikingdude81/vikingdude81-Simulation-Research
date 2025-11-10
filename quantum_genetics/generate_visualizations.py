"""
Generate visualizations for completed experiments
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sns.set_style("darkgrid")

def visualize_island_results():
    """Generate visualization for island evolution"""
    print("🎨 Generating Island Evolution visualization...")
    
    # Load the most recent island results
    import glob
    island_files = glob.glob("island_evolution_results_*.json")
    if not island_files:
        print("❌ No island evolution results found")
        return
    
    latest_file = max(island_files)
    print(f"   Loading: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Island comparison
    ax = axes[0, 0]
    island_ids = [s['island_id'] for s in data['island_stats']]
    best_fitness = [s['best_fitness'] for s in data['island_stats']]
    
    ax.bar(island_ids, best_fitness, alpha=0.7, color='steelblue')
    ax.set_title('Final Island Best Fitness', fontsize=12, fontweight='bold')
    ax.set_xlabel('Island ID')
    ax.set_ylabel('Best Fitness')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Champion info
    ax = axes[0, 1]
    ax.axis('off')
    champion = data['global_champion']
    genome = champion['genome']
    text = f"""
🏆 GLOBAL CHAMPION
Island: {champion['island_id']}
Fitness: {champion['fitness']:.6f}

Genome:
μ (mutation):    {genome[0]:.4f}
ω (oscillation): {genome[1]:.4f}
d (decoherence): {genome[2]:.6f}
φ (phase):       {genome[3]:.4f}

Total Agents: {data['total_agents']}
Generations: {data['generations']}
Migration Events: {len(data['migration_events'])}
"""
    ax.text(0.1, 0.5, text, fontsize=10, family='monospace',
           verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Parameter comparison
    ax = axes[0, 2]
    param_names = ['μ', 'ω', 'd', 'φ']
    genomes = [s['best_genome'] for s in data['island_stats']]
    
    for i, param_name in enumerate(param_names):
        values = [g[i] for g in genomes]
        ax.scatter([i]*len(values), values, alpha=0.6, s=50)
    
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names)
    ax.set_title('Parameter Distribution Across Islands', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter Value')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Diversity metrics
    ax = axes[1, 0]
    diversity = data['final_diversity']
    
    div_names = ['μ', 'ω', 'd', 'φ']
    div_values = [diversity['mu_std'], diversity['omega_std'], 
                  diversity['d_std'], diversity['phi_std']]
    
    ax.bar(div_names, div_values, alpha=0.7, color='coral')
    ax.set_title('Final Island Diversity (Std Dev)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standard Deviation')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Migration events
    ax = axes[1, 1]
    if data['migration_events']:
        migration_gens = [e['generation'] for e in data['migration_events']]
        ax.scatter(migration_gens, [1]*len(migration_gens), s=100, 
                  marker='|', color='purple', linewidths=3)
        ax.set_ylim(0.5, 1.5)
        ax.set_xlim(0, data['generations'])
    
    ax.set_title(f'Migration Events ({len(data["migration_events"])} total)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    # 6. Top 3 islands
    ax = axes[1, 2]
    ax.axis('off')
    
    sorted_islands = sorted(data['island_stats'], 
                          key=lambda x: x['best_fitness'], reverse=True)[:3]
    
    text = "🏆 TOP 3 ISLANDS\n\n"
    medals = ['🥇', '🥈', '🥉']
    for i, (medal, island) in enumerate(zip(medals, sorted_islands)):
        g = island['best_genome']
        text += f"{medal} Island {island['island_id']}: {island['best_fitness']:.6f}\n"
        text += f"   μ={g[0]:.2f} ω={g[1]:.2f} d={g[2]:.4f}\n\n"
    
    ax.text(0.1, 0.5, text, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.suptitle('Island Model Evolution Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f"island_evolution_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {filename}")
    plt.close()

def visualize_ultra_long_results():
    """Generate visualization for ultra-long evolution"""
    print("🎨 Generating Ultra-Long Evolution visualization...")
    
    # Load the most recent ultra-long results
    import glob
    ultra_files = glob.glob("ultra_long_analysis_*.json")
    if not ultra_files:
        print("❌ No ultra-long evolution results found")
        return
    
    latest_file = max(ultra_files)
    print(f"   Loading: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    snapshots = data['snapshots']
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    generations = [s['generation'] for s in snapshots]
    
    # 1. Fitness evolution
    ax1 = fig.add_subplot(gs[0, :2])
    
    max_fitness = [s['fitness_stats']['max'] for s in snapshots]
    mean_fitness = [s['fitness_stats']['mean'] for s in snapshots]
    top10 = [s['fitness_stats']['top10%'] for s in snapshots]
    
    ax1.plot(generations, max_fitness, 'g-', linewidth=2, label='Best', marker='o', markersize=2)
    ax1.plot(generations, top10, 'b--', linewidth=1.5, label='Top 10%', alpha=0.7)
    ax1.plot(generations, mean_fitness, 'r:', linewidth=1.5, label='Mean', alpha=0.7)
    
    if data.get('convergence_detected'):
        ax1.axvline(data['convergence_generation'], color='purple', 
                   linestyle='--', linewidth=2, alpha=0.5, 
                   label=f'Convergence (gen {data["convergence_generation"]})')
    
    ax1.set_title('Fitness Evolution Over 500 Generations', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Champion info
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    genome = data['champion_genome']
    text = f"""
🏆 CHAMPION (Gen 500)

Fitness: {data['champion_fitness']:.6f}

Genome:
μ (mutation):    {genome[0]:.4f}
ω (oscillation): {genome[1]:.4f}
d (decoherence): {genome[2]:.6f}
φ (phase):       {genome[3]:.4f}

Population: {data['population_size']}
Generations: {data['generations']}

Convergence: {data['convergence_detected']}
"""
    if data['convergence_detected']:
        text += f"At Gen: {data['convergence_generation']}\n"
    
    ax2.text(0.1, 0.5, text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Diversity evolution
    ax3 = fig.add_subplot(gs[1, 0])
    
    div_mu = [s['genome_diversity']['mu'] for s in snapshots]
    div_omega = [s['genome_diversity']['omega'] for s in snapshots]
    div_d = [s['genome_diversity']['d'] for s in snapshots]
    
    ax3.plot(generations, div_mu, label='μ diversity', linewidth=2)
    ax3.plot(generations, div_omega, label='ω diversity', linewidth=2)
    ax3.plot(generations, div_d, label='d diversity', linewidth=2)
    
    ax3.set_title('Genome Diversity', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Std Dev')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4-6. Parameter evolution
    for idx, (param_name, key_elite, key_non) in enumerate([
        ('μ (Mutation Rate)', 'elite_mu_mean', 'non_elite_mu_mean'),
        ('ω (Oscillation)', 'elite_omega_mean', 'non_elite_omega_mean'),
        ('d (Decoherence)', 'elite_d_mean', 'non_elite_d_mean')
    ]):
        ax = fig.add_subplot(gs[1, idx+1] if idx < 2 else gs[2, 0])
        
        elite_vals = [s['elite_comparison'][key_elite] for s in snapshots]
        non_elite_vals = [s['elite_comparison'][key_non] for s in snapshots]
        
        ax.plot(generations, elite_vals, 'g-', linewidth=2, label='Elite (Top 10%)', marker='o', markersize=2)
        ax.plot(generations, non_elite_vals, 'r--', linewidth=1.5, label='Non-Elite', alpha=0.7)
        
        ax.set_title(f'{param_name} Evolution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel(param_name.split()[0])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 7. Innovation events
    ax7 = fig.add_subplot(gs[2, 1:])
    
    if data.get('innovation_events'):
        innovations = data['innovation_events']
        innovation_gens = [i['generation'] for i in innovations]
        innovation_improvements = [i['improvement'] for i in innovations]
        
        ax7.scatter(innovation_gens, innovation_improvements, s=100, c='gold',
                   marker='*', edgecolors='black', linewidths=1.5, zorder=3)
        
        for gen, imp in zip(innovation_gens, innovation_improvements):
            if imp > 10:  # Only label significant ones
                ax7.annotate(f'+{imp:.0f}%', xy=(gen, imp), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)
    
    ax7.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax7.set_title(f'💡 Innovation Events ({len(data.get("innovation_events", []))} detected)', 
                 fontsize=14, fontweight='bold')
    ax7.set_xlabel('Generation', fontsize=12)
    ax7.set_ylabel('Fitness Improvement (%)', fontsize=12)
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Ultra-Long Evolution: 1000 Agents × 500 Generations', 
                fontsize=18, fontweight='bold', y=0.995)
    
    filename = f"ultra_long_evolution_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {filename}")
    plt.close()

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎨 GENERATING VISUALIZATIONS FOR COMPLETED EXPERIMENTS")
    print("="*80)
    print()
    
    visualize_island_results()
    print()
    visualize_ultra_long_results()
    
    print()
    print("="*80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
