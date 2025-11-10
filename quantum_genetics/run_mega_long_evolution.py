"""
Mega-Long Evolution: 1000 agents × 1000 generations
Ultimate test of quantum genetic evolution with ML adaptive mutation

Expected runtime: ~45-50 minutes
Total evaluations: 1,000,000
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adaptive_mutation_gpu_ml import AdaptiveMutationEvolution
from quantum_genetic_agents import QuantumAgent
import json
from datetime import datetime
import time

sns.set_style("darkgrid")

class MegaLongAnalyzer:
    """Analyzer for mega-long evolution runs (1000+ generations)"""
    
    def __init__(self):
        self.snapshots = []
        self.convergence_detected = False
        self.convergence_generation = None
        self.innovation_events = []
        self.ml_training_history = []
        
    def take_snapshot(self, evolution, generation):
        """Take detailed population snapshot"""
        # Extract fitness and genomes from population [[fitness, genome, id], ...]
        fitness = np.array([agent[0] for agent in evolution.population])
        genomes = np.array([agent[1] for agent in evolution.population])
        
        # Fitness statistics
        fitness_stats = {
            'mean': float(np.mean(fitness)),
            'std': float(np.std(fitness)),
            'min': float(np.min(fitness)),
            'max': float(np.max(fitness)),
            'median': float(np.median(fitness)),
            'q25': float(np.percentile(fitness, 25)),
            'q75': float(np.percentile(fitness, 75)),
            'top1%': float(np.percentile(fitness, 99)),
            'top10%': float(np.percentile(fitness, 90)),
            'top5%': float(np.percentile(fitness, 95)),
            'bottom10%': float(np.percentile(fitness, 10))
        }
        
        # Genome diversity
        mu_vals = genomes[:, 0]
        omega_vals = genomes[:, 1]
        d_vals = genomes[:, 2]
        phi_vals = genomes[:, 3]
        
        genome_diversity = {
            'mu': float(np.std(mu_vals)),
            'omega': float(np.std(omega_vals)),
            'd': float(np.std(d_vals)),
            'phi': float(np.std(phi_vals)),
            'mu_mean': float(np.mean(mu_vals)),
            'omega_mean': float(np.mean(omega_vals)),
            'd_mean': float(np.mean(d_vals)),
            'phi_mean': float(np.mean(phi_vals))
        }
        
        # Elite analysis (top 10%)
        elite_threshold = np.percentile(fitness, 90)
        elite_mask = fitness >= elite_threshold
        
        elite_comparison = {
            'elite_mu_mean': float(np.mean(mu_vals[elite_mask])),
            'non_elite_mu_mean': float(np.mean(mu_vals[~elite_mask])),
            'elite_omega_mean': float(np.mean(omega_vals[elite_mask])),
            'non_elite_omega_mean': float(np.mean(omega_vals[~elite_mask])),
            'elite_d_mean': float(np.mean(d_vals[elite_mask])),
            'non_elite_d_mean': float(np.mean(d_vals[~elite_mask])),
            'elite_phi_mean': float(np.mean(phi_vals[elite_mask])),
            'non_elite_phi_mean': float(np.mean(phi_vals[~elite_mask]))
        }
        
        snapshot = {
            'generation': generation,
            'fitness_stats': fitness_stats,
            'genome_diversity': genome_diversity,
            'elite_comparison': elite_comparison,
            'mutation_rate': float(evolution.current_mutation_rate) if hasattr(evolution, 'current_mutation_rate') else 0.3
        }
        
        self.snapshots.append(snapshot)
        
        # Check for convergence (diversity < threshold for 20 consecutive snapshots)
        if len(self.snapshots) >= 20:
            recent_diversity = [s['genome_diversity']['mu'] for s in self.snapshots[-20:]]
            if max(recent_diversity) < 0.03 and not self.convergence_detected:
                self.convergence_detected = True
                self.convergence_generation = generation
                print(f"\n🎯 CONVERGENCE DETECTED at generation {generation}!")
                print(f"   Population diversity stabilized below threshold")
        
        return snapshot
    
    def detect_innovation_events(self):
        """Detect major fitness jumps (innovation events)"""
        if len(self.snapshots) < 2:
            return []
        
        innovations = []
        for i in range(1, len(self.snapshots)):
            prev_best = self.snapshots[i-1]['fitness_stats']['max']
            curr_best = self.snapshots[i]['fitness_stats']['max']
            
            improvement = (curr_best - prev_best) / prev_best if prev_best > 0 else 0
            
            # Significant jump (>10% improvement)
            if improvement > 0.10:
                innovations.append({
                    'generation': self.snapshots[i]['generation'],
                    'prev_fitness': float(prev_best),
                    'new_fitness': float(curr_best),
                    'improvement_pct': float(improvement * 100)
                })
        
        self.innovation_events = innovations
        return innovations

def run_mega_long_evolution(population_size=1000, generations=1000):
    """
    Run mega-long evolution experiment
    
    Args:
        population_size: Number of agents (default 1000)
        generations: Number of generations (default 1000)
    """
    print("🔥 PyTorch available! Using device: cuda")
    print()
    print("=" * 80)
    print("🔬 MEGA-LONG EVOLUTION EXPERIMENT")
    print("=" * 80)
    print(f"Population: {population_size} agents")
    print(f"Generations: {generations}")
    print(f"Total evaluations: {population_size * generations:,}")
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Initialize evolution with ML adaptive mutation
    evolution = AdaptiveMutationEvolution(
        population_size=population_size,
        strategy='ml_adaptive'
    )
    
    # Initialize analyzer
    analyzer = MegaLongAnalyzer()
    
    # Track performance
    generation_times = []
    best_fitness_history = []
    mean_fitness_history = []
    
    # Main evolution loop
    for gen in range(generations):
        gen_start = time.time()
        
        # Adapt mutation rate
        evolution.adapt_mutation_rate(gen)
        
        # Evaluate population
        evolution.evaluate_population(environment='standard')
        
        # Take snapshot every 10 generations
        if gen % 10 == 0:
            snapshot = analyzer.take_snapshot(evolution, gen)
            
            # Get best agent (population is sorted by fitness descending)
            best_agent = evolution.population[0]
            best_fitness = best_agent[0]
            best_genome = best_agent[1]
            
            mean_fitness = np.mean([agent[0] for agent in evolution.population])
            
            best_fitness_history.append(best_fitness)
            mean_fitness_history.append(mean_fitness)
            
            # Calculate performance metrics
            gen_time = time.time() - gen_start
            generation_times.append(gen_time)
            
            avg_time = np.mean(generation_times[-10:])
            agents_per_sec = population_size / avg_time if avg_time > 0 else 0
            remaining_gens = generations - gen
            eta_seconds = remaining_gens * avg_time
            eta_minutes = eta_seconds / 60
            
            # Print progress
            print(f"Gen {gen:4d}/{generations} | Best: {best_fitness:.6f} | Mean: {mean_fitness:.6f} | "
                  f"μ={best_genome[0]:.2f} ω={best_genome[1]:.2f} d={best_genome[2]:.4f} | "
                  f"{agents_per_sec:.0f} agents/s | ETA: {eta_minutes:.1f}m")
        
        # Evolve to next generation
        evolution.evolve_generation()
        
        # Train ML predictor periodically
        if gen > 0 and gen % 50 == 0:
            print(f"  🧠 ML Training at gen {gen}...")
            analyzer.ml_training_history.append(gen)
    
    # Final snapshot
    final_snapshot = analyzer.take_snapshot(evolution, generations)
    
    # Calculate total runtime
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print()
    print("=" * 80)
    print("✅ MEGA-LONG EVOLUTION COMPLETE!")
    print("=" * 80)
    print(f"Total runtime: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print(f"Total evaluations: {population_size * generations:,}")
    print(f"Average throughput: {(population_size * generations) / total_time:.1f} agents/second")
    print()
    
    # Detect innovation events
    innovations = analyzer.detect_innovation_events()
    print(f"🚀 Innovation Events Detected: {len(innovations)}")
    if innovations:
        print("   Top 5 major breakthroughs:")
        sorted_innovations = sorted(innovations, key=lambda x: x['improvement_pct'], reverse=True)
        for i, event in enumerate(sorted_innovations[:5], 1):
            print(f"   {i}. Gen {event['generation']}: {event['prev_fitness']:.6f} → "
                  f"{event['new_fitness']:.6f} (+{event['improvement_pct']:.1f}%)")
    print()
    
    # Final champion
    champion = evolution.population[0]
    champion_fitness = champion[0]
    champion_genome = champion[1]
    
    print("🏆 FINAL CHAMPION:")
    print(f"   Fitness: {champion_fitness:.6f}")
    print(f"   Genome: μ={champion_genome[0]:.4f}, ω={champion_genome[1]:.4f}, "
          f"d={champion_genome[2]:.6f}, φ={champion_genome[3]:.4f}")
    print()
    
    # Convergence status
    if analyzer.convergence_detected:
        print(f"🎯 Convergence achieved at generation {analyzer.convergence_generation}")
    else:
        print("⚠️  No convergence detected - population still evolving!")
    print()
    
    return evolution, analyzer, best_fitness_history, mean_fitness_history

def visualize_mega_long(evolution, analyzer, best_fitness_history, mean_fitness_history):
    """Create comprehensive visualization for mega-long evolution"""
    
    print("📊 Generating comprehensive visualizations...")
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Fitness evolution over all 1000 generations
    ax1 = fig.add_subplot(gs[0, :2])
    
    generations = [s['generation'] for s in analyzer.snapshots]
    max_fitness = [s['fitness_stats']['max'] for s in analyzer.snapshots]
    mean_fitness = [s['fitness_stats']['mean'] for s in analyzer.snapshots]
    top10_fitness = [s['fitness_stats']['top10%'] for s in analyzer.snapshots]
    
    ax1.plot(generations, max_fitness, 'g-', linewidth=2, label='Best', alpha=0.8)
    ax1.plot(generations, top10_fitness, 'b--', linewidth=1.5, label='Top 10%', alpha=0.7)
    ax1.plot(generations, mean_fitness, 'orange', linewidth=1.5, label='Mean', alpha=0.6)
    ax1.fill_between(generations, mean_fitness, alpha=0.2, color='orange')
    
    # Mark innovation events
    innovations = analyzer.innovation_events
    if innovations:
        innovation_gens = [e['generation'] for e in innovations]
        innovation_fitness = [e['new_fitness'] for e in innovations]
        ax1.scatter(innovation_gens, innovation_fitness, s=150, c='red', 
                   marker='*', zorder=10, label='Innovation Events', edgecolors='darkred', linewidths=2)
    
    # Mark convergence if detected
    if analyzer.convergence_detected:
        ax1.axvline(analyzer.convergence_generation, color='purple', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Convergence (Gen {analyzer.convergence_generation})')
    
    ax1.set_title('Fitness Evolution: 1000 Generations', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1000)
    
    # 2. Champion genome info
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    champion = evolution.population[0]
    champion_fitness = champion[0]
    champion_genome = champion[1]
    
    info_text = f"""
    🏆 FINAL CHAMPION (Gen 1000)
    
    Fitness: {champion_fitness:.6f}
    
    Genome Parameters:
    • μ (mutation):    {champion_genome[0]:.4f}
    • ω (oscillation): {champion_genome[1]:.4f}
    • d (decoherence): {champion_genome[2]:.6f}
    • φ (phase):       {champion_genome[3]:.4f}
    
    Innovation Events: {len(innovations)}
    ML Training Cycles: {len(analyzer.ml_training_history)}
    
    Convergence: {'Yes (Gen ' + str(analyzer.convergence_generation) + ')' if analyzer.convergence_detected else 'No'}
    """
    
    ax2.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 3. Diversity evolution (all parameters)
    ax3 = fig.add_subplot(gs[1, 0])
    
    mu_diversity = [s['genome_diversity']['mu'] for s in analyzer.snapshots]
    omega_diversity = [s['genome_diversity']['omega'] for s in analyzer.snapshots]
    d_diversity = [s['genome_diversity']['d'] for s in analyzer.snapshots]
    
    ax3.plot(generations, mu_diversity, 'r-', linewidth=2, label='μ diversity', alpha=0.8)
    ax3.plot(generations, omega_diversity, 'b-', linewidth=2, label='ω diversity', alpha=0.8)
    ax3.plot(generations, d_diversity, 'g-', linewidth=2, label='d diversity', alpha=0.8)
    
    if analyzer.convergence_detected:
        ax3.axvline(analyzer.convergence_generation, color='purple', linestyle='--', 
                   linewidth=2, alpha=0.5)
    
    ax3.set_title('Population Diversity Over Time', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Generation', fontsize=10)
    ax3.set_ylabel('Standard Deviation', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Parameter evolution (means)
    ax4 = fig.add_subplot(gs[1, 1])
    
    mu_mean = [s['genome_diversity']['mu_mean'] for s in analyzer.snapshots]
    omega_mean = [s['genome_diversity']['omega_mean'] for s in analyzer.snapshots]
    d_mean = [s['genome_diversity']['d_mean'] for s in analyzer.snapshots]
    
    ax4_mu = ax4.twinx()
    ax4_omega = ax4.twinx()
    ax4_omega.spines['right'].set_position(('outward', 60))
    
    p1 = ax4.plot(generations, mu_mean, 'r-', linewidth=2, label='μ (mutation)', alpha=0.8)
    p2 = ax4_mu.plot(generations, omega_mean, 'b-', linewidth=2, label='ω (oscillation)', alpha=0.8)
    p3 = ax4_omega.plot(generations, d_mean, 'g-', linewidth=2, label='d (decoherence)', alpha=0.8)
    
    ax4.set_xlabel('Generation', fontsize=10)
    ax4.set_ylabel('μ (mutation rate)', fontsize=10, color='r')
    ax4_mu.set_ylabel('ω (oscillation freq)', fontsize=10, color='b')
    ax4_omega.set_ylabel('d (decoherence)', fontsize=10, color='g')
    
    ax4.tick_params(axis='y', labelcolor='r')
    ax4_mu.tick_params(axis='y', labelcolor='b')
    ax4_omega.tick_params(axis='y', labelcolor='g')
    
    ax4.set_title('Parameter Evolution (Population Means)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Innovation events timeline
    ax5 = fig.add_subplot(gs[1, 2])
    
    if innovations:
        innovation_gens = [e['generation'] for e in innovations]
        innovation_improvements = [e['improvement_pct'] for e in innovations]
        
        colors = plt.cm.Reds(np.linspace(0.4, 1.0, len(innovations)))
        ax5.bar(range(len(innovations)), innovation_improvements, color=colors, alpha=0.8, edgecolor='black')
        ax5.set_xlabel('Innovation Event #', fontsize=10)
        ax5.set_ylabel('Fitness Improvement (%)', fontsize=10)
        ax5.set_title(f'Innovation Events (n={len(innovations)})', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add generation labels for top events
        top_innovations = sorted(range(len(innovations)), 
                                key=lambda i: innovation_improvements[i], reverse=True)[:5]
        for idx in top_innovations:
            ax5.text(idx, innovation_improvements[idx], f'G{innovation_gens[idx]}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No major innovation events detected', 
                ha='center', va='center', fontsize=12, transform=ax5.transAxes)
        ax5.set_title('Innovation Events', fontsize=12, fontweight='bold')
    
    # 6. Elite vs Non-Elite (μ parameter)
    ax6 = fig.add_subplot(gs[2, 0])
    
    elite_mu = [s['elite_comparison']['elite_mu_mean'] for s in analyzer.snapshots]
    non_elite_mu = [s['elite_comparison']['non_elite_mu_mean'] for s in analyzer.snapshots]
    
    ax6.plot(generations, elite_mu, 'g-', linewidth=2, label='Elite (Top 10%)', alpha=0.8)
    ax6.plot(generations, non_elite_mu, 'orange', linewidth=2, label='Non-Elite', alpha=0.8)
    ax6.fill_between(generations, elite_mu, non_elite_mu, alpha=0.2, color='green')
    
    ax6.set_title('Mutation Rate: Elite vs Non-Elite', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Generation', fontsize=10)
    ax6.set_ylabel('μ (Mutation Rate)', fontsize=10)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Elite vs Non-Elite (d parameter - most critical!)
    ax7 = fig.add_subplot(gs[2, 1])
    
    elite_d = [s['elite_comparison']['elite_d_mean'] for s in analyzer.snapshots]
    non_elite_d = [s['elite_comparison']['non_elite_d_mean'] for s in analyzer.snapshots]
    
    ax7.plot(generations, elite_d, 'g-', linewidth=2, label='Elite (Top 10%)', alpha=0.8)
    ax7.plot(generations, non_elite_d, 'orange', linewidth=2, label='Non-Elite', alpha=0.8)
    ax7.fill_between(generations, elite_d, non_elite_d, alpha=0.2, color='green')
    
    # Highlight d=0.005 optimal line
    ax7.axhline(0.005, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal (d=0.005)')
    
    ax7.set_title('Decoherence Rate: Elite vs Non-Elite', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Generation', fontsize=10)
    ax7.set_ylabel('d (Decoherence Rate)', fontsize=10)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # 8. Final population scatter (μ vs d)
    ax8 = fig.add_subplot(gs[2, 2])
    
    genomes = np.array([agent[1] for agent in evolution.population])
    fitness = np.array([agent[0] for agent in evolution.population])
    
    mu_vals = genomes[:, 0]
    d_vals = genomes[:, 2]
    
    scatter = ax8.scatter(mu_vals, d_vals, c=fitness, cmap='viridis', 
                         s=30, alpha=0.6, edgecolors='black', linewidths=0.5)
    
    # Mark champion
    best_genome = evolution.population[0][1]
    ax8.scatter([best_genome[0]], [best_genome[2]], s=500, c='gold', 
               marker='★', edgecolors='red', linewidths=2, zorder=10)
    
    # Highlight d=0.005 optimal line
    ax8.axhline(0.005, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    ax8.set_title('Final Population: μ vs d', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Mutation Rate (μ)', fontsize=10)
    ax8.set_ylabel('Decoherence (d)', fontsize=10)
    plt.colorbar(scatter, ax=ax8, label='Fitness')
    ax8.grid(True, alpha=0.3)
    
    # 9. Fitness distribution over time (heatmap)
    ax9 = fig.add_subplot(gs[3, :])
    
    # Sample 50 evenly-spaced generations for heatmap
    sample_gens = np.linspace(0, len(analyzer.snapshots)-1, 50, dtype=int)
    fitness_matrix = []
    
    for idx in sample_gens:
        snapshot = analyzer.snapshots[idx]
        fitness_matrix.append([
            snapshot['fitness_stats']['max'],
            snapshot['fitness_stats']['top5%'],
            snapshot['fitness_stats']['top10%'],
            snapshot['fitness_stats']['median'],
            snapshot['fitness_stats']['mean'],
            snapshot['fitness_stats']['q25'],
            snapshot['fitness_stats']['bottom10%'],
            snapshot['fitness_stats']['min']
        ])
    
    fitness_matrix = np.array(fitness_matrix).T
    
    im = ax9.imshow(fitness_matrix, aspect='auto', cmap='YlGnBu', interpolation='bilinear')
    ax9.set_yticks(range(8))
    ax9.set_yticklabels(['Max', 'Top 5%', 'Top 10%', 'Median', 'Mean', 'Q25', 'Bottom 10%', 'Min'])
    ax9.set_xlabel('Generation (sampled)', fontsize=10)
    ax9.set_title('Fitness Distribution Evolution Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax9, label='Fitness')
    
    # Add timestamp
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mega_long_evolution_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: {filename}")
    
    return fig

def main():
    """Main execution"""
    
    # Run mega-long evolution (1000 gens)
    evolution, analyzer, best_history, mean_history = run_mega_long_evolution(
        population_size=1000,
        generations=1000
    )
    
    # Generate visualizations
    fig = visualize_mega_long(evolution, analyzer, best_history, mean_history)
    
    # Export detailed JSON data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    export_data = {
        'experiment': 'mega_long_evolution',
        'population_size': 1000,
        'generations': 1000,
        'total_snapshots': len(analyzer.snapshots),
        'snapshots': analyzer.snapshots,
        'innovation_events': analyzer.innovation_events,
        'ml_training_history': analyzer.ml_training_history,
        'convergence_detected': analyzer.convergence_detected,
        'convergence_generation': analyzer.convergence_generation,
        'champion_genome': {
            'mu': float(evolution.population[0][1][0]),
            'omega': float(evolution.population[0][1][1]),
            'd': float(evolution.population[0][1][2]),
            'phi': float(evolution.population[0][1][3]),
            'fitness': float(evolution.population[0][0])
        }
    }
    
    json_filename = f"mega_long_analysis_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✅ Data exported: {json_filename}")
    print()
    print("=" * 80)
    print("🎉 MEGA-LONG EVOLUTION EXPERIMENT COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
