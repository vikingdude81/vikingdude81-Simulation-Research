"""
Ultra-Long Evolution: 1000 agents × 500 generations
See what emerges with extended evolution time
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

class UltraLongAnalyzer:
    """Analyzer for ultra-long evolution runs"""
    
    def __init__(self):
        self.snapshots = []
        self.convergence_detected = False
        self.convergence_generation = None
        
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
            'phi': float(np.std(phi_vals))
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
        }
        
        snapshot = {
            'generation': generation,
            'fitness_stats': fitness_stats,
            'genome_diversity': genome_diversity,
            'elite_comparison': elite_comparison,
            'mutation_rate': float(evolution.current_mutation_rate) if hasattr(evolution, 'current_mutation_rate') else 0.3
        }
        
        self.snapshots.append(snapshot)
        
        # Check for convergence (diversity < threshold for multiple snapshots)
        if len(self.snapshots) >= 10:
            recent_diversity = [s['genome_diversity']['mu'] for s in self.snapshots[-10:]]
            if max(recent_diversity) < 0.05 and not self.convergence_detected:
                self.convergence_detected = True
                self.convergence_generation = generation
        
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
            
            if improvement > 0.1:  # 10% improvement = innovation
                innovations.append({
                    'generation': self.snapshots[i]['generation'],
                    'improvement': improvement * 100,
                    'new_fitness': curr_best
                })
        
        return innovations

def run_ultra_long_evolution(population_size=1000, generations=500):
    """Run ultra-long evolution experiment"""
    
    print("\n" + "="*80)
    print("🔬 ULTRA-LONG EVOLUTION EXPERIMENT")
    print("="*80)
    print(f"Population: {population_size} agents")
    print(f"Generations: {generations}")
    print(f"Total evaluations: {population_size * generations:,}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    
    # Initialize evolution
    evolution = AdaptiveMutationEvolution(
        population_size=population_size,
        strategy='ml_adaptive',
        use_gpu=True
    )
    
    analyzer = UltraLongAnalyzer()
    
    # Evolution loop
    start_time = time.time()
    
    for gen in range(generations):
        gen_start = time.time()
        
        # Adapt mutation rate
        evolution.adapt_mutation_rate(gen)
        
        # Evaluate population
        evolution.evaluate_population(environment='standard')
        
        # Evolve one generation
        evolution.evolve_generation()
        
        gen_time = time.time() - gen_start
        elapsed = time.time() - start_time
        
        # Take snapshot every 10 generations
        if gen % 10 == 0:
            snapshot = analyzer.take_snapshot(evolution, gen)
        
        # Progress display (population is sorted, best is at index 0)
        best_fitness = evolution.population[0][0]
        mean_fitness = np.mean([agent[0] for agent in evolution.population])
        best_genome = evolution.population[0][1]
        
        if gen % 25 == 0 or gen == generations - 1:
            eta = (elapsed / (gen + 1)) * (generations - gen - 1)
            throughput = population_size / gen_time
            
            print(f"Gen {gen:4d}/{generations} | "
                  f"Best: {best_fitness:.6f} | Mean: {mean_fitness:.6f} | "
                  f"μ={best_genome[0]:.2f} ω={best_genome[1]:.2f} d={best_genome[2]:.4f} | "
                  f"{throughput:.0f} agents/s | ETA: {eta/60:.1f}m")
            
            if analyzer.convergence_detected and gen == analyzer.convergence_generation:
                print(f"  🎯 CONVERGENCE DETECTED at generation {gen}")
        
        # Train ML predictor periodically
        if gen > 0 and gen % 50 == 0 and evolution.strategy == 'ml_adaptive':
            if hasattr(evolution, 'predictor') and evolution.predictor:
                print(f"  🧠 ML Training at gen {gen}...")
    
    # Final snapshot
    final_snapshot = analyzer.take_snapshot(evolution, generations)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"\n⏱️  Total time: {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print(f"Average time per generation: {total_time/generations:.3f}s")
    print(f"Average throughput: {population_size * generations / total_time:.1f} agents/sec")
    print()
    
    best_agent = evolution.population[0]  # Sorted, best is first
    best_genome = best_agent[1]
    best_fitness = best_agent[0]
    print(f"🏆 CHAMPION GENOME:")
    print(f"   Mutation rate (μ):      {best_genome[0]:.4f}")
    print(f"   Oscillation freq (ω):   {best_genome[1]:.4f}")
    print(f"   Decoherence rate (d):   {best_genome[2]:.6f}")
    print(f"   Phase offset (φ):       {best_genome[3]:.4f}")
    print(f"   Fitness:                {best_fitness:.6f}")
    print()
    
    # Elite analysis
    elite_data = final_snapshot['elite_comparison']
    print("🌟 ELITE VS NON-ELITE (Final Generation):")
    print(f"   Elite μ:     {elite_data['elite_mu_mean']:.4f} vs {elite_data['non_elite_mu_mean']:.4f}")
    print(f"   Elite ω:     {elite_data['elite_omega_mean']:.4f} vs {elite_data['non_elite_omega_mean']:.4f}")
    print(f"   Elite d:     {elite_data['elite_d_mean']:.6f} vs {elite_data['non_elite_d_mean']:.6f}")
    print()
    
    # Convergence info
    if analyzer.convergence_detected:
        print(f"🎯 Convergence detected at generation {analyzer.convergence_generation}")
    else:
        print("⚠️  No convergence detected - population still evolving")
    print()
    
    # Innovation events
    innovations = analyzer.detect_innovation_events()
    if innovations:
        print(f"💡 INNOVATION EVENTS ({len(innovations)} detected):")
        for event in innovations[-5:]:  # Show last 5
            print(f"   Gen {event['generation']:4d}: +{event['improvement']:.1f}% → {event['new_fitness']:.6f}")
        print()
    
    return evolution, analyzer

def visualize_ultra_long(evolution, analyzer):
    """Create comprehensive visualization"""
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    generations = [s['generation'] for s in analyzer.snapshots]
    
    # 1. Fitness evolution (top row, span 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    
    max_fitness = [s['fitness_stats']['max'] for s in analyzer.snapshots]
    mean_fitness = [s['fitness_stats']['mean'] for s in analyzer.snapshots]
    top10_fitness = [s['fitness_stats']['top10%'] for s in analyzer.snapshots]
    
    ax1.plot(generations, max_fitness, 'g-', linewidth=2, label='Best', marker='o', markersize=3)
    ax1.plot(generations, top10_fitness, 'b--', linewidth=1.5, label='Top 10%', alpha=0.7)
    ax1.plot(generations, mean_fitness, 'r:', linewidth=1.5, label='Mean', alpha=0.7)
    
    if analyzer.convergence_detected:
        ax1.axvline(analyzer.convergence_generation, color='purple', linestyle='--', 
                   linewidth=2, alpha=0.5, label=f'Convergence (gen {analyzer.convergence_generation})')
    
    ax1.set_title('Fitness Evolution Over 500 Generations', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Diversity evolution
    ax2 = fig.add_subplot(gs[0, 2])
    
    div_mu = [s['genome_diversity']['mu'] for s in analyzer.snapshots]
    div_omega = [s['genome_diversity']['omega'] for s in analyzer.snapshots]
    div_d = [s['genome_diversity']['d'] for s in analyzer.snapshots]
    
    ax2.plot(generations, div_mu, label='μ diversity', linewidth=2)
    ax2.plot(generations, div_omega, label='ω diversity', linewidth=2)
    ax2.plot(generations, div_d, label='d diversity', linewidth=2)
    
    ax2.set_title('Genome Diversity (Convergence Tracking)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Generation', fontsize=10)
    ax2.set_ylabel('Std Dev', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Parameter evolution (mu)
    ax3 = fig.add_subplot(gs[1, 0])
    elite_mu = [s['elite_comparison']['elite_mu_mean'] for s in analyzer.snapshots]
    non_elite_mu = [s['elite_comparison']['non_elite_mu_mean'] for s in analyzer.snapshots]
    
    ax3.plot(generations, elite_mu, 'g-', linewidth=2, label='Elite', marker='o', markersize=2)
    ax3.plot(generations, non_elite_mu, 'r--', linewidth=1.5, label='Non-Elite', alpha=0.7)
    ax3.set_title('Mutation Rate (μ) Evolution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Generation', fontsize=10)
    ax3.set_ylabel('μ', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Parameter evolution (omega)
    ax4 = fig.add_subplot(gs[1, 1])
    elite_omega = [s['elite_comparison']['elite_omega_mean'] for s in analyzer.snapshots]
    non_elite_omega = [s['elite_comparison']['non_elite_omega_mean'] for s in analyzer.snapshots]
    
    ax4.plot(generations, elite_omega, 'g-', linewidth=2, label='Elite', marker='o', markersize=2)
    ax4.plot(generations, non_elite_omega, 'r--', linewidth=1.5, label='Non-Elite', alpha=0.7)
    ax4.set_title('Oscillation Freq (ω) Evolution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Generation', fontsize=10)
    ax4.set_ylabel('ω', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Parameter evolution (d)
    ax5 = fig.add_subplot(gs[1, 2])
    elite_d = [s['elite_comparison']['elite_d_mean'] for s in analyzer.snapshots]
    non_elite_d = [s['elite_comparison']['non_elite_d_mean'] for s in analyzer.snapshots]
    
    ax5.plot(generations, elite_d, 'g-', linewidth=2, label='Elite', marker='o', markersize=2)
    ax5.plot(generations, non_elite_d, 'r--', linewidth=1.5, label='Non-Elite', alpha=0.7)
    ax5.set_title('Decoherence Rate (d) Evolution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Generation', fontsize=10)
    ax5.set_ylabel('d', fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Innovation timeline
    ax6 = fig.add_subplot(gs[2, :])
    
    innovations = analyzer.detect_innovation_events()
    if innovations:
        innovation_gens = [i['generation'] for i in innovations]
        innovation_improvements = [i['improvement'] for i in innovations]
        
        ax6.scatter(innovation_gens, innovation_improvements, s=100, c='gold', 
                   marker='*', edgecolors='black', linewidths=1.5, zorder=3)
        
        for gen, imp in zip(innovation_gens, innovation_improvements):
            ax6.annotate(f'+{imp:.1f}%', xy=(gen, imp), xytext=(5, 5),
                        textcoords='offset points', fontsize=8, fontweight='bold')
    
    ax6.plot(generations, [0]*len(generations), 'k--', alpha=0.3)
    ax6.set_title(f'💡 Innovation Events ({len(innovations)} major fitness jumps)', 
                 fontsize=14, fontweight='bold')
    ax6.set_xlabel('Generation', fontsize=12)
    ax6.set_ylabel('Fitness Improvement (%)', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    # 7. Final population distribution (mu vs d)
    ax7 = fig.add_subplot(gs[3, 0])
    
    mu_vals = np.array([agent[1][0] for agent in evolution.population])
    d_vals = np.array([agent[1][2] for agent in evolution.population])
    fitness = np.array([agent[0] for agent in evolution.population])
    
    scatter = ax7.scatter(mu_vals, d_vals, c=fitness, cmap='viridis', 
                         s=30, alpha=0.6, edgecolors='black', linewidths=0.5)
    
    # Mark champion
    best_genome = evolution.population[0][1]  # [fitness, genome, id] -> get genome
    ax7.scatter([best_genome[0]], [best_genome[2]], s=500, c='gold', 
               marker='*', edgecolors='red', linewidths=2, zorder=10)
    
    ax7.set_title('Final Population: μ vs d', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Mutation Rate (μ)', fontsize=10)
    ax7.set_ylabel('Decoherence (d)', fontsize=10)
    plt.colorbar(scatter, ax=ax7, label='Fitness')
    ax7.grid(True, alpha=0.3)
    
    # 8. Final population distribution (omega vs fitness)
    ax8 = fig.add_subplot(gs[3, 1])
    
    omega_vals = np.array([agent[1][1] for agent in evolution.population])
    
    best_genome = evolution.population[0][1]
    best_fitness = evolution.population[0][0]
    
    ax8.scatter(omega_vals, fitness, s=30, alpha=0.6, c=fitness, cmap='plasma',
               edgecolors='black', linewidths=0.5)
    ax8.scatter([best_genome[1]], [best_fitness], s=500, c='gold',
               marker='★', edgecolors='red', linewidths=2, zorder=10)
    
    ax8.set_title('Final Population: ω vs Fitness', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Oscillation Freq (ω)', fontsize=10)
    ax8.set_ylabel('Fitness', fontsize=10)
    ax8.grid(True, alpha=0.3)
    
    # 9. Fitness distribution histogram
    ax9 = fig.add_subplot(gs[3, 2])
    
    mean_fitness_val = float(np.mean(fitness))
    
    ax9.hist(fitness, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax9.axvline(best_fitness, color='gold', linestyle='--', 
               linewidth=2, label=f'Champion: {best_fitness:.6f}')
    ax9.axvline(mean_fitness_val, color='red', linestyle=':', 
               linewidth=2, label=f'Mean: {mean_fitness_val:.6f}')
    
    ax9.set_title('Final Fitness Distribution', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Fitness', fontsize=10)
    ax9.set_ylabel('Count', fontsize=10)
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Ultra-Long Evolution: 1000 Agents × 500 Generations', 
                fontsize=20, fontweight='bold', y=0.995)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ultra_long_evolution_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved: {filename}")
    
    return filename

def main():
    # Run evolution
    evolution, analyzer = run_ultra_long_evolution(population_size=1000, generations=500)
    
    # Visualize
    viz_file = visualize_ultra_long(evolution, analyzer)
    
    # Export data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = f"ultra_long_analysis_{timestamp}.json"
    
    export_data = {
        'timestamp': timestamp,
        'population_size': 1000,
        'generations': 500,
        'convergence_detected': analyzer.convergence_detected,
        'convergence_generation': analyzer.convergence_generation,
        'champion_genome': [float(x) for x in evolution.population[0][1]],
        'champion_fitness': float(evolution.population[0][0]),
        'innovation_events': analyzer.detect_innovation_events(),
        'snapshots': analyzer.snapshots
    }
    
    with open(data_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"📁 Data saved: {data_file}")
    print("\n✅ ULTRA-LONG EVOLUTION COMPLETE!")

if __name__ == "__main__":
    main()
