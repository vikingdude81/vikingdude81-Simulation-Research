"""
Island Model Evolution
Split 1000 agents into 10 islands, evolve independently with periodic migration
Test if specialized sub-populations emerge
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

class Island:
    """Single island in the archipelago"""
    
    def __init__(self, island_id, population_size, use_gpu=True):
        self.island_id = island_id
        self.population_size = population_size
        
        self.evolution = AdaptiveMutationEvolution(
            population_size=population_size,
            strategy='ml_adaptive',
            use_gpu=use_gpu
        )
        
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'best_genome': []
        }
    
    def evolve(self):
        """Evolve one generation"""
        # Adapt mutation, evaluate, then evolve
        self.evolution.adapt_mutation_rate(len(self.history['best_fitness']))
        self.evolution.evaluate_population(environment='standard')
        self.evolution.evolve_generation()
        
        # Population is [fitness, genome, id], sorted by fitness
        best_agent = self.evolution.population[0]
        best_genome = best_agent[1].copy()
        
        self.history['best_fitness'].append(best_agent[0])
        self.history['mean_fitness'].append(np.mean([a[0] for a in self.evolution.population]))
        self.history['best_genome'].append(best_genome)
    
    def get_best_agents(self, n=5):
        """Get top N agents for migration (population already sorted, best first)"""
        return [agent[1].copy() for agent in self.evolution.population[:n]]
    
    def receive_migrants(self, migrants):
        """Receive migrants and replace worst agents"""
        n_migrants = len(migrants)
        # Worst agents are at the end (population sorted)
        for i in range(n_migrants):
            # Replace worst agents with migrants
            idx = -(i + 1)
            self.evolution.population[idx][1] = migrants[i].copy()
            self.evolution.population[idx][0] = 0.0  # Will be re-evaluated
    
    def get_stats(self):
        """Get current island statistics"""
        best_agent = self.evolution.population[0]
        return {
            'best_fitness': best_agent[0],
            'mean_fitness': np.mean([a[0] for a in self.evolution.population]),
            'best_genome': best_agent[1].copy()
        }

class IslandArchipelago:
    """Archipelago of islands with migration"""
    
    def __init__(self, num_islands=10, agents_per_island=100, migration_interval=25, migrants_per_exchange=5):
        self.num_islands = num_islands
        self.agents_per_island = agents_per_island
        self.migration_interval = migration_interval
        self.migrants_per_exchange = migrants_per_exchange
        
        print(f"🏝️  Creating archipelago with {num_islands} islands")
        print(f"   Agents per island: {agents_per_island}")
        print(f"   Total agents: {num_islands * agents_per_island}")
        print(f"   Migration every {migration_interval} generations")
        print(f"   Migrants per exchange: {migrants_per_exchange}")
        
        # Create islands
        self.islands = [
            Island(i, agents_per_island, use_gpu=True)
            for i in range(num_islands)
        ]
        
        # Track migration events
        self.migration_events = []
        
    def evolve_generation(self, generation):
        """Evolve all islands for one generation"""
        
        # Evolve each island
        for island in self.islands:
            island.evolve()
        
        # Migration
        if generation > 0 and generation % self.migration_interval == 0:
            self.perform_migration(generation)
    
    def perform_migration(self, generation):
        """Perform ring migration between islands"""
        print(f"  🦜 Migration event at generation {generation}")
        
        # Ring topology: each island sends to next island
        migrants_list = []
        for island in self.islands:
            migrants = island.get_best_agents(self.migrants_per_exchange)
            migrants_list.append(migrants)
        
        # Distribute migrants (ring: island i receives from island i-1)
        for i, island in enumerate(self.islands):
            source_island = (i - 1) % self.num_islands
            island.receive_migrants(migrants_list[source_island])
        
        # Track event
        event = {
            'generation': generation,
            'migrants_per_island': self.migrants_per_exchange,
            'island_best_fitness': [island.evolution.population[0][0] for island in self.islands]
        }
        self.migration_events.append(event)
    
    def get_global_champion(self):
        """Find best agent across all islands"""
        best_island = None
        best_fitness = -np.inf
        best_genome = None
        
        for island in self.islands:
            island_best = island.evolution.population[0][0]
            if island_best > best_fitness:
                best_fitness = island_best
                best_genome = island.evolution.population[0][1].copy()
                best_island = island.island_id
        
        return {
            'island_id': best_island,
            'fitness': best_fitness,
            'genome': best_genome
        }
    
    def get_island_diversity(self):
        """Measure diversity between islands"""
        # Get best genome from each island
        island_genomes = []
        for island in self.islands:
            genome = island.evolution.population[0][1]  # Best agent's genome
            island_genomes.append(genome)
        
        island_genomes = np.array(island_genomes)
        
        # Calculate pairwise distances
        diversity = {
            'mu_std': float(np.std(island_genomes[:, 0])),
            'omega_std': float(np.std(island_genomes[:, 1])),
            'd_std': float(np.std(island_genomes[:, 2])),
            'phi_std': float(np.std(island_genomes[:, 3]))
        }
        
        return diversity

def run_island_evolution(num_islands=10, agents_per_island=100, generations=300):
    """Run island model evolution"""
    
    print("\n" + "="*80)
    print("🏝️  ISLAND MODEL EVOLUTION")
    print("="*80)
    print(f"Islands: {num_islands}")
    print(f"Agents per island: {agents_per_island}")
    print(f"Total agents: {num_islands * agents_per_island}")
    print(f"Generations: {generations}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    
    # Create archipelago
    archipelago = IslandArchipelago(
        num_islands=num_islands,
        agents_per_island=agents_per_island,
        migration_interval=25,
        migrants_per_exchange=5
    )
    
    print()
    
    # Track metrics
    global_best_history = []
    island_diversity_history = []
    
    start_time = time.time()
    
    # Evolution loop
    for gen in range(generations):
        gen_start = time.time()
        
        # Evolve all islands
        archipelago.evolve_generation(gen)
        
        # Track metrics
        champion = archipelago.get_global_champion()
        diversity = archipelago.get_island_diversity()
        
        global_best_history.append(champion['fitness'])
        island_diversity_history.append(diversity)
        
        gen_time = time.time() - gen_start
        elapsed = time.time() - start_time
        
        # Progress display
        if gen % 25 == 0 or gen == generations - 1:
            eta = (elapsed / (gen + 1)) * (generations - gen - 1)
            throughput = (num_islands * agents_per_island) / gen_time
            
            # Island statistics
            island_fitness = [island.evolution.population[0][0] for island in archipelago.islands]
            best_island_fitness = max(island_fitness)
            worst_island_fitness = min(island_fitness)
            mean_island_fitness = np.mean(island_fitness)
            
            print(f"Gen {gen:3d}/{generations} | Global Best: {champion['fitness']:.6f} (Island {champion['island_id']}) | "
                  f"Island Range: [{worst_island_fitness:.6f}, {best_island_fitness:.6f}] | "
                  f"Mean: {mean_island_fitness:.6f}")
            
            genome = champion['genome']
            print(f"         Champion: μ={genome[0]:.2f} ω={genome[1]:.2f} d={genome[2]:.4f} φ={genome[3]:.2f} | "
                  f"Diversity: μ_std={diversity['mu_std']:.3f} | "
                  f"{throughput:.0f} agents/s | ETA: {eta/60:.1f}m")
    
    total_time = time.time() - start_time
    
    # Final results
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per generation: {total_time/generations:.3f}s")
    print()
    
    final_champion = archipelago.get_global_champion()
    print(f"🏆 GLOBAL CHAMPION (from Island {final_champion['island_id']}):")
    genome = final_champion['genome']
    print(f"   Mutation rate (μ):      {genome[0]:.4f}")
    print(f"   Oscillation freq (ω):   {genome[1]:.4f}")
    print(f"   Decoherence rate (d):   {genome[2]:.6f}")
    print(f"   Phase offset (φ):       {genome[3]:.4f}")
    print(f"   Fitness:                {final_champion['fitness']:.6f}")
    print()
    
    # Island comparison
    print("🏝️  ISLAND COMPARISON (Final Generation):")
    print("-" * 80)
    island_stats = []
    for island in archipelago.islands:
        stats = island.get_stats()
        island_stats.append(stats)
        genome = stats['best_genome']
        print(f"Island {island.island_id:2d}: Best={stats['best_fitness']:.6f}, Mean={stats['mean_fitness']:.6f} | "
              f"μ={genome[0]:.2f} ω={genome[1]:.2f} d={genome[2]:.4f} φ={genome[3]:.2f}")
    
    # Sort islands by performance
    sorted_islands = sorted(enumerate(island_stats), key=lambda x: x[1]['best_fitness'], reverse=True)
    
    print()
    print("🥇 TOP 3 ISLANDS:")
    for rank, (island_id, stats) in enumerate(sorted_islands[:3], 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        genome = stats['best_genome']
        print(f"{emoji} Island {island_id}: {stats['best_fitness']:.6f} | "
              f"μ={genome[0]:.2f} ω={genome[1]:.2f} d={genome[2]:.4f} φ={genome[3]:.2f}")
    
    print()
    print(f"📊 Migration events: {len(archipelago.migration_events)}")
    
    final_diversity = archipelago.get_island_diversity()
    print(f"🌈 Final island diversity: μ_std={final_diversity['mu_std']:.3f}, "
          f"ω_std={final_diversity['omega_std']:.3f}, d_std={final_diversity['d_std']:.4f}")
    
    return archipelago, global_best_history, island_diversity_history

def visualize_island_evolution(archipelago, global_best_history, island_diversity_history):
    """Create comprehensive visualization"""
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    num_generations = len(global_best_history)
    generations = range(num_generations)
    
    # Color palette for islands
    colors = plt.cm.tab10(np.linspace(0, 1, len(archipelago.islands)))
    
    # 1. Global best fitness evolution
    ax1 = fig.add_subplot(gs[0, :2])
    
    ax1.plot(generations, global_best_history, 'g-', linewidth=3, label='Global Best', marker='o', markersize=2)
    
    # Mark migration events
    for event in archipelago.migration_events:
        ax1.axvline(event['generation'], color='purple', linestyle='--', alpha=0.3, linewidth=1)
    
    ax1.set_title('Global Best Fitness Evolution (with Migration Events)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Island diversity evolution
    ax2 = fig.add_subplot(gs[0, 2])
    
    mu_diversity = [d['mu_std'] for d in island_diversity_history]
    omega_diversity = [d['omega_std'] for d in island_diversity_history]
    d_diversity = [d['d_std'] for d in island_diversity_history]
    
    ax2.plot(generations, mu_diversity, label='μ diversity', linewidth=2)
    ax2.plot(generations, omega_diversity, label='ω diversity', linewidth=2)
    ax2.plot(generations, d_diversity, label='d diversity', linewidth=2)
    
    ax2.set_title('Island Diversity (Between-Island Variation)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Generation', fontsize=10)
    ax2.set_ylabel('Std Dev', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3-5. Individual island fitness trajectories (first 3 panels)
    for idx in range(min(3, len(archipelago.islands))):
        row = 1
        col = idx
        ax = fig.add_subplot(gs[row, col])
        
        for i, island in enumerate(archipelago.islands):
            ax.plot(island.history['best_fitness'], 
                   color=colors[i], linewidth=1.5, alpha=0.7, label=f'Island {i}')
        
        ax.set_title(f'All Island Fitness Trajectories', fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation', fontsize=10)
        ax.set_ylabel('Best Fitness', fontsize=10)
        if idx == 2:
            ax.legend(fontsize=6, ncol=2, loc='lower right')
        ax.grid(True, alpha=0.3)
        break  # Just one plot for all islands
    
    # 4. Island comparison (middle row, span remaining)
    ax4 = fig.add_subplot(gs[1, 1:])
    
    island_ids = range(len(archipelago.islands))
    final_best = [island.evolution.population[0][0] for island in archipelago.islands]
    final_mean = [np.mean([a[0] for a in island.evolution.population]) for island in archipelago.islands]
    
    x = np.arange(len(island_ids))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, final_best, width, label='Best Fitness', alpha=0.8, color='steelblue')
    bars2 = ax4.bar(x + width/2, final_mean, width, label='Mean Fitness', alpha=0.8, color='coral')
    
    ax4.set_title('Final Island Performance Comparison', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Island ID', fontsize=10)
    ax4.set_ylabel('Fitness', fontsize=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'#{i}' for i in island_ids])
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 6. Parameter comparison across islands (mu)
    ax6 = fig.add_subplot(gs[2, 0])
    
    island_mu = [island.get_stats()['best_genome'][0] for island in archipelago.islands]
    
    ax6.bar(island_ids, island_mu, alpha=0.7, color=colors)
    ax6.set_title('Mutation Rate (μ) by Island', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Island ID', fontsize=10)
    ax6.set_ylabel('μ', fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Parameter comparison across islands (omega)
    ax7 = fig.add_subplot(gs[2, 1])
    
    island_omega = [island.get_stats()['best_genome'][1] for island in archipelago.islands]
    
    ax7.bar(island_ids, island_omega, alpha=0.7, color=colors)
    ax7.set_title('Oscillation Freq (ω) by Island', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Island ID', fontsize=10)
    ax7.set_ylabel('ω', fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Parameter comparison across islands (d)
    ax8 = fig.add_subplot(gs[2, 2])
    
    island_d = [island.get_stats()['best_genome'][2] for island in archipelago.islands]
    
    ax8.bar(island_ids, island_d, alpha=0.7, color=colors)
    ax8.set_title('Decoherence (d) by Island', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Island ID', fontsize=10)
    ax8.set_ylabel('d', fontsize=10)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Migration impact analysis
    ax9 = fig.add_subplot(gs[3, :])
    
    if archipelago.migration_events:
        migration_gens = [e['generation'] for e in archipelago.migration_events]
        
        # Calculate fitness improvement after each migration
        improvements = []
        for i, gen in enumerate(migration_gens):
            if gen + 10 < len(global_best_history):
                before = global_best_history[gen]
                after = global_best_history[gen + 10]
                improvement = (after - before) / before * 100 if before > 0 else 0
                improvements.append(improvement)
            else:
                improvements.append(0)
        
        ax9.bar(migration_gens[:len(improvements)], improvements, alpha=0.7, color='purple', width=5)
        ax9.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax9.set_title('Migration Impact (Fitness improvement 10 gens after migration)', 
                     fontsize=14, fontweight='bold')
        ax9.set_xlabel('Migration Generation', fontsize=12)
        ax9.set_ylabel('Fitness Improvement (%)', fontsize=12)
        ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'Island Model Evolution: {len(archipelago.islands)} Islands × {archipelago.agents_per_island} Agents', 
                fontsize=20, fontweight='bold', y=0.995)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"island_evolution_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {filename}")
    
    return filename

def main():
    # Run island evolution
    archipelago, global_best_history, island_diversity_history = run_island_evolution(
        num_islands=10,
        agents_per_island=100,
        generations=300
    )
    
    # Visualize
    viz_file = visualize_island_evolution(archipelago, global_best_history, island_diversity_history)
    
    # Export data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = f"island_evolution_results_{timestamp}.json"
    
    champion = archipelago.get_global_champion()
    
    export_data = {
        'timestamp': timestamp,
        'num_islands': len(archipelago.islands),
        'agents_per_island': archipelago.agents_per_island,
        'total_agents': len(archipelago.islands) * archipelago.agents_per_island,
        'generations': len(global_best_history),
        'migration_interval': archipelago.migration_interval,
        'global_champion': {
            'island_id': champion['island_id'],
            'fitness': float(champion['fitness']),
            'genome': [float(x) for x in champion['genome']]
        },
        'island_stats': [
            {
                'island_id': island.island_id,
                'best_fitness': float(island.evolution.population[0][0]),
                'mean_fitness': float(np.mean([a[0] for a in island.evolution.population])),
                'best_genome': [float(x) for x in island.get_stats()['best_genome']]
            }
            for island in archipelago.islands
        ],
        'migration_events': archipelago.migration_events,
        'final_diversity': archipelago.get_island_diversity()
    }
    
    with open(data_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"📁 Data saved: {data_file}")
    print("\n✅ ISLAND MODEL EVOLUTION COMPLETE!")

if __name__ == "__main__":
    main()
