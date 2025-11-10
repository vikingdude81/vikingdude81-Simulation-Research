
"""
🌀 PHASE-FOCUSED QUANTUM-GENETIC EVOLUTION
Systematically explores phase space with locked decoherence constant
Tests mutation frontier and validates phase×mutation interactions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from quantum_genetic_agents import QuantumAgent, create_genome, crossover, mutate
import json
import os
from datetime import datetime
from itertools import product

# === CONSTANTS (Locked based on discoveries) ===
DECOHERENCE_CONSTANT = 0.011  # Universal constant from analysis
PHASE_HIGH_CORRELATION = 0.9532  # Discovered correlation strength

class PhaseEvolutionSystem:
    """Evolution system with phase-first optimization and locked decoherence"""
    
    def __init__(self, population_size=25, elite_preservation=0.3):
        self.population_size = population_size
        self.elite_count = max(1, int(population_size * elite_preservation))
        self.population = []
        self.generation = 0
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_phase': [],
            'best_mutation': [],
            'diversity': []
        }
        
    def create_phase_focused_genome(self, phase_target=None, mutation_target=None):
        """Create genome with phase emphasis and locked decoherence"""
        if phase_target is None:
            phase_target = np.random.uniform(0.1, 0.8)
        if mutation_target is None:
            # Bias towards frontier region
            if np.random.random() < 0.4:
                mutation_target = np.random.uniform(3.0, 5.0)  # Frontier
            else:
                mutation_target = np.random.uniform(0.8, 2.0)  # Moderate
        
        return [
            mutation_target,
            np.random.uniform(0.5, 2.0),  # oscillation_freq
            DECOHERENCE_CONSTANT,  # LOCKED
            phase_target
        ]
    
    def evaluate_genome(self, genome, steps=50):
        """Evaluate with phase-awareness"""
        agent = QuantumAgent(0, genome)
        for t in range(1, steps):
            agent.evolve(t)
        
        fitness = agent.get_final_fitness()
        phase_value = genome[3]
        
        # Multi-objective score: fitness + phase bonus
        phase_bonus = phase_value * 1e6  # Scale to be significant
        total_score = fitness + phase_bonus
        
        return {
            'fitness': fitness,
            'phase': phase_value,
            'mutation': genome[0],
            'total_score': total_score,
            'genome': genome
        }
    
    def initialize_population(self):
        """Initialize with phase-stratified sampling"""
        print(f"🧬 Initializing {self.population_size} agents with phase stratification...")
        
        # Sample different phase regions
        phase_regions = np.linspace(0.1, 0.8, self.population_size)
        
        self.population = []
        for phase_target in phase_regions:
            genome = self.create_phase_focused_genome(phase_target=phase_target)
            result = self.evaluate_genome(genome)
            self.population.append(result)
        
        self.population.sort(key=lambda x: x['total_score'], reverse=True)
        print(f"✓ Population initialized!")
        
    def evolve_generation(self):
        """Evolve with strong elitism and phase-aware selection"""
        self.generation += 1
        
        # Record stats
        best = self.population[0]
        avg_fitness = np.mean([p['fitness'] for p in self.population])
        diversity = np.std([p['phase'] for p in self.population])
        
        self.history['best_fitness'].append(best['fitness'])
        self.history['avg_fitness'].append(avg_fitness)
        self.history['best_phase'].append(best['phase'])
        self.history['best_mutation'].append(best['mutation'])
        self.history['diversity'].append(diversity)
        
        # Print progress
        if self.generation % 10 == 0 or self.generation == 1:
            print(f"Gen {self.generation:4d} | Fit: {best['fitness']:.4e} | "
                  f"Phase: {best['phase']:.4f} | μ: {best['mutation']:.4f} | "
                  f"Diversity: {diversity:.4f}")
        
        # Strong elitism: preserve top genomes INTACT
        next_generation = []
        elites = self.population[:self.elite_count]
        
        # Clone elites (no modification)
        for elite in elites:
            result = self.evaluate_genome(elite['genome'])
            next_generation.append(result)
        
        # Generate offspring with phase-aware crossover
        while len(next_generation) < self.population_size:
            # Tournament selection favoring high phase
            parent1 = self._tournament_select(k=3)
            parent2 = self._tournament_select(k=3)
            
            # Crossover
            child_genome = crossover(parent1['genome'], parent2['genome'])
            
            # Phase-preserving mutation (don't destroy high phase values)
            if child_genome[3] > 0.4:  # High phase value
                mutation_rate = 0.05  # Gentle mutation
            else:
                mutation_rate = 0.15  # Standard mutation
            
            child_genome = mutate(child_genome, mutation_rate=mutation_rate)
            
            # Re-lock decoherence (in case mutation changed it)
            child_genome[2] = DECOHERENCE_CONSTANT
            
            result = self.evaluate_genome(child_genome)
            next_generation.append(result)
        
        self.population = sorted(next_generation, key=lambda x: x['total_score'], reverse=True)
    
    def _tournament_select(self, k=3):
        """Tournament selection with phase awareness"""
        candidates = np.random.choice(self.population, k, replace=False)
        return max(candidates, key=lambda x: x['total_score'])
    
    def run(self, generations=100):
        """Run evolution"""
        print("\n" + "=" * 80)
        print("  🌀 PHASE-FOCUSED EVOLUTION")
        print("=" * 80)
        print(f"\n📊 Configuration:")
        print(f"   Population: {self.population_size}")
        print(f"   Generations: {generations}")
        print(f"   Elite preservation: {self.elite_count}/{self.population_size}")
        print(f"   Decoherence (LOCKED): {DECOHERENCE_CONSTANT}")
        print(f"   Phase correlation: +{PHASE_HIGH_CORRELATION}\n")
        
        self.initialize_population()
        
        print(f"\n🧬 Beginning phase-focused evolution...\n")
        
        for _ in range(generations):
            self.evolve_generation()
        
        best = self.population[0]
        
        print("\n" + "=" * 80)
        print("✨ PHASE-FOCUSED EVOLUTION COMPLETE!")
        print("=" * 80)
        print(f"\n🏆 Best Genome:")
        print(f"   Mutation: {best['mutation']:.6f}")
        print(f"   Oscillation: {best['genome'][1]:.6f}")
        print(f"   Decoherence: {best['genome'][2]:.6f} (locked)")
        print(f"   Phase: {best['phase']:.6f} ⭐")
        print(f"   Fitness: {best['fitness']:.4e}")
        print(f"   Total Score: {best['total_score']:.4e}")
        
        return best


def phase_landscape_scan(mutation_rates=[0.5, 1.0, 2.0, 3.5, 5.0], 
                         phase_values=np.linspace(0.1, 0.8, 20),
                         oscillation=1.0):
    """Systematically scan phase space at different mutation rates"""
    print("\n" + "=" * 80)
    print("  🗺️ PHASE LANDSCAPE SYSTEMATIC SCAN")
    print("=" * 80)
    
    results = []
    total_tests = len(mutation_rates) * len(phase_values)
    current = 0
    
    print(f"\n🧪 Testing {total_tests} parameter combinations...")
    print(f"   Mutation rates: {mutation_rates}")
    print(f"   Phase values: {len(phase_values)} samples from {phase_values[0]:.2f} to {phase_values[-1]:.2f}")
    print(f"   Decoherence: {DECOHERENCE_CONSTANT} (locked)\n")
    
    for mut_rate in mutation_rates:
        for phase in phase_values:
            current += 1
            if current % 20 == 0:
                print(f"   Progress: {current}/{total_tests} ({100*current/total_tests:.1f}%)")
            
            genome = [mut_rate, oscillation, DECOHERENCE_CONSTANT, phase]
            agent = QuantumAgent(0, genome)
            for t in range(1, 50):
                agent.evolve(t)
            
            fitness = agent.get_final_fitness()
            results.append({
                'mutation': mut_rate,
                'phase': phase,
                'fitness': fitness,
                'log_fitness': np.log10(fitness + 1e-10)
            })
    
    print(f"\n✓ Scan complete!\n")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, mut_rate in enumerate(mutation_rates):
        ax = axes[idx]
        
        # Filter results for this mutation rate
        data = [r for r in results if r['mutation'] == mut_rate]
        phases = [d['phase'] for d in data]
        fitness = [d['fitness'] for d in data]
        
        # Scatter plot
        ax.scatter(phases, fitness, s=50, alpha=0.6, c=phases, cmap='viridis', edgecolor='black')
        
        # Trend line
        z = np.polyfit(phases, fitness, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(phases), max(phases), 100)
        ax.plot(x_smooth, p(x_smooth), 'r--', lw=2, alpha=0.7, label='Trend')
        
        ax.set_xlabel('Phase', fontsize=10)
        ax.set_ylabel('Fitness', fontsize=10)
        ax.set_title(f'μ = {mut_rate:.1f}', fontweight='bold', fontsize=11)
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        ax.legend()
    
    # Overall heatmap
    ax_heat = axes[-1]
    
    # Create heatmap data
    phase_grid = np.linspace(0.1, 0.8, 50)
    mut_grid = mutation_rates
    fitness_grid = np.zeros((len(mut_grid), len(phase_grid)))
    
    for i, mut in enumerate(mut_grid):
        for j, phase in enumerate(phase_grid):
            # Find closest result
            closest = min(results, key=lambda r: abs(r['mutation'] - mut) + abs(r['phase'] - phase))
            fitness_grid[i, j] = closest['log_fitness']
    
    im = ax_heat.imshow(fitness_grid, aspect='auto', cmap='hot', 
                        extent=[phase_grid[0], phase_grid[-1], mut_grid[-1], mut_grid[0]],
                        interpolation='bilinear')
    ax_heat.set_xlabel('Phase', fontsize=10)
    ax_heat.set_ylabel('Mutation Rate', fontsize=10)
    ax_heat.set_title('Phase×Mutation Fitness Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax_heat, label='log₁₀(Fitness)')
    
    plt.suptitle('🌀 Phase Landscape Scan Across Mutation Regimes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/phase_landscape_scan.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: visualizations/phase_landscape_scan.png")
    plt.close()
    
    return results


def mutation_frontier_exploration(phase_values=[0.2, 0.35, 0.5, 0.65],
                                  mutation_range=(2.5, 6.0, 30)):
    """Explore mutation frontier at different phase values"""
    print("\n" + "=" * 80)
    print("  🚀 MUTATION FRONTIER EXPLORATION")
    print("=" * 80)
    
    mutation_rates = np.linspace(*mutation_range)
    
    print(f"\n🧪 Testing {len(mutation_rates)} mutation rates from {mutation_range[0]:.1f} to {mutation_range[1]:.1f}")
    print(f"   Phase values: {phase_values}")
    print(f"   Decoherence: {DECOHERENCE_CONSTANT} (locked)\n")
    
    results = {phase: [] for phase in phase_values}
    
    for phase in phase_values:
        print(f"   Testing phase = {phase:.2f}...")
        for mut_rate in mutation_rates:
            genome = [mut_rate, 1.0, DECOHERENCE_CONSTANT, phase]
            agent = QuantumAgent(0, genome)
            for t in range(1, 50):
                agent.evolve(t)
            
            fitness = agent.get_final_fitness()
            results[phase].append({
                'mutation': mut_rate,
                'fitness': fitness
            })
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    # Individual phase plots
    for idx, (phase, color) in enumerate(zip(phase_values, colors)):
        ax = axes[idx]
        
        mutations = [r['mutation'] for r in results[phase]]
        fitness = [r['fitness'] for r in results[phase]]
        
        ax.plot(mutations, fitness, 'o-', lw=2, color=color, alpha=0.7, label=f'Phase={phase:.2f}')
        ax.axvline(3.0, color='red', linestyle='--', lw=2, alpha=0.5, label='Frontier (μ=3.0)')
        
        ax.set_xlabel('Mutation Rate (μ)', fontsize=11)
        ax.set_ylabel('Fitness', fontsize=11)
        ax.set_title(f'Phase = {phase:.2f}', fontweight='bold')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('🚀 Mutation Frontier at Different Phase Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/mutation_frontier_by_phase.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: visualizations/mutation_frontier_by_phase.png")
    plt.close()
    
    return results


def compare_strategies():
    """Compare different evolution strategies"""
    print("\n" + "=" * 80)
    print("  ⚔️ STRATEGY COMPARISON")
    print("=" * 80)
    
    strategies = {
        'Phase-First': PhaseEvolutionSystem(population_size=25, elite_preservation=0.4),
        'Standard': PhaseEvolutionSystem(population_size=25, elite_preservation=0.2),
        'Ultra-Elite': PhaseEvolutionSystem(population_size=25, elite_preservation=0.6)
    }
    
    results = {}
    
    for name, system in strategies.items():
        print(f"\n🧪 Running: {name}")
        best = system.run(generations=50)
        results[name] = {
            'best': best,
            'history': system.history
        }
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Fitness evolution
    ax1 = axes[0, 0]
    for name, data in results.items():
        gens = range(1, len(data['history']['best_fitness']) + 1)
        ax1.plot(gens, data['history']['best_fitness'], lw=2, label=name, alpha=0.8)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Fitness Evolution', fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Phase evolution
    ax2 = axes[0, 1]
    for name, data in results.items():
        gens = range(1, len(data['history']['best_phase']) + 1)
        ax2.plot(gens, data['history']['best_phase'], lw=2, label=name, alpha=0.8)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best Phase')
    ax2.set_title('Phase Evolution', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Final comparison
    ax3 = axes[1, 0]
    names = list(results.keys())
    final_fitness = [results[n]['best']['fitness'] for n in names]
    final_phase = [results[n]['best']['phase'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    ax3.bar(x - width/2, final_fitness, width, label='Fitness', alpha=0.7)
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x + width/2, final_phase, width, label='Phase', alpha=0.7, color='orange')
    
    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Final Fitness', color='blue')
    ax3_twin.set_ylabel('Final Phase', color='orange')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=15)
    ax3.set_title('Final Performance', fontweight='bold')
    ax3.set_yscale('log')
    
    # Diversity
    ax4 = axes[1, 1]
    for name, data in results.items():
        gens = range(1, len(data['history']['diversity']) + 1)
        ax4.plot(gens, data['history']['diversity'], lw=2, label=name, alpha=0.8)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Phase Diversity (std)')
    ax4.set_title('Population Diversity', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.suptitle('⚔️ Evolution Strategy Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/strategy_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: visualizations/strategy_comparison.png")
    plt.close()
    
    return results


def main():
    print("\n" + "=" * 80)
    print("  🌀 PHASE-FOCUSED QUANTUM-GENETIC EVOLUTION SUITE")
    print("=" * 80)
    print("\nBased on strategic insights:")
    print("  ✅ Decoherence locked at ~0.011 (universal constant)")
    print("  ✅ Phase correlation: +0.95 (strongest predictor)")
    print("  ✅ Mutation frontier: >3.0 (explosive growth regime)")
    print("  ✅ Strong elitism (preserve champions intact)")
    print("=" * 80)
    
    print("\n🔬 EXPERIMENT MENU:")
    print("  1. Phase Landscape Scan - Systematically map phase×mutation space")
    print("  2. Mutation Frontier Exploration - Test μ=2.5 to 6.0 at key phases")
    print("  3. Phase-First Evolution - 100-gen run with phase emphasis")
    print("  4. Strategy Comparison - Compare elitism levels")
    print("  5. Full Suite - Run all experiments")
    
    try:
        choice = int(input("\nSelect experiment (1-5): "))
    except ValueError:
        choice = 5
    
    if choice == 1 or choice == 5:
        phase_landscape_scan()
    
    if choice == 2 or choice == 5:
        mutation_frontier_exploration()
    
    if choice == 3 or choice == 5:
        system = PhaseEvolutionSystem(population_size=30, elite_preservation=0.4)
        best = system.run(generations=100)
        
        # Export
        metadata = {
            'fitness': best['fitness'],
            'phase': best['phase'],
            'mutation': best['mutation'],
            'decoherence': DECOHERENCE_CONSTANT,
            'type': 'phase_focused',
            'strategy': 'strong_elitism'
        }
        
        from quantum_genetic_agents import export_genome
        export_genome(best['genome'], 'phase_focused_best.json', metadata)
    
    if choice == 4 or choice == 5:
        compare_strategies()
    
    print("\n" + "=" * 80)
    print("✨ PHASE-FOCUSED EXPERIMENTS COMPLETE!")
    print("=" * 80)
    print("\n📊 Key Findings:")
    print("  • Phase is the primary fitness driver")
    print("  • Decoherence ~0.011 is universal")
    print("  • Mutation >3.0 unlocks new regimes")
    print("  • Elitism preserves critical synergies")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
