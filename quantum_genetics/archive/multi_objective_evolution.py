
"""
🎯 Multi-Objective Evolution System
Evolve genomes optimized for different objectives beyond just fitness
Based on what made Gen 2117 successful (μ=6.27, ϕ=1.05, d=0.011, ω=1.0)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from quantum_genetic_agents import QuantumAgent, create_genome, crossover, mutate, export_genome
import seaborn as sns

class MultiObjectiveEvolution:
    """Evolve genomes for multiple objectives simultaneously"""
    
    OBJECTIVES = {
        'max_fitness': {
            'name': 'Maximum Fitness',
            'weight_fn': lambda agent: agent.get_final_fitness()
        },
        'max_stability': {
            'name': 'Maximum Stability',
            'weight_fn': lambda agent: 1.0 / (1.0 + np.std([state[3] for state in agent.history]))
        },
        'max_coherence': {
            'name': 'Coherence Preservation',
            'weight_fn': lambda agent: np.mean([state[1] for state in agent.history])
        },
        'max_energy': {
            'name': 'Energy Magnitude',
            'weight_fn': lambda agent: np.mean([abs(state[0]) for state in agent.history])
        },
        'min_variance': {
            'name': 'Minimum Variance',
            'weight_fn': lambda agent: -np.var([state[3] for state in agent.history])
        },
        'balanced': {
            'name': 'Balanced Performance',
            'weight_fn': lambda agent: (
                agent.get_final_fitness() * 0.4 +
                (1.0 / (1.0 + np.std([state[3] for state in agent.history]))) * 100000000 * 0.3 +
                np.mean([state[1] for state in agent.history]) * 10000000 * 0.3
            )
        }
    }
    
    def __init__(self, base_genome, population_size=50, simulation_steps=40):
        """Initialize with Gen 2117 as baseline"""
        self.base_genome = base_genome
        self.population_size = population_size
        self.simulation_steps = simulation_steps
        self.results = {}
        
    def evolve_for_objective(self, objective, generations=100, mutation_scale=0.1):
        """Evolve population optimized for specific objective"""
        print(f"\n🎯 Evolving for: {self.OBJECTIVES[objective]['name']}")
        print(f"   Base genome: μ={self.base_genome[0]:.3f}, ω={self.base_genome[1]:.3f}, "
              f"d={self.base_genome[2]:.6f}, ϕ={self.base_genome[3]:.3f}")
        
        # Initialize population with variations of base genome
        population = []
        for i in range(self.population_size):
            # Start with base genome and add small mutations
            genome = self.base_genome.copy()
            for j in range(len(genome)):
                if np.random.random() < 0.3:  # 30% chance to mutate each parameter
                    genome[j] = genome[j] + np.random.normal(0, genome[j] * mutation_scale)
                    genome[j] = max(0.01, genome[j])  # Keep positive
            
            agent = QuantumAgent(i, genome)
            for t in range(1, self.simulation_steps):
                agent.evolve(t)
            
            score = self.OBJECTIVES[objective]['weight_fn'](agent)
            population.append((score, agent))
        
        population.sort(key=lambda x: x[0], reverse=True)
        
        # Evolve
        best_history = []
        avg_history = []
        
        for gen in range(generations):
            best_score = population[0][0]
            avg_score = np.mean([s for s, _ in population])
            best_history.append(best_score)
            avg_history.append(avg_score)
            
            if gen % 20 == 0:
                print(f"   Gen {gen:3d}: Best={best_score:.4e}, Avg={avg_score:.4e}")
            
            # Create next generation
            next_gen = []
            
            # Elitism - keep top 5
            for score, agent in population[:5]:
                new_agent = QuantumAgent(agent.id, agent.genome)
                for t in range(1, self.simulation_steps):
                    new_agent.evolve(t)
                score = self.OBJECTIVES[objective]['weight_fn'](new_agent)
                next_gen.append((score, new_agent))
            
            # Breed and mutate
            while len(next_gen) < self.population_size:
                parent1 = population[np.random.randint(0, self.population_size // 2)][1]
                parent2 = population[np.random.randint(0, self.population_size // 2)][1]
                
                child_genome = crossover(parent1.genome, parent2.genome)
                child_genome = mutate(child_genome, mutation_rate=0.2)
                
                child_agent = QuantumAgent(len(next_gen), child_genome)
                for t in range(1, self.simulation_steps):
                    child_agent.evolve(t)
                
                score = self.OBJECTIVES[objective]['weight_fn'](child_agent)
                next_gen.append((score, child_agent))
            
            population = sorted(next_gen, key=lambda x: x[0], reverse=True)
        
        best_agent = population[0][1]
        self.results[objective] = {
            'genome': best_agent.genome,
            'score': population[0][0],
            'fitness': best_agent.get_final_fitness(),
            'history': {
                'best': best_history,
                'avg': avg_history
            }
        }
        
        print(f"   ✓ Final score: {population[0][0]:.4e}")
        print(f"   ✓ Genome: μ={best_agent.genome[0]:.3f}, ω={best_agent.genome[1]:.3f}, "
              f"d={best_agent.genome[2]:.6f}, ϕ={best_agent.genome[3]:.3f}")
        
        return best_agent.genome
    
    def visualize_pareto_frontier(self):
        """Visualize trade-offs between different objectives"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Evolution progress for each objective
        ax1 = axes[0, 0]
        for obj_name, data in self.results.items():
            ax1.plot(data['history']['best'], label=self.OBJECTIVES[obj_name]['name'], lw=2)
        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Objective Score', fontsize=11)
        ax1.set_title('Multi-Objective Evolution Progress', fontweight='bold', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.set_yscale('log')
        
        # Genome parameter comparison
        ax2 = axes[0, 1]
        objectives = list(self.results.keys())
        genomes = np.array([self.results[obj]['genome'] for obj in objectives])
        
        x = np.arange(4)
        width = 0.15
        colors = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6', '#1ABC9C']
        
        for i, obj in enumerate(objectives):
            offset = (i - len(objectives)/2) * width
            ax2.bar(x + offset, genomes[i], width, label=self.OBJECTIVES[obj]['name'][:15], 
                   color=colors[i % len(colors)], alpha=0.8)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Mutation', 'Oscillation', 'Decoherence', 'Phase'])
        ax2.set_ylabel('Parameter Value', fontsize=11)
        ax2.set_title('Genome Parameters by Objective', fontweight='bold', fontsize=12)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3, axis='y')
        
        # Fitness vs Stability scatter
        ax3 = axes[1, 0]
        fitness_scores = []
        stability_scores = []
        obj_labels = []
        
        for obj_name, data in self.results.items():
            fitness_scores.append(data['fitness'])
            # Calculate stability
            agent = QuantumAgent(0, data['genome'])
            for t in range(1, self.simulation_steps):
                agent.evolve(t)
            stability = 1.0 / (1.0 + np.std([state[3] for state in agent.history]))
            stability_scores.append(stability)
            obj_labels.append(self.OBJECTIVES[obj_name]['name'][:20])
        
        scatter = ax3.scatter(fitness_scores, stability_scores, s=200, alpha=0.6, 
                            c=range(len(objectives)), cmap='viridis', edgecolors='black', linewidth=2)
        
        for i, label in enumerate(obj_labels):
            ax3.annotate(label, (fitness_scores[i], stability_scores[i]), 
                        fontsize=8, ha='center', va='bottom')
        
        ax3.set_xlabel('Fitness', fontsize=11)
        ax3.set_ylabel('Stability', fontsize=11)
        ax3.set_title('Pareto Frontier: Fitness vs Stability', fontweight='bold', fontsize=12)
        ax3.grid(alpha=0.3)
        ax3.set_xscale('log')
        
        # Performance heatmap
        ax4 = axes[1, 1]
        metrics = ['Fitness', 'Stability', 'Coherence', 'Energy']
        metric_data = []
        
        for obj_name in objectives:
            agent = QuantumAgent(0, self.results[obj_name]['genome'])
            for t in range(1, self.simulation_steps):
                agent.evolve(t)
            
            row = [
                self.results[obj_name]['fitness'],
                1.0 / (1.0 + np.std([state[3] for state in agent.history])),
                np.mean([state[1] for state in agent.history]),
                np.mean([abs(state[0]) for state in agent.history])
            ]
            metric_data.append(row)
        
        # Normalize for heatmap
        metric_array = np.array(metric_data)
        metric_normalized = (metric_array - metric_array.min(axis=0)) / (metric_array.max(axis=0) - metric_array.min(axis=0) + 1e-10)
        
        sns.heatmap(metric_normalized, annot=False, cmap='RdYlGn', ax=ax4,
                   xticklabels=metrics, yticklabels=[self.OBJECTIVES[obj]['name'][:20] for obj in objectives],
                   cbar_kws={'label': 'Normalized Score'})
        ax4.set_title('Multi-Metric Performance Heatmap', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('visualizations/multi_objective_evolution.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved: visualizations/multi_objective_evolution.png")
        plt.close()

def main():
    print("\n" + "=" * 80)
    print("  🎯 MULTI-OBJECTIVE EVOLUTION SYSTEM")
    print("=" * 80)
    print("\nBased on Gen 2117 success: μ=6.27, ϕ=1.05, d=0.011, ω=1.0")
    print("\nEvolution objectives:")
    
    # Load Gen 2117 as baseline
    try:
        with open('co_evolved_best_gen_2117.json', 'r') as f:
            data = json.load(f)
            base_genome = [
                data['genome']['mutation_rate'],
                data['genome']['oscillation_freq'],
                data['genome']['decoherence_rate'],
                data['genome']['phase_offset']
            ]
        print(f"✓ Loaded Gen 2117 baseline")
    except:
        # Fallback to hardcoded Gen 2117 values
        base_genome = [6.27, 1.0, 0.011, 1.05]
        print(f"✓ Using hardcoded Gen 2117 baseline")
    
    # Initialize multi-objective evolution
    moe = MultiObjectiveEvolution(base_genome, population_size=30, simulation_steps=40)
    
    # Evolve for each objective
    for i, (obj_key, obj_info) in enumerate(moe.OBJECTIVES.items(), 1):
        print(f"\n{i}. {obj_info['name']}")
    
    print("\n" + "=" * 80)
    
    # Run evolution for all objectives
    for obj_key in moe.OBJECTIVES.keys():
        genome = moe.evolve_for_objective(obj_key, generations=100, mutation_scale=0.05)
        
        # Export specialized genome
        metadata = {
            'type': 'multi_objective',
            'objective': moe.OBJECTIVES[obj_key]['name'],
            'base_genome': 'Gen 2117',
            'score': moe.results[obj_key]['score'],
            'fitness': moe.results[obj_key]['fitness'],
            'export_timestamp': datetime.now().isoformat()
        }
        
        filename = f'specialized_{obj_key}_genome.json'
        export_genome(genome, filename, metadata)
    
    # Visualize results
    moe.visualize_pareto_frontier()
    
    # Summary
    print("\n" + "=" * 80)
    print("✨ MULTI-OBJECTIVE EVOLUTION COMPLETE!")
    print("=" * 80)
    
    print("\n📊 Specialized Genomes Created:")
    for obj_key, data in moe.results.items():
        print(f"\n   {moe.OBJECTIVES[obj_key]['name']}:")
        print(f"      Score: {data['score']:.4e}")
        print(f"      Fitness: {data['fitness']:.4e}")
        print(f"      μ={data['genome'][0]:.3f}, ω={data['genome'][1]:.3f}, "
              f"d={data['genome'][2]:.6f}, ϕ={data['genome'][3]:.3f}")
    
    print("\n💡 Use Cases:")
    print("   • max_fitness: Peak performance in standard environments")
    print("   • max_stability: Predictable, reliable behavior")
    print("   • max_coherence: Long-lasting quantum states")
    print("   • max_energy: High-power applications")
    print("   • min_variance: Consistent output")
    print("   • balanced: General-purpose deployment")
    
    print("\n🚀 Deploy any specialized genome via the dashboard!")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
