"""
Hybrid ML-Guided Evolution
===========================
Combines ML fitness prediction with actual simulation for 10x speedup.

Strategy:
1. Generate 1000 candidate genomes (mutations/crossovers)
2. ML predicts fitness for ALL candidates instantly (< 1 second)
3. Only SIMULATE top 100 by predicted fitness (10% of traditional)
4. Use actual simulation results for selection
5. Optional: Update ML with new data for continuous improvement

Expected speedup: 10x (3 seconds vs 30 seconds per generation)
"""

import numpy as np
import torch
import pickle
from pathlib import Path
from quantum_genetic_agents import QuantumAgent
from train_fitness_surrogate import FitnessSurrogate
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class MLGuidedEvolution:
    """Evolution with ML pre-filtering for 10x speedup"""
    
    def __init__(self, model_path, scaler_path, population_size=300, 
                 ml_filter_ratio=0.1, environment='standard'):
        """
        Args:
            model_path: Path to trained fitness surrogate model
            scaler_path: Path to feature scaler
            population_size: Size of population
            ml_filter_ratio: Fraction of candidates to actually simulate (0.1 = 10%)
            environment: Evolution environment
        """
        self.population_size = population_size
        self.ml_filter_ratio = ml_filter_ratio
        self.environment = environment
        self.num_to_simulate = int(population_size * ml_filter_ratio)
        
        print(f"\n🚀 Initializing ML-Guided Evolution")
        print(f"   Population: {population_size}")
        print(f"   ML Filter Ratio: {ml_filter_ratio:.1%}")
        print(f"   Simulations per gen: {self.num_to_simulate} (vs {population_size} traditional)")
        print(f"   Expected speedup: {1/ml_filter_ratio:.1f}x\n")
        
        # Load ML model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FitnessSurrogate(input_dim=4).to(self.device)
        checkpoint = torch.load(model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✓ ML model loaded (R² = {checkpoint.get('val_r2', 0):.3f})")
        print(f"✓ Device: {self.device}\n")
        
        # Statistics
        self.stats = {
            'ml_prediction_time': [],
            'simulation_time': [],
            'total_time': [],
            'best_fitness_history': [],
            'avg_fitness_history': [],
            'ml_predictions_per_gen': [],
            'simulations_per_gen': []
        }
    
    def predict_fitness_batch(self, genomes):
        """Predict fitness for multiple genomes using ML model"""
        # Convert to numpy array
        genomes_array = np.array(genomes, dtype=np.float32)
        
        # Scale features
        genomes_scaled = self.scaler.transform(genomes_array)
        
        # Convert to tensor
        genomes_tensor = torch.tensor(genomes_scaled, dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(genomes_tensor).cpu().numpy().flatten()
        
        return predictions
    
    def simulate_genome(self, genome, timesteps=100):
        """Actually simulate a genome to get true fitness"""
        agent = QuantumAgent(0, genome, self.environment)
        for t in range(timesteps):
            agent.evolve(t)
        return agent.get_final_fitness()
    
    def initialize_population(self):
        """Create initial random population"""
        population = []
        for _ in range(self.population_size):
            genome = np.array([
                np.random.uniform(1.0, 5.0),    # mu
                np.random.uniform(0.1, 2.5),    # omega
                np.random.uniform(0.001, 0.02), # d
                np.random.uniform(0.0, 2*np.pi) # phi
            ])
            population.append(genome)
        return population
    
    def mutate(self, genome, mutation_rate=0.1):
        """Mutate a genome"""
        mutated = genome.copy()
        for i in range(len(genome)):
            if np.random.random() < mutation_rate:
                mutated[i] += np.random.normal(0, 0.1 * genome[i])
        
        # Clip to valid ranges
        mutated[0] = np.clip(mutated[0], 1.0, 5.0)      # mu
        mutated[1] = np.clip(mutated[1], 0.1, 2.5)      # omega
        mutated[2] = np.clip(mutated[2], 0.0001, 0.03)  # d
        mutated[3] = mutated[3] % (2 * np.pi)           # phi
        
        return mutated
    
    def crossover(self, parent1, parent2):
        """Crossover two genomes"""
        point = np.random.randint(1, len(parent1))
        child = np.concatenate([parent1[:point], parent2[point:]])
        return child
    
    def evolve_generation_hybrid(self, population, generation):
        """
        Evolve one generation using hybrid ML-guided approach
        
        Returns:
            new_population: Selected genomes for next generation
            stats: Generation statistics
        """
        gen_start = time.time()
        
        # Step 1: Generate candidate genomes (mutations + crossovers)
        candidates = []
        
        # Keep top 10% elites unchanged
        elite_count = max(1, int(self.population_size * 0.1))
        
        # Generate rest as mutations/crossovers
        for _ in range(self.population_size - elite_count):
            if np.random.random() < 0.7:  # 70% mutation
                parent = population[np.random.randint(len(population))]
                candidates.append(self.mutate(parent))
            else:  # 30% crossover
                p1 = population[np.random.randint(len(population))]
                p2 = population[np.random.randint(len(population))]
                candidates.append(self.crossover(p1, p2))
        
        # Step 2: ML predicts fitness for ALL candidates
        ml_start = time.time()
        predicted_fitnesses = self.predict_fitness_batch(candidates)
        ml_time = time.time() - ml_start
        
        # Step 3: Select top N by predicted fitness to actually simulate
        top_indices = np.argsort(predicted_fitnesses)[-self.num_to_simulate:]
        top_candidates = [candidates[i] for i in top_indices]
        
        # Step 4: Simulate only the top candidates
        sim_start = time.time()
        actual_fitnesses = []
        for genome in top_candidates:
            fitness = self.simulate_genome(genome)
            actual_fitnesses.append(fitness)
        sim_time = time.time() - sim_start
        
        # Step 5: Also simulate current population for comparison
        current_fitnesses = []
        for genome in population[:elite_count]:
            fitness = self.simulate_genome(genome)
            current_fitnesses.append(fitness)
        
        # Combine simulated candidates with elites
        all_genomes = top_candidates + population[:elite_count]
        all_fitnesses = actual_fitnesses + current_fitnesses
        
        # Select top population_size for next generation
        sorted_indices = np.argsort(all_fitnesses)[-self.population_size:]
        new_population = [all_genomes[i] for i in sorted_indices]
        new_fitnesses = [all_fitnesses[i] for i in sorted_indices]
        
        gen_time = time.time() - gen_start
        
        # Statistics
        stats = {
            'generation': generation,
            'best_fitness': max(new_fitnesses),
            'avg_fitness': np.mean(new_fitnesses),
            'ml_prediction_time': ml_time,
            'simulation_time': sim_time,
            'total_time': gen_time,
            'ml_predictions': len(candidates),
            'actual_simulations': len(top_candidates) + elite_count,
            'speedup_factor': self.population_size / (len(top_candidates) + elite_count)
        }
        
        return new_population, new_fitnesses, stats
    
    def run(self, generations=50):
        """Run hybrid ML-guided evolution"""
        print(f"{'='*70}")
        print(f"HYBRID ML-GUIDED EVOLUTION")
        print(f"{'='*70}")
        print(f"  Generations: {generations}")
        print(f"  Population: {self.population_size}")
        print(f"  ML Filter: Top {self.ml_filter_ratio:.1%} simulated")
        print(f"  Environment: {self.environment}")
        print(f"{'='*70}\n")
        
        # Initialize population
        print("🧬 Initializing population...")
        population = self.initialize_population()
        
        # Initial fitnesses
        print("📊 Evaluating initial population...")
        fitnesses = [self.simulate_genome(g) for g in tqdm(population, desc="Initial eval")]
        
        best_genome = None
        best_fitness = -float('inf')
        
        print(f"\n{'='*70}")
        print(f"{'Gen':>4} {'Best':>12} {'Avg':>12} {'ML_ms':>8} {'Sim_ms':>8} {'Total_s':>8} {'Speedup':>8}")
        print(f"{'='*70}")
        
        # Evolution loop
        for gen in range(generations):
            population, fitnesses, stats = self.evolve_generation_hybrid(population, gen)
            
            # Track best
            if stats['best_fitness'] > best_fitness:
                best_fitness = stats['best_fitness']
                best_genome = population[np.argmax(fitnesses)]
            
            # Store statistics
            self.stats['ml_prediction_time'].append(stats['ml_prediction_time'])
            self.stats['simulation_time'].append(stats['simulation_time'])
            self.stats['total_time'].append(stats['total_time'])
            self.stats['best_fitness_history'].append(stats['best_fitness'])
            self.stats['avg_fitness_history'].append(stats['avg_fitness'])
            self.stats['ml_predictions_per_gen'].append(stats['ml_predictions'])
            self.stats['simulations_per_gen'].append(stats['actual_simulations'])
            
            # Print progress every 5 generations
            if gen % 5 == 0 or gen == generations - 1:
                print(f"{gen:4d} {stats['best_fitness']:12.2f} {stats['avg_fitness']:12.2f} "
                      f"{stats['ml_prediction_time']*1000:8.1f} {stats['simulation_time']*1000:8.1f} "
                      f"{stats['total_time']:8.2f} {stats['speedup_factor']:8.1f}x")
        
        print(f"{'='*70}\n")
        
        # Final summary
        total_time = sum(self.stats['total_time'])
        avg_time_per_gen = np.mean(self.stats['total_time'])
        total_sims = sum(self.stats['simulations_per_gen'])
        total_ml_preds = sum(self.stats['ml_predictions_per_gen'])
        
        print(f"✓ EVOLUTION COMPLETE")
        print(f"\n  Best Fitness: {best_fitness:.6f}")
        print(f"  Best Genome: {best_genome}")
        print(f"\n  Total Time: {total_time:.1f}s")
        print(f"  Avg Time/Gen: {avg_time_per_gen:.2f}s")
        print(f"  Total Simulations: {total_sims:,}")
        print(f"  Total ML Predictions: {total_ml_preds:,}")
        print(f"  Actual Speedup: {(generations * self.population_size) / total_sims:.1f}x")
        
        return {
            'best_genome': best_genome.tolist(),
            'best_fitness': float(best_fitness),
            'population': [g.tolist() for g in population],
            'fitnesses': [float(f) for f in fitnesses],
            'stats': self.stats,
            'config': {
                'generations': generations,
                'population_size': self.population_size,
                'ml_filter_ratio': self.ml_filter_ratio,
                'environment': self.environment
            },
            'performance': {
                'total_time': total_time,
                'avg_time_per_gen': avg_time_per_gen,
                'total_simulations': total_sims,
                'total_ml_predictions': total_ml_preds,
                'speedup_factor': (generations * self.population_size) / total_sims
            }
        }

def plot_hybrid_results(results, save_path):
    """Visualize hybrid evolution results"""
    stats = results['stats']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    generations = range(len(stats['best_fitness_history']))
    
    # Fitness evolution
    axes[0, 0].plot(generations, stats['best_fitness_history'], 'b-', linewidth=2, label='Best')
    axes[0, 0].plot(generations, stats['avg_fitness_history'], 'r--', linewidth=2, label='Average')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].set_title('Fitness Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Timing breakdown
    axes[0, 1].plot(generations, [t*1000 for t in stats['ml_prediction_time']], 'g-', label='ML Prediction', linewidth=2)
    axes[0, 1].plot(generations, [t*1000 for t in stats['simulation_time']], 'orange', label='Simulation', linewidth=2)
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Time (ms)')
    axes[0, 1].set_title('Time Breakdown per Generation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cumulative time
    cumulative_time = np.cumsum(stats['total_time'])
    axes[0, 2].plot(generations, cumulative_time, 'm-', linewidth=2)
    axes[0, 2].set_xlabel('Generation')
    axes[0, 2].set_ylabel('Cumulative Time (s)')
    axes[0, 2].set_title('Total Evolution Time')
    axes[0, 2].grid(True, alpha=0.3)
    
    # ML predictions vs simulations
    axes[1, 0].bar(generations[::5], [stats['ml_predictions_per_gen'][i] for i in range(0, len(generations), 5)], 
                   alpha=0.7, label='ML Predictions', color='green')
    axes[1, 0].bar(generations[::5], [stats['simulations_per_gen'][i] for i in range(0, len(generations), 5)], 
                   alpha=0.7, label='Actual Simulations', color='orange')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('ML Predictions vs Simulations')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Speedup factor
    speedup = np.array(stats['ml_predictions_per_gen']) / np.array(stats['simulations_per_gen'])
    axes[1, 1].plot(generations, speedup, 'c-', linewidth=2)
    axes[1, 1].axhline(y=10, color='r', linestyle='--', label='Target 10x', alpha=0.5)
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Speedup Factor')
    axes[1, 1].set_title('Achieved Speedup per Generation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Performance summary
    perf = results['performance']
    axes[1, 2].axis('off')
    summary_text = f"""
    PERFORMANCE SUMMARY
    
    Total Time: {perf['total_time']:.1f}s
    Avg Time/Gen: {perf['avg_time_per_gen']:.2f}s
    
    Total Simulations: {perf['total_simulations']:,}
    Total ML Predictions: {perf['total_ml_predictions']:,}
    
    Speedup Factor: {perf['speedup_factor']:.1f}x
    
    Best Fitness: {results['best_fitness']:.2f}
    
    ML Filter: {results['config']['ml_filter_ratio']:.1%}
    Population: {results['config']['population_size']}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results visualization saved to: {save_path}")
    plt.close()

def main():
    # Configuration
    script_dir = Path(__file__).parent
    
    MODEL_PATH = script_dir / 'fitness_surrogate_best.pth'
    SCALER_PATH = script_dir / 'fitness_surrogate_scaler.pkl'
    
    GENERATIONS = 50
    POPULATION_SIZE = 300
    ML_FILTER_RATIO = 0.1  # Simulate top 10% (30 out of 300)
    ENVIRONMENT = 'standard'
    
    # Run hybrid evolution
    evolution = MLGuidedEvolution(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        population_size=POPULATION_SIZE,
        ml_filter_ratio=ML_FILTER_RATIO,
        environment=ENVIRONMENT
    )
    
    results = evolution.run(generations=GENERATIONS)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = script_dir / f'hybrid_evolution_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    # Plot results
    plot_file = script_dir / f'hybrid_evolution_analysis_{timestamp}.png'
    plot_hybrid_results(results, plot_file)
    
    print(f"\n{'='*70}")
    print("🎉 HYBRID ML-GUIDED EVOLUTION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n  Check {results_file.name} for detailed results")
    print(f"  Check {plot_file.name} for visualizations")
    print(f"\n  Achieved {results['performance']['speedup_factor']:.1f}x speedup! 🚀")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
