"""
Multi-Environment ML-Guided Evolution

Trains genomes across multiple environments simultaneously to discover
more robust, generalizable quantum genetic algorithms.

Key Features:
- Multi-environment fitness evaluation
- Ensemble ML surrogate (one per environment)
- Pareto-optimal selection (must excel in ALL environments)
- Adaptive environment weighting
- Cross-environment diversity tracking
"""

import numpy as np
import json
import time
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Import quantum genetic agents
import sys
sys.path.append(str(Path(__file__).parent))
from quantum_genetic_agents import QuantumAgent


class FitnessSurrogate(nn.Module):
    """Neural network to predict fitness from genome parameters."""
    
    def __init__(self, input_dim=4, hidden_dims=[128, 64, 32]):
        super(FitnessSurrogate, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2 if hidden_dim > 32 else 0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MultiEnvironmentMLEvolution:
    """
    ML-guided evolution across multiple environments.
    
    Discovers genomes that perform well universally, not just in one environment.
    """
    
    def __init__(
        self,
        population_size=1000,
        generations=200,
        ml_filter_ratio=0.05,
        environments=None,
        model_paths=None,
        scaler_paths=None,
        elite_ratio=0.05,
        mutation_rate=0.7
    ):
        self.population_size = population_size
        self.generations = generations
        self.ml_filter_ratio = ml_filter_ratio
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        
        # Multiple environments
        if environments is None:
            self.environments = ['standard', 'gentle', 'harsh', 'chaotic']
        else:
            self.environments = environments
        
        self.num_environments = len(self.environments)
        
        # Load ML models for each environment
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        
        if model_paths and scaler_paths:
            for i, env in enumerate(self.environments):
                # Load model
                model = FitnessSurrogate().to(self.device)
                model.load_state_dict(torch.load(model_paths[i], weights_only=False))
                model.eval()
                self.models[env] = model
                
                # Load scaler
                with open(scaler_paths[i], 'rb') as f:
                    self.scalers[env] = pickle.load(f)
        else:
            # Use same model for all environments (single-environment trained)
            # This is a baseline - ideally train separate models per environment
            model_path = Path(__file__).parent / "fitness_surrogate_best.pth"
            scaler_path = Path(__file__).parent / "fitness_surrogate_scaler.pkl"
            
            if model_path.exists() and scaler_path.exists():
                base_model = FitnessSurrogate().to(self.device)
                checkpoint = torch.load(model_path, weights_only=False)
                # Handle checkpoint format (may have optimizer_state_dict, etc.)
                if 'model_state_dict' in checkpoint:
                    base_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    base_model.load_state_dict(checkpoint)
                base_model.eval()
                
                with open(scaler_path, 'rb') as f:
                    base_scaler = pickle.load(f)
                
                # Use same model for all environments (will retrain later)
                for env in self.environments:
                    self.models[env] = base_model
                    self.scalers[env] = base_scaler
            else:
                raise ValueError("No ML models found. Run train_fitness_surrogate.py first.")
        
        # Population and tracking
        self.population = []
        self.best_genome = None
        self.best_fitness_per_env = {env: -float('inf') for env in self.environments}
        self.best_overall_fitness = -float('inf')
        
        # History tracking
        self.history = {
            'generations': [],
            'best_fitness_per_env': {env: [] for env in self.environments},
            'avg_fitness_per_env': {env: [] for env in self.environments},
            'best_overall_fitness': [],
            'diversity': [],
            'time_per_generation': [],
            'ml_time': [],
            'sim_time': []
        }
        
        # Performance metrics
        self.total_simulations = 0
        self.total_ml_predictions = 0
        self.start_time = None
        
        print(f"🌍 Multi-Environment ML Evolution initialized")
        print(f"   Environments: {', '.join(self.environments)}")
        print(f"   Population: {population_size}")
        print(f"   Generations: {generations}")
        print(f"   ML Filter: {ml_filter_ratio*100}%")
        print(f"   Device: {self.device}")
    
    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        
        for _ in range(self.population_size):
            genome = [
                np.random.uniform(1.0, 5.0),  # mu
                np.random.uniform(0.1, 2.5),  # omega
                np.random.choice([
                    np.random.uniform(0.001, 0.01),  # 70% stable range
                    np.random.uniform(0.01, 0.03),   # 20% higher
                    np.random.uniform(0.0001, 0.001) # 10% very low
                ], p=[0.7, 0.2, 0.1]),
                np.random.uniform(0, 2 * np.pi)  # phi
            ]
            
            # Evaluate in all environments
            fitness_dict = self.evaluate_genome_all_envs(genome)
            
            self.population.append({
                'genome': genome,
                'fitness_per_env': fitness_dict,
                'overall_fitness': self.calculate_overall_fitness(fitness_dict)
            })
        
        self.total_simulations += self.population_size * self.num_environments
        
        # Sort by overall fitness
        self.population.sort(key=lambda x: x['overall_fitness'], reverse=True)
        self.best_genome = self.population[0]['genome']
        self.best_fitness_per_env = self.population[0]['fitness_per_env']
        self.best_overall_fitness = self.population[0]['overall_fitness']
        
        print(f"✅ Initialized population with {self.population_size} genomes")
        print(f"   Best overall fitness: {self.best_overall_fitness:.2f}")
        for env in self.environments:
            print(f"   Best {env}: {self.best_fitness_per_env[env]:.2f}")
    
    def evaluate_genome_all_envs(self, genome):
        """Simulate genome in all environments."""
        fitness_dict = {}
        
        for env in self.environments:
            # Create agent with genome [mu, omega, d, phi]
            agent = QuantumAgent(
                agent_id=0,
                genome=genome,
                environment=env
            )
            
            # Run simulation
            for timestep in range(100):
                agent.evolve(timestep)
            
            fitness_dict[env] = agent.get_final_fitness()
        
        return fitness_dict
    
    def calculate_overall_fitness(self, fitness_dict):
        """
        Calculate overall fitness across environments.
        
        Strategies:
        1. Minimum (worst-case) - ensures good performance everywhere
        2. Average - balanced performance
        3. Weighted average - prioritize certain environments
        
        Using MINIMUM for robustness.
        """
        return min(fitness_dict.values())  # Robust: must excel in ALL environments
    
    def predict_fitness_batch(self, genomes, environment):
        """Predict fitness for batch of genomes in specific environment."""
        with torch.no_grad():
            # Convert to numpy array
            X = np.array(genomes)
            
            # Scale features
            X_scaled = self.scalers[environment].transform(X)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Predict
            predictions = self.models[environment](X_tensor)
            
            return predictions.cpu().numpy().flatten()
    
    def predict_fitness_all_envs(self, genomes):
        """Predict fitness across all environments and calculate overall."""
        predictions_per_env = {}
        
        for env in self.environments:
            predictions_per_env[env] = self.predict_fitness_batch(genomes, env)
        
        # Calculate overall fitness (minimum across environments)
        overall_predictions = np.min([predictions_per_env[env] for env in self.environments], axis=0)
        
        return overall_predictions, predictions_per_env
    
    def mutate_genome(self, genome):
        """Apply mutation to genome."""
        mutated = genome.copy()
        
        # Mutation strength decreases over time
        mutation_strength = 0.2
        
        # Mutate each parameter with some probability
        if np.random.random() < 0.5:
            mutated[0] = np.clip(mutated[0] + np.random.normal(0, mutation_strength), 1.0, 5.0)
        
        if np.random.random() < 0.5:
            mutated[1] = np.clip(mutated[1] + np.random.normal(0, mutation_strength), 0.1, 2.5)
        
        if np.random.random() < 0.5:
            mutated[2] = np.clip(mutated[2] * np.random.uniform(0.5, 2.0), 0.0001, 0.03)
        
        if np.random.random() < 0.5:
            mutated[3] = (mutated[3] + np.random.normal(0, mutation_strength)) % (2 * np.pi)
        
        return mutated
    
    def crossover_genomes(self, parent1, parent2):
        """Create child genome from two parents."""
        child = []
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child
    
    def calculate_diversity(self):
        """Calculate population diversity."""
        genomes = np.array([ind['genome'] for ind in self.population])
        
        # Standard deviation across population for each parameter
        diversity = np.mean(np.std(genomes, axis=0))
        
        return diversity
    
    def evolve_generation(self, generation):
        """Evolve one generation with multi-environment ML guidance."""
        gen_start_time = time.time()
        ml_start_time = time.time()
        
        # Elite preservation
        elite_count = int(self.population_size * self.elite_ratio)
        elite = self.population[:elite_count]
        
        # Generate many candidates (2x population)
        candidates = []
        candidate_count = self.population_size * 2
        
        for _ in range(candidate_count):
            if np.random.random() < self.mutation_rate:
                # Mutation
                parent = self.population[np.random.randint(0, len(self.population))]
                candidate = self.mutate_genome(parent['genome'])
            else:
                # Crossover
                parent1 = self.population[np.random.randint(0, len(self.population))]
                parent2 = self.population[np.random.randint(0, len(self.population))]
                candidate = self.crossover_genomes(parent1['genome'], parent2['genome'])
            
            candidates.append(candidate)
        
        # ML prediction across all environments
        overall_predictions, predictions_per_env = self.predict_fitness_all_envs(candidates)
        self.total_ml_predictions += len(candidates) * self.num_environments
        
        ml_time = time.time() - ml_start_time
        sim_start_time = time.time()
        
        # Select top candidates by predicted overall fitness
        top_indices = np.argsort(overall_predictions)[-int(candidate_count * self.ml_filter_ratio):]
        top_candidates = [candidates[i] for i in top_indices]
        
        # Simulate only top candidates (in all environments)
        evaluated_candidates = []
        for genome in top_candidates:
            fitness_dict = self.evaluate_genome_all_envs(genome)
            overall_fitness = self.calculate_overall_fitness(fitness_dict)
            
            evaluated_candidates.append({
                'genome': genome,
                'fitness_per_env': fitness_dict,
                'overall_fitness': overall_fitness
            })
        
        self.total_simulations += len(top_candidates) * self.num_environments
        
        sim_time = time.time() - sim_start_time
        
        # Combine elite + evaluated candidates
        new_population = elite + evaluated_candidates
        
        # Sort and trim to population size
        new_population.sort(key=lambda x: x['overall_fitness'], reverse=True)
        self.population = new_population[:self.population_size]
        
        # Update best
        if self.population[0]['overall_fitness'] > self.best_overall_fitness:
            self.best_genome = self.population[0]['genome']
            self.best_fitness_per_env = self.population[0]['fitness_per_env']
            self.best_overall_fitness = self.population[0]['overall_fitness']
        
        # Track metrics
        gen_time = time.time() - gen_start_time
        diversity = self.calculate_diversity()
        
        self.history['generations'].append(generation)
        self.history['best_overall_fitness'].append(self.best_overall_fitness)
        self.history['diversity'].append(diversity)
        self.history['time_per_generation'].append(gen_time)
        self.history['ml_time'].append(ml_time)
        self.history['sim_time'].append(sim_time)
        
        # Per-environment tracking
        for env in self.environments:
            env_fitnesses = [ind['fitness_per_env'][env] for ind in self.population]
            self.history['best_fitness_per_env'][env].append(max(env_fitnesses))
            self.history['avg_fitness_per_env'][env].append(np.mean(env_fitnesses))
        
        # Progress report every 20 generations
        if generation % 20 == 0 or generation == 1:
            print(f"\n🔄 Generation {generation}/{self.generations}")
            print(f"   Best overall: {self.best_overall_fitness:.2f}")
            for env in self.environments:
                print(f"   Best {env}: {self.best_fitness_per_env[env]:.2f}")
            print(f"   Diversity: {diversity:.4f}")
            print(f"   Time: {gen_time:.3f}s (ML: {ml_time:.3f}s, Sim: {sim_time:.3f}s)")
    
    def run(self):
        """Run complete multi-environment evolution."""
        print("\n" + "="*70)
        print("🌍 MULTI-ENVIRONMENT ML-GUIDED EVOLUTION")
        print("="*70 + "\n")
        
        self.start_time = time.time()
        
        # Initialize
        print("🔧 Initializing population...")
        self.initialize_population()
        
        # Evolve
        print(f"\n🚀 Starting evolution for {self.generations} generations...")
        
        for gen in range(1, self.generations + 1):
            self.evolve_generation(gen)
        
        total_time = time.time() - self.start_time
        
        # Final report
        print("\n" + "="*70)
        print("✨ EVOLUTION COMPLETE!")
        print("="*70)
        print(f"\n🏆 BEST GENOME (Overall Fitness: {self.best_overall_fitness:.2f})")
        print(f"   Parameters: {self.best_genome}")
        print(f"\n📊 FITNESS PER ENVIRONMENT:")
        for env in self.environments:
            print(f"   {env:12s}: {self.best_fitness_per_env[env]:.2f}")
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Avg time/gen: {total_time/self.generations:.3f}s")
        print(f"   Total simulations: {self.total_simulations:,}")
        print(f"   Total ML predictions: {self.total_ml_predictions:,}")
        
        # Calculate speedup
        traditional_sims = self.population_size * self.generations * self.num_environments
        speedup = traditional_sims / self.total_simulations
        print(f"   Speedup factor: {speedup:.1f}x")
        print(f"   (vs {traditional_sims:,} traditional simulations)")
        
        return {
            'best_genome': self.best_genome,
            'best_fitness_per_env': self.best_fitness_per_env,
            'best_overall_fitness': self.best_overall_fitness,
            'history': self.history,
            'performance': {
                'total_time': total_time,
                'avg_time_per_gen': total_time / self.generations,
                'total_simulations': self.total_simulations,
                'total_ml_predictions': self.total_ml_predictions,
                'speedup_factor': speedup
            },
            'config': {
                'population_size': self.population_size,
                'generations': self.generations,
                'ml_filter_ratio': self.ml_filter_ratio,
                'environments': self.environments,
                'elite_ratio': self.elite_ratio,
                'mutation_rate': self.mutation_rate
            }
        }


def visualize_results(results, output_path):
    """Create comprehensive visualization of multi-environment results."""
    fig = plt.figure(figsize=(20, 12))
    
    environments = results['config']['environments']
    history = results['history']
    
    # 1. Overall fitness evolution
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history['generations'], history['best_overall_fitness'], 
             'b-', linewidth=2, label='Best Overall')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Overall Fitness (Min Across Envs)')
    ax1.set_title('Overall Fitness Evolution (Robust Performance)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Per-environment fitness evolution
    ax2 = plt.subplot(2, 3, 2)
    colors = plt.cm.viridis(np.linspace(0, 1, len(environments)))
    for i, env in enumerate(environments):
        ax2.plot(history['generations'], history['best_fitness_per_env'][env],
                linewidth=2, label=env, color=colors[i])
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best Fitness')
    ax2.set_title('Per-Environment Fitness Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Average fitness per environment
    ax3 = plt.subplot(2, 3, 3)
    for i, env in enumerate(environments):
        ax3.plot(history['generations'], history['avg_fitness_per_env'][env],
                linewidth=2, label=env, color=colors[i], alpha=0.7)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Average Fitness')
    ax3.set_title('Average Population Fitness Per Environment', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Diversity tracking
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(history['generations'], history['diversity'], 'g-', linewidth=2)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Population Diversity')
    ax4.set_title('Genetic Diversity Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Time breakdown
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(history['generations'], history['ml_time'], 'b-', 
             linewidth=2, label='ML Prediction', alpha=0.7)
    ax5.plot(history['generations'], history['sim_time'], 'r-', 
             linewidth=2, label='Simulation', alpha=0.7)
    ax5.set_xlabel('Generation')
    ax5.set_ylabel('Time (seconds)')
    ax5.set_title('Time Per Generation (ML vs Simulation)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Final fitness comparison (bar chart)
    ax6 = plt.subplot(2, 3, 6)
    env_names = list(results['best_fitness_per_env'].keys())
    env_fitness = list(results['best_fitness_per_env'].values())
    bars = ax6.bar(range(len(env_names)), env_fitness, color=colors, alpha=0.8)
    ax6.set_xticks(range(len(env_names)))
    ax6.set_xticklabels(env_names, rotation=45, ha='right')
    ax6.set_ylabel('Best Fitness')
    ax6.set_title('Final Best Fitness Per Environment', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, env_fitness)):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    plt.close()


def main():
    """Run multi-environment ML evolution."""
    
    # Configuration
    config = {
        'population_size': 1000,
        'generations': 200,
        'ml_filter_ratio': 0.05,  # 5% - only simulate top 50 per generation
        'environments': ['standard', 'gentle', 'harsh', 'chaotic'],
        'elite_ratio': 0.05,
        'mutation_rate': 0.7
    }
    
    # Create evolution
    evolution = MultiEnvironmentMLEvolution(**config)
    
    # Run
    results = evolution.run()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(__file__).parent / f"multi_env_ml_evolution_{timestamp}.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_serializable = json.loads(
        json.dumps(results, default=convert_numpy)
    )
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    # Visualize
    viz_path = Path(__file__).parent / f"multi_env_ml_analysis_{timestamp}.png"
    visualize_results(results, viz_path)
    
    print("\n✅ Multi-environment evolution complete!")
    print(f"\n🎯 Key Findings:")
    print(f"   - Best overall (robust): {results['best_overall_fitness']:.2f}")
    print(f"   - Speedup: {results['performance']['speedup_factor']:.1f}x")
    print(f"   - Total time: {results['performance']['total_time']:.2f}s")
    print(f"   - Simulations: {results['performance']['total_simulations']:,}")


if __name__ == "__main__":
    main()
