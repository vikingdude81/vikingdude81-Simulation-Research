"""
Enhanced Trainer with ML-Guided Parameter Adaptation

This is the Enhanced ML Predictor from the GA_CONDUCTOR_CONCEPT.md
It adds configuration context to the state, allowing the model to learn
strategic relationships like:
- High crossover → Low mutation (inverse relationship)
- Small population → High mutation (exploration boost)
- High selection pressure → Low mutation (preserve elite)

Input: 13 features (vs 10 baseline)
  - 10 existing: fitness metrics, diversity, generation, trends
  - 3 NEW: population_size, crossover_rate, mutation_rate (config context!)

Output: mutation_rate adjustment

This is Phase 2A - the stepping stone to full GA Conductor (25 inputs, multi-output)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path
from trading_specialist import TradingSpecialist
import matplotlib.pyplot as plt


class EnhancedMLPredictor(nn.Module):
    """
    Enhanced ML model that adapts mutation rate based on:
    1. Population state (fitness, diversity, convergence)
    2. Training dynamics (generation, trends, stagnation)
    3. **NEW**: Configuration context (population_size, crossover_rate, current_mutation)
    
    This allows learning strategic relationships between parameters!
    """
    
    def __init__(self, input_dim=13, hidden_dims=[128, 256, 128]):
        super().__init__()
        
        # Build network
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer (mutation rate: 0-2 range)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 0-1 range
        
        self.network = nn.Sequential(*layers)
        
        # Track normalization parameters
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('fitted', torch.tensor(False))
        
    def fit_normalization(self, data: np.ndarray):
        """Fit normalization parameters from training data"""
        self.input_mean = torch.tensor(np.mean(data, axis=0), dtype=torch.float32)
        self.input_std = torch.tensor(np.std(data, axis=0) + 1e-8, dtype=torch.float32)
        self.fitted = torch.tensor(True)
        
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input"""
        if not self.fitted:
            return x
        return (x - self.input_mean) / self.input_std
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Input: [batch, 13] features
        Output: [batch, 1] mutation rate (0-2 range)
        """
        x = self.normalize(x)
        out = self.network(x)
        return out * 2.0  # Scale from [0,1] to [0,2]


class EnhancedSpecialistTrainer:
    """
    Enhanced trainer that uses ML to adapt mutation rate during training
    
    Key differences from specialist_trainer.py:
    1. Tracks configuration context (population_size, crossover_rate, mutation_rate)
    2. Uses EnhancedMLPredictor to adjust mutation_rate dynamically
    3. Learns from baseline training data first (offline learning)
    4. Then applies learned policy during training (online application)
    """
    
    def __init__(
        self,
        regime_type: str,
        training_data: np.ndarray,
        predictions: np.ndarray,
        population_size: int = 200,
        generations: int = 300,
        initial_mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 10,
        tournament_size: int = 5,
        ml_model: Optional[EnhancedMLPredictor] = None,
        use_ml_adaptation: bool = True
    ):
        self.regime_type = regime_type
        self.training_data = training_data
        self.predictions = predictions
        self.population_size = population_size
        self.generations = generations
        self.initial_mutation_rate = initial_mutation_rate
        self.current_mutation_rate = initial_mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.use_ml_adaptation = use_ml_adaptation
        
        # ML model for adaptation
        self.ml_model = ml_model
        
        # Get parameter bounds for regime
        self.bounds = self.get_bounds_for_regime(regime_type)
        
        # Training history
        self.best_fitness = -np.inf
        self.best_genome = None
        self.fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        self.mutation_rate_history = []  # Track how mutation changes!
        self.generations_since_improvement = 0
        
    def get_bounds_for_regime(self, regime_type: str) -> Dict:
        """Get parameter bounds for specific regime"""
        base_bounds = {
            'stop_loss': (0.01, 0.05),
            'take_profit': (0.02, 0.20),
            'position_size': (0.01, 0.10),
            'entry_threshold': (0.0, 1.0),
            'exit_threshold': (0.0, 1.0),
            'max_hold_time': (1, 14),
            'volatility_scaling': (0.5, 2.0),
            'momentum_weight': (0.0, 1.0)
        }
        
        if regime_type == 'volatile':
            return {
                'stop_loss': (0.01, 0.03),
                'take_profit': (0.03, 0.10),
                'position_size': (0.02, 0.05),
                'entry_threshold': (0.5, 0.8),
                'exit_threshold': (0.3, 0.6),
                'max_hold_time': (1, 5),
                'volatility_scaling': (0.8, 1.5),
                'momentum_weight': (0.6, 0.9)
            }
        elif regime_type == 'trending':
            return {
                'stop_loss': (0.02, 0.05),
                'take_profit': (0.10, 0.25),
                'position_size': (0.05, 0.10),
                'entry_threshold': (0.6, 0.9),
                'exit_threshold': (0.4, 0.7),
                'max_hold_time': (5, 14),
                'volatility_scaling': (1.0, 2.0),
                'momentum_weight': (0.7, 1.0)
            }
        elif regime_type == 'ranging':
            return {
                'stop_loss': (0.02, 0.04),
                'take_profit': (0.03, 0.08),
                'position_size': (0.03, 0.07),
                'entry_threshold': (0.4, 0.7),
                'exit_threshold': (0.3, 0.6),
                'max_hold_time': (2, 7),
                'volatility_scaling': (0.5, 1.2),
                'momentum_weight': (0.2, 0.5)
            }
        else:
            return base_bounds
            
    def extract_state(self, population: List, fitness_scores: List, generation: int) -> np.ndarray:
        """
        Extract 13-feature state vector for ML model
        
        Features:
        1. avg_fitness (normalized by best ever)
        2. best_fitness (normalized)
        3. worst_fitness (normalized)
        4. diversity (genome variance)
        5. generation (normalized by total)
        6. generations_since_improvement (normalized)
        7. fitness_improvement_rate (recent trend)
        8. diversity_trend (recent trend)
        9. stagnation_indicator (0-1)
        10. convergence_speed (fitness change velocity)
        11. population_size (NEW - normalized)
        12. crossover_rate (NEW - already 0-1)
        13. current_mutation_rate (NEW - normalized)
        """
        
        # Basic fitness stats
        avg_fitness = np.mean(fitness_scores)
        best_fitness = np.max(fitness_scores)
        worst_fitness = np.min(fitness_scores)
        
        # Normalize by best ever (or 1 if first gen)
        normalization = max(abs(self.best_fitness), 1.0)
        avg_fitness_norm = avg_fitness / normalization
        best_fitness_norm = best_fitness / normalization
        worst_fitness_norm = worst_fitness / normalization
        
        # Diversity (standard deviation of all genome parameters)
        all_genomes = np.array(population)
        diversity = np.std(all_genomes)
        
        # Generation progress
        generation_norm = generation / self.generations
        
        # Time since improvement
        improvement_norm = min(self.generations_since_improvement / 50.0, 1.0)
        
        # Recent fitness trend (last 10 generations)
        if len(self.fitness_history) >= 10:
            recent_fitness = self.fitness_history[-10:]
            fitness_trend = (recent_fitness[-1] - recent_fitness[0]) / (abs(recent_fitness[0]) + 1e-8)
        else:
            fitness_trend = 0.0
            
        # Recent diversity trend
        if len(self.diversity_history) >= 10:
            recent_diversity = self.diversity_history[-10:]
            diversity_trend = (recent_diversity[-1] - recent_diversity[0]) / (recent_diversity[0] + 1e-8)
        else:
            diversity_trend = 0.0
            
        # Stagnation indicator
        stagnation = min(self.generations_since_improvement / 30.0, 1.0)
        
        # Convergence speed (how fast fitness is changing)
        if len(self.fitness_history) >= 5:
            recent_fitness = self.fitness_history[-5:]
            convergence_speed = np.std(recent_fitness) / (abs(np.mean(recent_fitness)) + 1e-8)
        else:
            convergence_speed = 1.0
            
        # NEW: Configuration context
        population_size_norm = self.population_size / 500.0  # Normalize by reasonable max
        crossover_rate_norm = self.crossover_rate  # Already 0-1
        mutation_rate_norm = self.current_mutation_rate / 2.0  # Normalize by max (2.0)
        
        return np.array([
            avg_fitness_norm,
            best_fitness_norm,
            worst_fitness_norm,
            diversity,
            generation_norm,
            improvement_norm,
            fitness_trend,
            diversity_trend,
            stagnation,
            convergence_speed,
            population_size_norm,      # NEW
            crossover_rate_norm,       # NEW
            mutation_rate_norm         # NEW
        ], dtype=np.float32)
        
    def adapt_mutation_rate(self, state: np.ndarray) -> float:
        """Use ML model to predict optimal mutation rate"""
        if self.ml_model is None or not self.use_ml_adaptation:
            return self.current_mutation_rate
            
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            predicted_rate = self.ml_model(state_tensor).item()
            
        # Clamp to reasonable range
        return np.clip(predicted_rate, 0.01, 2.0)
        
    def initialize_population(self) -> List[List[float]]:
        """Initialize random population within bounds"""
        population = []
        param_names = ['stop_loss', 'take_profit', 'position_size', 'entry_threshold',
                      'exit_threshold', 'max_hold_time', 'volatility_scaling', 'momentum_weight']
        
        for _ in range(self.population_size):
            genome = []
            for param in param_names:
                low, high = self.bounds[param]
                genome.append(np.random.uniform(low, high))
            population.append(genome)
            
        return population
        
    def evaluate_population(self, population: List) -> List[float]:
        """Evaluate fitness for all genomes"""
        fitness_scores = []
        for genome in population:
            specialist = TradingSpecialist(genome, self.regime_type)
            fitness = specialist.evaluate_fitness(self.training_data, self.predictions)
            fitness_scores.append(fitness)
        return fitness_scores
        
    def select_parents(self, population: List, fitness_scores: List) -> Tuple:
        """Tournament selection"""
        def tournament():
            indices = np.random.choice(len(population), self.tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            return population[winner_idx]
            
        parent1 = tournament()
        parent2 = tournament()
        return parent1, parent2
        
    def crossover(self, parent1: List, parent2: List) -> Tuple:
        """Single-point crossover"""
        if np.random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
            
        point = np.random.randint(1, len(parent1))
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
        
    def mutate(self, genome: List, mutation_rate: float) -> List:
        """Gaussian mutation within bounds"""
        mutated = genome[:]
        param_names = ['stop_loss', 'take_profit', 'position_size', 'entry_threshold',
                      'exit_threshold', 'max_hold_time', 'volatility_scaling', 'momentum_weight']
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                low, high = self.bounds[param_names[i]]
                range_size = high - low
                mutation = np.random.normal(0, range_size * 0.1)
                mutated[i] = np.clip(mutated[i] + mutation, low, high)
                
        return mutated
        
    def train(self) -> Tuple:
        """
        Main training loop with ML-guided mutation adaptation
        """
        print(f"\n{'='*70}")
        print(f"TRAINING {self.regime_type.upper()} SPECIALIST (ENHANCED ML)")
        print(f"{'='*70}")
        print(f"Training data: {len(self.training_data)} days")
        print(f"Population: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"Initial mutation rate: {self.initial_mutation_rate}")
        print(f"Crossover rate: {self.crossover_rate}")
        print(f"ML Adaptation: {'ENABLED' if self.use_ml_adaptation else 'DISABLED'}")
        print(f"{'='*70}\n")
        
        # Initialize population
        population = self.initialize_population()
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitness_scores = self.evaluate_population(population)
            
            # Extract state
            state = self.extract_state(population, fitness_scores, gen)
            
            # ML-guided mutation rate adaptation
            if self.use_ml_adaptation and gen > 0:
                self.current_mutation_rate = self.adapt_mutation_rate(state)
            
            # Track best
            gen_best = np.max(fitness_scores)
            gen_avg = np.mean(fitness_scores)
            
            if gen_best > self.best_fitness:
                self.best_fitness = gen_best
                self.best_genome = population[np.argmax(fitness_scores)]
                self.generations_since_improvement = 0
            else:
                self.generations_since_improvement += 1
                
            # Record history
            self.fitness_history.append(gen_best)
            self.avg_fitness_history.append(gen_avg)
            
            # Calculate diversity
            all_genomes = np.array(population)
            diversity = np.std(all_genomes)
            self.diversity_history.append(diversity)
            self.mutation_rate_history.append(self.current_mutation_rate)
            
            # Print progress
            if gen % 20 == 0 or gen == self.generations - 1:
                print(f"Gen {gen:4d}: Best={gen_best:8.2f}, Avg={gen_avg:8.2f}, "
                      f"Diversity={diversity:.4f}, Mutation={self.current_mutation_rate:.4f}")
                      
            # Create next generation
            next_population = []
            
            # Elitism - keep best genomes
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                next_population.append(population[idx])
                
            # Generate offspring
            while len(next_population) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, self.current_mutation_rate)
                child2 = self.mutate(child2, self.current_mutation_rate)
                next_population.extend([child1, child2])
                
            population = next_population[:self.population_size]
            
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"Best fitness: {self.best_fitness:.2f}")
        print(f"{'='*70}\n")
        
        return self.best_genome, self.best_fitness, {
            'fitness_history': self.fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'diversity_history': self.diversity_history,
            'mutation_rate_history': self.mutation_rate_history
        }
        
    def save_results(self, output_dir: str = 'outputs'):
        """Save training results"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Evaluate best genome
        specialist = TradingSpecialist(self.best_genome, self.regime_type)
        metrics = specialist.evaluate_fitness(self.training_data, self.predictions, return_metrics=True)
        
        results = {
            'regime_type': self.regime_type,
            'training_method': 'enhanced_ml_predictor',
            'genome': self.best_genome,
            'fitness': self.best_fitness,
            'metrics': metrics,
            'training_history': {
                'best_fitness': self.fitness_history,
                'avg_fitness': self.avg_fitness_history,
                'diversity': self.diversity_history,
                'mutation_rate': self.mutation_rate_history  # NEW!
            },
            'configuration': {
                'population_size': self.population_size,
                'generations': self.generations,
                'initial_mutation_rate': self.initial_mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elite_size': self.elite_size,
                'ml_adaptation': self.use_ml_adaptation
            }
        }
        
        filename = f"{output_dir}/specialist_enhanced_{self.regime_type}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"✅ Saved results to: {filename}")
        return filename
        
    def plot_training(self, output_dir: str = 'outputs'):
        """Plot training evolution with mutation rate overlay"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Enhanced Training Evolution - {self.regime_type.upper()} Specialist', 
                     fontsize=14, fontweight='bold')
        
        generations = range(len(self.fitness_history))
        
        # Fitness evolution
        ax = axes[0, 0]
        ax.plot(generations, self.fitness_history, label='Best Fitness', linewidth=2, color='green')
        ax.plot(generations, self.avg_fitness_history, label='Avg Fitness', linewidth=2, color='blue', alpha=0.7)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Diversity evolution
        ax = axes[0, 1]
        ax.plot(generations, self.diversity_history, linewidth=2, color='purple')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Diversity (Std Dev)')
        ax.set_title('Population Diversity')
        ax.grid(True, alpha=0.3)
        
        # Mutation rate adaptation (NEW!)
        ax = axes[1, 0]
        ax.plot(generations, self.mutation_rate_history, linewidth=2, color='red')
        ax.axhline(y=self.initial_mutation_rate, color='gray', linestyle='--', 
                   label=f'Initial ({self.initial_mutation_rate:.2f})')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mutation Rate')
        ax.set_title('ML-Adapted Mutation Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Fitness improvement rate
        ax = axes[1, 1]
        if len(self.fitness_history) > 10:
            improvement_rate = []
            window = 10
            for i in range(window, len(self.fitness_history)):
                rate = (self.fitness_history[i] - self.fitness_history[i-window]) / window
                improvement_rate.append(rate)
            ax.plot(range(window, len(self.fitness_history)), improvement_rate, 
                   linewidth=2, color='orange')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness Change / 10 Gens')
            ax.set_title('Fitness Improvement Rate')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{output_dir}/training_enhanced_{self.regime_type}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved training plot to: {filename}")
        return filename


if __name__ == '__main__':
    print("Enhanced ML Predictor Trainer")
    print("=" * 70)
    print("\nThis is Phase 2A: Enhanced ML Predictor")
    print("  - 13 input features (adds configuration context)")
    print("  - Learns strategic parameter relationships")
    print("  - Adapts mutation rate during training")
    print("\nNext: Phase 2B - Full GA Conductor (25 inputs, multi-output)")
    print("=" * 70)
