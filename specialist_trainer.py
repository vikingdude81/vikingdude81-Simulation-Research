"""
Specialist Trainer - Standard Genetic Algorithm

Trains TradingSpecialist agents using standard genetic algorithm.
This is the BASELINE implementation (no GA Conductor yet).

Uses regime-specific parameter bounds to guide evolution toward
strategies appropriate for each market regime.

Author: GA Conductor Research Team
Date: November 5, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from trading_specialist import TradingSpecialist
import json
from datetime import datetime
import matplotlib.pyplot as plt


class SpecialistTrainer:
    """
    Standard GA trainer for trading specialists
    
    Uses fixed mutation/crossover rates (no adaptation yet)
    """
    
    def __init__(self,
                 regime_type: str,
                 training_data: pd.DataFrame,
                 predictions: np.ndarray,
                 population_size: int = 100,
                 generations: int = 200,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elite_size: int = 10,
                 tournament_size: int = 5):
        """
        Initialize trainer
        
        Args:
            regime_type: 'volatile', 'trending', 'ranging', or 'crisis'
            training_data: Historical OHLCV data for this regime
            predictions: ML predictions (same length as data)
            population_size: Number of agents in population
            generations: Number of generations to evolve
            mutation_rate: Probability of gene mutation
            crossover_rate: Probability of crossover
            elite_size: Number of top agents to preserve
            tournament_size: Tournament selection size
        """
        self.regime_type = regime_type
        self.training_data = training_data
        self.predictions = predictions
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        # Get regime-specific bounds
        self.bounds = self.get_bounds_for_regime(regime_type)
        
        # Training history
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'generation': []
        }
        
        self.best_genome = None
        self.best_fitness = -np.inf
        
    def get_bounds_for_regime(self, regime_type: str) -> Dict:
        """
        Get parameter bounds optimized for each regime type
        
        These bounds guide initial population and mutations toward
        strategies that make sense for each regime.
        """
        
        if regime_type == 'volatile':
            # Volatile markets: tight stops, quick profits, small positions
            return {
                'stop_loss': (0.01, 0.03),        # 1-3% stops
                'take_profit': (0.03, 0.10),      # 3-10% targets
                'position_size': (0.02, 0.05),    # 2-5% positions
                'entry_threshold': (0.5, 0.8),    # Moderate-strong signals
                'exit_threshold': (0.3, 0.6),     # Quick exits
                'max_hold_time': (1, 5),          # 1-5 days
                'volatility_scaling': (0.8, 1.5), # Moderate scaling
                'momentum_weight': (0.6, 0.9)     # Favor momentum
            }
        
        elif regime_type == 'trending':
            # Trending markets: wider stops, let winners run, larger positions
            return {
                'stop_loss': (0.02, 0.05),        # 2-5% stops
                'take_profit': (0.10, 0.25),      # 10-25% targets (let it run!)
                'position_size': (0.05, 0.10),    # 5-10% positions
                'entry_threshold': (0.6, 0.9),    # Strong signals only
                'exit_threshold': (0.4, 0.7),     # Hold longer
                'max_hold_time': (5, 14),         # 5-14 days
                'volatility_scaling': (1.0, 2.0), # More scaling
                'momentum_weight': (0.7, 1.0)     # Strong trend-following
            }
        
        elif regime_type == 'ranging':
            # Ranging markets: medium everything, mean reversion
            return {
                'stop_loss': (0.02, 0.04),        # 2-4% stops
                'take_profit': (0.03, 0.08),      # 3-8% targets
                'position_size': (0.03, 0.07),    # 3-7% positions
                'entry_threshold': (0.4, 0.7),    # Moderate signals
                'exit_threshold': (0.3, 0.6),     # Moderate exits
                'max_hold_time': (2, 7),          # 2-7 days
                'volatility_scaling': (0.5, 1.2), # Less scaling
                'momentum_weight': (0.2, 0.5)     # Favor mean reversion
            }
        
        else:  # crisis
            # Crisis markets: very tight, minimal risk, quick in/out
            return {
                'stop_loss': (0.005, 0.015),      # 0.5-1.5% stops (tiny!)
                'take_profit': (0.01, 0.03),      # 1-3% targets (scalping)
                'position_size': (0.005, 0.02),   # 0.5-2% positions (minimal)
                'entry_threshold': (0.8, 0.95),   # Very strong signals only
                'exit_threshold': (0.5, 0.8),     # Fast exits
                'max_hold_time': (1, 3),          # 1-3 days max
                'volatility_scaling': (0.3, 0.8), # Heavy scaling
                'momentum_weight': (0.3, 0.6)     # Moderate momentum
            }
    
    def initialize_population(self) -> np.ndarray:
        """Create initial random population within bounds"""
        population = []
        
        for _ in range(self.population_size):
            genome = []
            for param, (low, high) in self.bounds.items():
                value = np.random.uniform(low, high)
                genome.append(value)
            population.append(genome)
        
        return np.array(population)
    
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness for all agents in population"""
        fitness_scores = []
        
        for genome in population:
            specialist = TradingSpecialist(genome, self.regime_type)
            fitness = specialist.evaluate_fitness(self.training_data, self.predictions)
            fitness_scores.append(fitness)
        
        return np.array(fitness_scores)
    
    def select_parents(self, population: np.ndarray, fitness_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tournament selection"""
        tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Single-point crossover"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy()
        
        crossover_point = np.random.randint(1, len(parent1))
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child
    
    def mutate(self, genome: np.ndarray) -> np.ndarray:
        """Mutate genes with Gaussian noise"""
        mutated = genome.copy()
        
        for i, (param, (low, high)) in enumerate(self.bounds.items()):
            if np.random.random() < self.mutation_rate:
                # Gaussian mutation with 10% std of range
                mutation = np.random.normal(0, (high - low) * 0.1)
                mutated[i] += mutation
                mutated[i] = np.clip(mutated[i], low, high)
        
        return mutated
    
    def train(self, verbose: bool = True) -> Tuple[np.ndarray, float, Dict]:
        """
        Train specialist using standard GA
        
        Returns:
            best_genome: Best genome found
            best_fitness: Best fitness achieved
            history: Training history dict
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"TRAINING {self.regime_type.upper()} SPECIALIST")
            print(f"{'='*70}")
            print(f"Training data: {len(self.training_data)} days")
            print(f"Population: {self.population_size}")
            print(f"Generations: {self.generations}")
            print(f"Mutation rate: {self.mutation_rate}")
            print(f"Crossover rate: {self.crossover_rate}")
            print(f"\nParameter bounds:")
            for param, (low, high) in self.bounds.items():
                print(f"  {param:20s}: [{low:.4f}, {high:.4f}]")
            print(f"{'='*70}\n")
        
        # Initialize population
        population = self.initialize_population()
        
        # Evolution loop
        for gen in range(self.generations):
            # Evaluate fitness
            fitness_scores = self.evaluate_population(population)
            
            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_genome = population[gen_best_idx].copy()
            
            # Calculate diversity (std of population)
            diversity = np.mean([np.std(population[:, i]) for i in range(population.shape[1])])
            
            # Record history
            self.history['generation'].append(gen)
            self.history['best_fitness'].append(self.best_fitness)
            self.history['avg_fitness'].append(np.mean(fitness_scores))
            self.history['diversity'].append(diversity)
            
            # Print progress
            if verbose and (gen % 20 == 0 or gen == self.generations - 1):
                print(f"Gen {gen:4d}: "
                      f"Best={self.best_fitness:8.2f}, "
                      f"Avg={np.mean(fitness_scores):8.2f}, "
                      f"Diversity={diversity:.4f}")
            
            # Create next generation
            next_population = []
            
            # Elitism - keep best performers
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            next_population.extend(population[elite_indices])
            
            # Fill rest with offspring
            while len(next_population) < self.population_size:
                # Select parents
                parent1 = self.select_parents(population, fitness_scores)
                parent2 = self.select_parents(population, fitness_scores)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutate(child)
                
                next_population.append(child)
            
            population = np.array(next_population[:self.population_size])
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"TRAINING COMPLETE!")
            print(f"Best fitness: {self.best_fitness:.2f}")
            print(f"{'='*70}\n")
        
        return self.best_genome, self.best_fitness, self.history
    
    def save_results(self, output_path: str = None):
        """Save training results and best genome"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'outputs/specialist_{self.regime_type}_{timestamp}.json'
        
        # Test best genome on training data
        best_specialist = TradingSpecialist(self.best_genome, self.regime_type)
        best_specialist.evaluate_fitness(self.training_data, self.predictions)
        metrics = best_specialist.get_metrics()
        
        results = {
            'regime_type': self.regime_type,
            'training_config': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elite_size': self.elite_size,
                'tournament_size': self.tournament_size
            },
            'best_genome': self.best_genome.tolist(),
            'best_fitness': float(self.best_fitness),
            'final_metrics': metrics.to_dict(),
            'bounds': {k: list(v) for k, v in self.bounds.items()},
            'training_history': {
                'generation': self.history['generation'],
                'best_fitness': [float(x) for x in self.history['best_fitness']],
                'avg_fitness': [float(x) for x in self.history['avg_fitness']],
                'diversity': [float(x) for x in self.history['diversity']]
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Saved results to: {output_path}")
        
        return results
    
    def plot_training(self, output_path: str = None):
        """Plot training progress"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'outputs/training_{self.regime_type}_{timestamp}.png'
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Fitness over time
        ax = axes[0, 0]
        ax.plot(self.history['generation'], self.history['best_fitness'], 
                'b-', linewidth=2, label='Best Fitness')
        ax.plot(self.history['generation'], self.history['avg_fitness'],
                'g--', linewidth=1, alpha=0.7, label='Avg Fitness')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title(f'{self.regime_type.upper()} Specialist Training - Fitness')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Diversity over time
        ax = axes[0, 1]
        ax.plot(self.history['generation'], self.history['diversity'],
                'r-', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Diversity (std)')
        ax.set_title('Population Diversity')
        ax.grid(True, alpha=0.3)
        
        # Fitness improvement rate
        ax = axes[1, 0]
        best_fitness_array = np.array(self.history['best_fitness'])
        improvement = np.diff(best_fitness_array)
        improvement = np.concatenate([[0], improvement])
        ax.plot(self.history['generation'], improvement, 'purple', linewidth=1)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Improvement')
        ax.set_title('Generation-to-Generation Improvement')
        ax.grid(True, alpha=0.3)
        
        # Best genome visualization
        ax = axes[1, 1]
        if self.best_genome is not None:
            gene_names = list(self.bounds.keys())
            gene_values = self.best_genome
            
            # Normalize to [0, 1] for visualization
            normalized = []
            for i, (low, high) in enumerate(self.bounds.values()):
                norm_val = (gene_values[i] - low) / (high - low)
                normalized.append(norm_val)
            
            bars = ax.barh(gene_names, normalized)
            
            # Color code by value
            for bar, val in zip(bars, normalized):
                bar.set_color(plt.cm.RdYlGn(val))
            
            ax.set_xlim(0, 1)
            ax.set_xlabel('Normalized Value')
            ax.set_title('Best Genome Parameters')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved training plot to: {output_path}")
        plt.close()


if __name__ == '__main__':
    # Test trainer with volatile data
    print("Testing SpecialistTrainer...\n")
    
    # Load volatile data
    try:
        df = pd.read_csv('DATA/yf_btc_1d_volatile.csv')
        print(f"Loaded {len(df)} days of volatile market data")
        
        # Add required indicators if missing
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            df['atr'] = high_low.rolling(14).mean()
        
        # Generate simple predictions (momentum-based)
        predictions = (df['returns'].rolling(5).mean() * 10).fillna(0).values
        
        # Create trainer (short test run)
        trainer = SpecialistTrainer(
            regime_type='volatile',
            training_data=df,
            predictions=predictions,
            population_size=50,      # Small for testing
            generations=50,          # Short for testing
            mutation_rate=0.15,
            crossover_rate=0.7
        )
        
        # Train
        best_genome, best_fitness, history = trainer.train()
        
        # Test best specialist
        print("\nTesting best specialist on training data...")
        specialist = TradingSpecialist(best_genome, 'volatile')
        specialist.evaluate_fitness(df, predictions)
        metrics = specialist.get_metrics()
        
        print(f"\nBest Specialist Performance:")
        print(f"  Genome: {best_genome}")
        print(f"  Fitness: {metrics.fitness:.2f}")
        print(f"  Total Return: {metrics.total_return*100:.2f}%")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Win Rate: {metrics.win_rate*100:.1f}%")
        print(f"  Num Trades: {metrics.num_trades}")
        
        # Save results
        trainer.save_results()
        trainer.plot_training()
        
        print("\n✅ SpecialistTrainer working!")
        
    except FileNotFoundError:
        print("⚠️  Volatile data not found. Run label_historical_regimes.py first.")
        print("Testing with dummy data instead...\n")
        
        # Create dummy data
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        df = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(200) * 2),
            'high': 100 + np.cumsum(np.random.randn(200) * 2) + 3,
            'low': 100 + np.cumsum(np.random.randn(200) * 2) - 3,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
        
        df['returns'] = df['close'].pct_change()
        predictions = (df['returns'].rolling(5).mean() * 10).fillna(0).values
        
        trainer = SpecialistTrainer(
            regime_type='volatile',
            training_data=df,
            predictions=predictions,
            population_size=30,
            generations=30
        )
        
        best_genome, best_fitness, history = trainer.train()
        print(f"\nBest fitness: {best_fitness:.2f}")
        print("✅ SpecialistTrainer class working!")
