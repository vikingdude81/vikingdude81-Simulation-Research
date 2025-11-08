"""
Conductor-Enhanced Specialist Trainer
Uses the trained GA Conductor to adaptively control all evolution parameters
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from trading_specialist import TradingSpecialist
from regime_detector import RegimeDetector
from phase2_complete_implementation import GAConductor


@dataclass
class ConductorControlSignals:
    """Control signals from GA Conductor"""
    # Evolution parameters
    mutation_rate: float
    crossover_rate: float
    selection_pressure: float
    
    # Population control
    population_delta: int
    immigration_count: int
    culling_count: int
    diversity_injection: float
    
    # Crisis management
    extinction_trigger: float
    elite_preservation: float
    restart_signal: float
    
    # Institutional
    welfare_amount: float
    tax_rate: float


class ConductorEnhancedTrainer:
    """
    Training with adaptive parameter control from GA Conductor
    """
    
    def __init__(self,
                 regime: str,
                 regime_data: pd.DataFrame,
                 conductor_model_path: str = "outputs/ga_conductor_best.pth",
                 population_size: int = 200,
                 generations: int = 300,
                 elite_size: int = 10,
                 tournament_size: int = 5):
        
        self.regime = regime
        self.regime_data = regime_data
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        # Load trained GA Conductor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading GA Conductor on {self.device}...")
        self.conductor = GAConductor().to(self.device)
        checkpoint = torch.load(conductor_model_path, map_location=self.device)
        self.conductor.load_state_dict(checkpoint['model_state_dict'])
        self.conductor.eval()
        val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'unknown'))
        print(f"✓ GA Conductor loaded (val_loss: {val_loss if isinstance(val_loss, str) else f'{val_loss:.6f}'})")
        
        # Initialize population
        self.population: List[TradingSpecialist] = []
        self.best_agent: Optional[TradingSpecialist] = None
        self.best_fitness: float = float('-inf')
        
        # Fitness cache for performance optimization
        self.fitness_cache: Dict[str, float] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        
        # Training history
        self.history = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'std_fitness': [],
            'diversity': [],
            'time_since_improvement': [],
            'stagnation_count': [],
            # Conductor control signals
            'mutation_rate': [],
            'crossover_rate': [],
            'selection_pressure': [],
            'population_delta': [],
            'immigration': [],
            'culling': [],
            'diversity_injection': [],
            'extinction_trigger': [],
            'elite_preservation': [],
            'restart_signal': [],
            'welfare': [],
            'tax': [],
            # Population metrics
            'population_size': [],
            'avg_age': [],
            'oldest_age': [],
            'young_pct': [],
            'wealth_gini': [],
        }
        
        self.time_since_improvement = 0
        self.stagnation_count = 0
        
    def _create_conductor_state(self, generation: int) -> torch.Tensor:
        """
        Create 25-dimensional state for GA Conductor
        """
        pop_stats = self._get_population_stats()
        
        # Population statistics (10)
        avg_fitness = pop_stats['avg_fitness']
        best_fitness = pop_stats['best_fitness']
        worst_fitness = pop_stats['worst_fitness']
        std_fitness = pop_stats['std_fitness']
        diversity = pop_stats['diversity']
        progress = generation / self.generations
        time_since_improvement_norm = min(self.time_since_improvement / 50.0, 1.0)
        
        fitness_trend = 0.0
        if len(self.history['best_fitness']) >= 2:
            recent = self.history['best_fitness'][-5:]
            if len(recent) >= 2:
                fitness_trend = (recent[-1] - recent[0]) / (len(recent) * max(abs(recent[-1]), 0.01))
        
        convergence_speed = abs(fitness_trend)
        stagnation = min(self.stagnation_count / 20.0, 1.0)
        
        # Wealth percentiles (6)
        # Get valid fitness values only
        valid_fitnesses = []
        for agent in self.population:
            if hasattr(agent, 'fitness') and agent.fitness is not None:
                if not np.isnan(agent.fitness) and not np.isinf(agent.fitness):
                    valid_fitnesses.append(agent.fitness)
        
        # If no valid fitnesses, use zeros
        if not valid_fitnesses:
            bottom_10 = 0.0
            bottom_25 = 0.0
            median = 0.0
            top_25 = 0.0
            top_10 = 0.0
            gini = 0.0
        else:
            fitnesses = np.array(valid_fitnesses)
            bottom_10 = np.percentile(fitnesses, 10)
            bottom_25 = np.percentile(fitnesses, 25)
            median = np.percentile(fitnesses, 50)
            top_25 = np.percentile(fitnesses, 75)
            top_10 = np.percentile(fitnesses, 90)
            
            # Gini coefficient
            sorted_fitness = np.sort(fitnesses)
            n = len(sorted_fitness)
            cumsum = np.cumsum(sorted_fitness)
            if cumsum[-1] != 0:
                gini = (2 * np.sum((np.arange(1, n+1)) * sorted_fitness)) / (n * cumsum[-1]) - (n + 1) / n
            else:
                gini = 0.0
        
        # Age metrics (3)
        ages = [getattr(agent, 'age', 0) for agent in self.population]
        avg_age = np.mean(ages)
        oldest_age = max(ages)
        young_agents_pct = sum(1 for age in ages if age < 10) / len(ages)
        
        # Strategy diversity (2)
        genomes = [agent.genome for agent in self.population]
        unique_strategies = len(set(str(g) for g in genomes))
        dominant_strategy_pct = max([str(g) for g in genomes].count(s) for s in set(str(g) for g in genomes)) / len(genomes)
        
        # Current configuration (4)
        current_mutation = self.history['mutation_rate'][-1] if self.history['mutation_rate'] else 0.1
        current_crossover = self.history['crossover_rate'][-1] if self.history['crossover_rate'] else 0.7
        current_selection = self.history['selection_pressure'][-1] if self.history['selection_pressure'] else 0.5
        current_pop_size = len(self.population)
        
        state = np.array([
            # Population stats (10)
            avg_fitness, best_fitness, worst_fitness, std_fitness, diversity,
            progress, time_since_improvement_norm, fitness_trend, convergence_speed, stagnation,
            # Wealth percentiles (6)
            bottom_10, bottom_25, median, top_25, top_10, gini,
            # Age metrics (3)
            avg_age / 100.0, oldest_age / 300.0, young_agents_pct,
            # Strategy diversity (2)
            unique_strategies / current_pop_size, dominant_strategy_pct,
            # Configuration (4)
            current_pop_size / 500.0, current_crossover, current_mutation, current_selection
        ], dtype=np.float32)
        
        # Replace any NaN or inf values with safe defaults
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def _get_conductor_controls(self, generation: int) -> ConductorControlSignals:
        """
        Get adaptive control signals from GA Conductor
        """
        with torch.no_grad():
            state = self._create_conductor_state(generation)
            
            # Check for NaN in state
            if torch.isnan(state).any():
                print(f"  ⚠️ Warning: NaN in conductor state at gen {generation}, using defaults")
                # Use default values if state is invalid
                return ConductorControlSignals(
                    mutation_rate=0.1,
                    crossover_rate=0.7,
                    selection_pressure=0.5,
                    population_delta=0,
                    immigration_count=0,
                    culling_count=0,
                    diversity_injection=0.0,
                    extinction_trigger=0.0,
                    elite_preservation=0.1,
                    restart_signal=0.0,
                    welfare_amount=0.0,
                    tax_rate=0.0
                )
            
            outputs = self.conductor(state)
            
            # Convert to numpy
            controls = {k: v.cpu().numpy()[0] for k, v in outputs.items()}
            
            # Map to control signals  
            # Extract scalars from arrays and handle NaN
            mutation_rate = float(controls['mutation_rate'].item())
            if np.isnan(mutation_rate):
                mutation_rate = 0.1
            
            crossover_rate = float(controls['crossover_rate'].item())
            if np.isnan(crossover_rate):
                crossover_rate = 0.7
                
            selection_pressure = float(controls['selection_pressure'].item())
            if np.isnan(selection_pressure):
                selection_pressure = 0.5
                
            population_delta = int(controls['population_delta'].item()) if not np.isnan(controls['population_delta'].item()) else 0
            immigration_rate = float(controls['immigration_rate'].item()) if not np.isnan(controls['immigration_rate'].item()) else 0.0
            culling_rate = float(controls['culling_rate'].item()) if not np.isnan(controls['culling_rate'].item()) else 0.0
            diversity_injection = float(controls['diversity_injection'].item()) if not np.isnan(controls['diversity_injection'].item()) else 0.0
            extinction_trigger = float(controls['extinction_trigger'].item()) if not np.isnan(controls['extinction_trigger'].item()) else 0.0
            elite_preservation = float(controls['elite_preservation'].item()) if not np.isnan(controls['elite_preservation'].item()) else 0.1
            restart_signal = float(controls['restart_signal'].item()) if not np.isnan(controls['restart_signal'].item()) else 0.0
            welfare_amount = float(controls['welfare_amount'].item()) if not np.isnan(controls['welfare_amount'].item()) else 0.0
            tax_rate = float(controls['tax_rate'].item()) if not np.isnan(controls['tax_rate'].item()) else 0.0
            
            # Convert rates to counts
            immigration_count = max(0, int(immigration_rate * 20))  # Up to 20 immigrants
            culling_count = max(0, int(culling_rate * 20))  # Up to 20 culled
            
            return ConductorControlSignals(
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                selection_pressure=selection_pressure,
                population_delta=population_delta,
                immigration_count=immigration_count,
                culling_count=culling_count,
                diversity_injection=diversity_injection,
                extinction_trigger=extinction_trigger,
                elite_preservation=elite_preservation,
                restart_signal=restart_signal,
                welfare_amount=welfare_amount,
                tax_rate=tax_rate
            )
    
    def _genome_hash(self, genome: np.ndarray) -> str:
        """Create hash string from genome for caching"""
        # Round to 6 decimals to avoid floating point precision issues
        rounded = np.round(genome, decimals=6)
        return ','.join(f'{x:.6f}' for x in rounded)
    
    def _initialize_population(self):
        """Create initial random population"""
        self.population = [
            TradingSpecialist(
                genome=np.random.random(8),  # 8 genes
                regime_type=self.regime
            ) for _ in range(self.population_size)
        ]
        # Set initial ages
        for i, agent in enumerate(self.population):
            agent.age = 0
            agent.regime_data = self.regime_data  # Attach data
    
    def _evaluate_population(self):
        """Evaluate fitness for all agents with caching"""
        predictions = self.regime_data['predictions'].values
        for agent in self.population:
            # Check if fitness already calculated for this genome
            genome_hash = self._genome_hash(agent.genome)
            
            if genome_hash in self.fitness_cache:
                # Use cached fitness
                agent.fitness = self.fitness_cache[genome_hash]
                self.cache_hits += 1
            elif not hasattr(agent, 'fitness') or agent.fitness is None or np.isnan(agent.fitness):
                # Calculate new fitness
                self.cache_misses += 1
                try:
                    agent.fitness = agent.evaluate_fitness(self.regime_data, predictions)
                    # If fitness is NaN, inf, or negative infinity, set to very low value
                    if np.isnan(agent.fitness) or np.isinf(agent.fitness):
                        agent.fitness = -1000.0
                    else:
                        # Cache the valid fitness
                        self.fitness_cache[genome_hash] = agent.fitness
                except Exception as e:
                    # If evaluation fails, set very low fitness
                    agent.fitness = -1000.0
                    self.fitness_cache[genome_hash] = -1000.0
    
    def _get_population_stats(self) -> Dict:
        """Calculate population statistics"""
        # Get fitnesses, filter out NaN/None values
        fitnesses = []
        for agent in self.population:
            if hasattr(agent, 'fitness') and agent.fitness is not None:
                if not np.isnan(agent.fitness) and not np.isinf(agent.fitness):
                    fitnesses.append(agent.fitness)
        
        # If no valid fitnesses, use defaults
        if not fitnesses:
            return {
                'avg_fitness': 0.0,
                'best_fitness': 0.0,
                'worst_fitness': 0.0,
                'std_fitness': 0.0,
                'diversity': 0.0
            }
        
        # Diversity: average pairwise distance
        diversity = 0.0
        if len(self.population) > 1:
            distances = []
            for i, agent1 in enumerate(self.population):
                for agent2 in self.population[i+1:]:
                    dist = np.sum(np.abs(np.array(agent1.genome) - np.array(agent2.genome)))
                    distances.append(dist)
            diversity = np.mean(distances) if distances else 0.0
        
        return {
            'avg_fitness': float(np.mean(fitnesses)),
            'best_fitness': float(np.max(fitnesses)),
            'worst_fitness': float(np.min(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'diversity': float(diversity)
        }
    
    def _tournament_selection(self, tournament_size: int) -> TradingSpecialist:
        """Tournament selection"""
        tournament = np.random.choice(self.population, size=tournament_size, replace=False)
        # Handle agents without fitness (shouldn't happen, but safety check)
        return max(tournament, key=lambda x: x.fitness if hasattr(x, 'fitness') and x.fitness is not None else -1e10)
    
    def _crossover(self, parent1: TradingSpecialist, parent2: TradingSpecialist) -> TradingSpecialist:
        """Single-point crossover"""
        point = np.random.randint(1, len(parent1.genome))
        child_genome = np.concatenate([parent1.genome[:point], parent2.genome[point:]])
        
        child = TradingSpecialist(
            genome=child_genome,
            regime_type=self.regime
        )
        child.age = 0
        child.regime_data = self.regime_data
        return child
    
    def _mutate(self, agent: TradingSpecialist, mutation_rate: float):
        """Mutate agent genome"""
        genome = agent.genome.copy()
        
        for i in range(len(genome)):
            if np.random.random() < mutation_rate:
                genome[i] += np.random.uniform(-0.1, 0.1)
                # Clip to valid range [0, 1]
                genome[i] = np.clip(genome[i], 0.0, 1.0)
        
        agent.genome = genome
        agent.fitness = None  # Mark for re-evaluation
    
    def train(self) -> Dict:
        """
        Main training loop with conductor-enhanced adaptive control
        """
        print(f"\n{'='*60}")
        print(f"Training {self.regime.upper()} Specialist (Conductor-Enhanced)")
        print(f"Population: {self.population_size} | Generations: {self.generations}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Initialize
        self._initialize_population()
        self._evaluate_population()
        
        # Training loop
        for gen in range(self.generations):
            # Get adaptive controls from conductor
            controls = self._get_conductor_controls(gen)
            
            # Evaluate population
            stats = self._get_population_stats()
            
            # Track best (filter valid fitnesses)
            valid_agents = [a for a in self.population 
                           if hasattr(a, 'fitness') and a.fitness is not None 
                           and not np.isnan(a.fitness) and not np.isinf(a.fitness)]
            
            if valid_agents:
                current_best = max(valid_agents, key=lambda x: x.fitness)
                if current_best.fitness > self.best_fitness:
                    self.best_fitness = current_best.fitness
                    self.best_agent = current_best
                    self.time_since_improvement = 0
                    self.stagnation_count = 0
                else:
                    self.time_since_improvement += 1
                    if self.time_since_improvement > 10:
                        self.stagnation_count += 1
            else:
                self.time_since_improvement += 1
                if self.time_since_improvement > 10:
                    self.stagnation_count += 1
            
            # Log progress
            if gen % 10 == 0:
                print(f"Gen {gen:3d} | Best: {stats['best_fitness']:7.2f} | "
                      f"Avg: {stats['avg_fitness']:7.2f} | Diversity: {stats['diversity']:6.2f} | "
                      f"M: {controls.mutation_rate:.3f} | C: {controls.crossover_rate:.3f}")
            
            # Record history
            self.history['generation'].append(gen)
            self.history['best_fitness'].append(stats['best_fitness'])
            self.history['avg_fitness'].append(stats['avg_fitness'])
            self.history['worst_fitness'].append(stats['worst_fitness'])
            self.history['std_fitness'].append(stats['std_fitness'])
            self.history['diversity'].append(stats['diversity'])
            self.history['time_since_improvement'].append(self.time_since_improvement)
            self.history['stagnation_count'].append(self.stagnation_count)
            
            # Record conductor controls
            self.history['mutation_rate'].append(controls.mutation_rate)
            self.history['crossover_rate'].append(controls.crossover_rate)
            self.history['selection_pressure'].append(controls.selection_pressure)
            self.history['population_delta'].append(controls.population_delta)
            self.history['immigration'].append(controls.immigration_count)
            self.history['culling'].append(controls.culling_count)
            self.history['diversity_injection'].append(controls.diversity_injection)
            self.history['extinction_trigger'].append(controls.extinction_trigger)
            self.history['elite_preservation'].append(controls.elite_preservation)
            self.history['restart_signal'].append(controls.restart_signal)
            self.history['welfare'].append(controls.welfare_amount)
            self.history['tax'].append(controls.tax_rate)
            
            # Population metrics
            ages = [getattr(agent, 'age', 0) for agent in self.population]
            self.history['population_size'].append(len(self.population))
            self.history['avg_age'].append(np.mean(ages))
            self.history['oldest_age'].append(max(ages))
            self.history['young_pct'].append(sum(1 for age in ages if age < 10) / len(ages))
            
            # Calculate Gini coefficient from valid fitnesses only
            valid_fitnesses = []
            for agent in self.population:
                if hasattr(agent, 'fitness') and agent.fitness is not None:
                    if not np.isnan(agent.fitness) and not np.isinf(agent.fitness):
                        valid_fitnesses.append(agent.fitness)
            
            if not valid_fitnesses or len(valid_fitnesses) < 2:
                gini = 0.0
            else:
                sorted_fitness = np.sort(valid_fitnesses)
                n = len(sorted_fitness)
                cumsum = np.cumsum(sorted_fitness)
                if cumsum[-1] != 0:
                    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_fitness)) / (n * cumsum[-1]) - (n + 1) / n
                else:
                    gini = 0.0
            self.history['wealth_gini'].append(gini)
            
            # Create next generation with adaptive parameters
            next_generation = []
            
            # Elitism (ensure all have fitness before sorting)
            elite_count = max(1, int(len(self.population) * controls.elite_preservation))
            elite = sorted(self.population, 
                          key=lambda x: x.fitness if hasattr(x, 'fitness') and x.fitness is not None else -1e10, 
                          reverse=True)[:elite_count]
            for agent in elite:
                if hasattr(agent, 'age'):
                    agent.age += 1
                else:
                    agent.age = 1
            next_generation.extend(elite)
            
            # Fill rest with crossover and mutation
            while len(next_generation) < self.population_size:
                if np.random.random() < controls.crossover_rate:
                    parent1 = self._tournament_selection(self.tournament_size)
                    parent2 = self._tournament_selection(self.tournament_size)
                    child = self._crossover(parent1, parent2)
                else:
                    # Clone from tournament winner
                    parent = self._tournament_selection(self.tournament_size)
                    child = TradingSpecialist(
                        genome=parent.genome.copy(),
                        regime_type=self.regime
                    )
                    child.age = 0
                    child.regime_data = self.regime_data
                
                self._mutate(child, controls.mutation_rate)
                next_generation.append(child)
            
            # Immigration (diversity injection)
            if controls.immigration_count > 0 and np.random.random() < controls.diversity_injection:
                num_immigrants = min(controls.immigration_count, len(next_generation) // 10)
                for _ in range(num_immigrants):
                    immigrant = TradingSpecialist(
                        genome=np.random.random(8),
                        regime_type=self.regime
                    )
                    immigrant.age = 0
                    immigrant.regime_data = self.regime_data
                    next_generation.append(immigrant)
            
            # Culling (remove worst performers)
            if controls.culling_count > 0 and len(next_generation) > self.population_size:
                next_generation = sorted(next_generation, 
                                        key=lambda x: x.fitness if hasattr(x, 'fitness') and x.fitness is not None else -1e10, 
                                        reverse=True)
                next_generation = next_generation[:self.population_size]
            
            # Crisis response (extinction event)
            if controls.extinction_trigger > 0.9 and self.stagnation_count > 20:
                print(f"  🔥 EXTINCTION EVENT triggered at gen {gen}!")
                # Keep only elite, reset rest
                keep_count = max(5, int(len(next_generation) * controls.elite_preservation))
                # Filter out agents with NaN fitness and sort safely
                valid_agents = [a for a in next_generation if hasattr(a, 'fitness') and a.fitness is not None and not np.isnan(a.fitness)]
                if not valid_agents:
                    print(f"  ⚠️ No valid agents! Restarting with random population...")
                    elite = []
                else:
                    elite = sorted(valid_agents, key=lambda x: x.fitness, reverse=True)[:keep_count]
                
                next_generation = elite.copy()
                while len(next_generation) < self.population_size:
                    new_agent = TradingSpecialist(
                        genome=np.random.random(8),
                        regime_type=self.regime
                    )
                    new_agent.age = 0
                    new_agent.regime_data = self.regime_data
                    next_generation.append(new_agent)
                
                self.stagnation_count = 0
            
            self.population = next_generation
            self._evaluate_population()
        
        # Final evaluation
        self._evaluate_population()
        final_best = max(self.population, key=lambda x: x.fitness if hasattr(x, 'fitness') and x.fitness is not None else -1e10)
        if hasattr(final_best, 'fitness') and final_best.fitness is not None and final_best.fitness > self.best_fitness:
            self.best_agent = final_best
            self.best_fitness = final_best.fitness
        
        # Re-evaluate best agent to get full metrics
        if self.best_agent:
            predictions = self.regime_data['predictions'].values
            self.best_agent.fitness = self.best_agent.evaluate_fitness(self.regime_data, predictions)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Fitness: {self.best_fitness:.2f}")
        
        # Cache statistics
        total_evaluations = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_evaluations * 100) if total_evaluations > 0 else 0
        print(f"\nFitness Cache Statistics:")
        print(f"  Cache Hits: {self.cache_hits:,}")
        print(f"  Cache Misses: {self.cache_misses:,}")
        print(f"  Hit Rate: {cache_hit_rate:.1f}%")
        print(f"  Evaluations Saved: {self.cache_hits:,}")
        
        if self.best_agent and hasattr(self.best_agent, 'total_return'):
            print(f"\nBest Agent Performance:")
            print(f"  Return: {self.best_agent.total_return:+.2f}%")
            print(f"  Sharpe: {self.best_agent.sharpe_ratio:.2f}")
            print(f"  Max DD: {self.best_agent.max_drawdown:.2f}%")
            print(f"  Trades: {self.best_agent.num_trades}")
        print(f"{'='*60}\n")
        
        return self.history


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_predictions(df: pd.DataFrame, regime_type: str) -> np.ndarray:
    """Generate trading predictions based on regime type"""
    df = df.copy()
    
    # Calculate indicators
    df['returns'] = df['close'].pct_change()
    df['sma_short'] = df['close'].rolling(5).mean()
    df['sma_long'] = df['close'].rolling(20).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    if regime_type == 'volatile' or regime_type == 'trending':
        # Momentum-based predictions
        crossover = (df['sma_short'] - df['sma_long']) / df['close']
        momentum = df['returns'].rolling(5).mean()
        predictions = (crossover * 5 + momentum * 10).fillna(0).values
        
    elif regime_type == 'ranging':
        # Mean-reversion predictions
        rsi_signal = (50 - df['rsi']) / 50
        price_dev = (df['close'] - df['sma_long']) / df['sma_long']
        predictions = (rsi_signal * 3 - price_dev * 5).fillna(0).values
        
    else:  # crisis
        # Very conservative
        strong_momentum = df['returns'].rolling(3).mean()
        volatility = df['returns'].rolling(10).std()
        predictions = (strong_momentum * 10 * (1 / (1 + volatility))).fillna(0).values
    
    return predictions


def main():
    """Train conductor-enhanced specialist and save results"""
    import sys
    
    # Get regime from command line argument, default to volatile
    regime = sys.argv[1] if len(sys.argv) > 1 else 'volatile'
    
    # Validate regime
    valid_regimes = ['volatile', 'trending', 'ranging']
    if regime not in valid_regimes:
        print(f"❌ ERROR: Invalid regime '{regime}'. Must be one of: {valid_regimes}")
        return
    
    # Load regime-specific data
    print(f"Loading {regime} regime data...")
    data_path = f'DATA/yf_btc_1d_{regime}.csv'
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} days of {regime} market data")
    except FileNotFoundError:
        print(f"❌ ERROR: {data_path} not found!")
        return
    
    # Add required indicators
    if 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()
    if 'atr' not in df.columns:
        high_low = df['high'] - df['low']
        df['atr'] = high_low.rolling(14).mean()
    
    # Generate predictions
    predictions = generate_predictions(df, regime)
    df['predictions'] = predictions
    regime_data = df
    
    trainer = ConductorEnhancedTrainer(
        regime=regime,
        regime_data=regime_data,
        population_size=200,
        generations=300
    )
    
    history = trainer.train()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"outputs/conductor_enhanced_{regime}_{timestamp}.json"
    
    # Prepare results dict with safety checks
    results = {
        'regime': regime,
        'conductor_enhanced': True,
        'population_size': trainer.population_size,
        'generations': trainer.generations,
        'best_fitness': float(trainer.best_fitness),
    }
    
    # Only add best_agent details if properly evaluated
    if trainer.best_agent and hasattr(trainer.best_agent, 'total_return'):
        results['best_agent'] = {
            'genome': trainer.best_agent.genome.tolist() if hasattr(trainer.best_agent.genome, 'tolist') else list(trainer.best_agent.genome),
            'fitness': float(trainer.best_agent.fitness),
            'total_return': float(trainer.best_agent.total_return),
            'sharpe_ratio': float(trainer.best_agent.sharpe_ratio),
            'max_drawdown': float(trainer.best_agent.max_drawdown),
            'num_trades': int(trainer.best_agent.num_trades),
            'win_rate': float(trainer.best_agent.win_rate)
        }
    else:
        results['best_agent'] = {
            'genome': trainer.best_agent.genome.tolist() if hasattr(trainer.best_agent.genome, 'tolist') else list(trainer.best_agent.genome),
            'fitness': float(trainer.best_fitness)
        }
    
    results['training_history'] = history
    results['timestamp'] = timestamp
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
