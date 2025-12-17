"""
Neuroevolution for SNN Trading Agents
NEAT-style evolution of SNN topologies
Combines genetic algorithms with spiking neural networks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from typing import List, Dict, Tuple
from pathlib import Path
import json
from datetime import datetime

from models.snn_trading_agent import SpikingTradingAgent


class SNNGenome:
    """Genome encoding SNN architecture and parameters."""
    
    def __init__(
        self,
        input_dim: int = 50,
        initial_hidden_dim: int = 100,
        initial_pathways: int = 2
    ):
        self.input_dim = input_dim
        self.hidden_dim = initial_hidden_dim
        self.num_pathways = initial_pathways
        
        # Genome: mutation-able parameters
        self.genes = {
            'hidden_dim': initial_hidden_dim,
            'num_pathways': initial_pathways,
            'threshold': 1.0,
            'decay': 0.9,
            'learning_rate': 0.001
        }
        
        self.fitness = 0.0
        self.id = np.random.randint(0, 1000000)
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate genome parameters."""
        if np.random.random() < mutation_rate:
            # Mutate hidden_dim
            self.genes['hidden_dim'] = max(
                50,
                int(self.genes['hidden_dim'] + np.random.randint(-20, 21))
            )
        
        if np.random.random() < mutation_rate:
            # Mutate num_pathways
            self.genes['num_pathways'] = max(
                1,
                self.genes['num_pathways'] + np.random.choice([-1, 0, 1])
            )
        
        if np.random.random() < mutation_rate:
            # Mutate threshold
            self.genes['threshold'] = max(
                0.5,
                min(2.0, self.genes['threshold'] + np.random.randn() * 0.1)
            )
        
        if np.random.random() < mutation_rate:
            # Mutate decay
            self.genes['decay'] = max(
                0.5,
                min(0.99, self.genes['decay'] + np.random.randn() * 0.05)
            )
        
        if np.random.random() < mutation_rate:
            # Mutate learning rate
            self.genes['learning_rate'] = max(
                0.0001,
                min(0.01, self.genes['learning_rate'] * np.random.uniform(0.5, 2.0))
            )
    
    def crossover(self, other: 'SNNGenome') -> 'SNNGenome':
        """Create offspring through crossover."""
        offspring = SNNGenome(self.input_dim)
        
        # Mix genes from both parents
        for gene_name in self.genes:
            if np.random.random() < 0.5:
                offspring.genes[gene_name] = self.genes[gene_name]
            else:
                offspring.genes[gene_name] = other.genes[gene_name]
        
        return offspring
    
    def create_agent(self, device: torch.device) -> SpikingTradingAgent:
        """Create SNN agent from genome."""
        agent = SpikingTradingAgent(
            input_dim=self.input_dim,
            hidden_dim=self.genes['hidden_dim'],
            output_dim=1,
            num_pathways=self.genes['num_pathways'],
            threshold=self.genes['threshold'],
            decay=self.genes['decay']
        ).to(device)
        
        return agent


class SNNNeuroevolution:
    """Evolutionary algorithm for SNN trading agents."""
    
    def __init__(
        self,
        population_size: int = 20,
        input_dim: int = 50,
        device: torch.device = None
    ):
        self.population_size = population_size
        self.input_dim = input_dim
        self.device = device or torch.device('cpu')
        
        # Initialize population
        self.population = [
            SNNGenome(input_dim)
            for _ in range(population_size)
        ]
        
        self.generation = 0
        self.best_fitness_history = []
    
    def evaluate_fitness(
        self,
        genome: SNNGenome,
        features: np.ndarray,
        returns: np.ndarray
    ) -> float:
        """Evaluate genome fitness on trading task."""
        # Create agent
        agent = genome.create_agent(self.device)
        
        # Simple training
        optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=genome.genes['learning_rate']
        )
        
        batch_size = 32
        num_epochs = 5  # Quick training for evolution
        
        for epoch in range(num_epochs):
            for i in range(0, len(features) - batch_size, batch_size):
                inputs = torch.FloatTensor(features[i:i+batch_size]).to(self.device)
                batch_returns = returns[i:i+batch_size]
                
                outputs, _ = agent(inputs, num_steps=10)
                decisions = torch.tanh(outputs.squeeze())
                
                batch_returns_tensor = torch.FloatTensor(batch_returns).to(self.device)
                fitness = (decisions * batch_returns_tensor).mean()
                
                loss = -fitness
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate final fitness
        agent.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(features).to(self.device)
            outputs, _ = agent(inputs, num_steps=10)
            decisions = torch.tanh(outputs.squeeze())
            
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            total_fitness = (decisions * returns_tensor).sum().item()
        
        return total_fitness
    
    def evolve_generation(
        self,
        features: np.ndarray,
        returns: np.ndarray
    ):
        """Evolve population for one generation."""
        print(f"\nGeneration {self.generation + 1}")
        
        # Evaluate fitness
        for i, genome in enumerate(self.population):
            genome.fitness = self.evaluate_fitness(genome, features, returns)
            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i + 1}/{self.population_size} genomes")
        
        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        best_fitness = self.population[0].fitness
        avg_fitness = np.mean([g.fitness for g in self.population])
        
        self.best_fitness_history.append(best_fitness)
        
        print(f"  Best fitness: {best_fitness:.4f}")
        print(f"  Avg fitness: {avg_fitness:.4f}")
        
        # Selection and reproduction
        # Keep top 20%
        elite_size = max(2, self.population_size // 5)
        elite = self.population[:elite_size]
        
        # Generate offspring
        offspring = []
        while len(offspring) < self.population_size - elite_size:
            # Tournament selection
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()
            
            # Crossover
            child = parent1.crossover(parent2)
            
            # Mutation
            child.mutate(mutation_rate=0.2)
            
            offspring.append(child)
        
        # New population
        self.population = elite + offspring
        self.generation += 1
    
    def tournament_select(self, tournament_size: int = 3) -> SNNGenome:
        """Tournament selection."""
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        return max(tournament, key=lambda g: g.fitness)
    
    def get_best_genome(self) -> SNNGenome:
        """Get best genome from current population."""
        return max(self.population, key=lambda g: g.fitness)


def main():
    """Main neuroevolution execution."""
    print("=" * 80)
    print("SNN NEUROEVOLUTION")
    print("=" * 80)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 50
    population_size = 10  # Small for demo
    num_generations = 5
    num_samples = 500  # Smaller dataset for faster evolution
    
    # Load mock data
    print("\nGenerating training data...")
    np.random.seed(42)
    features = np.random.randn(num_samples, input_dim) * 10 + 100
    prices = 100 + np.cumsum(np.random.randn(num_samples) * 0.5)
    returns = np.diff(prices)
    returns = np.concatenate([[0], returns])
    
    # Create neuroevolution
    print(f"\nInitializing population of {population_size} SNNs...")
    neuroevo = SNNNeuroevolution(
        population_size=population_size,
        input_dim=input_dim,
        device=device
    )
    
    # Evolve
    print("\nEvolving SNNs...")
    for gen in range(num_generations):
        neuroevo.evolve_generation(features, returns)
    
    # Get best genome
    best_genome = neuroevo.get_best_genome()
    
    print("\n" + "=" * 80)
    print("EVOLUTION COMPLETE")
    print("=" * 80)
    
    print(f"\nBest Genome:")
    print(f"  Hidden dim: {best_genome.genes['hidden_dim']}")
    print(f"  Num pathways: {best_genome.genes['num_pathways']}")
    print(f"  Threshold: {best_genome.genes['threshold']:.3f}")
    print(f"  Decay: {best_genome.genes['decay']:.3f}")
    print(f"  Fitness: {best_genome.fitness:.4f}")
    
    # Save results
    results = {
        'num_generations': num_generations,
        'population_size': population_size,
        'best_genome': best_genome.genes,
        'best_fitness': float(best_genome.fitness),
        'fitness_history': [float(f) for f in neuroevo.best_fitness_history]
    }
    
    output_dir = Path("outputs/ga_trading_agents")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"neuroevolution_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Check improvement
    if len(neuroevo.best_fitness_history) > 1:
        improvement = ((neuroevo.best_fitness_history[-1] - neuroevo.best_fitness_history[0]) / 
                      abs(neuroevo.best_fitness_history[0]) * 100)
        print(f"\nFitness improvement: {improvement:.2f}%")


if __name__ == "__main__":
    main()
