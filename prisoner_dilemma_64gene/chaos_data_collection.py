"""
🌊 Chaos Analysis Data Collection for 64-Gene Prisoner's Dilemma
================================================================

Generates 8,000-10,000 data points for chaos theory analysis.

Data collected:
1. Fitness trajectories across many evolution runs
2. Gene frequency dynamics (64 dimensions)
3. Strategy similarity measures over time
4. Population diversity metrics
5. Gene-level mutation patterns
6. Pairwise interaction outcomes

This generates the time-series data needed for:
- Lyapunov exponent calculation
- Attractor reconstruction
- Bifurcation analysis
- Entropy measures
- Correlation dimension
"""

import numpy as np
import random
import json
from prisoner_64gene import (
    AdvancedPrisonerAgent, 
    play_prisoner_dilemma,
    create_random_chromosome,
    create_tit_for_tat
)
from datetime import datetime

class ChaosDataCollector:
    """Collect extensive time-series data for chaos analysis"""
    
    def __init__(self, population_size=50, mutation_rate=0.01):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.generation = 0
        
        # Data storage for chaos analysis
        self.fitness_time_series = []  # Best fitness per generation
        self.gene_frequencies = []  # 64-dim vector per generation
        self.diversity_metrics = []  # Various diversity measures
        self.strategy_embeddings = []  # PCA-like representation
        self.mutation_events = []  # Track all mutations
        self.interaction_matrix = []  # Pairwise game outcomes
        
    def initialize_population(self):
        """Create initial random population"""
        self.population = [
            AdvancedPrisonerAgent(i, create_random_chromosome()) 
            for i in range(self.population_size)
        ]
        
    def evaluate_fitness(self):
        """Round-robin tournament"""
        for agent in self.population:
            agent.fitness = 0
        
        # Store full interaction matrix for chaos analysis
        interaction_scores = np.zeros((self.population_size, self.population_size))
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                score1, score2 = play_prisoner_dilemma(
                    self.population[i], 
                    self.population[j],
                    rounds=30
                )
                self.population[i].fitness += score1
                self.population[j].fitness += score2
                
                interaction_scores[i, j] = score1
                interaction_scores[j, i] = score2
        
        return interaction_scores
    
    def calculate_gene_frequencies(self):
        """Calculate frequency of 'C' at each of 64 positions"""
        frequencies = np.zeros(64)
        for agent in self.population:
            for i in range(64):
                if agent.chromosome[i] == 'C':
                    frequencies[i] += 1
        return frequencies / self.population_size
    
    def calculate_diversity(self):
        """Multiple diversity metrics"""
        metrics = {}
        
        # 1. Fitness diversity (standard deviation)
        fitnesses = [a.fitness for a in self.population]
        metrics['fitness_std'] = np.std(fitnesses)
        metrics['fitness_mean'] = np.mean(fitnesses)
        metrics['fitness_max'] = np.max(fitnesses)
        metrics['fitness_min'] = np.min(fitnesses)
        
        # 2. Genotypic diversity (average pairwise Hamming distance)
        total_distance = 0
        count = 0
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = sum(
                    1 for k in range(64) 
                    if self.population[i].chromosome[k] != self.population[j].chromosome[k]
                )
                total_distance += distance
                count += 1
        metrics['avg_hamming_distance'] = total_distance / count if count > 0 else 0
        
        # 3. Shannon entropy of gene frequencies
        gene_freqs = self.calculate_gene_frequencies()
        entropy = 0
        for freq in gene_freqs:
            if 0 < freq < 1:
                entropy += -freq * np.log2(freq) - (1-freq) * np.log2(1-freq)
        metrics['gene_entropy'] = entropy / 64  # Normalize
        
        # 4. Number of unique strategies
        unique_strategies = len(set(a.chromosome for a in self.population))
        metrics['unique_strategies'] = unique_strategies
        
        return metrics
    
    def calculate_strategy_embedding(self):
        """Create low-dimensional representation of population"""
        # Use first 10 principal components (simplified PCA)
        # Just take first 10 gene frequencies as proxy
        gene_freqs = self.calculate_gene_frequencies()
        return gene_freqs[:10].tolist()
    
    def select_parents(self):
        """Tournament selection"""
        tournament_size = 5
        selected = []
        
        for _ in range(2):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected[0], selected[1]
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if random.random() < 0.7:
            point = random.randint(1, 63)
            child_genes = parent1.chromosome[:point] + parent2.chromosome[point:]
        else:
            child_genes = parent1.chromosome
        
        child = AdvancedPrisonerAgent(
            agent_id=random.randint(10000, 99999),
            chromosome=child_genes
        )
        return child
    
    def mutate(self, agent):
        """Mutate and track mutations"""
        chrom_list = list(agent.chromosome)
        mutations = []
        
        for i in range(64):
            if random.random() < self.mutation_rate:
                old_gene = chrom_list[i]
                chrom_list[i] = 'D' if chrom_list[i] == 'C' else 'C'
                mutations.append({
                    'position': i,
                    'from': old_gene,
                    'to': chrom_list[i],
                    'generation': self.generation
                })
        
        agent.chromosome = "".join(chrom_list)
        return mutations
    
    def evolve_generation(self):
        """Evolve one generation and collect ALL data"""
        self.generation += 1
        
        # Evaluate fitness and get interaction matrix
        interaction_matrix = self.evaluate_fitness()
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Collect data BEFORE evolution
        best_fitness = self.population[0].fitness
        gene_freqs = self.calculate_gene_frequencies()
        diversity = self.calculate_diversity()
        embedding = self.calculate_strategy_embedding()
        
        # Store data
        self.fitness_time_series.append(best_fitness)
        self.gene_frequencies.append(gene_freqs.tolist())
        self.diversity_metrics.append(diversity)
        self.strategy_embeddings.append(embedding)
        self.interaction_matrix.append(interaction_matrix.tolist())
        
        # Elitism
        elite_size = 5
        new_population = self.population[:elite_size].copy()
        
        # Generate offspring with mutation tracking
        generation_mutations = []
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            mutations = self.mutate(child)
            generation_mutations.extend(mutations)
            new_population.append(child)
        
        self.mutation_events.append({
            'generation': self.generation,
            'mutations': generation_mutations,
            'mutation_count': len(generation_mutations)
        })
        
        self.population = new_population
        
        return best_fitness, diversity
    
    def print_progress(self, best_fitness, diversity):
        """Print progress every 50 generations"""
        if self.generation % 50 == 0:
            print(f"  Gen {self.generation:4d}: "
                  f"Fitness={int(best_fitness):5d}, "
                  f"Diversity={diversity['avg_hamming_distance']:.1f}, "
                  f"Unique={diversity['unique_strategies']:2d}")

def run_extended_collection(num_runs=100, generations_per_run=100):
    """
    Run multiple evolution experiments to collect 8,000-10,000 data points.
    
    Each run generates:
    - 100 fitness points
    - 100 x 64 gene frequency points
    - 100 diversity metric sets
    
    100 runs x 100 generations = 10,000 data points!
    """
    print("\n" + "="*70)
    print("🌊 CHAOS ANALYSIS DATA COLLECTION")
    print("="*70)
    print(f"\nGenerating data for chaos theory analysis...")
    print(f"  Runs: {num_runs}")
    print(f"  Generations per run: {generations_per_run}")
    print(f"  Expected data points: {num_runs * generations_per_run:,}")
    print("="*70)
    
    all_data = {
        'metadata': {
            'num_runs': num_runs,
            'generations_per_run': generations_per_run,
            'population_size': 50,
            'mutation_rate': 0.01,
            'timestamp': datetime.now().isoformat(),
            'total_generations': num_runs * generations_per_run
        },
        'runs': []
    }
    
    for run_idx in range(num_runs):
        print(f"\n🧬 Run {run_idx + 1}/{num_runs}")
        print("-" * 70)
        
        collector = ChaosDataCollector(population_size=50, mutation_rate=0.01)
        collector.initialize_population()
        
        for gen in range(generations_per_run):
            best_fitness, diversity = collector.evolve_generation()
            collector.print_progress(best_fitness, diversity)
        
        # Store run data
        run_data = {
            'run_id': run_idx,
            'fitness_trajectory': collector.fitness_time_series,
            'gene_frequency_matrix': collector.gene_frequencies,
            'diversity_history': collector.diversity_metrics,
            'strategy_embeddings': collector.strategy_embeddings,
            'mutation_summary': {
                'total_mutations': sum(m['mutation_count'] for m in collector.mutation_events),
                'mutations_per_generation': [m['mutation_count'] for m in collector.mutation_events]
            }
        }
        
        all_data['runs'].append(run_data)
        
        # Progress summary
        final_fitness = collector.fitness_time_series[-1]
        final_diversity = collector.diversity_metrics[-1]
        print(f"\n  ✅ Run {run_idx + 1} complete:")
        print(f"     Final fitness: {int(final_fitness)}")
        print(f"     Final diversity: {final_diversity['avg_hamming_distance']:.1f}")
        print(f"     Total mutations: {run_data['mutation_summary']['total_mutations']}")
    
    # Save complete dataset
    filename = f"chaos_dataset_{num_runs}runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    print("\n" + "="*70)
    print("💾 Saving dataset...")
    print("="*70)
    
    with open(filename, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"✅ Saved: {filename}")
    
    # Generate summary statistics
    print("\n" + "="*70)
    print("📊 DATASET SUMMARY")
    print("="*70)
    
    total_data_points = num_runs * generations_per_run
    total_gene_vectors = total_data_points * 64
    total_mutations = sum(
        run['mutation_summary']['total_mutations'] 
        for run in all_data['runs']
    )
    
    print(f"\n✅ Data points collected: {total_data_points:,}")
    print(f"   - Fitness values: {total_data_points:,}")
    print(f"   - Gene frequency vectors (64-dim): {total_data_points:,}")
    print(f"   - Diversity metric sets: {total_data_points:,}")
    print(f"   - Strategy embeddings (10-dim): {total_data_points:,}")
    print(f"\n✅ Total gene measurements: {total_gene_vectors:,}")
    print(f"✅ Total mutation events: {total_mutations:,}")
    
    # Analyze fitness convergence patterns
    converged_runs = 0
    diverged_runs = 0
    oscillating_runs = 0
    
    for run in all_data['runs']:
        trajectory = run['fitness_trajectory']
        final_third = trajectory[-33:]  # Last 1/3 of run
        
        if np.std(final_third) < 100:
            converged_runs += 1
        elif np.std(final_third) > 500:
            diverged_runs += 1
        else:
            oscillating_runs += 1
    
    print(f"\n📈 Evolution patterns:")
    print(f"   - Converged: {converged_runs}/{num_runs} ({converged_runs/num_runs*100:.1f}%)")
    print(f"   - Oscillating: {oscillating_runs}/{num_runs} ({oscillating_runs/num_runs*100:.1f}%)")
    print(f"   - Diverged: {diverged_runs}/{num_runs} ({diverged_runs/num_runs*100:.1f}%)")
    
    print("\n" + "="*70)
    print("🌊 READY FOR CHAOS ANALYSIS")
    print("="*70)
    print(f"\nDataset: {filename}")
    print("\nSuggested chaos analyses:")
    print("  1. Lyapunov exponents (fitness trajectories)")
    print("  2. Attractor reconstruction (gene frequency space)")
    print("  3. Correlation dimension (embedding vectors)")
    print("  4. Entropy measures (diversity metrics)")
    print("  5. Bifurcation analysis (mutation rate parameter sweep)")
    print("="*70)
    
    return all_data, filename

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 CHAOS DATA COLLECTION EXPERIMENT")
    print("="*70)
    print("\nThis will generate 10,000 data points from evolutionary dynamics.")
    print("Estimated time: 30-45 minutes")
    print("\nData collected:")
    print("  • Fitness time series")
    print("  • 64-dimensional gene frequency dynamics")
    print("  • Population diversity metrics")
    print("  • Mutation event tracking")
    print("\nReady for chaos theory analysis:")
    print("  • Lyapunov exponents")
    print("  • Strange attractors")
    print("  • Fractal dimensions")
    print("="*70)
    
    # Run experiment
    data, filename = run_extended_collection(num_runs=100, generations_per_run=100)
    
    print("\n" + "="*70)
    print("🎉 DATA COLLECTION COMPLETE!")
    print("="*70)
    print(f"\nNext step: Load '{filename}' into chaos analysis pipeline")
    print("="*70)
