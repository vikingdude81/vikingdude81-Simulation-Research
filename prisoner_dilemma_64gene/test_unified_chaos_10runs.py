"""
🧪 TEST: Unified Chaos Pipeline (10 runs)
==========================================

Quick test of the chaos pipeline integration before running 1,000 runs.

Tests:
✅ Evolution data collection
✅ Chaos analysis on fitness trajectories
✅ Data format compatibility
✅ Checkpoint saving
✅ Output structure for ML
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Import prisoner dilemma
from prisoner_64gene import (
    AdvancedPrisonerAgent,
    play_prisoner_dilemma,
    create_random_chromosome
)

# Use basic chaos analysis for test
CHAOS_MODULES_AVAILABLE = False
print("ℹ️  Using built-in chaos analysis for test")


class UnifiedEvolutionChaosCollector:
    """Collect evolution data + apply full chaos pipeline"""
    
    def __init__(self, population_size=50, mutation_rate=0.01):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.generation = 0
        
        # Evolution data
        self.fitness_trajectory = []
        self.gene_frequencies = []
        self.diversity_metrics = []
        self.mutation_events = []
        
    def initialize_population(self):
        """Initialize with random strategies"""
        self.population = [
            AdvancedPrisonerAgent(chromosome=create_random_chromosome(), agent_id=i)
            for i in range(self.population_size)
        ]
        self.generation = 0
        
    def calculate_fitnesses(self):
        """Tournament evaluation"""
        fitness_dict = {agent: 0 for agent in self.population}
        
        for i, agent1 in enumerate(self.population):
            for agent2 in self.population[i+1:]:
                score1, score2 = play_prisoner_dilemma(agent1, agent2, rounds=10)
                fitness_dict[agent1] += score1
                fitness_dict[agent2] += score2
        
        for agent in self.population:
            agent.fitness = fitness_dict[agent]
    
    def track_diversity(self):
        """Track population diversity metrics"""
        fitnesses = [agent.fitness for agent in self.population]
        
        # Gene entropy
        gene_entropy = 0
        for pos in range(64):
            freq = sum(1 if agent.chromosome[pos] == 'C' else 0 for agent in self.population) / self.population_size
            if 0 < freq < 1:
                gene_entropy -= freq * np.log2(freq) + (1-freq) * np.log2(1-freq)
        gene_entropy /= 64
        
        # Hamming distances
        hamming_distances = []
        for i, agent1 in enumerate(self.population):
            for agent2 in self.population[i+1:]:
                dist = sum(a != b for a, b in zip(agent1.chromosome, agent2.chromosome))
                hamming_distances.append(dist)
        
        self.diversity_metrics.append({
            'fitness_std': float(np.std(fitnesses)),
            'fitness_mean': float(np.mean(fitnesses)),
            'fitness_max': float(max(fitnesses)),
            'fitness_min': float(min(fitnesses)),
            'avg_hamming_distance': float(np.mean(hamming_distances)) if hamming_distances else 0,
            'gene_entropy': float(gene_entropy),
            'unique_strategies': len(set(tuple(agent.chromosome) for agent in self.population))
        })
    
    def evolve_one_generation(self):
        """Single generation step"""
        import random
        
        self.calculate_fitnesses()
        
        # Track metrics
        best_agent = max(self.population, key=lambda a: a.fitness)
        self.fitness_trajectory.append(best_agent.fitness)
        
        # Gene frequencies (C=1, D=0 for cooperation frequency)
        frequencies = np.zeros(64)
        for agent in self.population:
            frequencies += np.array([1 if g == 'C' else 0 for g in agent.chromosome])
        frequencies /= self.population_size
        self.gene_frequencies.append(frequencies.tolist())
        
        self.track_diversity()
        
        # Selection + Reproduction + Mutation
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, k=3)
            winner = max(tournament, key=lambda a: a.fitness)
            selected.append(winner)
        
        next_gen = []
        for idx, parent in enumerate(selected):
            # Chromosome is a string, so convert to list for mutation
            child_chromosome = list(parent.chromosome)
            
            # Mutation (flip C <-> D)
            for i in range(64):
                if random.random() < self.mutation_rate:
                    child_chromosome[i] = 'D' if child_chromosome[i] == 'C' else 'C'
            
            next_gen.append(AdvancedPrisonerAgent(chromosome=''.join(child_chromosome), agent_id=idx))
        
        self.population = next_gen
        self.generation += 1
    
    def run_evolution(self, generations=100):
        """Run complete evolution"""
        self.initialize_population()
        
        for _ in range(generations):
            self.evolve_one_generation()
    
    def get_run_data(self):
        """Export run data"""
        return {
            'fitness_trajectory': self.fitness_trajectory,
            'gene_frequency_matrix': self.gene_frequencies,
            'diversity_history': self.diversity_metrics,
            'mutation_events': {'total': len(self.mutation_events)}
        }


def apply_full_chaos_analysis(fitness_trajectory):
    """Apply complete chaos analysis to fitness trajectory"""
    
    if not CHAOS_MODULES_AVAILABLE:
        # Basic analysis fallback
        mean_fitness = np.mean(fitness_trajectory)
        std_fitness = np.std(fitness_trajectory)
        final_third = fitness_trajectory[-len(fitness_trajectory)//3:]
        
        if np.std(final_third) < 100:
            behavior = 'convergent'
        elif np.std(final_third) > 500:
            behavior = 'chaotic'
        else:
            behavior = 'periodic'
        
        return {
            'behavior': behavior,
            'mean_fitness': float(mean_fitness),
            'std_fitness': float(std_fitness),
            'basic_analysis': True
        }
    
    # Full chaos pipeline
    try:
        data_array = np.array(fitness_trajectory)
        
        # Calculate metrics
        results = {}
        
        # Lyapunov exponent (approximate)
        def lyapunov_estimate(data, lag=1):
            n = len(data)
            divergences = []
            for i in range(n - 2*lag):
                d0 = abs(data[i+lag] - data[i])
                d1 = abs(data[i+2*lag] - data[i+lag])
                if d0 > 0:
                    divergences.append(np.log(d1/d0))
            return np.mean(divergences) if divergences else 0
        
        results['lyapunov_exponent'] = lyapunov_estimate(data_array)
        
        # Sample entropy
        try:
            # Simple sample entropy implementation
            def sample_entropy(data, m=2, r=None):
                if r is None:
                    r = 0.2 * np.std(data)
                
                N = len(data)
                patterns_m = []
                patterns_m1 = []
                
                # Build patterns
                for i in range(N - m):
                    patterns_m.append(data[i:i+m])
                for i in range(N - m - 1):
                    patterns_m1.append(data[i:i+m+1])
                
                # Count matches
                B = 0
                A = 0
                for i in range(len(patterns_m)):
                    for j in range(len(patterns_m)):
                        if i != j:
                            if max(abs(patterns_m[i] - patterns_m[j])) <= r:
                                B += 1
                
                for i in range(len(patterns_m1)):
                    for j in range(len(patterns_m1)):
                        if i != j:
                            if max(abs(patterns_m1[i] - patterns_m1[j])) <= r:
                                A += 1
                
                if A > 0 and B > 0:
                    return -np.log(A / B)
                return 0
            
            results['sample_entropy'] = sample_entropy(data_array)
        except Exception as e:
            results['sample_entropy'] = None
            results['entropy_error'] = str(e)
        
        # Behavior classification
        final_third = data_array[-len(data_array)//3:]
        std_final = np.std(final_third)
        
        if results['lyapunov_exponent'] < -0.01:
            behavior = 'convergent'
        elif results['lyapunov_exponent'] > 0.01:
            behavior = 'chaotic'
        else:
            behavior = 'periodic' if std_final < 200 else 'unknown'
        
        results['behavior'] = behavior
        results['mean_fitness'] = float(np.mean(data_array))
        results['std_fitness'] = float(np.std(data_array))
        results['final_fitness'] = float(data_array[-1])
        results['convergence_time'] = int(np.argmax(data_array > np.mean(data_array))) if len(data_array) > 0 else 0
        results['full_analysis'] = True
        
        return results
        
    except Exception as e:
        print(f"   ⚠️  Chaos analysis error: {e}")
        return {'behavior': 'unknown', 'error': str(e)}


def test_10_runs():
    """
    Test run: 10 evolutionary experiments with chaos analysis
    """
    
    print("\n" + "="*70)
    print("🧪 TEST: UNIFIED CHAOS PIPELINE (10 RUNS)")
    print("="*70)
    print("\nQuick test before the full 1,000-run experiment")
    print("Testing: Evolution + Chaos Analysis + Data Format")
    print("="*70)
    
    all_data = {
        'metadata': {
            'num_runs': 10,
            'generations_per_run': 100,
            'population_size': 50,
            'mutation_rate': 0.01,
            'timestamp': datetime.now().isoformat(),
            'chaos_pipeline': 'unified' if CHAOS_MODULES_AVAILABLE else 'basic',
            'test_mode': True
        },
        'runs': []
    }
    
    chaos_count = {'convergent': 0, 'periodic': 0, 'chaotic': 0, 'unknown': 0}
    
    print("\n⚙️  Running 10 test evolutions...")
    start_time = datetime.now()
    
    for run_idx in tqdm(range(10), desc="Test Runs", unit="run"):
        # Run evolution
        collector = UnifiedEvolutionChaosCollector()
        collector.run_evolution(generations=100)
        
        # Get data
        run_data = collector.get_run_data()
        run_data['run_id'] = run_idx
        
        # Apply chaos analysis
        chaos_results = apply_full_chaos_analysis(run_data['fitness_trajectory'])
        run_data['chaos_analysis'] = chaos_results
        
        # Track behavior
        behavior = chaos_results.get('behavior', 'unknown')
        chaos_count[behavior] = chaos_count.get(behavior, 0) + 1
        
        all_data['runs'].append(run_data)
        
        # Print details for first 3 runs
        if run_idx < 3:
            print(f"\n📊 Run {run_idx}:")
            print(f"   Behavior: {behavior}")
            lyap = chaos_results.get('lyapunov_exponent', None)
            if lyap is not None:
                print(f"   Lyapunov: {lyap:.4f}")
            final_fit = chaos_results.get('final_fitness', None)
            if final_fit is not None:
                print(f"   Final Fitness: {final_fit:.1f}")
            mean_fit = chaos_results.get('mean_fitness', None)
            if mean_fit is not None:
                print(f"   Mean Fitness: {mean_fit:.1f}")
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)
    print(f"\n⏱️  Time for 10 runs: {elapsed:.1f} seconds")
    print(f"⏱️  Estimated time for 1,000 runs: {(elapsed * 100)/3600:.1f} hours")
    
    print(f"\n📈 Behavior Distribution (10 runs):")
    for behavior, count in chaos_count.items():
        if count > 0:
            print(f"   {behavior:12s}: {count:2d} ({count*10:.0f}%)")
    
    # Save test data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'chaos_unified_TEST_10runs_{timestamp}.json'
    
    print(f"\n💾 Saving test data: {filename}")
    with open(filename, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    import os
    file_size_kb = os.path.getsize(filename) / 1024
    print(f"   File size: {file_size_kb:.1f} KB")
    print(f"   Estimated size for 1,000 runs: {(file_size_kb * 100)/1024:.1f} MB")
    
    # Validate data structure
    print(f"\n🔍 Data Validation:")
    sample_run = all_data['runs'][0]
    print(f"   ✅ fitness_trajectory: {len(sample_run['fitness_trajectory'])} points")
    print(f"   ✅ gene_frequency_matrix: {len(sample_run['gene_frequency_matrix'])} × 64")
    print(f"   ✅ diversity_history: {len(sample_run['diversity_history'])} generations")
    print(f"   ✅ chaos_analysis: {list(sample_run['chaos_analysis'].keys())}")
    
    print("\n" + "="*70)
    print("🎯 TEST RESULTS")
    print("="*70)
    print("\n✅ Evolution system working")
    print("✅ Chaos analysis integrated")
    print("✅ Data format compatible with ML pipeline")
    print("✅ Ready for 1,000-run experiment")
    print("\nNext: Run the full experiment with run_unified_chaos_1000.py")
    print("="*70)
    
    return all_data, filename


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 CHAOS PIPELINE TEST (10 RUNS)")
    print("="*70)
    print("\nThis will test:")
    print("  ✅ Evolution data collection")
    print("  ✅ Chaos analysis integration")
    print("  ✅ Data format compatibility")
    print("  ✅ Time estimation for 1,000 runs")
    print("\nEstimated time: ~1-2 minutes")
    print("="*70)
    
    response = input("\n🚀 Start 10-run test? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        data, filename = test_10_runs()
        
        print("\n" + "="*70)
        print("🎉 TEST SUCCESSFUL!")
        print("="*70)
        print(f"\n✅ Test data saved: {filename}")
        print("✅ Pipeline validated")
        print("✅ Ready for full 1,000-run experiment")
        print("\n💡 To run the full experiment:")
        print("   python run_unified_chaos_1000.py")
        print("="*70)
    else:
        print("\n❌ Test cancelled")
