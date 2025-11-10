
"""
⚛️ Quantum Genome Test Suite
Evaluate evolved genomes against known quantum physics problems
"""

import numpy as np
from quantum_genetic_agents import QuantumAgent, load_genome
import json

class QuantumBenchmark:
    """Test genome performance on quantum physics tasks"""
    
    @staticmethod
    def coherence_preservation_test(genome, duration=100):
        """Test how well genome preserves coherence over time"""
        agent = QuantumAgent(0, genome)
        initial_coherence = agent.traits[1]
        
        for t in range(1, duration):
            agent.evolve(t)
        
        final_coherence = agent.traits[1]
        preservation_score = final_coherence / initial_coherence
        return preservation_score
    
    @staticmethod
    def energy_oscillation_test(genome, cycles=5):
        """Test energy oscillation quality (smoothness vs chaos)"""
        agent = QuantumAgent(0, genome)
        energies = []
        
        for t in range(1, cycles * 20):
            agent.evolve(t)
            energies.append(agent.traits[0])
        
        # Measure oscillation quality (lower variance in derivatives = smoother)
        derivatives = np.diff(energies)
        smoothness = 1.0 / (1.0 + np.std(derivatives))
        return smoothness
    
    @staticmethod
    def decoherence_constant_accuracy(genome, expected=0.011):
        """Test how close genome's decoherence is to discovered constant"""
        actual = genome[2]  # decoherence_rate
        error = abs(actual - expected)
        accuracy = np.exp(-error * 100)  # Exponential penalty
        return accuracy
    
    @staticmethod
    def parameter_synergy_test(genome):
        """Test parameter synergy by comparing to averaged version"""
        agent_original = QuantumAgent(0, genome)
        
        # Create averaged genome (destroy synergy)
        avg_genome = [np.mean([g]) for g in genome]
        agent_averaged = QuantumAgent(1, avg_genome)
        
        for t in range(1, 50):
            agent_original.evolve(t)
            agent_averaged.evolve(t)
        
        original_fitness = agent_original.get_final_fitness()
        averaged_fitness = agent_averaged.get_final_fitness()
        
        synergy_score = original_fitness / max(averaged_fitness, 1e-10)
        return synergy_score

def run_quantum_tests(genome_file):
    """Run all quantum tests on a genome"""
    print(f"\n⚛️ Testing: {genome_file}")
    print("=" * 60)
    
    genome, metadata = load_genome(genome_file)
    
    tests = {
        'Coherence Preservation': QuantumBenchmark.coherence_preservation_test,
        'Energy Oscillation Quality': QuantumBenchmark.energy_oscillation_test,
        'Decoherence Accuracy': QuantumBenchmark.decoherence_constant_accuracy,
        'Parameter Synergy': QuantumBenchmark.parameter_synergy_test
    }
    
    results = {}
    for test_name, test_func in tests.items():
        score = test_func(genome)
        results[test_name] = score
        print(f"  {test_name:30s}: {score:.6f}")
    
    # Overall quantum score
    overall = np.mean(list(results.values()))
    print(f"\n  Overall Quantum Score: {overall:.6f}")
    
    # Export results
    export_data = {
        'genome_file': genome_file,
        'test_results': results,
        'overall_score': overall,
        'genome': metadata.get('genome', {})
    }
    
    output_file = genome_file.replace('.json', '_quantum_test.json')
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    return results

if __name__ == '__main__':
    import glob
    
    print("\n" + "=" * 60)
    print("⚛️ QUANTUM GENOME TEST SUITE")
    print("=" * 60)
    
    genome_files = glob.glob('*_genome.json')
    
    if not genome_files:
        print("\n❌ No genome files found!")
    else:
        all_results = {}
        for genome_file in genome_files:
            try:
                results = run_quantum_tests(genome_file)
                all_results[genome_file] = results
            except Exception as e:
                print(f"\n❌ Error testing {genome_file}: {e}")
        
        # Rank genomes
        print("\n" + "=" * 60)
        print("🏆 QUANTUM BENCHMARK RANKINGS")
        print("=" * 60)
        
        for test_name in ['Coherence Preservation', 'Energy Oscillation Quality', 
                         'Decoherence Accuracy', 'Parameter Synergy']:
            print(f"\n{test_name}:")
            ranked = sorted(all_results.items(), 
                          key=lambda x: x[1].get(test_name, 0), 
                          reverse=True)
            for i, (genome, scores) in enumerate(ranked[:3], 1):
                print(f"  {i}. {genome:40s}: {scores.get(test_name, 0):.6f}")
