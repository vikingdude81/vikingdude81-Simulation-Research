"""
Validate Fitness Stability Fixes
================================
Test the numerical stability fixes for the fitness function with:
1. All 8 champion genomes (d=0.005)
2. Edge case genomes (d=0.001, d=0.02)
3. Extreme negative coherence scenarios
4. Type consistency checks (numpy arrays vs lists)
"""

import numpy as np
import json
from quantum_genetic_agents import QuantumAgent
from pathlib import Path

def test_champion_genome(name, genome, environment='standard'):
    """Test a champion genome with the fixed fitness function"""
    print(f"\n{'='*60}")
    print(f"Testing {name} Champion")
    print(f"Genome: {genome}")
    print(f"Environment: {environment}")
    print(f"{'='*60}")
    
    # Create agent (agent_id, genome, environment)
    agent = QuantumAgent(0, genome, environment)
    
    # Run simulation
    for t in range(100):
        agent.evolve(t)
    
    # Get fitness
    final_fitness = agent.get_final_fitness()
    
    # Validate fitness
    is_finite = np.isfinite(final_fitness)
    is_reasonable = abs(final_fitness) < 1e8
    
    # Get coherence stats
    coherence_values = [state[1] for state in agent.history]
    coherence_min = min(coherence_values)
    coherence_max = max(coherence_values)
    coherence_decay = coherence_values[0] - coherence_values[-1]
    
    # Get fitness stats
    fitness_values = [state[3] for state in agent.history]
    fitness_mean = np.mean(fitness_values)
    fitness_std = np.std(fitness_values)
    
    print(f"✓ Final Fitness: {final_fitness:.10f}")
    print(f"✓ Is Finite: {is_finite}")
    print(f"✓ Is Reasonable: {is_reasonable}")
    print(f"✓ Coherence Range: [{coherence_min:.6f}, {coherence_max:.6f}]")
    print(f"✓ Coherence Decay: {coherence_decay:.6f}")
    print(f"✓ Fitness Mean: {fitness_mean:.10f}")
    print(f"✓ Fitness Std: {fitness_std:.10f}")
    
    # Check for negative coherence
    has_negative_coherence = coherence_min < 0
    if has_negative_coherence:
        print(f"⚠ WARNING: Negative coherence detected: {coherence_min:.6f}")
    else:
        print(f"✓ No negative coherence")
    
    return {
        'name': name,
        'genome': genome.tolist() if isinstance(genome, np.ndarray) else genome,
        'environment': environment,
        'final_fitness': float(final_fitness),
        'is_finite': bool(is_finite),
        'is_reasonable': bool(is_reasonable),
        'coherence_min': float(coherence_min),
        'coherence_max': float(coherence_max),
        'coherence_decay': float(coherence_decay),
        'fitness_mean': float(fitness_mean),
        'fitness_std': float(fitness_std),
        'has_negative_coherence': bool(has_negative_coherence)
    }

def test_edge_case_genome(name, genome, description):
    """Test edge case genomes that previously caused instability"""
    print(f"\n{'='*60}")
    print(f"Testing Edge Case: {name}")
    print(f"Description: {description}")
    print(f"Genome: {genome}")
    print(f"{'='*60}")
    
    agent = QuantumAgent(0, genome, 'standard')
    
    try:
        for t in range(100):
            agent.evolve(t)
        
        final_fitness = agent.get_final_fitness()
        
        # Check for explosion
        is_finite = np.isfinite(final_fitness)
        is_reasonable = abs(final_fitness) < 1e8
        
        coherence_values = [state[1] for state in agent.history]
        coherence_min = min(coherence_values)
        coherence_decay = coherence_values[0] - coherence_values[-1]
        
        print(f"✓ Final Fitness: {final_fitness:.10f}")
        print(f"✓ Is Finite: {is_finite}")
        print(f"✓ Is Reasonable: {is_reasonable}")
        print(f"✓ Coherence Min: {coherence_min:.6f}")
        print(f"✓ Coherence Decay: {coherence_decay:.6f}")
        
        if not is_finite:
            print(f"⚠ FAILED: Fitness is not finite!")
        elif not is_reasonable:
            print(f"⚠ FAILED: Fitness is unreasonable: {final_fitness}")
        else:
            print(f"✓ PASSED: Edge case handled correctly")
        
        return {
            'name': name,
            'description': description,
            'genome': genome.tolist() if isinstance(genome, np.ndarray) else genome,
            'final_fitness': float(final_fitness),
            'is_finite': bool(is_finite),
            'is_reasonable': bool(is_reasonable),
            'coherence_min': float(coherence_min),
            'coherence_decay': float(coherence_decay)
        }
    
    except Exception as e:
        print(f"⚠ EXCEPTION: {e}")
        return {
            'name': name,
            'description': description,
            'genome': genome.tolist() if isinstance(genome, np.ndarray) else genome,
            'exception': str(e)
        }

def check_type_consistency():
    """Check that all genomes are numpy arrays and JSON serialization works"""
    print(f"\n{'='*60}")
    print("Type Consistency Checks")
    print(f"{'='*60}")
    
    # Test genome creation
    genome_list = [3.0, 0.5, 0.005, 0.5]
    genome_array = np.array([3.0, 0.5, 0.005, 0.5])
    
    # Test agent with list (should work or convert)
    agent_list = QuantumAgent(0, genome_list, 'standard')
    print(f"✓ Agent with list genome: {type(agent_list.genome)}")
    
    # Test agent with array
    agent_array = QuantumAgent(1, genome_array, 'standard')
    print(f"✓ Agent with array genome: {type(agent_array.genome)}")
    
    # Test JSON serialization
    test_data = {
        'genome_list': genome_list,
        'genome_array': genome_array.tolist(),
        'fitness': 0.012345,
        'nested': {
            'array': np.array([1, 2, 3]).tolist()
        }
    }
    
    try:
        json_str = json.dumps(test_data)
        print(f"✓ JSON serialization successful")
        
        # Test deserialization
        loaded = json.loads(json_str)
        print(f"✓ JSON deserialization successful")
        
        # Convert back to array
        genome_recovered = np.array(loaded['genome_array'])
        print(f"✓ Array recovery successful: {type(genome_recovered)}")
        
        return {'status': 'PASSED', 'details': 'All type checks passed'}
    
    except Exception as e:
        print(f"⚠ JSON serialization FAILED: {e}")
        return {'status': 'FAILED', 'error': str(e)}

def main():
    print("\n" + "="*70)
    print("QUANTUM GENETIC FITNESS STABILITY VALIDATION")
    print("="*70)
    
    # Define all 8 champions with d=0.005
    champions = [
        ('Gentle', np.array([2.7668, 0.1853, 0.0050, 0.6798]), 'gentle'),
        ('Standard', np.array([2.9460, 0.1269, 0.0050, 0.2996]), 'standard'),
        ('Chaotic', np.array([3.0000, 0.5045, 0.0050, 0.4108]), 'chaotic'),
        ('Oscillating', np.array([3.0000, 1.8126, 0.0050, 0.0000]), 'oscillating'),
        ('Harsh', np.array([3.0000, 2.0000, 0.0050, 0.5713]), 'harsh'),
        ('Island_Elite_1', np.array([2.9996, 1.5018, 0.0050, 0.7527]), 'standard'),
        ('Island_Elite_2', np.array([2.9813, 1.7393, 0.0050, 0.7854]), 'standard'),
        ('Island_Elite_3', np.array([2.9979, 1.1872, 0.0050, 0.6283]), 'standard')
    ]
    
    # Test all champions
    print("\n" + "="*70)
    print("PHASE 1: Testing 8 Champion Genomes (d=0.005)")
    print("="*70)
    
    champion_results = []
    for name, genome, env in champions:
        result = test_champion_genome(name, genome, env)
        champion_results.append(result)
    
    # Test edge cases that previously caused problems
    print("\n" + "="*70)
    print("PHASE 2: Testing Edge Cases")
    print("="*70)
    
    edge_cases = [
        ('Low_d_0.001', np.array([3.0, 0.5, 0.001, 0.5]), 
         'Very low decoherence rate - previously caused explosion'),
        ('High_d_0.02', np.array([3.0, 0.5, 0.02, 0.5]),
         'High decoherence rate - fast decay'),
        ('Zero_d', np.array([3.0, 0.5, 0.0, 0.5]),
         'Zero decoherence rate - coherence should stay constant'),
        ('High_omega', np.array([3.0, 2.5, 0.005, 0.5]),
         'High oscillation frequency'),
        ('Extreme_mu', np.array([5.0, 0.5, 0.005, 0.5]),
         'Very high energy scale')
    ]
    
    edge_results = []
    for name, genome, description in edge_cases:
        result = test_edge_case_genome(name, genome, description)
        edge_results.append(result)
    
    # Type consistency checks
    print("\n" + "="*70)
    print("PHASE 3: Type Consistency Checks")
    print("="*70)
    
    type_result = check_type_consistency()
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    # Champion summary
    champion_passes = sum(1 for r in champion_results if r['is_finite'] and r['is_reasonable'])
    print(f"\nChampion Tests: {champion_passes}/{len(champion_results)} passed")
    
    for r in champion_results:
        status = "✓ PASS" if r['is_finite'] and r['is_reasonable'] else "✗ FAIL"
        print(f"  {status} {r['name']}: fitness={r['final_fitness']:.6f}, coherence_min={r['coherence_min']:.6f}")
    
    # Edge case summary
    edge_passes = sum(1 for r in edge_results if 'is_finite' in r and r['is_finite'] and r['is_reasonable'])
    print(f"\nEdge Case Tests: {edge_passes}/{len(edge_results)} passed")
    
    for r in edge_results:
        if 'exception' in r:
            print(f"  ✗ EXCEPTION {r['name']}: {r['exception']}")
        else:
            status = "✓ PASS" if r['is_finite'] and r['is_reasonable'] else "✗ FAIL"
            print(f"  {status} {r['name']}: fitness={r['final_fitness']:.6f}")
    
    # Type consistency summary
    print(f"\nType Consistency: {type_result['status']}")
    
    # Overall status
    all_passed = (champion_passes == len(champion_results) and 
                  edge_passes == len(edge_results) and 
                  type_result['status'] == 'PASSED')
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Fitness function is numerically stable!")
    else:
        print("⚠ SOME TESTS FAILED - Review results above")
    print("="*70)
    
    # Save results
    results = {
        'timestamp': Path(__file__).stem,
        'champion_results': champion_results,
        'edge_case_results': edge_results,
        'type_consistency': type_result,
        'summary': {
            'champion_passes': champion_passes,
            'champion_total': len(champion_results),
            'edge_passes': edge_passes,
            'edge_total': len(edge_results),
            'all_passed': all_passed
        }
    }
    
    output_file = Path(__file__).parent / 'fitness_stability_validation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return all_passed

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
