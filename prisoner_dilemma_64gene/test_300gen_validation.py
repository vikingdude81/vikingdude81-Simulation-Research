"""
300-Generation Validation Test
Tests if multi-quantum ensemble continues to scale beyond 150 generations
"""

import json
from datetime import datetime
from multi_quantum_controller import create_default_ensemble, MetaController, MultiQuantumGodController
from prisoner_echo_god import run_god_echo_simulation
from quantum_god_controller import QuantumGodController
import statistics

def calculate_score(population):
    """Calculate composite score: wealth + cooperation - inequality"""
    if not hasattr(population, 'resources') or len(population.resources) == 0:
        return 0
    
    avg_wealth = statistics.mean(population.resources)
    
    # Cooperation rate
    coop_actions = sum(1 for a in population.agents if a.is_cooperating)
    coop_rate = coop_actions / len(population.agents) if population.agents else 0
    
    # Gini coefficient (inequality)
    sorted_wealth = sorted(population.resources)
    n = len(sorted_wealth)
    cumsum = sum((i + 1) * wealth for i, wealth in enumerate(sorted_wealth))
    gini = (2 * cumsum) / (n * sum(sorted_wealth)) - (n + 1) / n if sum(sorted_wealth) > 0 else 0
    
    # Composite score (weights tuned for prisoner's dilemma)
    score = (avg_wealth * 50) + (coop_rate * 100) - (gini * 30)
    return score

def run_300gen_test(strategy, run_number=1):
    """Run a single 300-generation test"""
    
    print(f"\n{'='*70}")
    print(f"🧪 RUNNING 300-GEN TEST: {strategy.upper()} (Run #{run_number})")
    print(f"{'='*70}")
    
    # Create ensemble
    ensemble = create_default_ensemble()
    meta = MetaController(ensemble, strategy=strategy)
    
    # Run simulation in phases (like the working test does)
    start_time = datetime.now()
    
    # For 300 gen, break into phases: 0-100, 100-200, 200-300
    phase_segments = [(0, 100), (100, 200), (200, 300)]
    total_score = 0
    total_cooperation = 0
    total_population = 0
    
    for phase_num, (start_gen, end_gen) in enumerate(phase_segments):
        phase_gens = end_gen - start_gen
        genome = meta.select_genome(start_gen, 300)
        
        print(f"\n📍 Phase {phase_num + 1}: Gen {start_gen}-{end_gen} using {genome.name}")
        
        # Temporarily modify quantum_god_controller.py's genome for this phase
        import quantum_god_controller
        original_genome = quantum_god_controller.QUANTUM_GENOME.copy()
        quantum_god_controller.QUANTUM_GENOME = genome.genome
        
        population = run_god_echo_simulation(
            generations=phase_gens,
            initial_size=1000,
            god_mode="ML_BASED",
            update_frequency=10,
            prompt_style="neutral"
        )
        
        # Restore original genome
        quantum_god_controller.QUANTUM_GENOME = original_genome
        
        # Calculate phase score
        stats = calculate_stats(population)
        total_score += stats['score']
        total_cooperation += stats['cooperation']
        total_population += stats['population']
        
        print(f"  Phase Score: {stats['score']:,.0f}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Calculate final score
    final_score = calculate_score(population)
    
    # Get specialist usage
    specialist_usage = meta_controller.get_statistics()
    
    result = {
        'strategy': strategy,
        'run_number': run_number,
        'generations': 300,
        'final_score': final_score,
        'per_gen_efficiency': final_score / 300,
        'avg_wealth': statistics.mean(population.resources) if population.resources else 0,
        'cooperation_rate': sum(1 for a in population.agents if a.is_cooperating) / len(population.agents),
        'gini_coefficient': calculate_gini(population.resources),
        'duration_seconds': duration,
        'specialist_usage': specialist_usage,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n✅ COMPLETED in {duration:.1f}s")
    print(f"   Final Score: {final_score:,.0f}")
    print(f"   Efficiency: {result['per_gen_efficiency']:.1f} per gen")
    print(f"   Avg Wealth: ${result['avg_wealth']:.2f}")
    print(f"   Cooperation: {result['cooperation_rate']:.1%}")
    print(f"   Gini: {result['gini_coefficient']:.3f}")
    
    return result

def calculate_gini(resources):
    """Calculate Gini coefficient"""
    if not resources or len(resources) == 0:
        return 0
    
    sorted_wealth = sorted(resources)
    n = len(sorted_wealth)
    cumsum = sum((i + 1) * wealth for i, wealth in enumerate(sorted_wealth))
    total = sum(sorted_wealth)
    
    if total == 0:
        return 0
    
    gini = (2 * cumsum) / (n * total) - (n + 1) / n
    return gini

def run_baseline_300gen(controller_type):
    """Run baseline comparison at 300 generations"""
    
    print(f"\n{'='*70}")
    print(f"🔬 BASELINE: {controller_type.upper()} (300 generations)")
    print(f"{'='*70}")
    
    start_time = datetime.now()
    
    if controller_type == "single_50gen":
        # Use the 50-gen trained quantum controller
        population = run_god_echo_simulation(
            generations=300,
            initial_size=1000,
            god_mode="ML_BASED",
            update_frequency=10,
            prompt_style="neutral"
        )
    elif controller_type == "gpt4":
        # Use GPT-4 controller
        population = run_god_echo_simulation(
            generations=300,
            initial_size=1000,
            god_mode="ML_BASED",  # Will need to adapt for GPT-4
            update_frequency=10,
            prompt_style="neutral"
        )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    final_score = calculate_score(population)
    
    result = {
        'controller': controller_type,
        'generations': 300,
        'final_score': final_score,
        'per_gen_efficiency': final_score / 300,
        'avg_wealth': statistics.mean(population.resources) if population.resources else 0,
        'cooperation_rate': sum(1 for a in population.agents if a.is_cooperating) / len(population.agents),
        'gini_coefficient': calculate_gini(population.resources),
        'duration_seconds': duration,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n✅ COMPLETED in {duration:.1f}s")
    print(f"   Final Score: {final_score:,.0f}")
    print(f"   Efficiency: {result['per_gen_efficiency']:.1f} per gen")
    
    return result

def analyze_scaling(results_150gen, results_300gen):
    """Compare 150-gen vs 300-gen performance"""
    
    print(f"\n{'='*70}")
    print("📊 SCALING ANALYSIS: 150-gen vs 300-gen")
    print(f"{'='*70}\n")
    
    # Load 150-gen results
    with open('outputs/god_ai/multi_quantum_ensemble_20251104_171322.json', 'r') as f:
        data_150 = json.load(f)
    
    # Extract 150-gen phase-based efficiency
    phase_150_scores = []
    for result in data_150.get('detailed_results', []):
        if result['strategy'] == 'phase_based' and result['generations'] == 150:
            phase_150_scores.append(result['final_score'])
    
    avg_150 = statistics.mean(phase_150_scores) if phase_150_scores else 0
    efficiency_150 = avg_150 / 150
    
    # 300-gen efficiency
    phase_300_scores = [r['final_score'] for r in results_300gen if r['strategy'] == 'phase_based']
    avg_300 = statistics.mean(phase_300_scores) if phase_300_scores else 0
    efficiency_300 = avg_300 / 300
    
    improvement = ((efficiency_300 - efficiency_150) / efficiency_150 * 100) if efficiency_150 > 0 else 0
    
    print(f"Phase-Based Ensemble:")
    print(f"  150-gen efficiency: {efficiency_150:.1f} per gen")
    print(f"  300-gen efficiency: {efficiency_300:.1f} per gen")
    print(f"  Change: {improvement:+.1f}%")
    print(f"")
    
    if improvement > 0:
        print(f"✅ SCALING CONFIRMED! Multi-quantum continues to improve at 300 gen (+{improvement:.1f}%)")
    elif improvement > -5:
        print(f"✅ PLATEAU REACHED. Performance stable at 300 gen ({improvement:+.1f}%)")
    else:
        print(f"⚠️ DEGRADATION DETECTED. Performance declined at 300 gen ({improvement:+.1f}%)")
    
    return {
        'efficiency_150': efficiency_150,
        'efficiency_300': efficiency_300,
        'improvement_percent': improvement,
        'scaling_status': 'improving' if improvement > 0 else ('stable' if improvement > -5 else 'degrading')
    }

def main():
    """Run complete 300-generation validation"""
    
    print("\n" + "="*70)
    print("🚀 300-GENERATION VALIDATION TEST")
    print("="*70)
    print("\nObjective: Validate that multi-quantum ensemble continues to scale")
    print("Expected: +50-100% advantage over single controller at 300 gen")
    print("Estimated time: ~30 minutes for all tests")
    print("\n" + "="*70)
    
    all_results = []
    
    # Test 1: Phase-Based Ensemble
    result1 = run_300gen_test('phase_based', run_number=1)
    all_results.append(result1)
    
    # Test 2: Adaptive Ensemble
    result2 = run_300gen_test('adaptive', run_number=1)
    all_results.append(result2)
    
    # Test 3: Single 50-gen Baseline
    result3 = run_baseline_300gen('single_50gen')
    all_results.append(result3)
    
    # Analyze results
    print(f"\n{'='*70}")
    print("🏆 FINAL RESULTS - 300 GENERATION TEST")
    print(f"{'='*70}\n")
    
    phase_score = next(r['final_score'] for r in all_results if r.get('strategy') == 'phase_based')
    adaptive_score = next(r['final_score'] for r in all_results if r.get('strategy') == 'adaptive')
    single_score = next(r['final_score'] for r in all_results if r.get('controller') == 'single_50gen')
    
    phase_vs_single = ((phase_score - single_score) / single_score * 100) if single_score > 0 else 0
    adaptive_vs_single = ((adaptive_score - single_score) / single_score * 100) if single_score > 0 else 0
    
    print(f"Phase-Based Ensemble:    {phase_score:>10,.0f}  ({phase_score/300:.1f}/gen)")
    print(f"Adaptive Ensemble:       {adaptive_score:>10,.0f}  ({adaptive_score/300:.1f}/gen)")
    print(f"Single 50-gen ML:        {single_score:>10,.0f}  ({single_score/300:.1f}/gen)")
    print(f"")
    print(f"Phase-Based vs Single:   {phase_vs_single:>10.1f}%")
    print(f"Adaptive vs Single:      {adaptive_vs_single:>10.1f}%")
    print(f"")
    
    winner = "Phase-Based" if phase_score > adaptive_score else "Adaptive"
    print(f"🏆 Winner: {winner}")
    
    # Scaling analysis
    scaling_analysis = analyze_scaling([], all_results)
    
    # Save results
    output = {
        'test_type': '300_generation_validation',
        'test_date': datetime.now().isoformat(),
        'results': all_results,
        'comparison': {
            'phase_based_vs_single': phase_vs_single,
            'adaptive_vs_single': adaptive_vs_single,
            'winner': winner
        },
        'scaling_analysis': scaling_analysis,
        'conclusion': (
            f"Multi-quantum ensemble achieved {phase_vs_single:+.1f}% improvement "
            f"over single controller at 300 generations. "
            f"Scaling status: {scaling_analysis['scaling_status']}"
        )
    }
    
    filename = f"outputs/god_ai/validation_300gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    print(f"\n{'='*70}")
    print("✅ VALIDATION COMPLETE!")
    print(f"{'='*70}\n")
    
    # Final recommendation
    if phase_vs_single >= 50:
        print("🎯 RECOMMENDATION: Multi-quantum ensemble validated for long-term use.")
        print("   Performance advantage maintained at 300 generations.")
        print("   ✅ Ready to proceed with trading system implementation!")
    elif phase_vs_single >= 20:
        print("⚠️  RECOMMENDATION: Multi-quantum shows improvement but less than predicted.")
        print("   Consider additional tuning or specialist optimization.")
    else:
        print("❌ RECOMMENDATION: Multi-quantum advantage diminished at 300 generations.")
        print("   May need to investigate why scaling didn't continue.")

if __name__ == "__main__":
    main()
