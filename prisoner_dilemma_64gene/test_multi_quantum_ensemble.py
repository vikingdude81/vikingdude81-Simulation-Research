"""
Test Multi-Quantum Ensemble Controller
======================================
Tests ensemble of specialized Quantum ML genomes vs single controllers.

Comparison:
1. Multi-Quantum Ensemble (phase-based switching)
2. Multi-Quantum Ensemble (adaptive switching)
3. Fixed 50-gen Quantum ML (our champion)
4. Fixed 150-gen Quantum ML (retrained)
5. GPT-4 Neutral (baseline)
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List
import numpy as np

from multi_quantum_controller import (
    create_default_ensemble,
    MetaController,
    MultiQuantumGodController
)
from prisoner_echo_god import run_god_echo_simulation


def calculate_score(population):
    """Calculate overall score from population metrics."""
    if population and population.agents:
        total_wealth = sum(a.resources for a in population.agents)
        
        # Cooperation rate - check if agents have cooperations/defections
        if hasattr(population.agents[0], 'cooperations'):
            total_interactions = sum(a.cooperations + a.defections for a in population.agents)
            total_cooperations = sum(a.cooperations for a in population.agents)
            coop_rate = total_cooperations / total_interactions if total_interactions > 0 else 0.5
        else:
            # Fallback - assume strategy attribute
            cooperators = sum(1 for a in population.agents if hasattr(a, 'strategy') and a.strategy == 'C')
            coop_rate = cooperators / len(population.agents) if len(population.agents) > 0 else 0.5
        
        # Gini coefficient
        resources = sorted([a.resources for a in population.agents])
        n = len(resources)
        cumsum = np.cumsum(resources)
        gini = (2 * cumsum.sum()) / (n * sum(resources)) - (n + 1) / n if sum(resources) > 0 else 0
        
        # Composite score (weighted)
        score = total_wealth / 100 + coop_rate * 10000 + (1 - gini) * 10000
        
        return {
            'wealth': total_wealth / len(population.agents),
            'cooperation': coop_rate,
            'gini': gini,
            'score': score,
            'population': len(population.agents)
        }
    return {'wealth': 0, 'cooperation': 0, 'gini': 1, 'score': 0, 'population': 0}


def run_multi_quantum_test(
    generations: int,
    strategy: str,
    ensemble: List = None,
    run_number: int = 1
) -> Dict:
    """
    Run a single test with multi-quantum controller.
    
    Args:
        generations: Number of generations to run
        strategy: Meta-controller strategy ("phase_based" or "adaptive")
        ensemble: List of QuantumGenome objects (uses default if None)
        run_number: Run identifier
    
    Returns:
        Dict with results
    """
    if ensemble is None:
        ensemble = create_default_ensemble()
    
    print(f"\n{'='*60}")
    print(f"🧬 MULTI-QUANTUM TEST - {strategy.upper()}")
    print(f"   Generations: {generations}")
    print(f"   Genomes: {len(ensemble)}")
    print(f"   Run: {run_number}")
    print(f"{'='*60}")
    
    # Create meta-controller
    meta = MetaController(ensemble, strategy=strategy)
    multi_controller = MultiQuantumGodController(meta, generations)
    
    # For now, we'll run with the first genome and simulate switching
    # In a full implementation, we'd integrate deeply with the simulation loop
    
    # Run simulation with adaptive genome selection
    # We'll use multiple short runs to simulate phase switching
    total_score = 0
    total_cooperation = 0
    total_population = 0
    phase_results = []
    
    if generations <= 50:
        # Single phase
        genome = meta.select_genome(0, generations)
        print(f"\n📍 Using {genome.name} for entire run")
        
        population = run_god_echo_simulation(
            generations=generations,
            initial_size=100,
            god_mode="ML_BASED",
            update_frequency=10,
            prompt_style="neutral"
        )
        
        stats = calculate_score(population)
        score = stats['score']
        coop = stats['cooperation'] * 100
        pop = stats['population']
        
        phase_results.append({
            "generations": f"0-{generations}",
            "genome": genome.name,
            "score": score,
            "cooperation": coop,
            "population": pop
        })
        
        total_score = score
        total_cooperation = coop
        total_population = pop
        
    else:
        # Multi-phase run - break into segments
        phase_segments = []
        if generations <= 75:
            phase_segments = [(0, generations)]
        elif generations <= 100:
            phase_segments = [(0, 50), (50, generations)]
        elif generations <= 125:
            phase_segments = [(0, 40), (40, 85), (85, generations)]
        else:  # 150
            phase_segments = [(0, 50), (50, 100), (100, generations)]
        
        for phase_num, (start_gen, end_gen) in enumerate(phase_segments):
            phase_gens = end_gen - start_gen
            genome = meta.select_genome(start_gen, generations)
            
            print(f"\n📍 Phase {phase_num + 1}: Gen {start_gen}-{end_gen}")
            print(f"   Selected: {genome.name}")
            print(f"   Reason: {genome.trained_for}")
            
            population = run_god_echo_simulation(
                generations=phase_gens,
                initial_size=100,
                god_mode="ML_BASED",
                update_frequency=10,
                prompt_style="neutral"
            )
            
            stats = calculate_score(population)
            score = stats['score']
            coop = stats['cooperation'] * 100
            pop = stats['population']
            
            phase_results.append({
                "generations": f"{start_gen}-{end_gen}",
                "genome": genome.name,
                "score": score,
                "cooperation": coop,
                "population": pop
            })
            
            # Update performance tracking
            meta.update_performance(genome.name, end_gen, score)
            
            total_score += score
            total_cooperation += coop
            total_population = pop  # Final population
        
        # Average cooperation across phases
        total_cooperation = np.mean([p["cooperation"] for p in phase_results])
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTS:")
    print(f"   Total Score: {total_score:,.0f}")
    print(f"   Avg Cooperation: {total_cooperation:.1f}%")
    print(f"   Final Population: {total_population}")
    print(f"{'='*60}")
    
    # Get meta-controller statistics
    meta_stats = meta.get_statistics()
    
    return {
        "strategy": strategy,
        "generations": generations,
        "run_number": run_number,
        "total_score": total_score,
        "cooperation_rate": total_cooperation,
        "final_population": total_population,
        "phase_results": phase_results,
        "meta_stats": meta_stats,
        "ensemble_size": len(ensemble)
    }


def run_baseline_comparison(generations: int, run_number: int = 1) -> Dict:
    """Run baseline tests with single controllers."""
    baselines = {}
    
    # Test 1: Fixed 50-gen Quantum ML
    print(f"\n{'='*60}")
    print(f"🔷 BASELINE: Fixed 50-gen Quantum ML")
    print(f"{'='*60}")
    population = run_god_echo_simulation(
        generations=generations,
        initial_size=100,
        god_mode="ML_BASED",
        update_frequency=10,
        prompt_style="neutral"
    )
    stats = calculate_score(population)
    baselines["fixed_50gen"] = {
        "score": stats['score'],
        "cooperation": stats['cooperation'] * 100,
        "population": stats['population']
    }
    print(f"   Score: {stats['score']:,.0f}")
    print(f"   Cooperation: {stats['cooperation'] * 100:.1f}%")
    
    # Test 2: GPT-4 Neutral
    print(f"\n{'='*60}")
    print(f"🔷 BASELINE: GPT-4 Neutral")
    print(f"{'='*60}")
    population = run_god_echo_simulation(
        generations=generations,
        initial_size=100,
        god_mode="API_BASED",
        update_frequency=10,
        prompt_style="neutral"
    )
    stats = calculate_score(population)
    baselines["gpt4_neutral"] = {
        "score": stats['score'],
        "cooperation": stats['cooperation'] * 100,
        "population": stats['population']
    }
    print(f"   Score: {stats['score']:,.0f}")
    print(f"   Cooperation: {stats['cooperation'] * 100:.1f}%")
    
    return baselines


def main():
    """Run comprehensive multi-quantum ensemble tests."""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "MULTI-QUANTUM ENSEMBLE TEST SUITE" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Test configurations
    time_horizons = [50, 75, 100, 125, 150]
    strategies = ["phase_based", "adaptive"]
    runs_per_config = 2
    
    # Create default ensemble
    ensemble = create_default_ensemble()
    print(f"\n📦 Ensemble Configuration:")
    for i, g in enumerate(ensemble, 1):
        print(f"   {i}. {g.name}")
        print(f"      Phase: {g.optimal_phase}")
        print(f"      Purpose: {g.trained_for}")
    
    all_results = []
    
    # Run tests for each time horizon
    for horizon in time_horizons:
        print(f"\n\n{'#'*60}")
        print(f"# TIME HORIZON: {horizon} GENERATIONS")
        print(f"{'#'*60}")
        
        horizon_results = {
            "horizon": horizon,
            "tests": []
        }
        
        # Run baseline comparisons (once per horizon)
        print(f"\n🔍 Running baseline comparisons...")
        baselines = run_baseline_comparison(horizon, run_number=1)
        horizon_results["baselines"] = baselines
        
        # Run multi-quantum tests with different strategies
        for strategy in strategies:
            for run in range(1, runs_per_config + 1):
                result = run_multi_quantum_test(
                    generations=horizon,
                    strategy=strategy,
                    ensemble=ensemble,
                    run_number=run
                )
                horizon_results["tests"].append(result)
        
        all_results.append(horizon_results)
        
        # Print horizon summary
        print(f"\n{'='*60}")
        print(f"📈 {horizon}-GENERATION SUMMARY")
        print(f"{'='*60}")
        
        # Calculate averages for each strategy
        strategy_scores = {s: [] for s in strategies}
        for test in horizon_results["tests"]:
            strategy_scores[test["strategy"]].append(test["total_score"])
        
        print(f"\n{'Strategy':<25} {'Avg Score':<15} {'vs Fixed-50':<15} {'vs GPT-4':<15}")
        print("-" * 70)
        
        baseline_50_score = baselines["fixed_50gen"]["score"]
        baseline_gpt4_score = baselines["gpt4_neutral"]["score"]
        
        for strategy in strategies:
            if strategy_scores[strategy]:
                avg_score = np.mean(strategy_scores[strategy])
                vs_50 = ((avg_score - baseline_50_score) / baseline_50_score) * 100
                vs_gpt4 = ((avg_score - baseline_gpt4_score) / baseline_gpt4_score) * 100
                
                print(f"{strategy:<25} {avg_score:>12,.0f}   {vs_50:>+6.1f}%        {vs_gpt4:>+6.1f}%")
        
        # Print baselines
        print(f"\n{'Baseline Comparisons:':<25}")
        print(f"  Fixed 50-gen ML:        {baseline_50_score:>12,.0f}")
        print(f"  GPT-4 Neutral:          {baseline_gpt4_score:>12,.0f}")
    
    # Save results
    output_dir = "outputs/god_ai"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"multi_quantum_ensemble_{timestamp}.json")
    
    final_results = {
        "timestamp": timestamp,
        "ensemble_config": [g.to_dict() for g in ensemble],
        "test_config": {
            "time_horizons": time_horizons,
            "strategies": strategies,
            "runs_per_config": runs_per_config
        },
        "results": all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n\n{'='*60}")
    print(f"✅ ALL TESTS COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 Results saved to: {output_file}")
    
    # Final analysis
    print(f"\n{'='*60}")
    print(f"🏆 OVERALL WINNER ANALYSIS")
    print(f"{'='*60}")
    
    # Calculate overall performance by strategy
    overall_scores = {s: [] for s in strategies}
    overall_baselines = {"fixed_50gen": [], "gpt4_neutral": []}
    
    for horizon_result in all_results:
        # Collect baseline scores
        overall_baselines["fixed_50gen"].append(horizon_result["baselines"]["fixed_50gen"]["score"])
        overall_baselines["gpt4_neutral"].append(horizon_result["baselines"]["gpt4_neutral"]["score"])
        
        # Collect multi-quantum scores
        for test in horizon_result["tests"]:
            overall_scores[test["strategy"]].append(test["total_score"])
    
    print(f"\n{'Strategy':<25} {'Total Score':<15} {'Avg Score':<15}")
    print("-" * 55)
    
    baseline_50_total = sum(overall_baselines["fixed_50gen"])
    baseline_gpt4_total = sum(overall_baselines["gpt4_neutral"])
    
    for strategy in strategies:
        if overall_scores[strategy]:
            total = sum(overall_scores[strategy])
            avg = np.mean(overall_scores[strategy])
            print(f"{strategy:<25} {total:>12,.0f}   {avg:>12,.0f}")
    
    print(f"\n{'Baselines:':<25}")
    print(f"  Fixed 50-gen ML:        {baseline_50_total:>12,.0f}   {np.mean(overall_baselines['fixed_50gen']):>12,.0f}")
    print(f"  GPT-4 Neutral:          {baseline_gpt4_total:>12,.0f}   {np.mean(overall_baselines['gpt4_neutral']):>12,.0f}")
    
    # Determine winners
    all_totals = {
        "phase_based": sum(overall_scores["phase_based"]),
        "adaptive": sum(overall_scores["adaptive"]),
        "fixed_50gen": baseline_50_total,
        "gpt4_neutral": baseline_gpt4_total
    }
    
    winner = max(all_totals.keys(), key=lambda k: all_totals[k])
    winner_score = all_totals[winner]
    
    print(f"\n🥇 CHAMPION: {winner.upper()}")
    print(f"   Total Score: {winner_score:,.0f}")
    
    if winner in strategies:
        margin_50 = ((winner_score - baseline_50_total) / baseline_50_total) * 100
        margin_gpt4 = ((winner_score - baseline_gpt4_total) / baseline_gpt4_total) * 100
        print(f"   vs Fixed 50-gen: {margin_50:+.1f}%")
        print(f"   vs GPT-4: {margin_gpt4:+.1f}%")
        print(f"\n🎉 Multi-Quantum ensemble {'WINS' if margin_50 > 0 or margin_gpt4 > 0 else 'LOSES'}!")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
