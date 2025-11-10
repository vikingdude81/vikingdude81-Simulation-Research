"""
🏆 ULTIMATE GOD CONTROLLER SHOWDOWN

Comparative test: DISABLED vs RULE_BASED vs QUANTUM_ML vs GPT-4_API

This test will pit all four god controllers against each other to see which
produces the best economic outcomes. We'll run multiple trials and compare:
- Final wealth
- Cooperation rates
- Inequality (Gini coefficient)
- Intervention efficiency
- Cost (API calls for GPT-4)

WARNING: This test makes real API calls to OpenAI! 
Cost: ~$0.24 per 100-generation run with GPT-4
"""

from prisoner_echo_god import run_god_echo_simulation
import json
from datetime import datetime
import os
import time

def run_ultimate_showdown(generations=100, initial_size=300, runs_per_mode=2):
    """
    Run the ultimate comparison test with all 4 god modes.
    
    Args:
        generations: Number of generations per run
        initial_size: Initial population
        runs_per_mode: How many times to run each mode
    
    Cost estimate: runs_per_mode * $0.24 for API mode
    """
    print("\n" + "="*70)
    print("🏆 ULTIMATE GOD CONTROLLER SHOWDOWN")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"   Generations: {generations}")
    print(f"   Initial population: {initial_size}")
    print(f"   Runs per mode: {runs_per_mode}")
    print(f"   Total runs: {runs_per_mode * 4} ({runs_per_mode} × 4 modes)")
    
    # Estimate cost
    estimated_cost = (runs_per_mode * generations / 10) * 0.003  # Rough estimate
    print(f"\n💰 Estimated API cost: ${estimated_cost:.2f} (GPT-4 only)")
    print(f"   (Based on ~1 intervention per 10 generations)")
    
    response = input("\n⚠️  Continue with real GPT-4 API calls? (yes/no): ").strip().lower()
    if response != 'yes':
        print("\n❌ Test cancelled")
        return
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ OPENAI_API_KEY not found in environment")
        print("   Set it with: $env:OPENAI_API_KEY='sk-...'")
        return
    
    print(f"\n✅ API key found: sk-...{api_key[-4:]}")
    print("\n🚀 Starting ultimate showdown...\n")
    
    modes = ["DISABLED", "RULE_BASED", "ML_BASED", "API_BASED"]
    mode_names = {
        "DISABLED": "No God (Baseline)",
        "RULE_BASED": "Rule-Based God",
        "ML_BASED": "Quantum ML God",
        "API_BASED": "GPT-4 API God"
    }
    
    all_results = {}
    
    for mode in modes:
        print("\n" + "="*70)
        print(f"🎯 TESTING: {mode_names[mode]}")
        print("="*70)
        
        mode_results = []
        
        for run in range(1, runs_per_mode + 1):
            print(f"\n📊 Run {run}/{runs_per_mode}")
            
            start_time = time.time()
            
            try:
                result = run_god_echo_simulation(
                    generations=generations,
                    initial_size=initial_size,
                    god_mode=mode,
                    update_frequency=25  # Less frequent updates
                )
                
                elapsed = time.time() - start_time
                
                # Calculate metrics
                final_pop = len(result.agents)
                
                if final_pop > 0:
                    avg_wealth = sum(a.resources for a in result.agents) / final_pop
                    total_actions = sum(a.cooperations + a.defections for a in result.agents)
                    total_coop = sum(a.cooperations for a in result.agents)
                    coop_rate = (total_coop / total_actions) if total_actions > 0 else 0
                    
                    # Calculate Gini coefficient
                    resources = sorted([a.resources for a in result.agents])
                    n = len(resources)
                    cumsum = 0
                    for i, r in enumerate(resources):
                        cumsum += (i + 1) * r
                    gini = (2 * cumsum) / (n * sum(resources)) - (n + 1) / n
                    
                    survival_rate = final_pop / initial_size
                    
                    # Get intervention count
                    intervention_count = 0
                    if result.god:
                        intervention_count = len(result.god.intervention_history)
                    
                    # API stats for GPT-4
                    api_calls = 0
                    failed_calls = 0
                    if mode == "API_BASED" and result.god and result.god.llm_controller:
                        stats = result.god.llm_controller.get_statistics()
                        api_calls = stats['total_api_calls']
                        failed_calls = stats['failed_calls']
                    
                    mode_results.append({
                        'run': run,
                        'final_population': final_pop,
                        'avg_wealth': avg_wealth,
                        'cooperation_rate': coop_rate,
                        'gini_coefficient': gini,
                        'survival_rate': survival_rate,
                        'interventions': intervention_count,
                        'elapsed_time': elapsed,
                        'api_calls': api_calls,
                        'failed_api_calls': failed_calls
                    })
                    
                    print(f"   ✅ Completed in {elapsed:.1f}s")
                    print(f"      Population: {final_pop}/{initial_size}")
                    print(f"      Avg wealth: ${avg_wealth:.0f}")
                    print(f"      Cooperation: {coop_rate:.1%}")
                    print(f"      Gini: {gini:.3f}")
                    print(f"      Interventions: {intervention_count}")
                    if mode == "API_BASED":
                        print(f"      API calls: {api_calls} (failed: {failed_calls})")
                
                else:
                    print(f"   ❌ Population collapsed!")
                    mode_results.append({
                        'run': run,
                        'final_population': 0,
                        'survival_rate': 0,
                        'elapsed_time': elapsed,
                        'collapsed': True
                    })
            
            except Exception as e:
                print(f"   ❌ Error: {e}")
                mode_results.append({
                    'run': run,
                    'error': str(e)
                })
        
        all_results[mode] = mode_results
    
    # Calculate aggregated results
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    
    summary = {}
    for mode in modes:
        results = all_results[mode]
        successful = [r for r in results if 'final_population' in r and r['final_population'] > 0]
        
        if successful:
            summary[mode] = {
                'name': mode_names[mode],
                'success_rate': len(successful) / len(results),
                'avg_population': sum(r['final_population'] for r in successful) / len(successful),
                'avg_wealth': sum(r['avg_wealth'] for r in successful) / len(successful),
                'avg_cooperation': sum(r['cooperation_rate'] for r in successful) / len(successful),
                'avg_gini': sum(r['gini_coefficient'] for r in successful) / len(successful),
                'avg_interventions': sum(r['interventions'] for r in successful) / len(successful),
                'avg_time': sum(r['elapsed_time'] for r in results) / len(results),
                'runs': len(successful)
            }
            
            if mode == "API_BASED":
                summary[mode]['total_api_calls'] = sum(r.get('api_calls', 0) for r in successful)
                summary[mode]['total_failed_calls'] = sum(r.get('failed_api_calls', 0) for r in successful)
                estimated_cost = summary[mode]['total_api_calls'] * 0.003
                summary[mode]['estimated_cost'] = estimated_cost
        else:
            summary[mode] = {
                'name': mode_names[mode],
                'success_rate': 0,
                'runs': 0
            }
    
    # Display comparison
    print("\n" + "─"*70)
    for mode in modes:
        s = summary[mode]
        print(f"\n🎯 {s['name']}:")
        print(f"   Success rate: {s['success_rate']:.0%} ({s['runs']}/{runs_per_mode})")
        
        if s['runs'] > 0:
            print(f"   Avg population: {s['avg_population']:.0f}")
            print(f"   Avg wealth: ${s['avg_wealth']:.0f}")
            print(f"   Avg cooperation: {s['avg_cooperation']:.1%}")
            print(f"   Avg Gini: {s['avg_gini']:.3f}")
            print(f"   Avg interventions: {s['avg_interventions']:.1f}")
            print(f"   Avg time: {s['avg_time']:.1f}s")
            
            if mode == "API_BASED" and 'estimated_cost' in s:
                print(f"   Total API calls: {s['total_api_calls']} (failed: {s['total_failed_calls']})")
                print(f"   Estimated cost: ${s['estimated_cost']:.3f}")
    
    # Calculate winner
    print("\n" + "="*70)
    print("🏆 WINNER")
    print("="*70)
    
    # Score = wealth + cooperation*50 + (1-gini)*100 - time_penalty
    scores = {}
    for mode in modes:
        s = summary[mode]
        if s['runs'] > 0:
            score = (
                s['avg_wealth'] / 100 +
                s['avg_cooperation'] * 50 +
                (1 - s['avg_gini']) * 100 +
                s['success_rate'] * 100
            )
            scores[mode] = score
    
    if scores:
        winner = max(scores, key=scores.get)
        print(f"\n🥇 Winner: {mode_names[winner]}")
        print(f"   Score: {scores[winner]:.1f}")
        print(f"\n📊 All scores:")
        for mode in sorted(scores, key=scores.get, reverse=True):
            print(f"   {mode_names[mode]}: {scores[mode]:.1f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/god_ai/ultimate_showdown_{timestamp}.json"
    os.makedirs("outputs/god_ai", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'generations': generations,
                'initial_size': initial_size,
                'runs_per_mode': runs_per_mode
            },
            'results': all_results,
            'summary': summary,
            'scores': scores,
            'winner': winner if scores else None
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n✅ Ultimate showdown complete!")


if __name__ == "__main__":
    # Run with relatively short test (to manage API costs)
    run_ultimate_showdown(
        generations=50,  # Shorter test to manage costs
        initial_size=300,
        runs_per_mode=2  # 2 runs per mode = 8 total runs
    )
