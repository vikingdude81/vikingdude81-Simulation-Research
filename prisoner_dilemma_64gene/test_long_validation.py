"""
🔬 LONG-TERM VALIDATION TEST (100 Generations)

This test runs a comprehensive 100-generation comparison to validate
the findings from the 50-generation ultimate showdown:
- Quantum ML God should maintain highest cooperation (71.4%+)
- Rule-Based should remain consistent (~70%)
- GPT-4 should show sophisticated reasoning but be slower
- All interventions should improve outcomes vs baseline

This validation test uses 100 generations to see if patterns hold over time.
"""

from prisoner_echo_god import run_god_echo_simulation
import json
from datetime import datetime
import os
import time

def run_long_validation(generations=100, initial_size=300, runs_per_mode=2):
    """
    Run extended validation test with 100+ generations.
    
    Args:
        generations: Number of generations per run (default 100)
        initial_size: Initial population
        runs_per_mode: How many times to run each mode
    """
    print("\n" + "="*70)
    print("🔬 LONG-TERM VALIDATION TEST")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"   Generations: {generations} (extended test)")
    print(f"   Initial population: {initial_size}")
    print(f"   Runs per mode: {runs_per_mode}")
    print(f"   Total runs: {runs_per_mode * 4} ({runs_per_mode} × 4 modes)")
    
    # Estimate time and cost
    estimated_time = (runs_per_mode * 3 * 2.2) + (runs_per_mode * generations / 2.2)  # Rule/ML + API
    estimated_cost = (runs_per_mode * generations / 10) * 0.003
    
    print(f"\n⏱️  Estimated time: {estimated_time/60:.1f} minutes")
    print(f"💰 Estimated API cost: ${estimated_cost:.2f} (GPT-4 only)")
    
    response = input("\n⚠️  Continue with real GPT-4 API calls? (yes/no): ").strip().lower()
    if response != 'yes':
        print("\n❌ Test cancelled")
        return
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ OPENAI_API_KEY not found in environment")
        return
    
    print(f"\n✅ API key found: sk-...{api_key[-4:]}")
    print("\n🚀 Starting long-term validation...\n")
    print("⏳ This will take several minutes. Please be patient...")
    
    modes = ["DISABLED", "RULE_BASED", "ML_BASED", "API_BASED"]
    mode_names = {
        "DISABLED": "No God (Baseline)",
        "RULE_BASED": "Rule-Based God",
        "ML_BASED": "Quantum ML God",
        "API_BASED": "GPT-4 API God"
    }
    
    all_results = {}
    overall_start = time.time()
    
    for mode_idx, mode in enumerate(modes):
        print("\n" + "="*70)
        print(f"🎯 TESTING: {mode_names[mode]} ({mode_idx+1}/4)")
        print("="*70)
        
        mode_results = []
        
        for run in range(1, runs_per_mode + 1):
            print(f"\n📊 Run {run}/{runs_per_mode} - {generations} generations")
            
            start_time = time.time()
            
            try:
                result = run_god_echo_simulation(
                    generations=generations,
                    initial_size=initial_size,
                    god_mode=mode,
                    update_frequency=50  # Update every 50 generations
                )
                
                elapsed = time.time() - start_time
                
                # Calculate comprehensive metrics
                final_pop = len(result.agents)
                
                if final_pop > 0:
                    # Basic metrics
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
                    
                    # Age distribution
                    avg_age = sum(a.age for a in result.agents) / final_pop
                    max_age = max(a.age for a in result.agents)
                    
                    # Wealth distribution
                    min_wealth = min(a.resources for a in result.agents)
                    max_wealth = max(a.resources for a in result.agents)
                    
                    survival_rate = final_pop / initial_size
                    
                    # Get intervention count and breakdown
                    intervention_count = 0
                    intervention_types = {}
                    if result.god:
                        intervention_count = len(result.god.intervention_history)
                        for record in result.god.intervention_history:
                            itype = record.intervention_type.name
                            intervention_types[itype] = intervention_types.get(itype, 0) + 1
                    
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
                        'min_wealth': min_wealth,
                        'max_wealth': max_wealth,
                        'cooperation_rate': coop_rate,
                        'gini_coefficient': gini,
                        'avg_age': avg_age,
                        'max_age': max_age,
                        'survival_rate': survival_rate,
                        'interventions': intervention_count,
                        'intervention_types': intervention_types,
                        'elapsed_time': elapsed,
                        'generations_per_second': generations / elapsed,
                        'api_calls': api_calls,
                        'failed_api_calls': failed_calls
                    })
                    
                    print(f"\n   ✅ Completed in {elapsed:.1f}s ({generations/elapsed:.1f} gen/s)")
                    print(f"      Population: {final_pop}/{initial_size} ({survival_rate:.1%})")
                    print(f"      Avg wealth: ${avg_wealth:.0f} (range: ${min_wealth:.0f}-${max_wealth:.0f})")
                    print(f"      Cooperation: {coop_rate:.1%}")
                    print(f"      Gini: {gini:.3f}")
                    print(f"      Avg age: {avg_age:.1f} (max: {max_age})")
                    print(f"      Interventions: {intervention_count} ({', '.join(f'{k}: {v}' for k, v in intervention_types.items())})")
                    if mode == "API_BASED":
                        print(f"      API calls: {api_calls} (failed: {failed_calls})")
                        print(f"      Estimated cost: ${api_calls * 0.003:.3f}")
                
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
                import traceback
                traceback.print_exc()
                mode_results.append({
                    'run': run,
                    'error': str(e)
                })
        
        all_results[mode] = mode_results
        
        # Show progress
        elapsed_overall = time.time() - overall_start
        remaining_modes = 4 - (mode_idx + 1)
        if mode_idx > 0:
            avg_time_per_mode = elapsed_overall / (mode_idx + 1)
            estimated_remaining = avg_time_per_mode * remaining_modes
            print(f"\n   ⏱️  Progress: {mode_idx+1}/4 modes complete")
            print(f"      Time so far: {elapsed_overall/60:.1f} min")
            print(f"      Estimated remaining: {estimated_remaining/60:.1f} min")
    
    # Calculate aggregated results
    print("\n" + "="*70)
    print("📊 FINAL RESULTS - 100 GENERATION VALIDATION")
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
                'avg_age': sum(r['avg_age'] for r in successful) / len(successful),
                'max_age': max(r['max_age'] for r in successful),
                'avg_interventions': sum(r['interventions'] for r in successful) / len(successful),
                'avg_time': sum(r['elapsed_time'] for r in results) / len(results),
                'avg_speed': sum(r['generations_per_second'] for r in successful) / len(successful),
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
            print(f"   Avg Gini: {s['avg_gini']:.3f} (inequality)")
            print(f"   Avg age: {s['avg_age']:.1f} (max: {s['max_age']})")
            print(f"   Avg interventions: {s['avg_interventions']:.1f}")
            print(f"   Avg speed: {s['avg_speed']:.1f} gen/s")
            print(f"   Avg time: {s['avg_time']:.1f}s")
            
            if mode == "API_BASED" and 'estimated_cost' in s:
                print(f"   Total API calls: {s['total_api_calls']} (failed: {s['total_failed_calls']})")
                print(f"   Total cost: ${s['estimated_cost']:.3f}")
    
    # Calculate winner
    print("\n" + "="*70)
    print("🏆 VALIDATION RESULTS")
    print("="*70)
    
    # Score = wealth/100 + cooperation*50 + (1-gini)*100 + survival*100
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
        for rank, mode in enumerate(sorted(scores, key=scores.get, reverse=True), 1):
            emoji = ["🥇", "🥈", "🥉", "4️⃣"][rank-1]
            print(f"   {emoji} {mode_names[mode]}: {scores[mode]:.1f}")
        
        # Validation check
        print(f"\n" + "="*70)
        print("✅ VALIDATION CHECK: 100-Gen vs 50-Gen Results")
        print("="*70)
        
        # Expected from 50-gen test
        expected_50gen = {
            "DISABLED": {"coop": 0.659, "wealth": 4832},
            "RULE_BASED": {"coop": 0.700, "wealth": 5152},
            "ML_BASED": {"coop": 0.714, "wealth": 5134},
            "API_BASED": {"coop": 0.640, "wealth": 4900}
        }
        
        print("\nCooperation Rate Comparison:")
        print(f"{'Mode':<20} {'50-Gen':<12} {'100-Gen':<12} {'Change':<12}")
        print("-"*70)
        for mode in modes:
            if summary[mode]['runs'] > 0:
                coop_50 = expected_50gen[mode]['coop']
                coop_100 = summary[mode]['avg_cooperation']
                change = ((coop_100 - coop_50) / coop_50) * 100
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"{mode_names[mode]:<20} {coop_50:<11.1%} {coop_100:<11.1%} {arrow} {abs(change):.1f}%")
        
        print("\nWealth Comparison:")
        print(f"{'Mode':<20} {'50-Gen':<12} {'100-Gen':<12} {'Change':<12}")
        print("-"*70)
        for mode in modes:
            if summary[mode]['runs'] > 0:
                wealth_50 = expected_50gen[mode]['wealth']
                wealth_100 = summary[mode]['avg_wealth']
                change = ((wealth_100 - wealth_50) / wealth_50) * 100
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"{mode_names[mode]:<20} ${wealth_50:<11.0f} ${wealth_100:<11.0f} {arrow} {abs(change):.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/god_ai/long_validation_{timestamp}.json"
    os.makedirs("outputs/god_ai", exist_ok=True)
    
    total_time = time.time() - overall_start
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'generations': generations,
                'initial_size': initial_size,
                'runs_per_mode': runs_per_mode,
                'total_time_minutes': total_time / 60
            },
            'results': all_results,
            'summary': summary,
            'scores': scores,
            'winner': winner if scores else None
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"\n⏱️  Total test time: {total_time/60:.1f} minutes")
    print("\n✅ Long-term validation complete!")


if __name__ == "__main__":
    run_long_validation(
        generations=100,  # Extended test
        initial_size=300,
        runs_per_mode=2  # 2 runs per mode = 8 total runs
    )
