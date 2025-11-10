"""
Comprehensive God Controller Testing Suite

Experiments:
1. Test original Quantum ML (50-gen trained) at 50, 100, 200 generations
2. Test retrained Quantum ML (200-gen trained) at 50, 100, 200 generations  
3. Test GPT-4 with conservative prompts at 50, 100, 200 generations
4. Test GPT-4 with neutral prompts at 50, 100, 200 generations
5. Test GPT-4 with aggressive prompts at 50, 100, 200 generations
6. Compare all results

This will definitively show:
- Whether retraining fixes Quantum ML performance drop
- How much prompt bias affects GPT-4
- Optimal strategies for different time horizons
"""

import json
import time
from datetime import datetime
from prisoner_echo_god import run_god_echo_simulation
from quantum_god_controller import QuantumGodController
from llm_god_controller import LLMGodController
import os

def run_single_test(mode, generations, run_num, genome=None, prompt_style=None):
    """Run a single test and return results."""
    print(f"\n  Run {run_num}/{2}: ", end="", flush=True)
    
    try:
        # Set up controller based on mode
        if mode.startswith("QUANTUM_200GEN"):
            # Use retrained genome
            controller = QuantumGodController(environment='standard')
            controller.champion_genome = genome
            result = run_god_echo_simulation(
                generations=generations,
                initial_size=300,
                god_mode="ML_BASED",
                update_frequency=9999
            )
        elif mode.startswith("QUANTUM_50GEN"):
            # Use original genome
            result = run_god_echo_simulation(
                generations=generations,
                initial_size=300,
                god_mode="ML_BASED",
                update_frequency=9999
            )
        elif mode.startswith("GPT4"):
            # Use GPT-4 with specified prompt style
            result = run_god_echo_simulation(
                generations=generations,
                initial_size=300,
                god_mode="API_BASED",
                update_frequency=9999,
                prompt_style=prompt_style  # Will pass to LLM controller
            )
        else:
            # Baseline
            result = run_god_echo_simulation(
                generations=generations,
                initial_size=300,
                god_mode="DISABLED",
                update_frequency=9999
            )
        
        # Calculate metrics
        if len(result.agents) > 0:
            final_pop = len(result.agents)
            avg_wealth = sum(a.resources for a in result.agents) / final_pop
            total_wealth = sum(a.resources for a in result.agents)
            
            total_actions = sum(a.cooperations + a.defections for a in result.agents)
            total_coop = sum(a.cooperations for a in result.agents)
            coop_rate = (total_coop / total_actions * 100) if total_actions > 0 else 0
            
            # Gini
            resources = sorted([a.resources for a in result.agents])
            n = len(resources)
            cumsum = sum((i + 1) * r for i, r in enumerate(resources))
            gini = (2 * cumsum) / (n * sum(resources)) - (n + 1) / n
            
            # Interventions
            interventions = len(result.god.intervention_history) if hasattr(result.god, 'intervention_history') else 0
            
            score = total_wealth / 100 + coop_rate
            
            print(f"✅ Wealth=${avg_wealth:.0f}, Coop={coop_rate:.1f}%, Gini={gini:.3f}, Score={score:.1f}")
            
            return {
                'success': True,
                'final_population': final_pop,
                'avg_wealth': avg_wealth,
                'total_wealth': total_wealth,
                'cooperation_rate': coop_rate,
                'gini': gini,
                'interventions': interventions,
                'score': score
            }
        else:
            print("❌ EXTINCT")
            return {'success': False, 'score': 0}
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {'success': False, 'error': str(e), 'score': 0}

def run_experiment_set(name, mode, test_lengths, genome=None, prompt_style=None):
    """Run a set of experiments across different generation lengths."""
    print(f"\n{'='*80}")
    print(f"🧪 {name}")
    print(f"{'='*80}")
    
    results = {}
    
    for gens in test_lengths:
        print(f"\n📊 Testing at {gens} generations:")
        
        runs = []
        for run in range(2):
            run_result = run_single_test(mode, gens, run + 1, genome, prompt_style)
            runs.append(run_result)
        
        # Calculate averages
        successful_runs = [r for r in runs if r.get('success', False)]
        if successful_runs:
            avg_result = {
                'runs': runs,
                'avg_wealth': sum(r['avg_wealth'] for r in successful_runs) / len(successful_runs),
                'avg_cooperation': sum(r['cooperation_rate'] for r in successful_runs) / len(successful_runs),
                'avg_gini': sum(r['gini'] for r in successful_runs) / len(successful_runs),
                'avg_interventions': sum(r['interventions'] for r in successful_runs) / len(successful_runs),
                'avg_score': sum(r['score'] for r in successful_runs) / len(successful_runs),
                'success_rate': len(successful_runs) / len(runs) * 100
            }
            
            print(f"\n  📈 {gens}-gen Average:")
            print(f"     Wealth: ${avg_result['avg_wealth']:.0f}")
            print(f"     Cooperation: {avg_result['avg_cooperation']:.1f}%")
            print(f"     Gini: {avg_result['avg_gini']:.3f}")
            print(f"     Interventions: {avg_result['avg_interventions']:.1f}")
            print(f"     Score: {avg_result['avg_score']:.1f}")
        else:
            avg_result = {'success_rate': 0, 'avg_score': 0}
            print(f"\n  ❌ All runs failed at {gens} generations")
        
        results[f"{gens}_gen"] = avg_result
    
    return results

def main():
    """Run all experiments."""
    print("\n" + "="*80)
    print("🔬 COMPREHENSIVE GOD CONTROLLER TESTING SUITE")
    print("="*80)
    
    # Check for retrained genome
    retrained_genome = None
    retrained_available = False
    
    # Look for most recent 200-gen training results
    import glob
    training_files = glob.glob("outputs/god_ai/quantum_evolution_200gen_*.json")
    if training_files:
        latest_file = max(training_files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            data = json.load(f)
            retrained_genome = data['champion']['genome']
            retrained_available = True
            print(f"\n✅ Found retrained 200-gen genome from: {latest_file}")
            print(f"   Champion score: {data['champion']['score']:.1f}")
    else:
        print(f"\n⚠️  No retrained 200-gen genome found!")
        print(f"   Run train_quantum_200gen.py first to generate one.")
        print(f"   For now, we'll skip Quantum 200-gen trained tests.\n")
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    gpt4_available = api_key is not None
    
    if not gpt4_available:
        print(f"\n⚠️  OPENAI_API_KEY not set!")
        print(f"   GPT-4 tests will be skipped.\n")
    
    # Confirm testing plan
    test_lengths = [50, 100, 200]
    
    experiments = []
    experiments.append(("Baseline (No God)", "DISABLED", None, None))
    experiments.append(("Quantum ML (50-gen trained)", "QUANTUM_50GEN", None, None))
    
    if retrained_available:
        experiments.append(("Quantum ML (200-gen trained)", "QUANTUM_200GEN", retrained_genome, None))
    
    if gpt4_available:
        experiments.append(("GPT-4 Conservative Prompt", "GPT4_CONSERVATIVE", None, "conservative"))
        experiments.append(("GPT-4 Neutral Prompt", "GPT4_NEUTRAL", None, "neutral"))
        experiments.append(("GPT-4 Aggressive Prompt", "GPT4_AGGRESSIVE", None, "aggressive"))
    
    print(f"\n📋 Test Plan:")
    print(f"   Test lengths: {test_lengths} generations")
    print(f"   Experiments: {len(experiments)}")
    print(f"   Total runs: {len(experiments) * len(test_lengths) * 2}")
    print(f"   Estimated time: ~{len(experiments) * len(test_lengths) * 2 * 2 / 60:.0f} minutes")
    
    print(f"\n🧪 Experiments to run:")
    for i, (name, _, _, _) in enumerate(experiments, 1):
        print(f"   {i}. {name}")
    
    print("\n" + "="*80)
    response = input("\n⚠️  Continue with all experiments? (yes/no): ").strip().lower()
    if response != 'yes':
        print("❌ Testing cancelled")
        return
    
    print("\n🚀 Starting comprehensive testing...\n")
    start_time = time.time()
    
    # Run all experiments
    all_results = {}
    
    for name, mode, genome, prompt_style in experiments:
        results = run_experiment_set(name, mode, test_lengths, genome, prompt_style)
        all_results[name] = results
    
    elapsed = time.time() - start_time
    
    # Save results
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'test_lengths': test_lengths,
        'experiments': list(all_results.keys()),
        'results': all_results,
        'elapsed_time': elapsed
    }
    
    output_file = f"outputs/god_ai/comprehensive_tests_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    for test_length in test_lengths:
        print(f"\n{'='*80}")
        print(f"🎯 {test_length} GENERATION TESTS")
        print(f"{'='*80}")
        
        scores = []
        for exp_name, exp_results in all_results.items():
            gen_key = f"{test_length}_gen"
            if gen_key in exp_results:
                result = exp_results[gen_key]
                score = result.get('avg_score', 0)
                coop = result.get('avg_cooperation', 0)
                wealth = result.get('avg_wealth', 0)
                gini = result.get('avg_gini', 0)
                
                scores.append((score, exp_name, coop, wealth, gini))
        
        # Sort by score
        scores.sort(reverse=True)
        
        for rank, (score, name, coop, wealth, gini) in enumerate(scores, 1):
            medal = ["🥇", "🥈", "🥉"][rank-1] if rank <= 3 else f"{rank}️⃣"
            print(f"\n{medal} {name}")
            print(f"   Score: {score:.1f}")
            print(f"   Wealth: ${wealth:.0f}")
            print(f"   Cooperation: {coop:.1f}%")
            print(f"   Gini: {gini:.3f}")
    
    # Key insights
    print("\n" + "="*80)
    print("💡 KEY INSIGHTS")
    print("="*80)
    
    # Check if retraining helped
    if retrained_available:
        print("\n🧬 QUANTUM ML RETRAINING IMPACT:")
        for test_length in test_lengths:
            old_score = all_results.get("Quantum ML (50-gen trained)", {}).get(f"{test_length}_gen", {}).get('avg_score', 0)
            new_score = all_results.get("Quantum ML (200-gen trained)", {}).get(f"{test_length}_gen", {}).get('avg_score', 0)
            
            if old_score and new_score:
                improvement = ((new_score - old_score) / old_score * 100) if old_score > 0 else 0
                symbol = "✅" if improvement > 0 else "❌"
                print(f"   {test_length} gen: {old_score:.1f} → {new_score:.1f} ({improvement:+.1f}%) {symbol}")
    
    # Check prompt impact
    if gpt4_available:
        print("\n🎭 GPT-4 PROMPT BIAS IMPACT:")
        for test_length in test_lengths:
            conservative = all_results.get("GPT-4 Conservative Prompt", {}).get(f"{test_length}_gen", {}).get('avg_score', 0)
            neutral = all_results.get("GPT-4 Neutral Prompt", {}).get(f"{test_length}_gen", {}).get('avg_score', 0)
            aggressive = all_results.get("GPT-4 Aggressive Prompt", {}).get(f"{test_length}_gen", {}).get('avg_score', 0)
            
            if conservative and neutral and aggressive:
                print(f"   {test_length} gen:")
                print(f"     Conservative: {conservative:.1f}")
                print(f"     Neutral: {neutral:.1f}")
                print(f"     Aggressive: {aggressive:.1f}")
                
                if neutral > conservative and neutral > aggressive:
                    print(f"     → Neutral wins! ✅")
                elif conservative > neutral and conservative > aggressive:
                    print(f"     → Conservative wins! (original)")
                else:
                    print(f"     → Aggressive wins!")
    
    print("\n" + "="*80)
    print(f"✅ ALL EXPERIMENTS COMPLETE!")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
