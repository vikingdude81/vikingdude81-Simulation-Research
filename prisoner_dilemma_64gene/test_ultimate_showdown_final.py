"""
ULTIMATE SHOWDOWN: Quantum ML (50-gen vs 150-gen) vs GPT-4 (All Prompts)

Tests all controllers at 50, 100, and 150 generations:
1. Quantum ML (50-gen trained) - Original champion
2. Quantum ML (150-gen trained) - NEW retrained champion
3. GPT-4 Conservative - Original prompt
4. GPT-4 Neutral - WINNER from prompt bias test
5. Baseline (No God) - Natural selection

This will definitively answer:
- Does retraining fix Quantum ML's performance drop?
- Which controller is best at each time horizon?
- Is GPT-4 or Quantum ML better overall?
"""

import json
import os
import glob
from datetime import datetime
from prisoner_echo_god import run_god_echo_simulation
from quantum_god_controller import QuantumGodController

def run_single_test(name, mode, generations, run_num, genome=None, prompt_style=None):
    """Run a single test."""
    print(f"  Run {run_num}/2: ", end="", flush=True)
    
    try:
        if mode == "QUANTUM_150GEN":
            # Use NEW retrained genome
            controller = QuantumGodController(environment='standard')
            controller.champion_genome = genome
            result = run_god_echo_simulation(
                generations=generations,
                initial_size=300,
                god_mode="ML_BASED",
                update_frequency=99999
            )
        elif mode == "QUANTUM_50GEN":
            # Use ORIGINAL genome (default in quantum_god_controller.py)
            result = run_god_echo_simulation(
                generations=generations,
                initial_size=300,
                god_mode="ML_BASED",
                update_frequency=99999
            )
        elif mode.startswith("GPT4"):
            # Use GPT-4 with specified prompt
            result = run_god_echo_simulation(
                generations=generations,
                initial_size=300,
                god_mode="API_BASED",
                update_frequency=99999,
                prompt_style=prompt_style
            )
        else:
            # Baseline
            result = run_god_echo_simulation(
                generations=generations,
                initial_size=300,
                god_mode="DISABLED",
                update_frequency=99999
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
            
            interventions = len(result.god.intervention_history) if hasattr(result.god, 'intervention_history') else 0
            
            score = total_wealth / 100 + coop_rate
            
            print(f"✅ ${avg_wealth:.0f} {coop_rate:.1f}% gini={gini:.3f} score={score:.1f}")
            
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

def ultimate_showdown():
    """Run comprehensive comparison."""
    
    print("\n" + "="*80)
    print("⚔️  ULTIMATE SHOWDOWN: QUANTUM ML vs GPT-4")
    print("="*80)
    
    # Load NEW 150-gen trained genome
    training_files = glob.glob("outputs/god_ai/quantum_evolution_150gen_*.json")
    if not training_files:
        print("\n❌ ERROR: No 150-gen trained genome found!")
        print("   Run train_quantum_150gen_fast.py first!")
        return
    
    latest_file = max(training_files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        data = json.load(f)
        retrained_genome = data['champion']['genome']
        retrained_score = data['champion']['score']
    
    print(f"\n✅ Loaded retrained genome from: {os.path.basename(latest_file)}")
    print(f"   Training score: {retrained_score:.1f}")
    
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not set - GPT-4 tests will be skipped")
        print("   Set with: $env:OPENAI_API_KEY='your-key-here'")
        gpt4_available = False
    else:
        gpt4_available = True
        print(f"\n✅ GPT-4 API available")
    
    # Test configuration
    test_lengths = [50, 100, 150]
    
    controllers = [
        ("Baseline (No God)", "BASELINE", None, None),
        ("Quantum ML (50-gen trained)", "QUANTUM_50GEN", None, None),
        ("Quantum ML (150-gen trained)", "QUANTUM_150GEN", retrained_genome, None),
    ]
    
    if gpt4_available:
        controllers.extend([
            ("GPT-4 Conservative", "GPT4_CONSERVATIVE", None, "conservative"),
            ("GPT-4 Neutral", "GPT4_NEUTRAL", None, "neutral"),
        ])
    
    print(f"\n📋 Test Plan:")
    print(f"   Test lengths: {test_lengths} generations")
    print(f"   Controllers: {len(controllers)}")
    print(f"   Runs per test: 2")
    print(f"   Total runs: {len(controllers) * len(test_lengths) * 2}")
    
    if gpt4_available:
        print(f"   Estimated time: ~{len(controllers) * len(test_lengths) * 2 * 1.5 / 60:.0f} minutes")
        print(f"   Estimated cost: ~${len(test_lengths) * 2 * 2 * 0.03:.2f}")
    else:
        print(f"   Estimated time: ~{(len(controllers)) * len(test_lengths) * 2 * 0.5 / 60:.0f} minutes")
    
    print("\n🎯 Controllers:")
    for name, _, _, _ in controllers:
        print(f"   • {name}")
    
    print("\n" + "="*80)
    
    # Run tests
    all_results = {}
    
    for name, mode, genome, prompt_style in controllers:
        print(f"\n{'='*80}")
        print(f"🧪 TESTING: {name}")
        print(f"{'='*80}")
        
        controller_results = {}
        
        for gens in test_lengths:
            print(f"\n📊 {gens} generations:")
            
            runs = []
            for run in range(2):
                run_result = run_single_test(name, mode, gens, run + 1, genome, prompt_style)
                runs.append(run_result)
            
            # Calculate averages
            successful = [r for r in runs if r.get('success', False)]
            if successful:
                avg_result = {
                    'runs': runs,
                    'avg_wealth': sum(r['avg_wealth'] for r in successful) / len(successful),
                    'avg_cooperation': sum(r['cooperation_rate'] for r in successful) / len(successful),
                    'avg_gini': sum(r['gini'] for r in successful) / len(successful),
                    'avg_interventions': sum(r['interventions'] for r in successful) / len(successful),
                    'avg_score': sum(r['score'] for r in successful) / len(successful),
                    'success_rate': len(successful) / len(runs) * 100
                }
                
                print(f"  📈 Average: ${avg_result['avg_wealth']:.0f} {avg_result['avg_cooperation']:.1f}% score={avg_result['avg_score']:.1f}")
            else:
                avg_result = {'success_rate': 0, 'avg_score': 0}
                print(f"  ❌ All runs failed")
            
            controller_results[f"{gens}_gen"] = avg_result
        
        all_results[name] = controller_results
    
    # Final comparison
    print("\n" + "="*80)
    print("🏆 FINAL RANKINGS")
    print("="*80)
    
    for test_length in test_lengths:
        print(f"\n{'='*80}")
        print(f"📊 {test_length} GENERATION TESTS")
        print(f"{'='*80}")
        
        scores = []
        for name, results in all_results.items():
            gen_key = f"{test_length}_gen"
            if gen_key in results:
                result = results[gen_key]
                score = result.get('avg_score', 0)
                coop = result.get('avg_cooperation', 0)
                wealth = result.get('avg_wealth', 0)
                gini = result.get('avg_gini', 0)
                
                scores.append((score, name, coop, wealth, gini))
        
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
    
    # Compare Quantum ML 50-gen vs 150-gen
    print("\n🧬 QUANTUM ML RETRAINING IMPACT:")
    for test_length in test_lengths:
        old_score = all_results.get("Quantum ML (50-gen trained)", {}).get(f"{test_length}_gen", {}).get('avg_score', 0)
        new_score = all_results.get("Quantum ML (150-gen trained)", {}).get(f"{test_length}_gen", {}).get('avg_score', 0)
        
        if old_score and new_score:
            improvement = ((new_score - old_score) / old_score * 100) if old_score > 0 else 0
            symbol = "✅" if improvement > 0 else "❌"
            print(f"   {test_length} gen: {old_score:.1f} → {new_score:.1f} ({improvement:+.1f}%) {symbol}")
    
    # Compare GPT-4 prompts
    if gpt4_available:
        print("\n🎭 GPT-4 PROMPT COMPARISON:")
        for test_length in test_lengths:
            conservative = all_results.get("GPT-4 Conservative", {}).get(f"{test_length}_gen", {}).get('avg_score', 0)
            neutral = all_results.get("GPT-4 Neutral", {}).get(f"{test_length}_gen", {}).get('avg_score', 0)
            
            if conservative and neutral:
                diff = neutral - conservative
                print(f"   {test_length} gen: Neutral {'+' if diff > 0 else ''}{diff:.1f} vs Conservative")
    
    # Overall winner
    print("\n🏆 OVERALL CHAMPION:")
    all_scores = {}
    for name, results in all_results.items():
        total = sum(r.get('avg_score', 0) for r in results.values())
        all_scores[name] = total
    
    winner = max(all_scores.items(), key=lambda x: x[1])
    print(f"   {winner[0]} - Total Score: {winner[1]:.1f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'test_lengths': test_lengths,
        'controllers': [name for name, _, _, _ in controllers],
        'results': all_results,
        'retrained_genome': retrained_genome
    }
    
    output_file = f"outputs/god_ai/ultimate_showdown_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved: {output_file}")
    print("\n" + "="*80)
    print("✅ ULTIMATE SHOWDOWN COMPLETE!")
    print("="*80 + "\n")

if __name__ == "__main__":
    ultimate_showdown()
