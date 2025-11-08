"""
PROMPT BIAS EXPERIMENT - Test GPT-4 with Different Prompt Styles

This directly answers the user's question: "is chat gpt prompt making it act a certain way?"

We'll test GPT-4 with 3 prompt styles at 100 generations:
1. Conservative (current): "Intervene only when necessary"
2. Neutral: "Analyze and decide"
3. Aggressive: "Intervene frequently"

This will show if the prompt creates bias and which style is actually optimal.
"""

import json
import os
from datetime import datetime
from prisoner_echo_god import run_god_echo_simulation

def test_prompt_styles():
    """Test GPT-4 with different prompt styles."""
    
    print("\n" + "="*80)
    print("🎭 PROMPT BIAS EXPERIMENT")
    print("="*80)
    print("\nTesting how GPT-4's prompt affects governance decisions...\n")
    
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set!")
        print("   Set it with: $env:OPENAI_API_KEY='your-key-here'")
        return
    
    # Test parameters
    generations = 100
    runs_per_style = 2
    
    styles = [
        ("conservative", "Intervene only when necessary"),
        ("neutral", "Analyze and decide"),
        ("aggressive", "Intervene frequently")
    ]
    
    print(f"Configuration:")
    print(f"  Generations: {generations}")
    print(f"  Runs per style: {runs_per_style}")
    print(f"  Total runs: {len(styles) * runs_per_style}")
    print(f"  Estimated time: ~{len(styles) * runs_per_style * 0.5:.0f} minutes")
    print(f"  Estimated cost: ~${len(styles) * runs_per_style * 0.03:.2f}")
    
    print("\n📋 Prompt Styles to Test:")
    for style, desc in styles:
        print(f"   {style.upper()}: '{desc}'")
    
    print("\n" + "="*80)
    
    # Run experiments
    all_results = {}
    
    for style, desc in styles:
        print(f"\n{'='*80}")
        print(f"🎭 TESTING: {style.upper()} PROMPT")
        print(f"   Guidance: '{desc}'")
        print(f"{'='*80}")
        
        runs = []
        for run in range(runs_per_style):
            print(f"\n  Run {run + 1}/{runs_per_style}:")
            
            try:
                result = run_god_echo_simulation(
                    generations=generations,
                    initial_size=300,
                    god_mode="API_BASED",
                    update_frequency=9999,  # Suppress output
                    prompt_style=style
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
                    
                    print(f"     ✅ Wealth=${avg_wealth:.0f}, Coop={coop_rate:.1f}%, Gini={gini:.3f}")
                    print(f"        Interventions={interventions}, Score={score:.1f}")
                    
                    runs.append({
                        'success': True,
                        'final_population': final_pop,
                        'avg_wealth': avg_wealth,
                        'total_wealth': total_wealth,
                        'cooperation_rate': coop_rate,
                        'gini': gini,
                        'interventions': interventions,
                        'score': score
                    })
                else:
                    print(f"     ❌ EXTINCT")
                    runs.append({'success': False, 'score': 0})
            
            except Exception as e:
                print(f"     ❌ ERROR: {e}")
                runs.append({'success': False, 'error': str(e), 'score': 0})
        
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
            
            print(f"\n  📊 {style.upper()} Average:")
            print(f"     Wealth: ${avg_result['avg_wealth']:.0f}")
            print(f"     Cooperation: {avg_result['avg_cooperation']:.1f}%")
            print(f"     Gini: {avg_result['avg_gini']:.3f}")
            print(f"     Interventions: {avg_result['avg_interventions']:.1f}")
            print(f"     Score: {avg_result['avg_score']:.1f}")
        else:
            avg_result = {'success_rate': 0, 'avg_score': 0}
            print(f"\n  ❌ All runs failed for {style}")
        
        all_results[style] = avg_result
    
    # Final comparison
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON - 100 GENERATIONS")
    print("="*80)
    
    # Sort by score
    ranked = [(style, all_results[style].get('avg_score', 0), 
               all_results[style].get('avg_cooperation', 0),
               all_results[style].get('avg_interventions', 0))
              for style in ['conservative', 'neutral', 'aggressive']]
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (style, score, coop, interventions) in enumerate(ranked, 1):
        medal = ["🥇", "🥈", "🥉"][rank-1]
        print(f"\n{medal} {style.upper()} PROMPT")
        print(f"   Score: {score:.1f}")
        print(f"   Cooperation: {coop:.1f}%")
        print(f"   Interventions: {interventions:.1f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'generations': generations,
        'runs_per_style': runs_per_style,
        'results': all_results,
        'ranking': [{'rank': i+1, 'style': s, 'score': sc} 
                    for i, (s, sc, _, _) in enumerate(ranked)]
    }
    
    output_file = f"outputs/god_ai/prompt_bias_experiment_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Analysis
    print("\n" + "="*80)
    print("💡 KEY INSIGHTS")
    print("="*80)
    
    conservative_score = all_results['conservative'].get('avg_score', 0)
    neutral_score = all_results['neutral'].get('avg_score', 0)
    aggressive_score = all_results['aggressive'].get('avg_score', 0)
    
    if neutral_score > conservative_score and neutral_score > aggressive_score:
        print("\n✅ NEUTRAL PROMPT WINS!")
        print("   The current conservative bias was suboptimal.")
        print("   Removing 'intervene only when necessary' improves outcomes.")
    elif conservative_score > neutral_score and conservative_score > aggressive_score:
        print("\n✅ CONSERVATIVE PROMPT WINS!")
        print("   The original prompt bias was actually optimal.")
        print("   'Intervene only when necessary' leads to best outcomes.")
    else:
        print("\n✅ AGGRESSIVE PROMPT WINS!")
        print("   Frequent interventions lead to better outcomes.")
        print("   Conservative approach was too hands-off.")
    
    conservative_interventions = all_results['conservative'].get('avg_interventions', 0)
    neutral_interventions = all_results['neutral'].get('avg_interventions', 0)
    aggressive_interventions = all_results['aggressive'].get('avg_interventions', 0)
    
    print(f"\n📊 INTERVENTION FREQUENCY:")
    print(f"   Conservative: {conservative_interventions:.1f} interventions")
    print(f"   Neutral: {neutral_interventions:.1f} interventions")
    print(f"   Aggressive: {aggressive_interventions:.1f} interventions")
    
    if aggressive_interventions > neutral_interventions > conservative_interventions:
        print("   → Prompt DOES control intervention frequency! ✅")
    else:
        print("   → Prompt has mixed effects on intervention frequency")
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETE!")
    print("="*80 + "\n")
    
    return all_results

if __name__ == "__main__":
    results = test_prompt_styles()
