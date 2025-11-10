"""
🧬⚖️ COMPARATIVE GOD-AI TESTING

Compare performance of different God controller types:
1. DISABLED (baseline - no interventions)
2. RULE_BASED (hard-coded logic)
3. ML_BASED (quantum genetic evolution)

Metrics compared:
- Population survival
- Average wealth
- Cooperation rate
- Intervention count
- Wealth inequality
- Simulation dynamics
"""

import json
from datetime import datetime
from pathlib import Path
from prisoner_echo_god import run_god_echo_simulation
import matplotlib.pyplot as plt
import seaborn as sns

def run_comparative_test(generations=100, initial_size=300, runs_per_mode=3):
    """
    Run comparative tests across different God modes.
    
    Args:
        generations: Length of each simulation
        initial_size: Starting population
        runs_per_mode: Number of runs per mode for statistical reliability
    """
    print("\n" + "="*70)
    print("🧬⚖️ COMPARATIVE GOD-AI TESTING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"   Generations: {generations}")
    print(f"   Initial population: {initial_size}")
    print(f"   Runs per mode: {runs_per_mode}")
    
    modes = ["DISABLED", "RULE_BASED", "ML_BASED"]
    results = {mode: [] for mode in modes}
    
    # Run simulations for each mode
    for mode in modes:
        print(f"\n{'-'*70}")
        print(f"🎮 Testing {mode} mode...")
        print(f"{'-'*70}")
        
        for run in range(runs_per_mode):
            print(f"\n   Run {run+1}/{runs_per_mode}...")
            
            try:
                population = run_god_echo_simulation(
                    generations=generations,
                    initial_size=initial_size,
                    god_mode=mode,
                    update_frequency=25  # Less frequent updates for speed
                )
                
                # Calculate metrics
                final_pop = len(population.agents)
                survived = final_pop > 0
                
                if survived:
                    avg_wealth = sum(a.resources for a in population.agents) / final_pop
                    total_actions = sum(a.cooperations + a.defections for a in population.agents)
                    total_coop = sum(a.cooperations for a in population.agents)
                    coop_rate = (total_coop / total_actions) if total_actions > 0 else 0
                    
                    resources = [a.resources for a in population.agents]
                    gini = calculate_gini(resources)
                else:
                    avg_wealth = 0
                    coop_rate = 0
                    gini = 0
                
                result = {
                    'run': run + 1,
                    'survived': survived,
                    'final_population': final_pop,
                    'avg_wealth': float(avg_wealth),
                    'cooperation_rate': float(coop_rate),
                    'gini_coefficient': float(gini),
                    'interventions': population.god.total_interventions,
                    'interventions_by_type': {
                        k.value if hasattr(k, 'value') else str(k): v 
                        for k, v in population.god.interventions_by_type.items()
                    }
                }
                
                # Add quantum stats if available
                if hasattr(population.god, 'quantum_controller') and population.god.quantum_controller:
                    stats = population.god.quantum_controller.get_statistics()
                    result['quantum_stats'] = {
                        'timesteps': stats['timesteps'],
                        'fitness': float(stats['fitness']),
                        'final_traits': stats['current_traits']
                    }
                
                results[mode].append(result)
                
                print(f"      ✅ Complete: pop={final_pop}, wealth={avg_wealth:.0f}, coop={coop_rate:.1%}")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                results[mode].append({
                    'run': run + 1,
                    'error': str(e)
                })
    
    # Aggregate results
    print(f"\n{'='*70}")
    print("📊 COMPARATIVE RESULTS")
    print(f"{'='*70}\n")
    
    summary = {}
    for mode in modes:
        successful_runs = [r for r in results[mode] if 'error' not in r]
        
        if not successful_runs:
            print(f"❌ {mode}: All runs failed")
            continue
        
        survived_runs = [r for r in successful_runs if r['survived']]
        
        summary[mode] = {
            'total_runs': len(results[mode]),
            'successful_runs': len(successful_runs),
            'survival_rate': len(survived_runs) / len(successful_runs) if successful_runs else 0,
            'avg_final_population': sum(r['final_population'] for r in survived_runs) / len(survived_runs) if survived_runs else 0,
            'avg_wealth': sum(r['avg_wealth'] for r in survived_runs) / len(survived_runs) if survived_runs else 0,
            'avg_cooperation': sum(r['cooperation_rate'] for r in survived_runs) / len(survived_runs) if survived_runs else 0,
            'avg_gini': sum(r['gini_coefficient'] for r in survived_runs) / len(survived_runs) if survived_runs else 0,
            'avg_interventions': sum(r['interventions'] for r in successful_runs) / len(successful_runs) if successful_runs else 0
        }
        
        print(f"🎯 {mode}:")
        print(f"   Survival rate: {summary[mode]['survival_rate']:.1%}")
        print(f"   Avg population: {summary[mode]['avg_final_population']:.0f}")
        print(f"   Avg wealth: {summary[mode]['avg_wealth']:.1f}")
        print(f"   Avg cooperation: {summary[mode]['avg_cooperation']:.1%}")
        print(f"   Avg Gini: {summary[mode]['avg_gini']:.3f}")
        print(f"   Avg interventions: {summary[mode]['avg_interventions']:.1f}\n")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'generations': generations,
            'initial_size': initial_size,
            'runs_per_mode': runs_per_mode
        },
        'detailed_results': results,
        'summary': summary
    }
    
    output_dir = Path("outputs/god_ai")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"comparative_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    
    # Create visualization
    create_comparison_chart(summary, output_dir)
    
    return output

def calculate_gini(resources):
    """Calculate Gini coefficient for wealth inequality."""
    if len(resources) == 0:
        return 0
    
    sorted_resources = sorted(resources)
    n = len(sorted_resources)
    cumsum = 0
    for i, val in enumerate(sorted_resources):
        cumsum += (i + 1) * val
    
    return (2 * cumsum) / (n * sum(sorted_resources)) - (n + 1) / n if sum(sorted_resources) > 0 else 0

def create_comparison_chart(summary, output_dir):
    """Create comparison visualization."""
    import numpy as np
    
    modes = list(summary.keys())
    metrics = ['survival_rate', 'avg_final_population', 'avg_wealth', 'avg_cooperation', 'avg_interventions']
    metric_names = ['Survival Rate', 'Final Population', 'Avg Wealth', 'Cooperation Rate', 'Interventions']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('🧬 God-AI Comparative Performance', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        values = [summary[mode][metric] for mode in modes]
        colors = ['gray', 'blue', 'purple']
        
        bars = ax.bar(modes, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}' if metric != 'survival_rate' else f'{height:.0%}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(name, fontweight='bold')
        ax.set_ylabel(name)
        ax.grid(axis='y', alpha=0.3)
        
        # Special formatting for survival rate
        if metric == 'survival_rate':
            ax.set_ylim(0, 1.1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    chart_file = output_dir / f"comparison_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to: {chart_file}")
    plt.close()

if __name__ == "__main__":
    # Run quick test
    results = run_comparative_test(
        generations=100,
        initial_size=300,
        runs_per_mode=3
    )
    
    print("\n✅ Comparative test complete!")
    print(f"\n🎯 Key Findings:")
    
    # Determine winner
    summary = results['summary']
    
    if summary:
        # Score each mode
        scores = {}
        for mode in summary:
            score = (
                summary[mode]['survival_rate'] * 100 +
                summary[mode]['avg_cooperation'] * 50 +
                (summary[mode]['avg_final_population'] / 10) +
                (summary[mode]['avg_wealth'] / 100) -
                (summary[mode]['avg_gini'] * 20)
            )
            scores[mode] = score
        
        best_mode = max(scores, key=scores.get)
        print(f"\n🏆 Best performing mode: {best_mode}")
        print(f"   Overall score: {scores[best_mode]:.1f}")
        
        for mode in sorted(scores, key=scores.get, reverse=True):
            print(f"   {mode}: {scores[mode]:.1f}")
