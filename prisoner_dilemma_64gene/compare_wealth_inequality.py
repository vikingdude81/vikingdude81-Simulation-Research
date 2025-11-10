"""
🏛️💰 GOVERNMENT WEALTH INEQUALITY COMPARISON
============================================

Compare wealth inequality and elite emergence across all 12 government types.

Research Questions:
1. Which governments create the most inequality? (Gini coefficient)
2. Which governments produce "super citizens"?
3. Do cooperators or defectors become wealthy under each system?
4. Does wealth concentration correlate with cooperation rates?
5. Which governments have the most wealth mobility?
"""

import sys
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from enhanced_government_styles import EnhancedGovernmentController, EnhancedGovernmentStyle
from ultimate_echo_simulation import UltimateEchoSimulation
from wealth_inequality_tracker import WealthInequalityTracker


def run_government_with_wealth_tracking(
    government_style: EnhancedGovernmentStyle,
    generations: int = 300,
    population: int = 200
) -> dict:
    """
    Run simulation with comprehensive wealth inequality tracking.
    
    Args:
        government_style: Which government to test
        generations: Number of generations
        population: Initial population size
        
    Returns:
        dict with cooperation metrics AND wealth inequality metrics
    """
    print(f"\n{'='*70}")
    print(f"🏛️💰 Testing: {government_style.value.upper()}")
    print(f"{'='*70}")
    
    # Create simulation
    sim = UltimateEchoSimulation(
        initial_size=population,
        grid_size=(50, 50)
    )
    
    # Create enhanced government controller
    gov_controller = EnhancedGovernmentController(government_style)
    
    # Create wealth inequality tracker
    wealth_tracker = WealthInequalityTracker()
    
    # Track metrics
    cooperation_history = []
    diversity_history = []
    population_history = []
    
    # Run simulation
    for gen in range(generations):
        # Step simulation
        sim.step()
        
        # Apply government policy
        if len(sim.agents) > 0:
            action = gov_controller.apply_policy(sim.agents, sim.grid_size)
            sim.agents = [a for a in sim.agents if a.wealth > -9999]
        
        # Track cooperation/diversity
        if len(sim.agents) > 0:
            cooperation_rate = sum(1 for a in sim.agents if a.get_strategy() == 1) / len(sim.agents)
            cooperation_history.append(cooperation_rate * 100)
            
            chromosomes = [tuple(a.chromosome.tolist()) for a in sim.agents]
            diversity = len(set(chromosomes)) / len(chromosomes) if len(chromosomes) > 1 else 0
            diversity_history.append(diversity)
            
            population_history.append(len(sim.agents))
            
            # WEALTH TRACKING (every 10 generations to reduce overhead)
            if gen % 10 == 0:
                wealth_snapshot = wealth_tracker.capture_snapshot(sim.agents, gen)
        
        # Progress
        if (gen + 1) % 50 == 0:
            latest_snapshot = wealth_tracker.snapshots[-1] if wealth_tracker.snapshots else None
            gini_str = f", Gini={latest_snapshot.gini_coefficient:.3f}" if latest_snapshot else ""
            print(f"  Gen {gen+1}/{generations}: Pop={len(sim.agents)}, Coop={cooperation_history[-1]:.1f}%{gini_str}")
    
    # Final wealth snapshot
    if len(sim.agents) > 0:
        final_wealth_snapshot = wealth_tracker.capture_snapshot(sim.agents, generations)
    
    # Get wealth summary
    wealth_summary = wealth_tracker.get_summary()
    
    # Get government summary
    gov_summary = gov_controller.get_summary()
    
    # Plot wealth inequality for this government
    plot_path = f"wealth_inequality_{government_style.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    wealth_tracker.plot_wealth_inequality(plot_path)
    
    print(f"\n📊 Results:")
    print(f"  Final Cooperation: {cooperation_history[-1]:.1f}%")
    print(f"  Final Gini: {wealth_summary['final_state']['gini']:.3f}")
    print(f"  Top 1% Wealth Share: {wealth_summary['final_state']['top_1_share']:.1f}%")
    print(f"  Super Citizens: {wealth_summary['super_citizens']['total_emerged']}")
    print(f"  Wealth Mobility: {wealth_summary['average_inequality']['wealth_mobility']:.1f}%")
    
    # Elite analysis
    if wealth_summary['super_citizens']['total_emerged'] > 0:
        print(f"\n💎 Elite Analysis:")
        print(f"  Cooperator Elites: {wealth_summary['super_citizens']['cooperators']}")
        print(f"  Defector Elites: {wealth_summary['super_citizens']['defectors']}")
        print(f"  Max Wealth Ever: {wealth_summary['super_citizens']['max_wealth_ever']:.1f}")
    
    return {
        'government': government_style.value,
        'parameters': gov_summary.get('parameters', {}),
        
        # Cooperation metrics
        'final_cooperation': cooperation_history[-1] if cooperation_history else 0,
        'final_diversity': diversity_history[-1] if diversity_history else 0,
        'final_population': population_history[-1] if population_history else 0,
        
        # Wealth inequality metrics
        'gini_coefficient': wealth_summary['final_state']['gini'],
        'top_1_percent_share': wealth_summary['final_state']['top_1_share'],
        'top_10_percent_share': wealth_summary['average_inequality']['top_10_percent_share'],
        'bottom_50_percent_share': wealth_summary['average_inequality'].get('bottom_50_share', 0),
        'wealth_mobility': wealth_summary['average_inequality']['wealth_mobility'],
        
        # Elite emergence
        'super_citizens_total': wealth_summary['super_citizens']['total_emerged'],
        'super_citizens_cooperators': wealth_summary['super_citizens']['cooperators'],
        'super_citizens_defectors': wealth_summary['super_citizens']['defectors'],
        'max_wealth_ever': wealth_summary['super_citizens']['max_wealth_ever'],
        'richest_strategy': wealth_summary['final_state']['richest_strategy'],
        
        # Wealth by strategy
        'cooperator_mean_wealth': wealth_summary['wealth_by_strategy']['cooperator_mean'],
        'defector_mean_wealth': wealth_summary['wealth_by_strategy']['defector_mean'],
        'wealth_gap': wealth_summary['wealth_by_strategy']['wealth_gap'],
        
        # Policy actions
        'policy_actions': gov_summary['total_actions'],
        
        # Full histories
        'cooperation_history': cooperation_history,
        'diversity_history': diversity_history,
        'wealth_snapshots': [vars(s) for s in wealth_tracker.snapshots]
    }


def main():
    """Run wealth inequality comparison for all 12 governments."""
    print("\n" + "="*70)
    print("🏛️💰 GOVERNMENT WEALTH INEQUALITY COMPARISON")
    print("   Testing all 12 government types + wealth tracking")
    print("="*70)
    
    # All government types
    governments_to_test = [
        EnhancedGovernmentStyle.LIBERTARIAN,
        EnhancedGovernmentStyle.LAISSEZ_FAIRE,
        EnhancedGovernmentStyle.WELFARE_STATE,
        EnhancedGovernmentStyle.SOCIAL_DEMOCRACY,
        EnhancedGovernmentStyle.COMMUNIST,
        EnhancedGovernmentStyle.AUTHORITARIAN,
        EnhancedGovernmentStyle.FASCIST,
        EnhancedGovernmentStyle.OLIGARCHY,
        EnhancedGovernmentStyle.THEOCRACY,
        EnhancedGovernmentStyle.CENTRAL_BANKER,
        EnhancedGovernmentStyle.TECHNOCRACY,
        EnhancedGovernmentStyle.MIXED_ECONOMY,
    ]
    
    results = []
    for gov_style in governments_to_test:
        try:
            result = run_government_with_wealth_tracking(
                government_style=gov_style,
                generations=300,
                population=200
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error testing {gov_style.value}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"wealth_inequality_comparison_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'generations': 300,
            'initial_population': 200,
            'results': results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {filename}")
    
    # COMPARISON TABLES
    print("\n" + "="*70)
    print("📊 WEALTH INEQUALITY COMPARISON")
    print("="*70)
    
    # Sort by Gini coefficient (inequality)
    results_by_gini = sorted(results, key=lambda x: x['gini_coefficient'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Government':<20} {'Gini':<7} {'Top 1%':<8} {'Super':<6} {'Coop%':<8}")
    print("-" * 70)
    for i, r in enumerate(results_by_gini, 1):
        print(f"{i:<5} {r['government']:<20} {r['gini_coefficient']:<7.3f} {r['top_1_percent_share']:<8.1f} {r['super_citizens_total']:<6} {r['final_cooperation']:<8.1f}")
    
    # KEY INSIGHTS
    print("\n" + "="*70)
    print("🔍 KEY INSIGHTS")
    print("="*70)
    
    # Most unequal
    most_unequal = max(results, key=lambda x: x['gini_coefficient'])
    print(f"\n📈 Most Unequal: {most_unequal['government'].upper()}")
    print(f"   Gini: {most_unequal['gini_coefficient']:.3f}")
    print(f"   Top 1% owns: {most_unequal['top_1_percent_share']:.1f}% of wealth")
    print(f"   Super Citizens: {most_unequal['super_citizens_total']}")
    
    # Most equal
    most_equal = min(results, key=lambda x: x['gini_coefficient'])
    print(f"\n⚖️  Most Equal: {most_equal['government'].upper()}")
    print(f"   Gini: {most_equal['gini_coefficient']:.3f}")
    print(f"   Top 1% owns: {most_equal['top_1_percent_share']:.1f}% of wealth")
    print(f"   Wealth Mobility: {most_equal['wealth_mobility']:.1f}%")
    
    # Most super citizens
    most_elites = max(results, key=lambda x: x['super_citizens_total'])
    print(f"\n💎 Most Elite Formation: {most_elites['government'].upper()}")
    print(f"   Total Super Citizens: {most_elites['super_citizens_total']}")
    print(f"   Cooperator Elites: {most_elites['super_citizens_cooperators']}")
    print(f"   Defector Elites: {most_elites['super_citizens_defectors']}")
    print(f"   Max Wealth: {most_elites['max_wealth_ever']:.1f}")
    
    # Best wealth mobility
    most_mobile = max(results, key=lambda x: x['wealth_mobility'])
    print(f"\n🔄 Most Wealth Mobility: {most_mobile['government'].upper()}")
    print(f"   Mobility: {most_mobile['wealth_mobility']:.1f}% agents changed class")
    print(f"   Gini: {most_mobile['gini_coefficient']:.3f}")
    
    # Cooperator vs Defector wealth
    print(f"\n💰 Wealth by Strategy:")
    for r in sorted(results, key=lambda x: x['wealth_gap'], reverse=True)[:5]:
        gap = r['wealth_gap']
        winner = "Defectors" if gap > 0 else "Cooperators"
        print(f"   {r['government']:<20} {winner} richer by {abs(gap):.2f}")
    
    # Correlation: Inequality vs Cooperation
    print(f"\n📊 Correlation Analysis:")
    ginis = [r['gini_coefficient'] for r in results]
    coops = [r['final_cooperation'] for r in results]
    correlation = np.corrcoef(ginis, coops)[0, 1]
    print(f"   Inequality vs Cooperation: r={correlation:.3f}")
    
    if correlation < -0.3:
        print(f"   → More inequality = LESS cooperation")
    elif correlation > 0.3:
        print(f"   → More inequality = MORE cooperation")
    else:
        print(f"   → No strong correlation")
    
    print("\n" + "="*70)
    print("✅ Wealth inequality comparison complete!")
    print("="*70)


if __name__ == "__main__":
    main()
