"""
🏛️ COMPREHENSIVE GOVERNMENT COMPARISON
==========================================

Compare all 12 government types including:
- Original 5 (Laissez-Faire, Welfare, Authoritarian, Central Banker, Mixed)
- New 7 (Communist, Fascist, Social Democracy, Libertarian, Theocracy, Oligarchy, Technocracy)

Tests each government across multiple metrics:
- Cooperation rate
- Genetic diversity
- Population stability
- Wealth inequality (Gini coefficient)
- Average wealth
- Policy intervention frequency
"""

import sys
import os
from datetime import datetime
import json
import numpy as np

# Import enhanced government styles
from enhanced_government_styles import EnhancedGovernmentController, EnhancedGovernmentStyle
from ultimate_echo_simulation import UltimateEchoSimulation


def run_government_test(
    government_style: EnhancedGovernmentStyle,
    generations: int = 300,
    population: int = 200
) -> dict:
    """
    Run a single government test and collect comprehensive metrics.
    
    Args:
        government_style: Which government type to test
        generations: How many generations to run
        population: Initial population size
        
    Returns:
        dict with results
    """
    print(f"\n{'='*70}")
    print(f"🏛️  Testing: {government_style.value.upper()}")
    print(f"{'='*70}")
    
    # Create simulation (UltimateEchoSimulation doesn't use government_style in __init__)
    sim = UltimateEchoSimulation(
        initial_size=population,
        grid_size=(50, 50)
    )
    
    # Create government controller with enhanced parameters (REPLACE simulation's controller)
    gov_controller = EnhancedGovernmentController(government_style)
    
    # Track metrics over time
    cooperation_history = []
    diversity_history = []
    population_history = []
    gini_history = []
    wealth_history = []
    
    # Run simulation
    for gen in range(generations):
        # Step simulation
        sim.step()
        
        # Apply government policy (using enhanced controller)
        if len(sim.agents) > 0:
            # Apply enhanced government policy
            action = gov_controller.apply_policy(sim.agents, sim.grid_size)
            
            # Remove agents marked for deletion (wealth = -9999)
            sim.agents = [a for a in sim.agents if a.wealth > -9999]
        
        # Track metrics
        if len(sim.agents) > 0:
            cooperation_rate = sum(1 for a in sim.agents if a.get_strategy() == 1) / len(sim.agents)
            cooperation_history.append(cooperation_rate * 100)
            
            # Calculate genetic diversity manually
            chromosomes = [tuple(a.chromosome.tolist()) for a in sim.agents]
            if len(chromosomes) > 1:
                unique_genes = len(set(chromosomes))
                diversity = unique_genes / len(chromosomes)
            else:
                diversity = 0
            diversity_history.append(diversity)
            
            population_history.append(len(sim.agents))
            
            # Calculate Gini coefficient
            wealths = sorted([a.wealth for a in sim.agents if a.wealth > 0])
            if len(wealths) > 1:
                n = len(wealths)
                gini = sum((i + 1) * w for i, w in enumerate(wealths))
                gini = 2 * gini / (n * sum(wealths)) - (n + 1) / n
                gini_history.append(gini)
            else:
                gini_history.append(0)
            
            avg_wealth = np.mean([a.wealth for a in sim.agents])
            wealth_history.append(avg_wealth)
        
        # Progress indicator
        if (gen + 1) % 50 == 0:
            print(f"  Gen {gen+1}/{generations}: Pop={len(sim.agents)}, Coop={cooperation_history[-1]:.1f}%, Div={diversity_history[-1]:.3f}")
    
    # Calculate final metrics
    final_cooperation = cooperation_history[-1] if cooperation_history else 0
    final_diversity = diversity_history[-1] if diversity_history else 0
    final_population = population_history[-1] if population_history else 0
    final_gini = gini_history[-1] if gini_history else 0
    final_wealth = wealth_history[-1] if wealth_history else 0
    
    # Get government summary
    gov_summary = gov_controller.get_summary()
    
    # Calculate stability metrics
    coop_std = np.std(cooperation_history[-100:]) if len(cooperation_history) >= 100 else 0
    div_std = np.std(diversity_history[-100:]) if len(diversity_history) >= 100 else 0
    pop_std = np.std(population_history[-100:]) if len(population_history) >= 100 else 0
    
    print(f"\n📊 Results:")
    print(f"  Final Cooperation: {final_cooperation:.1f}%")
    print(f"  Final Diversity: {final_diversity:.3f}")
    print(f"  Final Population: {final_population}")
    print(f"  Final Gini: {final_gini:.3f}")
    print(f"  Final Avg Wealth: {final_wealth:.1f}")
    print(f"  Policy Actions: {gov_summary['total_actions']}")
    
    return {
        'government': government_style.value,
        'parameters': gov_summary.get('parameters', {}),
        'final_cooperation': final_cooperation,
        'final_diversity': final_diversity,
        'final_population': final_population,
        'final_gini': final_gini,
        'final_wealth': final_wealth,
        'cooperation_stability': coop_std,
        'diversity_stability': div_std,
        'population_stability': pop_std,
        'policy_actions': gov_summary['total_actions'],
        'action_breakdown': gov_summary.get('action_breakdown', {}),
        'total_wealth_transferred': gov_summary.get('total_wealth_transferred', 0),
        'cooperation_history': cooperation_history,
        'diversity_history': diversity_history,
        'population_history': population_history,
        'gini_history': gini_history,
        'wealth_history': wealth_history
    }


def main():
    """Run comprehensive comparison of all 12 government types."""
    print("\n" + "="*70)
    print("🏛️  COMPREHENSIVE GOVERNMENT COMPARISON")
    print("   Testing all 12 government types")
    print("="*70)
    
    # All government types to test
    governments_to_test = [
        # Original 5
        EnhancedGovernmentStyle.LAISSEZ_FAIRE,
        EnhancedGovernmentStyle.WELFARE_STATE,
        EnhancedGovernmentStyle.AUTHORITARIAN,
        EnhancedGovernmentStyle.CENTRAL_BANKER,
        EnhancedGovernmentStyle.MIXED_ECONOMY,
        
        # New ideological types
        EnhancedGovernmentStyle.COMMUNIST,
        EnhancedGovernmentStyle.FASCIST,
        EnhancedGovernmentStyle.SOCIAL_DEMOCRACY,
        EnhancedGovernmentStyle.LIBERTARIAN,
        EnhancedGovernmentStyle.THEOCRACY,
        EnhancedGovernmentStyle.OLIGARCHY,
        EnhancedGovernmentStyle.TECHNOCRACY,
    ]
    
    # Run all tests
    results = []
    for gov_style in governments_to_test:
        try:
            result = run_government_test(
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
    filename = f"government_comparison_all_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'generations': 300,
            'initial_population': 200,
            'results': results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {filename}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("📊 FINAL COMPARISON TABLE")
    print("="*70)
    
    # Sort by cooperation rate
    results_sorted = sorted(results, key=lambda x: x['final_cooperation'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Government':<20} {'Coop%':<8} {'Div':<7} {'Gini':<7} {'Pop':<6} {'Actions':<8}")
    print("-" * 70)
    
    for i, result in enumerate(results_sorted, 1):
        print(f"{i:<5} {result['government']:<20} {result['final_cooperation']:<8.1f} {result['final_diversity']:<7.3f} {result['final_gini']:<7.3f} {result['final_population']:<6.0f} {result['policy_actions']:<8}")
    
    # Print best performers
    print("\n" + "="*70)
    print("🏆 BEST PERFORMERS")
    print("="*70)
    
    best_cooperation = max(results, key=lambda x: x['final_cooperation'])
    print(f"\n🥇 Best Cooperation: {best_cooperation['government'].upper()}")
    print(f"   Cooperation: {best_cooperation['final_cooperation']:.1f}%")
    print(f"   Diversity: {best_cooperation['final_diversity']:.3f}")
    print(f"   Gini: {best_cooperation['final_gini']:.3f}")
    
    best_diversity = max(results, key=lambda x: x['final_diversity'])
    print(f"\n🌟 Best Diversity: {best_diversity['government'].upper()}")
    print(f"   Diversity: {best_diversity['final_diversity']:.3f}")
    print(f"   Cooperation: {best_diversity['final_cooperation']:.1f}%")
    print(f"   Gini: {best_diversity['final_gini']:.3f}")
    
    best_equality = min(results, key=lambda x: x['final_gini'])
    print(f"\n⚖️  Best Equality: {best_equality['government'].upper()}")
    print(f"   Gini: {best_equality['final_gini']:.3f}")
    print(f"   Cooperation: {best_equality['final_cooperation']:.1f}%")
    print(f"   Diversity: {best_equality['final_diversity']:.3f}")
    
    # Key insights
    print("\n" + "="*70)
    print("💡 KEY INSIGHTS")
    print("="*70)
    
    # Compare authoritarian variants
    authoritarian_results = [r for r in results if r['government'] in ['authoritarian', 'fascist', 'communist']]
    if authoritarian_results:
        print("\n📌 Authoritarian Variants:")
        for r in authoritarian_results:
            print(f"   {r['government']:<20} Coop: {r['final_cooperation']:.1f}%, Div: {r['final_diversity']:.3f}, Gini: {r['final_gini']:.3f}")
    
    # Compare welfare variants
    welfare_results = [r for r in results if r['government'] in ['welfare_state', 'social_democracy', 'communist']]
    if welfare_results:
        print("\n📌 Redistributive Variants:")
        for r in welfare_results:
            tax_rate = r['parameters'].get('wealth_tax_rate', 0) * 100 if r['parameters'] else 0
            print(f"   {r['government']:<20} Tax: {tax_rate:.0f}%, Coop: {r['final_cooperation']:.1f}%, Gini: {r['final_gini']:.3f}")
    
    # Compare minimal intervention
    minimal_results = [r for r in results if r['government'] in ['libertarian', 'laissez_faire', 'oligarchy']]
    if minimal_results:
        print("\n📌 Minimal Intervention Variants:")
        for r in minimal_results:
            print(f"   {r['government']:<20} Coop: {r['final_cooperation']:.1f}%, Div: {r['final_diversity']:.3f}, Pop: {r['final_population']:.0f}")
    
    print("\n" + "="*70)
    print("✅ Comprehensive comparison complete!")
    print("="*70)


if __name__ == "__main__":
    main()
