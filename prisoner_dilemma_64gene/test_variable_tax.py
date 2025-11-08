"""
Variable Tax Rate Government Test
Compare different tax rates to find optimal balance
"""

from ultimate_echo_simulation import UltimateEchoSimulation
from government_styles import GovernmentStyle
import json
from datetime import datetime

def test_tax_rate(tax_rate, run_name, generations=300, population=200):
    """Test a specific tax rate"""
    print(f"\n{'='*60}")
    print(f"Testing: {run_name} (Tax Rate: {tax_rate*100}%)")
    print(f"{'='*60}")
    
    sim = UltimateEchoSimulation(
        initial_size=population,
        grid_size=(75, 75),
        government_style=GovernmentStyle.WELFARE_STATE  # Use welfare as base
    )
    
    # Override the welfare tax rate
    sim.government_params['tax_rate'] = tax_rate
    
    for gen in range(generations):
        sim.step()
        
        if gen % 50 == 0:
            coop_rate = sim.get_cooperation_rate()
            diversity = sim.calculate_genetic_diversity()
            print(f"  Gen {gen:3d}: Pop={len(sim.agents):4d} | "
                  f"Coop={coop_rate*100:5.1f}% | Div={diversity:.3f}")
    
    # Final statistics
    final_coop = sim.get_cooperation_rate()
    final_diversity = sim.calculate_genetic_diversity()
    final_pop = len(sim.agents)
    
    # Calculate inequality
    if sim.agents:
        sorted_wealth = sorted([a.wealth for a in sim.agents])
        total_wealth = sum(sorted_wealth)
        cumsum = 0
        gini_sum = 0
        for i, wealth in enumerate(sorted_wealth):
            cumsum += wealth
            gini_sum += (i + 1) * wealth
        gini = (2 * gini_sum) / (len(sorted_wealth) * total_wealth) - (len(sorted_wealth) + 1) / len(sorted_wealth) if total_wealth > 0 else 0
    else:
        gini = 0
    
    cooperators = [a for a in sim.agents if a.traits.strategy == 1]
    defectors = [a for a in sim.agents if a.traits.strategy == 0]
    coop_avg_wealth = sum(a.wealth for a in cooperators) / len(cooperators) if cooperators else 0
    defector_avg_wealth = sum(a.wealth for a in defectors) / len(defectors) if defectors else 0
    
    results = {
        'tax_rate': tax_rate,
        'run_name': run_name,
        'final_cooperation': final_coop * 100,
        'final_diversity': final_diversity,
        'final_population': final_pop,
        'gini_coefficient': gini,
        'cooperator_avg_wealth': coop_avg_wealth,
        'defector_avg_wealth': defector_avg_wealth,
        'wealth_gap': defector_avg_wealth - coop_avg_wealth
    }
    
    print(f"\n  ✅ Final Results:")
    print(f"     Cooperation: {final_coop*100:.1f}%")
    print(f"     Diversity: {final_diversity:.3f}")
    print(f"     Population: {final_pop}")
    print(f"     Gini: {gini:.3f}")
    print(f"     Cooperator Wealth: {coop_avg_wealth:.1f}")
    print(f"     Defector Wealth: {defector_avg_wealth:.1f}")
    print(f"     Wealth Gap: {defector_avg_wealth - coop_avg_wealth:.1f}")
    
    return results


def main():
    """Test different tax rates"""
    print("\n" + "="*60)
    print("🔬 VARIABLE TAX RATE EXPERIMENT")
    print("Testing: Laissez-Faire, Low Tax, Medium Tax, High Tax, Very High Tax")
    print("="*60)
    
    tax_scenarios = [
        (0.0, "Laissez-Faire (0% tax)"),
        (0.1, "Low Tax (10%)"),
        (0.2, "Medium-Low Tax (20%)"),
        (0.3, "Medium Tax (30% - baseline)"),
        (0.4, "Medium-High Tax (40%)"),
        (0.5, "High Tax (50%)"),
        (0.6, "Very High Tax (60%)"),
        (0.7, "Extreme Tax (70%)"),
    ]
    
    all_results = []
    
    for tax_rate, name in tax_scenarios:
        result = test_tax_rate(tax_rate, name)
        all_results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"variable_tax_experiment_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'scenarios': all_results
        }, f, indent=2)
    
    # Summary comparison
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Tax Rate':<20} {'Cooperation':<15} {'Diversity':<15} {'Gini':<10}")
    print("-"*60)
    
    best_coop = max(all_results, key=lambda x: x['final_cooperation'])
    best_div = max(all_results, key=lambda x: x['final_diversity'])
    best_equality = min(all_results, key=lambda x: x['gini_coefficient'])
    
    for result in all_results:
        coop_str = f"{result['final_cooperation']:.1f}%"
        if result == best_coop:
            coop_str += " 🏆"
        
        div_str = f"{result['final_diversity']:.3f}"
        if result == best_div:
            div_str += " 🌟"
        
        gini_str = f"{result['gini_coefficient']:.3f}"
        if result == best_equality:
            gini_str += " ⚖️"
        
        print(f"{result['run_name']:<20} {coop_str:<15} {div_str:<15} {gini_str:<10}")
    
    print(f"\n✅ Results saved to: {filename}")
    print(f"\n🔍 Key Findings:")
    print(f"   • Best Cooperation: {best_coop['run_name']} ({best_coop['final_cooperation']:.1f}%)")
    print(f"   • Best Diversity: {best_div['run_name']} ({best_div['final_diversity']:.3f})")
    print(f"   • Best Equality: {best_equality['run_name']} (Gini: {best_equality['gini_coefficient']:.3f})")


if __name__ == "__main__":
    main()
