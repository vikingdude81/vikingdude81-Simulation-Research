"""
📊 COMPREHENSIVE ANALYSIS OF GOVERNMENT COMPARISON RESULTS
===========================================================

Analyzing: government_comparison_all_20251101_223924.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results():
    """Load the government comparison JSON results."""
    with open('government_comparison_all_20251101_223924.json', 'r') as f:
        return json.load(f)

def analyze_time_series(data):
    """Analyze time-series patterns for each government."""
    print("\n" + "="*80)
    print("⏱️  TIME-SERIES ANALYSIS - How Governments Evolve Over 300 Generations")
    print("="*80)
    
    for result in data['results']:
        gov = result['government']
        coop_history = result['cooperation_history']
        div_history = result['diversity_history']
        gini_history = result['gini_history']
        
        # Calculate trends
        coop_start = np.mean(coop_history[:50])
        coop_end = np.mean(coop_history[-50:])
        coop_change = coop_end - coop_start
        
        div_start = np.mean(div_history[:50])
        div_end = np.mean(div_history[-50:])
        
        print(f"\n🏛️  {gov.upper()}")
        print(f"   Cooperation: {coop_start:.1f}% → {coop_end:.1f}% (change: {coop_change:+.1f}%)")
        print(f"   Diversity:   {div_start:.3f} → {div_end:.3f}")
        print(f"   Gini:        {gini_history[0]:.3f} → {gini_history[-1]:.3f}")
        
        # Identify pattern
        if abs(coop_change) < 5:
            pattern = "✅ STABLE"
        elif coop_change > 0:
            pattern = "📈 IMPROVING"
        else:
            pattern = "📉 DECLINING"
        print(f"   Pattern: {pattern}")

def analyze_parameter_correlations(data):
    """Analyze which parameters correlate with outcomes."""
    print("\n" + "="*80)
    print("🔬 PARAMETER CORRELATION ANALYSIS")
    print("="*80)
    
    # Extract data
    govs = []
    tax_rates = []
    enforcement_severity = []
    ubi_amounts = []
    cooperation = []
    diversity = []
    gini = []
    wealth = []
    
    for result in data['results']:
        govs.append(result['government'])
        params = result['parameters']
        tax_rates.append(params['wealth_tax_rate'])
        enforcement_severity.append(params['enforcement_severity'])
        ubi_amounts.append(params['universal_basic_income'])
        cooperation.append(result['final_cooperation'])
        diversity.append(result['final_diversity'])
        gini.append(result['final_gini'])
        wealth.append(result['final_wealth'])
    
    # Tax rate vs Cooperation
    print("\n📊 TAX RATE vs COOPERATION")
    sorted_by_tax = sorted(zip(tax_rates, cooperation, govs), key=lambda x: x[0])
    for tax, coop, gov in sorted_by_tax:
        bar = "█" * int(coop / 2)
        print(f"   {tax:4.0%} tax → {coop:5.1f}% coop {bar:20} ({gov})")
    
    # Correlation coefficient
    corr_tax_coop = np.corrcoef(tax_rates, cooperation)[0, 1]
    print(f"   Correlation: {corr_tax_coop:+.3f}")
    
    # Enforcement vs Cooperation
    print("\n⚖️  ENFORCEMENT SEVERITY vs COOPERATION")
    sorted_by_enf = sorted(zip(enforcement_severity, cooperation, govs), key=lambda x: x[0])
    for enf, coop, gov in sorted_by_enf:
        bar = "█" * int(coop / 2)
        print(f"   {enf:.1f} severity → {coop:5.1f}% coop {bar:20} ({gov})")
    
    corr_enf_coop = np.corrcoef(enforcement_severity, cooperation)[0, 1]
    print(f"   Correlation: {corr_enf_coop:+.3f}")
    
    # UBI vs Diversity
    print("\n💰 UNIVERSAL BASIC INCOME vs DIVERSITY")
    sorted_by_ubi = sorted(zip(ubi_amounts, diversity, govs), key=lambda x: x[0])
    for ubi, div, gov in sorted_by_ubi:
        bar = "█" * int(div * 40)
        print(f"   ${ubi:3.0f} UBI → {div:.3f} diversity {bar:30} ({gov})")
    
    corr_ubi_div = np.corrcoef(ubi_amounts, diversity)[0, 1]
    print(f"   Correlation: {corr_ubi_div:+.3f}")

def analyze_wealth_concentration(data):
    """Analyze wealth distribution patterns."""
    print("\n" + "="*80)
    print("💎 WEALTH CONCENTRATION ANALYSIS")
    print("="*80)
    
    wealth_data = []
    for result in data['results']:
        wealth_data.append({
            'gov': result['government'],
            'avg_wealth': result['final_wealth'],
            'gini': result['final_gini'],
            'cooperation': result['final_cooperation']
        })
    
    # Sort by average wealth
    wealth_data.sort(key=lambda x: x['avg_wealth'], reverse=True)
    
    print("\n📈 GOVERNMENTS BY AVERAGE WEALTH")
    for i, item in enumerate(wealth_data, 1):
        print(f"   {i:2}. {item['gov']:<20} ${item['avg_wealth']:>8.1f}  "
              f"(Gini: {item['gini']:.3f}, Coop: {item['cooperation']:.1f}%)")
    
    # Wealth-Inequality tradeoff
    print("\n⚖️  WEALTH-INEQUALITY TRADEOFF")
    print("   High wealth + Low inequality = Optimal")
    for item in wealth_data:
        # Score: wealth / (1 + gini)
        score = item['avg_wealth'] / (1 + item['gini'])
        bar = "█" * int(score / 30)
        print(f"   {item['gov']:<20} Score: {score:6.1f} {bar}")

def analyze_stability(data):
    """Analyze which governments are most stable over time."""
    print("\n" + "="*80)
    print("📊 STABILITY ANALYSIS - Which Governments Are Most Predictable?")
    print("="*80)
    
    stability_data = []
    for result in data['results']:
        # Calculate coefficient of variation (lower = more stable)
        coop_std = np.std(result['cooperation_history'][-100:])
        coop_mean = np.mean(result['cooperation_history'][-100:])
        coop_cv = coop_std / coop_mean if coop_mean > 0 else 999
        
        div_std = np.std(result['diversity_history'][-100:])
        div_mean = np.mean(result['diversity_history'][-100:])
        div_cv = div_std / div_mean if div_mean > 0 else 999
        
        stability_data.append({
            'gov': result['government'],
            'coop_stability': coop_cv,
            'div_stability': div_cv,
            'combined_stability': coop_cv + div_cv
        })
    
    # Sort by combined stability
    stability_data.sort(key=lambda x: x['combined_stability'])
    
    print("\n✅ MOST STABLE GOVERNMENTS (Low Variance)")
    for i, item in enumerate(stability_data[:6], 1):
        stability_rating = "⭐" * (6 - i)
        print(f"   {i}. {item['gov']:<20} {stability_rating:10} "
              f"(CoopCV: {item['coop_stability']:.4f}, DivCV: {item['div_stability']:.4f})")
    
    print("\n❌ MOST VOLATILE GOVERNMENTS (High Variance)")
    for i, item in enumerate(reversed(stability_data[-6:]), 1):
        volatility_rating = "💥" * i
        print(f"   {i}. {item['gov']:<20} {volatility_rating:10} "
              f"(CoopCV: {item['coop_stability']:.4f}, DivCV: {item['div_stability']:.4f})")

def analyze_policy_effectiveness(data):
    """Analyze which policy combinations work best."""
    print("\n" + "="*80)
    print("🎯 POLICY EFFECTIVENESS ANALYSIS")
    print("="*80)
    
    # Group by policy features
    enforced = []
    redistributive = []
    minimal = []
    
    for result in data['results']:
        params = result['parameters']
        coop = result['final_cooperation']
        
        if params['enforcement_severity'] >= 0.7:
            enforced.append((result['government'], coop))
        
        if params['wealth_tax_rate'] >= 0.3:
            redistributive.append((result['government'], coop))
        
        if params['wealth_tax_rate'] == 0 and params['enforcement_severity'] == 0:
            minimal.append((result['government'], coop))
    
    print("\n🔒 ENFORCEMENT-HEAVY GOVERNMENTS (severity ≥ 0.7)")
    for gov, coop in sorted(enforced, key=lambda x: x[1], reverse=True):
        print(f"   {gov:<20} {coop:5.1f}% cooperation")
    print(f"   Average: {np.mean([c for _, c in enforced]):.1f}%")
    
    print("\n💰 REDISTRIBUTIVE GOVERNMENTS (tax ≥ 30%)")
    for gov, coop in sorted(redistributive, key=lambda x: x[1], reverse=True):
        print(f"   {gov:<20} {coop:5.1f}% cooperation")
    print(f"   Average: {np.mean([c for _, c in redistributive]):.1f}%")
    
    print("\n🆓 MINIMAL INTERVENTION GOVERNMENTS")
    for gov, coop in sorted(minimal, key=lambda x: x[1], reverse=True):
        print(f"   {gov:<20} {coop:5.1f}% cooperation")
    if minimal:
        print(f"   Average: {np.mean([c for _, c in minimal]):.1f}%")

def identify_surprises(data):
    """Identify surprising or counterintuitive results."""
    print("\n" + "="*80)
    print("🤯 SURPRISING FINDINGS")
    print("="*80)
    
    results_dict = {r['government']: r for r in data['results']}
    
    print("\n1. ⚖️  COMMUNIST OUTPERFORMS WELFARE STATE")
    communist = results_dict['communist']
    welfare = results_dict['welfare_state']
    print(f"   Communist:     {communist['final_cooperation']:.1f}% coop, {communist['final_gini']:.3f} gini, ${communist['final_wealth']:.1f} wealth")
    print(f"   Welfare State: {welfare['final_cooperation']:.1f}% coop, {welfare['final_gini']:.3f} gini, ${welfare['final_wealth']:.1f} wealth")
    print(f"   → Communist achieves {communist['final_cooperation'] - welfare['final_cooperation']:.1f}% higher cooperation!")
    print(f"   → Communist has PERFECT equality (0.000 Gini) vs Welfare's {welfare['final_gini']:.3f}")
    
    print("\n2. 💰 SOCIAL DEMOCRACY: HIGH DIVERSITY + HIGH WEALTH, LOW COOPERATION")
    socdem = results_dict['social_democracy']
    print(f"   Diversity: {socdem['final_diversity']:.3f} (HIGHEST OF ALL!)")
    print(f"   Wealth: ${socdem['final_wealth']:.1f} (2nd highest)")
    print(f"   Cooperation: {socdem['final_cooperation']:.1f}% (only 6th place)")
    print(f"   → UBI + bonuses create wealth but don't enforce cooperation")
    
    print("\n3. 😱 CENTRAL BANKER CATASTROPHE")
    cb = results_dict['central_banker']
    print(f"   Cooperation: {cb['final_cooperation']:.1f}% (WORST!)")
    print(f"   → Reactive-only policy fails completely")
    print(f"   → No proactive cooperation support = defectors dominate")
    
    print("\n4. 🏛️  FASCIST vs AUTHORITARIAN")
    fascist = results_dict['fascist']
    auth = results_dict['authoritarian']
    print(f"   Both achieve 100% cooperation through enforcement")
    print(f"   Fascist: {fascist['final_diversity']:.3f} diversity, ${fascist['final_wealth']:.1f} wealth, {fascist['final_gini']:.3f} gini")
    print(f"   Authoritarian: {auth['final_diversity']:.3f} diversity, ${auth['final_wealth']:.1f} wealth, {auth['final_gini']:.3f} gini")
    print(f"   → Fascist creates MORE wealth (+${fascist['final_wealth'] - auth['final_wealth']:.1f}) but also MORE inequality")
    
    print("\n5. 📉 OLIGARCHY BACKFIRES")
    oligarchy = results_dict['oligarchy']
    laissez = results_dict['laissez_faire']
    print(f"   Oligarchy: {oligarchy['final_cooperation']:.1f}% (2nd worst)")
    print(f"   Laissez-Faire: {laissez['final_cooperation']:.1f}%")
    print(f"   → Helping ONLY the rich is WORSE than doing nothing!")
    print(f"   → Elite capture destroys cooperation ({laissez['final_cooperation'] - oligarchy['final_cooperation']:.1f}% difference)")

def create_visualization_summary(data):
    """Create summary charts."""
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATION SUMMARY...")
    print("="*80)
    
    # Extract data
    govs = [r['government'] for r in data['results']]
    coop = [r['final_cooperation'] for r in data['results']]
    div = [r['final_diversity'] for r in data['results']]
    gini = [r['final_gini'] for r in data['results']]
    wealth = [r['final_wealth'] for r in data['results']]
    
    # Sort by cooperation
    sorted_indices = np.argsort(coop)[::-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Government Comparison - Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cooperation
    ax = axes[0, 0]
    ax.barh([govs[i] for i in sorted_indices], [coop[i] for i in sorted_indices], color='skyblue')
    ax.set_xlabel('Cooperation %')
    ax.set_title('Final Cooperation Rate')
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Diversity
    ax = axes[0, 1]
    ax.barh(govs, div, color='lightgreen')
    ax.set_xlabel('Genetic Diversity (0-1)')
    ax.set_title('Final Genetic Diversity')
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Equality (inverse Gini)
    ax = axes[1, 0]
    colors = ['red' if g > 0.5 else 'orange' if g > 0.3 else 'green' for g in gini]
    ax.barh(govs, gini, color=colors)
    ax.set_xlabel('Gini Coefficient (0=perfect equality)')
    ax.set_title('Wealth Inequality')
    ax.grid(axis='x', alpha=0.3)
    
    # 4. Average Wealth
    ax = axes[1, 1]
    ax.barh(govs, wealth, color='gold')
    ax.set_xlabel('Average Wealth')
    ax.set_title('Economic Prosperity')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('government_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: government_comparison_analysis.png")
    
    # Create scatter plot matrix
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Government Trade-offs - Scatter Plots', fontsize=16, fontweight='bold')
    
    # Cooperation vs Diversity
    ax = axes[0, 0]
    for i, gov in enumerate(govs):
        ax.scatter(coop[i], div[i], s=100, alpha=0.6)
        ax.annotate(gov, (coop[i], div[i]), fontsize=8, alpha=0.7)
    ax.set_xlabel('Cooperation %')
    ax.set_ylabel('Diversity')
    ax.set_title('Cooperation vs Diversity')
    ax.grid(alpha=0.3)
    
    # Cooperation vs Equality
    ax = axes[0, 1]
    for i, gov in enumerate(govs):
        ax.scatter(coop[i], gini[i], s=100, alpha=0.6)
        ax.annotate(gov, (coop[i], gini[i]), fontsize=8, alpha=0.7)
    ax.set_xlabel('Cooperation %')
    ax.set_ylabel('Gini (higher = more inequality)')
    ax.set_title('Cooperation vs Equality')
    ax.grid(alpha=0.3)
    
    # Cooperation vs Wealth
    ax = axes[0, 2]
    for i, gov in enumerate(govs):
        ax.scatter(coop[i], wealth[i], s=100, alpha=0.6)
        ax.annotate(gov, (coop[i], wealth[i]), fontsize=8, alpha=0.7)
    ax.set_xlabel('Cooperation %')
    ax.set_ylabel('Average Wealth')
    ax.set_title('Cooperation vs Wealth')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    
    # Diversity vs Equality
    ax = axes[1, 0]
    for i, gov in enumerate(govs):
        ax.scatter(div[i], gini[i], s=100, alpha=0.6)
        ax.annotate(gov, (div[i], gini[i]), fontsize=8, alpha=0.7)
    ax.set_xlabel('Diversity')
    ax.set_ylabel('Gini (higher = more inequality)')
    ax.set_title('Diversity vs Equality')
    ax.grid(alpha=0.3)
    
    # Diversity vs Wealth
    ax = axes[1, 1]
    for i, gov in enumerate(govs):
        ax.scatter(div[i], wealth[i], s=100, alpha=0.6)
        ax.annotate(gov, (div[i], wealth[i]), fontsize=8, alpha=0.7)
    ax.set_xlabel('Diversity')
    ax.set_ylabel('Average Wealth')
    ax.set_title('Diversity vs Wealth')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    
    # Equality vs Wealth
    ax = axes[1, 2]
    for i, gov in enumerate(govs):
        ax.scatter(gini[i], wealth[i], s=100, alpha=0.6)
        ax.annotate(gov, (gini[i], wealth[i]), fontsize=8, alpha=0.7)
    ax.set_xlabel('Gini (higher = more inequality)')
    ax.set_ylabel('Average Wealth')
    ax.set_title('Equality vs Wealth')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('government_tradeoffs_scatter.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: government_tradeoffs_scatter.png")

def main():
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE GOVERNMENT COMPARISON ANALYSIS")
    print("   Analyzing: government_comparison_all_20251101_223924.json")
    print("="*80)
    
    # Load data
    data = load_results()
    
    # Run analyses
    analyze_time_series(data)
    analyze_parameter_correlations(data)
    analyze_wealth_concentration(data)
    analyze_stability(data)
    analyze_policy_effectiveness(data)
    identify_surprises(data)
    create_visualization_summary(data)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 government_comparison_analysis.png")
    print("  📈 government_tradeoffs_scatter.png")

if __name__ == "__main__":
    main()
