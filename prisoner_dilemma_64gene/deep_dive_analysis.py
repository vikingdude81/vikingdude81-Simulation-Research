"""
🔬 DEEP DIVE ANALYSIS - Government Comparison Study
===================================================

Analyzes:
1. Genetic diversity across governments
2. Policy switching patterns (Mixed Economy)
3. Inequality (Gini coefficient) trends
4. Agent survival analysis
5. Wealth distribution evolution
6. Strategy evolution over time

Run this to generate comprehensive insights from all 5 government experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from datetime import datetime
from ultimate_echo_simulation import UltimateEchoSimulation
from government_styles import GovernmentStyle
from genetic_traits import ExtendedChromosome, GeneticTraits, ExtendedEchoAgent
from collections import defaultdict

def calculate_gini(wealth_values):
    """Calculate Gini coefficient (0=perfect equality, 1=perfect inequality)"""
    if len(wealth_values) == 0:
        return 0.0
    sorted_wealth = np.sort(wealth_values)
    n = len(sorted_wealth)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_wealth)) / (n * np.sum(sorted_wealth)) - (n + 1) / n

def analyze_genetic_diversity(agents):
    """Analyze genetic diversity in population"""
    if len(agents) == 0:
        return {
            'unique_chromosomes': 0,
            'diversity_score': 0.0,
            'trait_distributions': {}
        }
    
    # Convert numpy arrays to tuples for hashing
    chromosomes = [tuple(agent.chromosome.flatten()) for agent in agents]
    unique_chromosomes = len(set(chromosomes))
    diversity_score = unique_chromosomes / len(chromosomes)
    
    # Analyze trait distributions
    trait_distributions = defaultdict(lambda: defaultdict(int))
    for agent in agents:
        traits = ExtendedChromosome.decode(agent.chromosome)
        trait_distributions['strategy'][traits.strategy] += 1
        trait_distributions['metabolism'][traits.metabolism] += 1
        trait_distributions['vision'][traits.vision] += 1
        trait_distributions['lifespan'][traits.lifespan] += 1
    
    return {
        'unique_chromosomes': unique_chromosomes,
        'diversity_score': diversity_score,
        'trait_distributions': dict(trait_distributions)
    }

def run_detailed_simulation(government_type, generations=300, initial_pop=200):
    """Run simulation with detailed data collection"""
    print(f"\n{'='*80}")
    print(f"🔬 ANALYZING: {government_type.name}")
    print(f"{'='*80}\n")
    
    sim = UltimateEchoSimulation(
        initial_size=initial_pop,
        grid_size=(75, 75),
        government_style=government_type
    )
    
    # Data collectors
    history = {
        'generations': [],
        'population': [],
        'cooperation': [],
        'gini': [],
        'diversity': [],
        'avg_wealth': [],
        'cooperator_wealth': [],
        'defector_wealth': [],
        'policy_actions': [],
        'trait_evolution': defaultdict(list)
    }
    
    for gen in range(generations):
        # Step simulation
        sim.step()
        
        # Collect basic stats
        cooperators = [a for a in sim.agents if a.traits.strategy == 1]
        defectors = [a for a in sim.agents if a.traits.strategy == 0]
        wealth_values = [a.wealth for a in sim.agents]
        
        cooperation_rate = len(cooperators) / len(sim.agents) if sim.agents else 0
        gini = calculate_gini(wealth_values)
        diversity_data = analyze_genetic_diversity(sim.agents)
        
        history['generations'].append(gen)
        history['population'].append(len(sim.agents))
        history['cooperation'].append(cooperation_rate * 100)
        history['gini'].append(gini)
        history['diversity'].append(diversity_data['diversity_score'])
        history['avg_wealth'].append(np.mean(wealth_values) if wealth_values else 0)
        history['cooperator_wealth'].append(
            np.mean([a.wealth for a in cooperators]) if cooperators else 0
        )
        history['defector_wealth'].append(
            np.mean([a.wealth for a in defectors]) if defectors else 0
        )
        
        # Track policy actions (for mixed economy)
        if hasattr(sim, 'government'):
            history['policy_actions'].append(1 if gen > 0 else 0)  # Simplified tracking
        
        # Track trait evolution
        if gen % 10 == 0:  # Sample every 10 generations
            for agent in sim.agents:
                history['trait_evolution']['metabolism'].append(agent.traits.metabolism)
                history['trait_evolution']['vision'].append(agent.traits.vision)
                history['trait_evolution']['lifespan'].append(agent.traits.lifespan)
        
        # Progress indicator
        if gen % 50 == 0:
            print(f"  Gen {gen:3d}: Pop={len(sim.agents):4d}, "
                  f"Coop={cooperation_rate*100:5.1f}%, "
                  f"Gini={gini:.3f}, "
                  f"Diversity={diversity_data['diversity_score']:.3f}")
    
    # Final analysis
    final_diversity = analyze_genetic_diversity(sim.agents)
    
    print(f"\n✅ {government_type.name} Analysis Complete!")
    print(f"   Final Cooperation: {history['cooperation'][-1]:.1f}%")
    print(f"   Final Gini: {history['gini'][-1]:.3f}")
    print(f"   Final Diversity: {history['diversity'][-1]:.3f}")
    print(f"   Unique Chromosomes: {final_diversity['unique_chromosomes']}")
    
    return history, final_diversity

def create_comprehensive_visualization(all_results):
    """Create comprehensive visualization of all analyses"""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('🔬 DEEP DIVE ANALYSIS - Government Comparison', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    governments = list(all_results.keys())
    colors = {
        'AUTHORITARIAN': 'red',
        'MIXED_ECONOMY': 'purple',
        'WELFARE_STATE': 'blue',
        'LAISSEZ_FAIRE': 'gray',
        'CENTRAL_BANKER': 'orange'
    }
    
    # 1. Cooperation Evolution
    ax1 = fig.add_subplot(gs[0, 0])
    for gov in governments:
        history = all_results[gov]['history']
        ax1.plot(history['generations'], history['cooperation'], 
                label=gov.replace('_', ' ').title(), 
                color=colors[gov], linewidth=2, alpha=0.8)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Cooperation Rate (%)')
    ax1.set_title('Cooperation Evolution')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    
    # 2. Genetic Diversity Evolution
    ax2 = fig.add_subplot(gs[0, 1])
    for gov in governments:
        history = all_results[gov]['history']
        ax2.plot(history['generations'], history['diversity'], 
                label=gov.replace('_', ' ').title(), 
                color=colors[gov], linewidth=2, alpha=0.8)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Diversity Score')
    ax2.set_title('Genetic Diversity Evolution')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    
    # 3. Inequality (Gini) Evolution
    ax3 = fig.add_subplot(gs[0, 2])
    for gov in governments:
        history = all_results[gov]['history']
        ax3.plot(history['generations'], history['gini'], 
                label=gov.replace('_', ' ').title(), 
                color=colors[gov], linewidth=2, alpha=0.8)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Gini Coefficient')
    ax3.set_title('Inequality Evolution (0=equal, 1=unequal)')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    
    # 4. Population Dynamics
    ax4 = fig.add_subplot(gs[1, 0])
    for gov in governments:
        history = all_results[gov]['history']
        ax4.plot(history['generations'], history['population'], 
                label=gov.replace('_', ' ').title(), 
                color=colors[gov], linewidth=2, alpha=0.8)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Population')
    ax4.set_title('Population Dynamics')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    
    # 5. Wealth Evolution (Cooperators vs Defectors)
    ax5 = fig.add_subplot(gs[1, 1])
    for gov in governments:
        history = all_results[gov]['history']
        ax5.plot(history['generations'], history['cooperator_wealth'], 
                color=colors[gov], linewidth=2, alpha=0.8, linestyle='-',
                label=f"{gov.replace('_', ' ').title()} (Coop)")
        ax5.plot(history['generations'], history['defector_wealth'], 
                color=colors[gov], linewidth=1, alpha=0.5, linestyle='--')
    ax5.set_xlabel('Generation')
    ax5.set_ylabel('Average Wealth')
    ax5.set_title('Wealth: Cooperators (solid) vs Defectors (dashed)')
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.3)
    
    # 6. Final State Comparison (Bar chart)
    ax6 = fig.add_subplot(gs[1, 2])
    final_coop = [all_results[g]['history']['cooperation'][-1] for g in governments]
    final_div = [all_results[g]['history']['diversity'][-1] * 100 for g in governments]
    final_gini = [all_results[g]['history']['gini'][-1] * 100 for g in governments]
    
    x = np.arange(len(governments))
    width = 0.25
    ax6.bar(x - width, final_coop, width, label='Cooperation %', alpha=0.8)
    ax6.bar(x, final_div, width, label='Diversity %', alpha=0.8)
    ax6.bar(x + width, final_gini, width, label='Inequality %', alpha=0.8)
    ax6.set_xlabel('Government')
    ax6.set_ylabel('Percentage')
    ax6.set_title('Final State Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels([g.replace('_', '\n') for g in governments], fontsize=8)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # 7-9. Unique Chromosomes (Bottom row)
    ax7 = fig.add_subplot(gs[2, 0])
    unique_chroms = [all_results[g]['final_diversity']['unique_chromosomes'] for g in governments]
    bars = ax7.bar([g.replace('_', '\n') for g in governments], unique_chroms,
                   color=[colors[g] for g in governments], alpha=0.7, edgecolor='black')
    ax7.set_ylabel('Unique Chromosomes')
    ax7.set_title('Genetic Diversity (Final)')
    ax7.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, unique_chroms):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha='center', fontweight='bold')
    
    # 8. Cooperation-Diversity Tradeoff Scatter
    ax8 = fig.add_subplot(gs[2, 1])
    for gov in governments:
        final_coop = all_results[gov]['history']['cooperation'][-1]
        final_div = all_results[gov]['history']['diversity'][-1]
        ax8.scatter(final_coop, final_div, s=300, 
                   color=colors[gov], alpha=0.6, edgecolor='black', linewidth=2)
        ax8.text(final_coop, final_div, gov.replace('_', '\n'), 
                fontsize=7, ha='center', va='center', fontweight='bold')
    ax8.set_xlabel('Cooperation Rate (%)')
    ax8.set_ylabel('Diversity Score')
    ax8.set_title('Cooperation-Diversity Tradeoff')
    ax8.grid(alpha=0.3)
    
    # 9. Summary Statistics Table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    table_data = []
    for gov in governments:
        history = all_results[gov]['history']
        coop = history['cooperation'][-1]
        div = history['diversity'][-1]
        gini = history['gini'][-1]
        unique = all_results[gov]['final_diversity']['unique_chromosomes']
        table_data.append([
            gov.replace('_', ' ')[:12],
            f"{coop:.1f}%",
            f"{div:.3f}",
            f"{gini:.3f}",
            str(unique)
        ])
    
    table = ax9.table(cellText=table_data,
                     colLabels=['Government', 'Coop', 'Div', 'Gini', 'Uniq'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax9.set_title('Final Statistics Summary', pad=20, fontweight='bold')
    
    # 10-12. Bottom row additional analyses
    
    # 10. Wealth Distribution Comparison
    ax10 = fig.add_subplot(gs[3, 0])
    for gov in governments:
        history = all_results[gov]['history']
        avg_wealth = history['avg_wealth']
        ax10.plot(history['generations'], avg_wealth,
                 label=gov.replace('_', ' ').title(),
                 color=colors[gov], linewidth=2, alpha=0.8)
    ax10.set_xlabel('Generation')
    ax10.set_ylabel('Average Wealth')
    ax10.set_title('Average Wealth Evolution')
    ax10.legend(fontsize=8)
    ax10.grid(alpha=0.3)
    
    # 11. Cooperation vs Inequality Scatter
    ax11 = fig.add_subplot(gs[3, 1])
    for gov in governments:
        history = all_results[gov]['history']
        ax11.scatter(history['cooperation'], history['gini'],
                    s=5, color=colors[gov], alpha=0.3, label=gov.replace('_', ' ').title())
    ax11.set_xlabel('Cooperation Rate (%)')
    ax11.set_ylabel('Gini Coefficient')
    ax11.set_title('Cooperation vs Inequality (all generations)')
    ax11.legend(fontsize=8)
    ax11.grid(alpha=0.3)
    
    # 12. Diversity vs Population Scatter
    ax12 = fig.add_subplot(gs[3, 2])
    for gov in governments:
        history = all_results[gov]['history']
        ax12.scatter(history['population'], history['diversity'],
                    s=5, color=colors[gov], alpha=0.3, label=gov.replace('_', ' ').title())
    ax12.set_xlabel('Population')
    ax12.set_ylabel('Diversity Score')
    ax12.set_title('Population vs Diversity (all generations)')
    ax12.legend(fontsize=8)
    ax12.grid(alpha=0.3)
    
    plt.savefig('deep_dive_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Comprehensive visualization saved: deep_dive_analysis.png")
    
    return fig

def generate_insights_report(all_results):
    """Generate detailed insights report"""
    print("\n" + "="*80)
    print("📊 DEEP DIVE INSIGHTS REPORT")
    print("="*80 + "\n")
    
    # 1. Genetic Diversity Analysis
    print("1️⃣  GENETIC DIVERSITY ANALYSIS")
    print("-" * 80)
    for gov in all_results.keys():
        diversity = all_results[gov]['final_diversity']
        history = all_results[gov]['history']
        print(f"\n{gov.replace('_', ' ').title()}:")
        print(f"   Final Unique Chromosomes: {diversity['unique_chromosomes']}")
        print(f"   Final Diversity Score: {history['diversity'][-1]:.3f}")
        print(f"   Diversity Change: {history['diversity'][0]:.3f} → {history['diversity'][-1]:.3f}")
    
    # Find most/least diverse
    diversities = {g: all_results[g]['history']['diversity'][-1] for g in all_results}
    most_diverse_val = max(diversities.values())
    least_diverse_val = min(diversities.values())
    most_diverse = [g for g, v in diversities.items() if v == most_diverse_val][0]
    least_diverse = [g for g, v in diversities.items() if v == least_diverse_val][0]
    
    print(f"\n🏆 Most Diverse: {most_diverse.replace('_', ' ').title()} ({diversities[most_diverse]:.3f})")
    print(f"⚠️  Least Diverse: {least_diverse.replace('_', ' ').title()} ({diversities[least_diverse]:.3f})")
    
    # 2. Inequality Analysis
    print("\n\n2️⃣  INEQUALITY ANALYSIS (Gini Coefficient)")
    print("-" * 80)
    for gov in all_results.keys():
        history = all_results[gov]['history']
        gini_start = history['gini'][0]
        gini_end = history['gini'][-1]
        gini_change = gini_end - gini_start
        print(f"\n{gov.replace('_', ' ').title()}:")
        print(f"   Final Gini: {gini_end:.3f}")
        print(f"   Change: {gini_change:+.3f} ({gini_start:.3f} → {gini_end:.3f})")
        if gini_end < 0.3:
            print(f"   Status: ✅ Low inequality (equal)")
        elif gini_end < 0.5:
            print(f"   Status: ⚠️  Moderate inequality")
        else:
            print(f"   Status: 🔴 High inequality")
    
    # 3. Cooperation-Diversity Tradeoff
    print("\n\n3️⃣  COOPERATION-DIVERSITY TRADEOFF")
    print("-" * 80)
    print("\nGovernment          Cooperation  Diversity   Tradeoff Score")
    print("-" * 80)
    for gov in all_results.keys():
        history = all_results[gov]['history']
        coop = history['cooperation'][-1]
        div = history['diversity'][-1]
        # Harmonic mean as tradeoff score
        tradeoff = 2 * (coop/100 * div) / ((coop/100) + div) if (coop + div*100) > 0 else 0
        print(f"{gov.replace('_', ' ').title():20s} {coop:6.1f}%     {div:6.3f}      {tradeoff:.3f}")
    
    # 4. Wealth Gap Analysis
    print("\n\n4️⃣  COOPERATOR vs DEFECTOR WEALTH GAP")
    print("-" * 80)
    for gov in all_results.keys():
        history = all_results[gov]['history']
        if history['cooperator_wealth'] and history['defector_wealth']:
            coop_wealth = history['cooperator_wealth'][-1]
            def_wealth = history['defector_wealth'][-1]
            gap = coop_wealth - def_wealth
            print(f"\n{gov.replace('_', ' ').title()}:")
            print(f"   Cooperator Wealth: {coop_wealth:.1f}")
            print(f"   Defector Wealth: {def_wealth:.1f}")
            print(f"   Gap: {gap:+.1f} ({'Cooperators win' if gap > 0 else 'Defectors win'})")
    
    print("\n" + "="*80)

def main():
    """Run complete deep dive analysis"""
    print("\n" + "="*80)
    print("🔬 DEEP DIVE ANALYSIS - Starting Complete Analysis")
    print("="*80)
    print("\nThis will run detailed simulations for all 5 governments.")
    print("Estimated time: 5-10 minutes\n")
    
    input("Press Enter to start...")
    
    all_results = {}
    
    # Run analysis for each government
    governments = [
        GovernmentStyle.AUTHORITARIAN,
        GovernmentStyle.MIXED_ECONOMY,
        GovernmentStyle.WELFARE_STATE,
        GovernmentStyle.LAISSEZ_FAIRE,
        GovernmentStyle.CENTRAL_BANKER
    ]
    
    for gov in governments:
        history, final_diversity = run_detailed_simulation(gov)
        all_results[gov.name] = {
            'history': history,
            'final_diversity': final_diversity
        }
    
    # Create comprehensive visualization
    print("\n" + "="*80)
    print("📊 Creating Comprehensive Visualization...")
    print("="*80)
    create_comprehensive_visualization(all_results)
    
    # Generate insights report
    generate_insights_report(all_results)
    
    # Save data
    print("\n💾 Saving analysis data...")
    with open('deep_dive_analysis_data.json', 'w') as f:
        # Convert to JSON-serializable format
        json_data = {}
        for gov, data in all_results.items():
            json_data[gov] = {
                'history': {k: v if isinstance(v, list) else list(v) 
                           for k, v in data['history'].items()},
                'final_diversity': data['final_diversity']
            }
        json.dump(json_data, f, indent=2)
    print("✅ Data saved to: deep_dive_analysis_data.json")
    
    print("\n" + "="*80)
    print("✅ DEEP DIVE ANALYSIS COMPLETE!")
    print("="*80)
    print("\nFiles created:")
    print("  1. deep_dive_analysis.png - Comprehensive visualization")
    print("  2. deep_dive_analysis_data.json - Raw data")
    print("\nNext: Option C - ML-Based Governance")
    print("      Use this data to train RL agent!")
    print("="*80 + "\n")
    
    plt.show()

if __name__ == '__main__':
    main()
