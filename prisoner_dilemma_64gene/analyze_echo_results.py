"""
Comprehensive Analysis of Echo Model Results
Analyzes the saved JSON data and creates visualizations for:
- Population growth (50→500)
- Resource accumulation (5K→13.1M)
- Cooperation rate stability
- Tag dominance over time
- Age distribution evolution
- Birth/death rate timeline (showing transition to immortality)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter
import seaborn as sns

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_latest_results():
    """Load the most recent echo dashboard results."""
    # Find the most recent result file
    result_files = list(Path('.').glob('echo_dashboard_*.json'))
    if not result_files:
        raise FileNotFoundError("No echo dashboard results found!")
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"📊 Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def analyze_population_growth(history):
    """Analyze and plot population growth over time."""
    populations = history['population']
    generations = list(range(len(populations)))
    
    plt.subplot(3, 3, 1)
    plt.plot(generations, populations, linewidth=2, color='#2E86AB')
    plt.fill_between(generations, populations, alpha=0.3, color='#2E86AB')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Population Size', fontsize=12)
    plt.title('Population Growth Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key milestones
    if len(populations) > 0:
        plt.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='Population Cap')
        plt.legend()
    
    return {
        'initial': populations[0] if populations else 0,
        'final': populations[-1] if populations else 0,
        'max': max(populations) if populations else 0,
        'growth_rate': (populations[-1] - populations[0]) / len(populations) if populations else 0
    }

def analyze_resource_accumulation(history):
    """Analyze and plot resource accumulation over time."""
    total_resources = history['resources']
    populations = history['population']
    avg_resources = [total / pop if pop > 0 else 0 for total, pop in zip(total_resources, populations)]
    generations = list(range(len(total_resources)))
    
    # Total resources (log scale for better visibility)
    plt.subplot(3, 3, 2)
    plt.plot(generations, total_resources, linewidth=2, color='#A23B72', label='Total Resources')
    plt.fill_between(generations, total_resources, alpha=0.3, color='#A23B72')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Total Resources (log scale)', fontsize=12)
    plt.yscale('log')
    plt.title('Resource Accumulation (Exponential Growth)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    
    # Average resources per agent
    plt.subplot(3, 3, 3)
    plt.plot(generations, avg_resources, linewidth=2, color='#F18F01')
    plt.fill_between(generations, avg_resources, alpha=0.3, color='#F18F01')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Avg Resources per Agent', fontsize=12)
    plt.title('Average Wealth per Agent', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    return {
        'initial_total': total_resources[0] if total_resources else 0,
        'final_total': total_resources[-1] if total_resources else 0,
        'growth_factor': total_resources[-1] / total_resources[0] if total_resources and total_resources[0] > 0 else 0,
        'final_avg': avg_resources[-1] if avg_resources else 0
    }

def analyze_cooperation_rate(history):
    """Analyze and plot cooperation rate stability."""
    coop_rates = [rate * 100 for rate in history['cooperation']]
    generations = list(range(len(coop_rates)))
    
    plt.subplot(3, 3, 4)
    plt.plot(generations, coop_rates, linewidth=2, color='#06A77D')
    plt.fill_between(generations, coop_rates, alpha=0.3, color='#06A77D')
    plt.axhline(y=np.mean(coop_rates), color='red', linestyle='--', alpha=0.5, 
                label=f'Mean: {np.mean(coop_rates):.1f}%')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Cooperation Rate (%)', fontsize=12)
    plt.title('Cooperation Rate Stability', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return {
        'mean': np.mean(coop_rates),
        'std': np.std(coop_rates),
        'min': min(coop_rates),
        'max': max(coop_rates),
        'stability': np.std(coop_rates)  # Lower = more stable
    }

def analyze_tag_dominance(history):
    """Analyze and visualize tag dominance evolution over time."""
    generations = [h['generation'] for h in history]
    
    # Track top 5 tags over time
    tag_evolution = {}
    for h in history:
        for tag, count in h['top_tags'][:5]:  # Top 5
            if tag not in tag_evolution:
                tag_evolution[tag] = [0] * len(history)
            idx = generations.index(h['generation'])
            tag_evolution[tag][idx] = count
    
    plt.subplot(3, 3, 5)
    colors = plt.cm.Set3(np.linspace(0, 1, len(tag_evolution)))
    
    for (tag, counts), color in zip(tag_evolution.items(), colors):
        plt.plot(generations, counts, linewidth=2, label=tag, color=color, alpha=0.8)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Tag Count', fontsize=12)
    plt.title('Tag Dominance Evolution (Top 5)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc='best')
    
    # Final tag diversity
    final_tags = Counter(dict(history[-1]['top_tags']))
    
    return {
        'dominant_tag': history[-1]['top_tags'][0][0] if history[-1]['top_tags'] else None,
        'dominant_count': history[-1]['top_tags'][0][1] if history[-1]['top_tags'] else 0,
        'unique_tags': len(history[-1]['top_tags']),
        'diversity_index': len([t for t, c in history[-1]['top_tags'] if c > 5])  # Tags with >5 agents
    }

def analyze_age_distribution(history):
    """Analyze age distribution evolution."""
    generations = [h['generation'] for h in history]
    avg_ages = [h['avg_age'] for h in history]
    max_ages = [h['max_age'] for h in history]
    
    plt.subplot(3, 3, 6)
    plt.plot(generations, avg_ages, linewidth=2, label='Average Age', color='#8338EC')
    plt.plot(generations, max_ages, linewidth=2, label='Max Age', color='#FF006E', linestyle='--')
    plt.fill_between(generations, avg_ages, alpha=0.3, color='#8338EC')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Age (generations)', fontsize=12)
    plt.title('Age Distribution Evolution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return {
        'final_avg_age': avg_ages[-1] if avg_ages else 0,
        'final_max_age': max_ages[-1] if max_ages else 0,
        'age_acceleration': (avg_ages[-1] - avg_ages[0]) / len(avg_ages) if avg_ages else 0
    }

def analyze_birth_death_timeline(history):
    """Analyze birth/death rates showing transition to immortality."""
    generations = [h['generation'] for h in history]
    births = [h['births'] for h in history]
    deaths = [h['deaths'] for h in history]
    
    plt.subplot(3, 3, 7)
    plt.plot(generations, births, linewidth=2, label='Births', color='#06A77D', marker='o', markersize=3)
    plt.plot(generations, deaths, linewidth=2, label='Deaths', color='#D62828', marker='x', markersize=3)
    plt.fill_between(generations, births, alpha=0.2, color='#06A77D')
    plt.fill_between(generations, deaths, alpha=0.2, color='#D62828')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Birth/Death Timeline (Transition to Immortality)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Find when immortality began (sustained period of 0 deaths)
    immortality_start = None
    zero_count = 0
    for i, d in enumerate(deaths):
        if d == 0:
            zero_count += 1
            if zero_count >= 10 and immortality_start is None:  # 10 consecutive zero deaths
                immortality_start = generations[i - 9]
        else:
            zero_count = 0
    
    if immortality_start:
        plt.axvline(x=immortality_start, color='gold', linestyle='--', linewidth=2, 
                   label=f'Immortality at Gen {immortality_start}')
        plt.legend()
    
    return {
        'immortality_generation': immortality_start,
        'total_births': sum(births),
        'total_deaths': sum(deaths),
        'final_birth_rate': births[-1] if births else 0,
        'final_death_rate': deaths[-1] if deaths else 0
    }

def analyze_strategy_evolution(history):
    """Analyze strategy evolution over time."""
    generations = [h['generation'] for h in history]
    
    # Track if TFT appears in top strategies
    tft_pattern = "CDCDCDCDCDCDCDCD"
    tft_counts = []
    
    for h in history:
        tft_count = 0
        for strategy, count in h['top_strategies']:
            if strategy.startswith(tft_pattern):
                tft_count = count
                break
        tft_counts.append(tft_count)
    
    plt.subplot(3, 3, 8)
    plt.plot(generations, tft_counts, linewidth=2, color='#3A86FF', marker='o', markersize=3)
    plt.fill_between(generations, tft_counts, alpha=0.3, color='#3A86FF')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Tit-for-Tat Count', fontsize=12)
    plt.title('Tit-for-Tat Strategy Survival', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    return {
        'tft_survived': tft_counts[-1] > 0 if tft_counts else False,
        'final_tft_count': tft_counts[-1] if tft_counts else 0,
        'tft_peak': max(tft_counts) if tft_counts else 0
    }

def create_summary_statistics(history, results):
    """Create a summary statistics panel."""
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    summary_text = f"""
    📊 ECHO MODEL FINAL RESULTS
    
    🧬 Population Dynamics:
    • Start: {results['population']['initial']} → Final: {results['population']['final']}
    • Growth Rate: {results['population']['growth_rate']:.1f} agents/gen
    
    💰 Resource Economy:
    • Start: {results['resources']['initial_total']:,.0f} → Final: {results['resources']['final_total']:,.0f}
    • Growth Factor: {results['resources']['growth_factor']:.1f}x
    • Avg Wealth: {results['resources']['final_avg']:,.0f} per agent
    
    🤝 Cooperation:
    • Mean Rate: {results['cooperation']['mean']:.1f}%
    • Stability (σ): {results['cooperation']['std']:.2f}%
    • Range: {results['cooperation']['min']:.1f}%-{results['cooperation']['max']:.1f}%
    
    🏷️ Tag Tribalism:
    • Dominant Tag: {results['tags']['dominant_tag']}
    • Dominance: {results['tags']['dominant_count']} agents
    • Unique Tags: {results['tags']['unique_tags']}
    
    👴 Age Distribution:
    • Final Avg Age: {results['age']['final_avg_age']:.1f} gens
    • Final Max Age: {results['age']['final_max_age']:.0f} gens
    
    ⚰️ Immortality:
    • Achieved at Gen: {results['births_deaths']['immortality_generation']}
    • Total Births: {results['births_deaths']['total_births']}
    • Total Deaths: {results['births_deaths']['total_deaths']}
    
    🎯 Tit-for-Tat:
    • Survived: {'✅ Yes' if results['strategy']['tft_survived'] else '❌ No'}
    • Final Count: {results['strategy']['final_tft_count']} agents
    • Peak Count: {results['strategy']['tft_peak']} agents
    """
    
    plt.text(0.1, 0.95, summary_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def main():
    """Main analysis function."""
    print("🔬 Starting Echo Model Analysis...")
    print("=" * 60)
    
    # Load data
    data = load_latest_results()
    history = data['history']
    
    print(f"\n📈 Analyzing {len(history)} generations of data...")
    print(f"⏱️  Simulation ran for {data['metadata']['elapsed_time']:.1f}s")
    avg_speed = data['metadata']['generations'] / data['metadata']['elapsed_time']
    print(f"⚡ Average speed: {avg_speed:.2f} gen/s")
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('🧬 Echo Model Evolution Analysis - Complete Timeline', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Run all analyses
    results = {}
    
    print("\n🔍 Analyzing components:")
    print("  1. Population growth...")
    results['population'] = analyze_population_growth(history)
    
    print("  2. Resource accumulation...")
    results['resources'] = analyze_resource_accumulation(history)
    
    print("  3. Cooperation rate stability...")
    results['cooperation'] = analyze_cooperation_rate(history)
    
    print("  4. Tag dominance evolution...")
    results['tags'] = analyze_tag_dominance(history)
    
    print("  5. Age distribution...")
    results['age'] = analyze_age_distribution(history)
    
    print("  6. Birth/death timeline...")
    results['births_deaths'] = analyze_birth_death_timeline(history)
    
    print("  7. Strategy evolution...")
    results['strategy'] = analyze_strategy_evolution(history)
    
    print("  8. Summary statistics...")
    create_summary_statistics(history, results)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"echo_analysis_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 Analysis saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    # Print detailed findings
    print("\n" + "=" * 60)
    print("🎯 KEY FINDINGS:")
    print("=" * 60)
    
    print(f"\n1️⃣ POPULATION DYNAMICS:")
    print(f"   • Grew from {results['population']['initial']} to {results['population']['final']} agents")
    print(f"   • Growth rate: {results['population']['growth_rate']:.1f} agents/generation")
    
    print(f"\n2️⃣ RESOURCE ECONOMY:")
    print(f"   • {results['resources']['growth_factor']:.1f}x resource multiplication!")
    print(f"   • Final wealth: {results['resources']['final_avg']:,.0f} per agent")
    print(f"   • Total economy: {results['resources']['final_total']:,.0f} resources")
    
    print(f"\n3️⃣ COOPERATION TRIUMPH:")
    print(f"   • Stable cooperation at {results['cooperation']['mean']:.1f}%")
    print(f"   • Very low variance (σ={results['cooperation']['std']:.2f}%) = highly stable!")
    
    print(f"\n4️⃣ TAG TRIBALISM:")
    print(f"   • Dominant tag '{results['tags']['dominant_tag']}' has {results['tags']['dominant_count']} agents")
    print(f"   • {results['tags']['unique_tags']} unique tags maintained diversity")
    
    print(f"\n5️⃣ ANCIENT IMMORTAL SOCIETY:")
    print(f"   • Average age: {results['age']['final_avg_age']:.1f} generations (ancient!)")
    print(f"   • Immortality achieved at generation {results['births_deaths']['immortality_generation']}")
    print(f"   • Zero deaths for {len(history) - results['births_deaths']['immortality_generation']} generations")
    
    print(f"\n6️⃣ TIT-FOR-TAT LEGACY:")
    if results['strategy']['tft_survived']:
        print(f"   • ✅ Survived! {results['strategy']['final_tft_count']} agents still use TFT")
        print(f"   • Peak population: {results['strategy']['tft_peak']} agents")
    else:
        print(f"   • ❌ Did not survive to final generation")
    
    print("\n" + "=" * 60)
    print("✨ Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
