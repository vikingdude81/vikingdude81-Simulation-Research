"""
📊 GOD-AI COMPARATIVE ANALYSIS

Runs multiple simulations in parallel to compare:
1. NO GOD (baseline - pure evolution + external shocks)
2. RULE-BASED GOD (simple if/then interventions)
3. Future: ML-BASED GOD
4. Future: API-BASED GOD

Measures effectiveness across:
- Population survival
- Average wealth
- Cooperation rate
- Tribe diversity
- Wealth inequality
- Resilience to shocks
"""

import json
import time
import numpy as np
from datetime import datetime
from colorama import init, Fore, Style
import matplotlib.pyplot as plt
from typing import Dict, List
import os

# Import our God-controlled simulation
from prisoner_echo_god import run_god_echo_simulation, GodEchoPopulation

init(autoreset=True)

def run_controlled_experiment(
    generations: int = 500,
    trials: int = 5,
    initial_size: int = 100
) -> Dict:
    """
    Run controlled experiment comparing God vs No-God.
    
    Returns dictionary with results from both conditions.
    """
    
    print(f"\n{Fore.CYAN}{'='*100}")
    print(f"{Fore.YELLOW}🔬 CONTROLLED EXPERIMENT: GOD vs NO-GOD{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")
    print(f"{Fore.WHITE}Trials per condition: {Fore.YELLOW}{trials}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Generations per trial: {Fore.YELLOW}{generations}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Initial population: {Fore.YELLOW}{initial_size}{Style.RESET_ALL}\n")
    
    results = {
        'config': {
            'generations': generations,
            'trials': trials,
            'initial_size': initial_size
        },
        'no_god_trials': [],
        'rule_god_trials': []
    }
    
    # === CONDITION 1: NO GOD (Baseline) ===
    print(f"{Fore.CYAN}{'─'*100}")
    print(f"{Fore.YELLOW}📋 CONDITION 1: NO GOD (Baseline){Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'─'*100}{Style.RESET_ALL}\n")
    
    for trial in range(trials):
        print(f"{Fore.WHITE}Running trial {trial+1}/{trials}...{Style.RESET_ALL}")
        
        population = run_god_echo_simulation(
            generations=generations,
            initial_size=initial_size,
            god_mode="DISABLED",
            update_frequency=100  # Minimal updates for speed
        )
        
        trial_results = extract_trial_results(population, trial+1, "NO_GOD")
        results['no_god_trials'].append(trial_results)
        
        print(f"{Fore.GREEN}✓ Trial {trial+1} complete{Style.RESET_ALL}\n")
    
    # === CONDITION 2: RULE-BASED GOD ===
    print(f"\n{Fore.CYAN}{'─'*100}")
    print(f"{Fore.MAGENTA}📋 CONDITION 2: RULE-BASED GOD{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'─'*100}{Style.RESET_ALL}\n")
    
    for trial in range(trials):
        print(f"{Fore.WHITE}Running trial {trial+1}/{trials}...{Style.RESET_ALL}")
        
        population = run_god_echo_simulation(
            generations=generations,
            initial_size=initial_size,
            god_mode="RULE_BASED",
            update_frequency=100
        )
        
        trial_results = extract_trial_results(population, trial+1, "RULE_GOD")
        results['rule_god_trials'].append(trial_results)
        
        print(f"{Fore.GREEN}✓ Trial {trial+1} complete{Style.RESET_ALL}\n")
    
    return results

def extract_trial_results(population: GodEchoPopulation, trial_num: int, condition: str) -> Dict:
    """Extract key metrics from a completed simulation."""
    
    survived = len(population.agents) > 0
    
    if survived:
        final_pop = len(population.agents)
        final_wealth = np.mean([a.resources for a in population.agents])
        final_coop = population.history['cooperation'][-1] if population.history['cooperation'] else 0
        final_clustering = population.history['clustering'][-1] if population.history['clustering'] else 0
        
        # Calculate wealth inequality (Gini-like)
        resources = sorted([a.resources for a in population.agents])
        wealth_inequality = resources[-1] / max(resources[0], 1)
        
        # Count unique tribes
        unique_tags = len(set(a.tag for a in population.agents))
        tribe_diversity = unique_tags / len(population.agents)
        
        # Total births/deaths
        total_births = sum(population.history['births'])
        total_deaths = sum(population.history['deaths'])
        
        # Shock survival
        shock_count = sum(1 for s in population.history['shocks'] if s is not None)
        
    else:
        final_pop = 0
        final_wealth = 0
        final_coop = 0
        final_clustering = 0
        wealth_inequality = 0
        tribe_diversity = 0
        total_births = sum(population.history['births'])
        total_deaths = sum(population.history['deaths'])
        shock_count = sum(1 for s in population.history['shocks'] if s is not None)
    
    # God stats (if applicable)
    god_stats = population.god.get_summary_stats()
    
    return {
        'trial': trial_num,
        'condition': condition,
        'survived': survived,
        'final_population': final_pop,
        'final_avg_wealth': final_wealth,
        'final_cooperation': final_coop,
        'final_clustering': final_clustering,
        'wealth_inequality': wealth_inequality,
        'tribe_diversity': tribe_diversity,
        'total_births': total_births,
        'total_deaths': total_deaths,
        'shock_count': shock_count,
        'god_interventions': god_stats.get('total_interventions', 0),
        'population_history': population.history['population'],
        'cooperation_history': population.history['cooperation'],
        'wealth_history': population.history['resources']
    }

def analyze_results(results: Dict):
    """Analyze and print comparison between conditions."""
    
    print(f"\n{Fore.CYAN}{'='*100}")
    print(f"{Fore.YELLOW}📊 COMPARATIVE ANALYSIS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")
    
    no_god = results['no_god_trials']
    rule_god = results['rule_god_trials']
    
    # === SURVIVAL RATE ===
    no_god_survival = sum(1 for t in no_god if t['survived']) / len(no_god) * 100
    rule_god_survival = sum(1 for t in rule_god if t['survived']) / len(rule_god) * 100
    
    print(f"{Fore.CYAN}{'─'*100}")
    print(f"{Fore.YELLOW}1. SURVIVAL RATE{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'─'*100}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}NO GOD:      {Fore.RED if no_god_survival < 100 else Fore.GREEN}{no_god_survival:.1f}%{Style.RESET_ALL} survived")
    print(f"{Fore.WHITE}RULE GOD:    {Fore.GREEN}{rule_god_survival:.1f}%{Style.RESET_ALL} survived")
    
    if rule_god_survival > no_god_survival:
        improvement = rule_god_survival - no_god_survival
        print(f"{Fore.GREEN}✓ God improved survival by {improvement:.1f}%{Style.RESET_ALL}")
    
    # === AVERAGE METRICS (only from survivors) ===
    print(f"\n{Fore.CYAN}{'─'*100}")
    print(f"{Fore.YELLOW}2. AVERAGE FINAL METRICS (Survivors Only){Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'─'*100}{Style.RESET_ALL}")
    
    metrics = ['final_population', 'final_avg_wealth', 'final_cooperation', 'final_clustering', 'tribe_diversity']
    metric_names = ['Population', 'Avg Wealth', 'Cooperation', 'Clustering', 'Tribe Diversity']
    
    for metric, name in zip(metrics, metric_names):
        no_god_survivors = [t[metric] for t in no_god if t['survived']]
        rule_god_survivors = [t[metric] for t in rule_god if t['survived']]
        
        if no_god_survivors and rule_god_survivors:
            no_god_avg = np.mean(no_god_survivors)
            rule_god_avg = np.mean(rule_god_survivors)
            
            # Format based on metric type
            if metric in ['final_cooperation', 'final_clustering', 'tribe_diversity']:
                format_str = "{:.1%}"
            else:
                format_str = "{:.1f}"
            
            print(f"{Fore.WHITE}{name:20} NO GOD: {Fore.CYAN}{format_str.format(no_god_avg)}{Style.RESET_ALL}  |  ", end="")
            print(f"RULE GOD: {Fore.GREEN}{format_str.format(rule_god_avg)}{Style.RESET_ALL}")
            
            # Calculate improvement
            if no_god_avg > 0:
                improvement = ((rule_god_avg - no_god_avg) / no_god_avg) * 100
                color = Fore.GREEN if improvement > 0 else Fore.RED
                print(f"{color}   → {'+'if improvement > 0 else ''}{improvement:.1f}% change{Style.RESET_ALL}")
    
    # === SHOCK RESILIENCE ===
    print(f"\n{Fore.CYAN}{'─'*100}")
    print(f"{Fore.YELLOW}3. SHOCK RESILIENCE{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'─'*100}{Style.RESET_ALL}")
    
    no_god_shocks = np.mean([t['shock_count'] for t in no_god])
    rule_god_shocks = np.mean([t['shock_count'] for t in rule_god])
    
    print(f"{Fore.WHITE}Avg External Shocks Survived:{Style.RESET_ALL}")
    print(f"  NO GOD:   {Fore.CYAN}{no_god_shocks:.1f}{Style.RESET_ALL}")
    print(f"  RULE GOD: {Fore.GREEN}{rule_god_shocks:.1f}{Style.RESET_ALL}")
    
    # === GOD INTERVENTION STATS ===
    print(f"\n{Fore.CYAN}{'─'*100}")
    print(f"{Fore.MAGENTA}4. GOD-AI INTERVENTION STATS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'─'*100}{Style.RESET_ALL}")
    
    avg_interventions = np.mean([t['god_interventions'] for t in rule_god])
    print(f"{Fore.WHITE}Avg Interventions per Trial: {Fore.YELLOW}{avg_interventions:.1f}{Style.RESET_ALL}")
    
    # === STATISTICAL SIGNIFICANCE (t-test) ===
    print(f"\n{Fore.CYAN}{'─'*100}")
    print(f"{Fore.YELLOW}5. STATISTICAL SIGNIFICANCE{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'─'*100}{Style.RESET_ALL}")
    
    from scipy import stats
    
    # Test cooperation rate difference
    no_god_coop = [t['final_cooperation'] for t in no_god if t['survived']]
    rule_god_coop = [t['final_cooperation'] for t in rule_god if t['survived']]
    
    if len(no_god_coop) > 1 and len(rule_god_coop) > 1:
        t_stat, p_value = stats.ttest_ind(no_god_coop, rule_god_coop)
        print(f"{Fore.WHITE}Cooperation Rate t-test:{Style.RESET_ALL}")
        print(f"  t-statistic: {Fore.CYAN}{t_stat:.3f}{Style.RESET_ALL}")
        print(f"  p-value: {Fore.CYAN}{p_value:.4f}{Style.RESET_ALL}")
        
        if p_value < 0.05:
            print(f"  {Fore.GREEN}✓ Statistically significant difference (p < 0.05){Style.RESET_ALL}")
        else:
            print(f"  {Fore.YELLOW}⚠ No significant difference (p >= 0.05){Style.RESET_ALL}")
    
    return no_god, rule_god

def visualize_comparison(results: Dict, output_dir: str = 'outputs/god_ai'):
    """Create comparison visualizations."""
    
    print(f"\n{Fore.CYAN}{'─'*100}")
    print(f"{Fore.YELLOW}📈 GENERATING VISUALIZATIONS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'─'*100}{Style.RESET_ALL}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    no_god = results['no_god_trials']
    rule_god = results['rule_god_trials']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('God-AI Controller vs Baseline Comparison', fontsize=16, fontweight='bold')
    
    # 1. Population Over Time (average across trials)
    ax = axes[0, 0]
    max_len = max(max(len(t['population_history']) for t in no_god),
                  max(len(t['population_history']) for t in rule_god))
    
    no_god_pop_avg = []
    rule_god_pop_avg = []
    
    for gen in range(max_len):
        no_god_vals = [t['population_history'][gen] for t in no_god if gen < len(t['population_history'])]
        rule_god_vals = [t['population_history'][gen] for t in rule_god if gen < len(t['population_history'])]
        
        if no_god_vals:
            no_god_pop_avg.append(np.mean(no_god_vals))
        if rule_god_vals:
            rule_god_pop_avg.append(np.mean(rule_god_vals))
    
    ax.plot(no_god_pop_avg, label='No God', color='red', linewidth=2, alpha=0.7)
    ax.plot(rule_god_pop_avg, label='Rule-Based God', color='green', linewidth=2, alpha=0.7)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Population')
    ax.set_title('Population Over Time (Averaged)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Cooperation Over Time
    ax = axes[0, 1]
    no_god_coop_avg = []
    rule_god_coop_avg = []
    
    for gen in range(max_len):
        no_god_vals = [t['cooperation_history'][gen] for t in no_god if gen < len(t['cooperation_history'])]
        rule_god_vals = [t['cooperation_history'][gen] for t in rule_god if gen < len(t['cooperation_history'])]
        
        if no_god_vals:
            no_god_coop_avg.append(np.mean(no_god_vals))
        if rule_god_vals:
            rule_god_coop_avg.append(np.mean(rule_god_vals))
    
    ax.plot(no_god_coop_avg, label='No God', color='red', linewidth=2, alpha=0.7)
    ax.plot(rule_god_coop_avg, label='Rule-Based God', color='green', linewidth=2, alpha=0.7)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title('Cooperation Over Time (Averaged)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Final Metrics Comparison (Box plots)
    ax = axes[0, 2]
    metrics_data = [
        [t['final_cooperation'] for t in no_god if t['survived']],
        [t['final_cooperation'] for t in rule_god if t['survived']]
    ]
    ax.boxplot(metrics_data, labels=['No God', 'Rule God'])
    ax.set_ylabel('Final Cooperation Rate')
    ax.set_title('Final Cooperation Distribution')
    ax.grid(alpha=0.3)
    
    # 4. Survival Rate
    ax = axes[1, 0]
    survival_rates = [
        sum(1 for t in no_god if t['survived']) / len(no_god) * 100,
        sum(1 for t in rule_god if t['survived']) / len(rule_god) * 100
    ]
    bars = ax.bar(['No God', 'Rule God'], survival_rates, color=['red', 'green'], alpha=0.7)
    ax.set_ylabel('Survival Rate (%)')
    ax.set_title('Population Survival Rate')
    ax.set_ylim([0, 110])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 5. Average Wealth Comparison
    ax = axes[1, 1]
    wealth_data = [
        [t['final_avg_wealth'] for t in no_god if t['survived']],
        [t['final_avg_wealth'] for t in rule_god if t['survived']]
    ]
    ax.boxplot(wealth_data, labels=['No God', 'Rule God'])
    ax.set_ylabel('Final Average Wealth')
    ax.set_title('Wealth Distribution')
    ax.grid(alpha=0.3)
    
    # 6. Tribe Diversity
    ax = axes[1, 2]
    diversity_data = [
        [t['tribe_diversity'] for t in no_god if t['survived']],
        [t['tribe_diversity'] for t in rule_god if t['survived']]
    ]
    ax.boxplot(diversity_data, labels=['No God', 'Rule God'])
    ax.set_ylabel('Tribe Diversity')
    ax.set_title('Genetic Diversity')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'god_comparison_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"{Fore.GREEN}✓ Visualization saved: {filepath}{Style.RESET_ALL}")
    
    plt.close()

def save_experiment_results(results: Dict, output_dir: str = 'outputs/god_ai'):
    """Save full experiment results to JSON."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'experiment_results_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"{Fore.GREEN}✓ Results saved: {filepath}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    # Run comparative experiment
    results = run_controlled_experiment(
        generations=500,
        trials=5,
        initial_size=100
    )
    
    # Analyze results
    analyze_results(results)
    
    # Create visualizations
    visualize_comparison(results)
    
    # Save results
    save_experiment_results(results)
    
    print(f"\n{Fore.CYAN}{'='*100}")
    print(f"{Fore.GREEN}✅ EXPERIMENT COMPLETE!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")
