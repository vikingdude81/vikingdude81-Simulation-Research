"""
🏆 ULTRA-SCALE CHAMPION vs KNOWN GENOMES - SIMPLIFIED
======================================================

Compare the 1000-agent champion against known good genomes.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_genetic_agents import QuantumAgent

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    class Style:
        BRIGHT = RESET_ALL = ''


def evaluate_genome(genome, environment='standard', num_trials=50):
    """Evaluate a genome over multiple trials"""
    fitnesses = []
    
    for trial in range(num_trials):
        agent = QuantumAgent(trial, genome, environment)
        
        # Simulate for 100 timesteps
        for t in range(100):
            agent.evolve(t)
        
        # Get final fitness
        fitness = agent.get_final_fitness()
        fitnesses.append(fitness)
    
    return {
        'mean': np.mean(fitnesses),
        'std': np.std(fitnesses),
        'min': np.min(fitnesses),
        'max': np.max(fitnesses),
        'median': np.median(fitnesses)
    }


def compare_champion():
    """Compare ultra-scale champion with known genomes"""
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}🏆 CHAMPION vs KNOWN GENOMES COMPARISON{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    # Define genomes to compare
    genomes = [
        {
            'name': '🥇 Ultra-Scale Champion',
            'genome': [3.0000, 0.1000, 0.0050, 0.1842],
            'source': '1000 agents × 200 gens, ML adaptive',
            'is_champion': True
        },
        {
            'name': 'Gen 5878 Historic Best',
            'genome': [0.65, 1.077, 0.011, 2.486],
            'source': 'Historical best from analysis',
            'is_champion': False
        },
        {
            'name': 'Low Mutation Stable',
            'genome': [0.3, 0.7, 0.01, 0.5],
            'source': 'Conservative strategy',
            'is_champion': False
        },
        {
            'name': 'High Mutation Explorer',
            'genome': [3.0, 0.5, 0.01, 1.0],
            'source': 'Aggressive exploration',
            'is_champion': False
        },
        {
            'name': 'Optimal Decoherence',
            'genome': [1.5, 1.0, 0.011, 1.5],
            'source': 'd=0.011 optimized',
            'is_champion': False
        },
        {
            'name': 'Balanced Strategy',
            'genome': [1.5, 1.2, 0.02, 2.0],
            'source': 'Balanced params',
            'is_champion': False
        },
        {
            'name': 'Ultra-Low Decoherence',
            'genome': [2.5, 0.2, 0.005, 0.3],
            'source': 'd=0.005 like champion',
            'is_champion': False
        }
    ]
    
    print(f'{Fore.CYAN}Evaluating {len(genomes)} genomes (50 trials each)...{Style.RESET_ALL}\n')
    
    # Evaluate each genome
    results = []
    for i, genome_data in enumerate(genomes, 1):
        print(f'{Fore.YELLOW}[{i}/{len(genomes)}] {genome_data["name"]}{Style.RESET_ALL}')
        g = genome_data['genome']
        print(f'  Genome: μ={g[0]:.4f}, ω={g[1]:.4f}, d={g[2]:.4f}, φ={g[3]:.4f}')
        print(f'  Source: {genome_data["source"]}')
        
        eval_results = evaluate_genome(g, environment='standard', num_trials=50)
        
        print(f'  {Fore.GREEN}Mean: {eval_results["mean"]:.6f}{Style.RESET_ALL} | '
              f'Median: {eval_results["median"]:.6f} | '
              f'Std: {eval_results["std"]:.6f}')
        print()
        
        results.append({
            **genome_data,
            'results': eval_results
        })
    
    # Rankings
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}🏅 FINAL RANKINGS{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    ranked = sorted(results, key=lambda x: x['results']['mean'], reverse=True)
    
    for rank, entry in enumerate(ranked, 1):
        res = entry['results']
        g = entry['genome']
        
        if rank == 1:
            medal = '🥇'
            rank_color = Fore.YELLOW + Style.BRIGHT
        elif rank == 2:
            medal = '🥈'
            rank_color = Fore.WHITE + Style.BRIGHT
        elif rank == 3:
            medal = '🥉'
            rank_color = Fore.RED + Style.BRIGHT
        else:
            medal = f'{rank}.'
            rank_color = Fore.WHITE
        
        name_display = f'{Fore.GREEN}{Style.BRIGHT}{entry["name"]}{Style.RESET_ALL}' if entry['is_champion'] else entry['name']
        
        print(f'{rank_color}{medal:3s}{Style.RESET_ALL} {name_display}')
        print(f'     Mean: {Fore.GREEN}{Style.BRIGHT}{res["mean"]:.6f}{Style.RESET_ALL} | '
              f'Median: {Fore.YELLOW}{res["median"]:.6f}{Style.RESET_ALL} | '
              f'Std: {Fore.WHITE}{res["std"]:.6f}{Style.RESET_ALL}')
        print(f'     Genome: {Fore.RED}μ={g[0]:.4f}{Style.RESET_ALL} '
              f'{Fore.GREEN}ω={g[1]:.4f}{Style.RESET_ALL} '
              f'{Fore.BLUE}d={g[2]:.4f}{Style.RESET_ALL} '
              f'{Fore.MAGENTA}φ={g[3]:.4f}{Style.RESET_ALL}')
        print()
    
    # Analysis
    champion_rank = [i for i, r in enumerate(ranked, 1) if r['is_champion']][0]
    champion_result = [r for r in ranked if r['is_champion']][0]
    
    print(f'{Fore.CYAN}{Style.BRIGHT}📊 CHAMPION ANALYSIS:{Style.RESET_ALL}\n')
    print(f'{Fore.WHITE}Champion Rank:{Style.RESET_ALL}        #{champion_rank} out of {len(genomes)}')
    print(f'{Fore.WHITE}Champion Mean Fitness:{Style.RESET_ALL} {champion_result["results"]["mean"]:.6f}')
    
    if champion_rank == 1:
        second_best = ranked[1]['results']['mean']
        improvement = ((champion_result['results']['mean'] - second_best) / second_best * 100)
        print(f'{Fore.GREEN}{Style.BRIGHT}✅ CHAMPION WINS!{Style.RESET_ALL}')
        print(f'{Fore.WHITE}Advantage over #2:{Style.RESET_ALL}    {Fore.GREEN}{improvement:.2f}%{Style.RESET_ALL} better')
    else:
        best_mean = ranked[0]['results']['mean']
        difference = ((best_mean - champion_result['results']['mean']) / best_mean * 100)
        print(f'{Fore.YELLOW}⚠️  Champion ranked #{champion_rank}{Style.RESET_ALL}')
        print(f'{Fore.WHITE}Gap to #1:{Style.RESET_ALL}            {Fore.RED}{difference:.2f}%{Style.RESET_ALL} behind "{ranked[0]["name"]}"')
    
    # Genome pattern analysis
    print(f'\n{Fore.CYAN}{Style.BRIGHT}🧬 GENOME PATTERN INSIGHTS:{Style.RESET_ALL}\n')
    
    top3 = ranked[:3]
    top3_genomes = np.array([r['genome'] for r in top3])
    
    avg_params = np.mean(top3_genomes, axis=0)
    print(f'{Fore.YELLOW}Top 3 Average Parameters:{Style.RESET_ALL}')
    print(f'  μ (mutation):     {Fore.RED}{avg_params[0]:.4f}{Style.RESET_ALL}')
    print(f'  ω (oscillation):  {Fore.GREEN}{avg_params[1]:.4f}{Style.RESET_ALL}')
    print(f'  d (decoherence):  {Fore.BLUE}{avg_params[2]:.4f}{Style.RESET_ALL}')
    print(f'  φ (phase):        {Fore.MAGENTA}{avg_params[3]:.4f}{Style.RESET_ALL}')
    
    # Key findings
    print(f'\n{Fore.YELLOW}Key Findings:{Style.RESET_ALL}')
    
    # Decoherence analysis
    d_values = [r['genome'][2] for r in top3]
    print(f'  • Top 3 decoherence (d): {Fore.CYAN}{min(d_values):.4f} - {max(d_values):.4f}{Style.RESET_ALL}')
    print(f'    → Lower decoherence (d < 0.012) correlates with better fitness')
    
    # Mutation analysis
    mu_values = [r['genome'][0] for r in top3]
    print(f'  • Top 3 mutation (μ): {Fore.CYAN}{min(mu_values):.4f} - {max(mu_values):.4f}{Style.RESET_ALL}')
    if avg_params[0] > 2.0:
        print(f'    → High mutation strategy (μ > 2.0) dominates!')
    elif avg_params[0] < 1.0:
        print(f'    → Low mutation strategy (μ < 1.0) preferred')
    else:
        print(f'    → Moderate mutation strategy (1.0 < μ < 2.0)')
    
    # Oscillation analysis
    omega_values = [r['genome'][1] for r in top3]
    print(f'  • Top 3 oscillation (ω): {Fore.CYAN}{min(omega_values):.4f} - {max(omega_values):.4f}{Style.RESET_ALL}')
    if avg_params[1] < 0.5:
        print(f'    → Very slow oscillations (ω < 0.5) are optimal!')
    
    # Visualization
    print(f'\n{Fore.CYAN}📊 Generating comparison visualization...{Style.RESET_ALL}')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🏆 Champion vs Known Genomes Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean fitness bar chart
    ax = axes[0, 0]
    names = [r['name'][:25] for r in ranked]
    means = [r['results']['mean'] for r in ranked]
    colors = ['#2ECC71' if r['is_champion'] else '#3498DB' for r in ranked]
    
    bars = ax.barh(range(len(names)), means, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Mean Fitness (50 trials)')
    ax.set_title('Fitness Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Fitness with error bars
    ax = axes[0, 1]
    x_pos = range(len(ranked))
    means = [r['results']['mean'] for r in ranked]
    stds = [r['results']['std'] for r in ranked]
    colors = ['#2ECC71' if r['is_champion'] else '#3498DB' for r in ranked]
    
    ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r['name'][:15] for r in ranked], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Fitness ± Std')
    ax.set_title('Fitness with Variability', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Parameter comparison (top 5)
    ax = axes[1, 0]
    params = ['μ', 'ω', 'd', 'φ']
    x = np.arange(len(params))
    width = 0.15
    
    for i in range(min(5, len(ranked))):
        genome = ranked[i]['genome']
        offset = (i - 2) * width
        color = '#2ECC71' if ranked[i]['is_champion'] else f'C{i}'
        ax.bar(x + offset, genome, width, label=ranked[i]['name'][:20], 
              color=color, alpha=0.8)
    
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Value')
    ax.set_title('Genome Parameters (Top 5)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Scatter - mutation vs fitness
    ax = axes[1, 1]
    mus = [r['genome'][0] for r in ranked]
    fitness_means = [r['results']['mean'] for r in ranked]
    colors = ['#2ECC71' if r['is_champion'] else '#3498DB' for r in ranked]
    
    ax.scatter(mus, fitness_means, c=colors, s=150, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Highlight champion with star
    champ_mu = champion_result['genome'][0]
    champ_fitness = champion_result['results']['mean']
    ax.scatter([champ_mu], [champ_fitness], c='gold', s=500, marker='*', 
              edgecolors='red', linewidths=3, label='Champion', zorder=5)
    
    ax.set_xlabel('Mutation Rate (μ)')
    ax.set_ylabel('Mean Fitness')
    ax.set_title('Mutation vs Fitness', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = 'champion_comparison_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'{Fore.GREEN}✅ Saved: {filename}{Style.RESET_ALL}')
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}✅ Comparison Complete!{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    return ranked


if __name__ == "__main__":
    try:
        results = compare_champion()
    except Exception as e:
        print(f'\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}\n')
        raise
