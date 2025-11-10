"""
🏆 ULTRA-SCALE CHAMPION vs PRODUCTION GENOMES
==============================================

Compare the 1000-agent ultra-scale champion against existing production genomes
to see if we've discovered a superior agent!
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    class Style:
        BRIGHT = RESET_ALL = ''

from quantum_genetic_agents import QuantumAgent


def load_production_genomes():
    """Load all production genome files"""
    genome_files = list(Path('.').glob('production_genome_gen*.json'))
    
    genomes = []
    for file in sorted(genome_files):
        with open(file, 'r') as f:
            data = json.load(f)
            genomes.append({
                'file': file.name,
                'generation': data['generation'],
                'genome': data['genome'],
                'fitness': data['fitness'],
                'environment': data['environment']
            })
    
    return genomes


def evaluate_agent(agent_id, genome, environment='standard', num_trials=100):
    """Evaluate an agent's fitness over multiple trials"""
    fitnesses = []
    
    for _ in range(num_trials):
        agent = QuantumAgent(agent_id, genome, environment)
        
        # Simulate for fixed timesteps
        for t in range(100):
            agent.evolve(t)
        
        # Calculate final fitness
        agent.calculate_fitness()
        fitnesses.append(agent.fitness)
    
    return {
        'mean': np.mean(fitnesses),
        'std': np.std(fitnesses),
        'min': np.min(fitnesses),
        'max': np.max(fitnesses),
        'median': np.median(fitnesses)
    }


def compare_genomes():
    """Compare ultra-scale champion with production genomes"""
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}🏆 ULTRA-SCALE CHAMPION vs PRODUCTION GENOMES{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    # Ultra-scale champion from 1000-agent evolution
    champion = {
        'name': 'Ultra-Scale Champion (1000 agents)',
        'genome': [3.0000, 0.1000, 0.0050, 0.1842],
        'source': '1000-agent evolution, Gen 200',
        'color': '#2ECC71'
    }
    
    print(f'{Fore.CYAN}Loading production genomes...{Style.RESET_ALL}')
    production_genomes = load_production_genomes()
    print(f'{Fore.GREEN}✅ Loaded {len(production_genomes)} production genomes{Style.RESET_ALL}\n')
    
    # Evaluate champion
    print(f'{Fore.YELLOW}{Style.BRIGHT}Evaluating Ultra-Scale Champion...{Style.RESET_ALL}')
    print(f'{Fore.WHITE}Genome: μ={champion["genome"][0]:.4f}, ω={champion["genome"][1]:.4f}, '
          f'd={champion["genome"][2]:.4f}, φ={champion["genome"][3]:.4f}{Style.RESET_ALL}')
    
    champion_results = evaluate_agent(champion['genome'], num_trials=100)
    
    print(f'{Fore.GREEN}Results (100 trials):{Style.RESET_ALL}')
    print(f'  Mean:   {Fore.GREEN}{Style.BRIGHT}{champion_results["mean"]:.6f}{Style.RESET_ALL}')
    print(f'  Median: {Fore.YELLOW}{champion_results["median"]:.6f}{Style.RESET_ALL}')
    print(f'  Std:    {Fore.WHITE}{champion_results["std"]:.6f}{Style.RESET_ALL}')
    print(f'  Range:  {champion_results["min"]:.6f} - {champion_results["max"]:.6f}')
    
    # Evaluate production genomes
    print(f'\n{Fore.YELLOW}{Style.BRIGHT}Evaluating Production Genomes...{Style.RESET_ALL}\n')
    
    all_results = [{'name': champion['name'], 'results': champion_results, 'genome_data': champion}]
    
    for i, genome_data in enumerate(production_genomes, 1):
        print(f'{Fore.CYAN}[{i}/{len(production_genomes)}] {genome_data["file"]}{Style.RESET_ALL}')
        print(f'  Gen: {genome_data["generation"]}, Env: {genome_data["environment"]}')
        print(f'  Genome: μ={genome_data["genome"][0]:.4f}, ω={genome_data["genome"][1]:.4f}, '
              f'd={genome_data["genome"][2]:.4f}, φ={genome_data["genome"][3]:.4f}')
        
        results = evaluate_agent(genome_data['genome'], num_trials=100)
        
        print(f'  Mean: {results["mean"]:.6f}, Median: {results["median"]:.6f}, '
              f'Std: {results["std"]:.6f}\n')
        
        all_results.append({
            'name': f'Gen {genome_data["generation"]} ({genome_data["environment"]})',
            'results': results,
            'genome_data': genome_data
        })
    
    # Rankings
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}🏅 FINAL RANKINGS (by Mean Fitness){Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    # Sort by mean fitness
    ranked = sorted(all_results, key=lambda x: x['results']['mean'], reverse=True)
    
    for rank, entry in enumerate(ranked, 1):
        results = entry['results']
        
        if rank == 1:
            rank_color = Fore.YELLOW + Style.BRIGHT
            medal = '🥇'
        elif rank == 2:
            rank_color = Fore.WHITE + Style.BRIGHT
            medal = '🥈'
        elif rank == 3:
            rank_color = Fore.RED + Style.BRIGHT
            medal = '🥉'
        else:
            rank_color = Fore.WHITE
            medal = f'  {rank}.'
        
        # Highlight champion
        if 'Ultra-Scale' in entry['name']:
            name_display = f'{Fore.GREEN}{Style.BRIGHT}{entry["name"]}{Style.RESET_ALL} ⭐'
        else:
            name_display = entry['name']
        
        print(f'{rank_color}{medal}{Style.RESET_ALL} {name_display}')
        print(f'    Mean: {Fore.GREEN}{Style.BRIGHT}{results["mean"]:.6f}{Style.RESET_ALL} | '
              f'Median: {Fore.YELLOW}{results["median"]:.6f}{Style.RESET_ALL} | '
              f'Std: {Fore.WHITE}{results["std"]:.6f}{Style.RESET_ALL}')
        
        # Show genome
        genome = entry['genome_data']['genome']
        print(f'    Genome: {Fore.RED}μ={genome[0]:.4f}{Style.RESET_ALL} '
              f'{Fore.GREEN}ω={genome[1]:.4f}{Style.RESET_ALL} '
              f'{Fore.BLUE}d={genome[2]:.4f}{Style.RESET_ALL} '
              f'{Fore.MAGENTA}φ={genome[3]:.4f}{Style.RESET_ALL}')
        print()
    
    # Statistical comparison
    champion_mean = champion_results['mean']
    best_production = max([r['results']['mean'] for r in all_results[1:]])
    best_production_name = [r['name'] for r in all_results[1:] 
                           if r['results']['mean'] == best_production][0]
    
    print(f'\n{Fore.CYAN}{Style.BRIGHT}📊 STATISTICAL COMPARISON:{Style.RESET_ALL}\n')
    
    if champion_mean > best_production:
        improvement = ((champion_mean - best_production) / best_production * 100)
        print(f'{Fore.GREEN}{Style.BRIGHT}✅ CHAMPION WINS!{Style.RESET_ALL}')
        print(f'Ultra-Scale Champion is {Fore.GREEN}{Style.BRIGHT}{improvement:.2f}%{Style.RESET_ALL} '
              f'better than best production genome')
        print(f'Champion: {champion_mean:.6f} vs Best Production: {best_production:.6f}')
    else:
        difference = ((best_production - champion_mean) / best_production * 100)
        print(f'{Fore.YELLOW}⚠️  Production genome "{best_production_name}" still leads{Style.RESET_ALL}')
        print(f'Best Production: {best_production:.6f} vs Champion: {champion_mean:.6f}')
        print(f'Difference: {difference:.2f}%')
    
    print(f'\n{Fore.CYAN}Champion Position:{Style.RESET_ALL}')
    champion_rank = [i for i, r in enumerate(ranked, 1) if 'Ultra-Scale' in r['name']][0]
    print(f'  Ranked #{champion_rank} out of {len(ranked)} total genomes')
    
    # Genome pattern analysis
    print(f'\n{Fore.CYAN}{Style.BRIGHT}🧬 GENOME PATTERN ANALYSIS:{Style.RESET_ALL}\n')
    
    # Calculate average parameters for top 3
    top3 = ranked[:3]
    top3_genomes = [r['genome_data']['genome'] for r in top3]
    
    avg_mu = np.mean([g[0] for g in top3_genomes])
    avg_omega = np.mean([g[1] for g in top3_genomes])
    avg_d = np.mean([g[2] for g in top3_genomes])
    avg_phi = np.mean([g[3] for g in top3_genomes])
    
    print(f'{Fore.YELLOW}Top 3 Average Parameters:{Style.RESET_ALL}')
    print(f'  μ (mutation):     {Fore.RED}{avg_mu:.4f}{Style.RESET_ALL}')
    print(f'  ω (oscillation):  {Fore.GREEN}{avg_omega:.4f}{Style.RESET_ALL}')
    print(f'  d (decoherence):  {Fore.BLUE}{avg_d:.4f}{Style.RESET_ALL}')
    print(f'  φ (phase):        {Fore.MAGENTA}{avg_phi:.4f}{Style.RESET_ALL}')
    
    print(f'\n{Fore.YELLOW}Champion vs Top 3 Average:{Style.RESET_ALL}')
    print(f'  μ: {champion["genome"][0]:.4f} vs {avg_mu:.4f} '
          f'({Fore.GREEN if champion["genome"][0] > avg_mu else Fore.RED}'
          f'{(champion["genome"][0] - avg_mu):+.4f}{Style.RESET_ALL})')
    print(f'  ω: {champion["genome"][1]:.4f} vs {avg_omega:.4f} '
          f'({Fore.GREEN if champion["genome"][1] < avg_omega else Fore.RED}'
          f'{(champion["genome"][1] - avg_omega):+.4f}{Style.RESET_ALL})')
    print(f'  d: {champion["genome"][2]:.4f} vs {avg_d:.4f} '
          f'({Fore.GREEN if champion["genome"][2] < avg_d else Fore.RED}'
          f'{(champion["genome"][2] - avg_d):+.4f}{Style.RESET_ALL})')
    
    # Visualization
    print(f'\n{Fore.CYAN}📊 Generating comparison visualization...{Style.RESET_ALL}')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🏆 Ultra-Scale Champion vs Production Genomes', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Mean fitness comparison
    ax = axes[0, 0]
    names = [r['name'][:30] for r in ranked]  # Truncate long names
    means = [r['results']['mean'] for r in ranked]
    colors = ['#2ECC71' if 'Ultra-Scale' in r['name'] else '#3498DB' for r in ranked]
    
    bars = ax.barh(range(len(names)), means, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Mean Fitness (100 trials)')
    ax.set_title('Mean Fitness Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Fitness distribution box plot
    ax = axes[0, 1]
    top5_names = [r['name'][:20] for r in ranked[:5]]
    top5_means = [r['results']['mean'] for r in ranked[:5]]
    top5_stds = [r['results']['std'] for r in ranked[:5]]
    
    ax.bar(range(5), top5_means, yerr=top5_stds, 
          color=['#2ECC71' if 'Ultra-Scale' in ranked[i]['name'] else '#3498DB' for i in range(5)],
          alpha=0.8, capsize=5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(top5_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Fitness ± Std Dev')
    ax.set_title('Top 5 Agents with Error Bars', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Parameter comparison (top 5)
    ax = axes[1, 0]
    params = ['μ', 'ω', 'd', 'φ']
    x = np.arange(len(params))
    width = 0.15
    
    for i in range(min(5, len(ranked))):
        genome = ranked[i]['genome_data']['genome']
        offset = (i - 2) * width
        color = '#2ECC71' if 'Ultra-Scale' in ranked[i]['name'] else f'C{i}'
        ax.bar(x + offset, genome, width, label=ranked[i]['name'][:20], 
              color=color, alpha=0.8)
    
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Value')
    ax.set_title('Genome Parameter Comparison (Top 5)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Scatter: mutation rate vs fitness
    ax = axes[1, 1]
    mus = [r['genome_data']['genome'][0] for r in ranked]
    fitness_means = [r['results']['mean'] for r in ranked]
    point_colors = ['#2ECC71' if 'Ultra-Scale' in r['name'] else '#3498DB' for r in ranked]
    
    ax.scatter(mus, fitness_means, c=point_colors, s=100, alpha=0.6, edgecolors='black')
    
    # Highlight champion
    champ_mu = champion['genome'][0]
    champ_fitness = champion_results['mean']
    ax.scatter([champ_mu], [champ_fitness], c='#2ECC71', s=300, 
              marker='*', edgecolors='gold', linewidths=2, label='Champion', zorder=5)
    
    ax.set_xlabel('Mutation Rate (μ)')
    ax.set_ylabel('Mean Fitness')
    ax.set_title('Mutation Rate vs Fitness', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = 'champion_vs_production_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'{Fore.GREEN}✅ Saved: {filename}{Style.RESET_ALL}')
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}✅ Comparison Complete!{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    return ranked


if __name__ == "__main__":
    try:
        results = compare_genomes()
    except FileNotFoundError as e:
        print(f'\n{Fore.RED}❌ Error: Could not find production genome files{Style.RESET_ALL}')
        print(f'{Fore.YELLOW}Make sure you are in the quantum_genetics directory{Style.RESET_ALL}')
        print(f'{Fore.YELLOW}Looking for files: production_genome_gen*.json{Style.RESET_ALL}\n')
    except Exception as e:
        print(f'\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}\n')
        raise
