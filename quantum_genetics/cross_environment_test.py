"""
Cross-Environment Genome Testing

Tests both single-environment and multi-environment champions
across all environments to determine generalization vs overfitting.
"""

import numpy as np
import json
from pathlib import Path
from quantum_genetic_agents import QuantumAgent
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_genome_in_environment(genome, environment, timesteps=100):
    """Evaluate a genome in a specific environment."""
    agent = QuantumAgent(
        agent_id=0,
        genome=genome,
        environment=environment
    )
    
    for t in range(timesteps):
        agent.evolve(t)
    
    return agent.get_final_fitness()


def test_genome_all_environments(genome, name):
    """Test a genome across all environments."""
    environments = ['standard', 'gentle', 'harsh', 'chaotic', 'oscillating', 'unstable', 'extreme', 'mixed']
    
    results = {
        'name': name,
        'genome': genome,
        'fitness_per_env': {}
    }
    
    print(f"\n🧬 Testing: {name}")
    print(f"   Genome: {genome}")
    print(f"\n   Results:")
    
    for env in environments:
        fitness = evaluate_genome_in_environment(genome, env)
        results['fitness_per_env'][env] = float(fitness)
        print(f"      {env:12s}: {fitness:10.2f}")
    
    # Calculate statistics
    fitness_values = list(results['fitness_per_env'].values())
    results['stats'] = {
        'min': float(np.min(fitness_values)),
        'max': float(np.max(fitness_values)),
        'mean': float(np.mean(fitness_values)),
        'std': float(np.std(fitness_values)),
        'range': float(np.max(fitness_values) - np.min(fitness_values))
    }
    
    print(f"\n   📊 Statistics:")
    print(f"      Min:   {results['stats']['min']:10.2f}")
    print(f"      Max:   {results['stats']['max']:10.2f}")
    print(f"      Mean:  {results['stats']['mean']:10.2f}")
    print(f"      Std:   {results['stats']['std']:10.2f}")
    print(f"      Range: {results['stats']['range']:10.2f}")
    
    return results


def visualize_comparison(results_list, output_path):
    """Create comparison visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data
    environments = list(results_list[0]['fitness_per_env'].keys())
    
    # 1. Grouped bar chart
    x = np.arange(len(environments))
    width = 0.35
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, results in enumerate(results_list):
        fitness_values = [results['fitness_per_env'][env] for env in environments]
        offset = (i - len(results_list)/2 + 0.5) * width
        bars = ax1.bar(x + offset, fitness_values, width, 
                      label=results['name'], alpha=0.8, color=colors[i % len(colors)])
        
        # Add value labels on top of bars
        for bar, val in zip(bars, fitness_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Environment', fontweight='bold')
    ax1.set_ylabel('Fitness', fontweight='bold')
    ax1.set_title('Fitness Across Environments', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(environments, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Statistics comparison (box plot style)
    names = [r['name'] for r in results_list]
    stats_data = {
        'Min': [r['stats']['min'] for r in results_list],
        'Mean': [r['stats']['mean'] for r in results_list],
        'Max': [r['stats']['max'] for r in results_list]
    }
    
    x_pos = np.arange(len(names))
    bar_width = 0.25
    
    for i, (stat_name, values) in enumerate(stats_data.items()):
        ax2.bar(x_pos + i * bar_width, values, bar_width, 
               label=stat_name, alpha=0.8)
    
    ax2.set_xlabel('Genome', fontweight='bold')
    ax2.set_ylabel('Fitness', fontweight='bold')
    ax2.set_title('Performance Statistics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos + bar_width)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    plt.close()


def main():
    """Run cross-environment testing."""
    print("\n" + "="*70)
    print("🧪 CROSS-ENVIRONMENT GENOME TESTING")
    print("="*70)
    
    # Define genomes to test
    genomes = [
        {
            'name': 'Single-Env Champion (Ultra-Scale)',
            'genome': [5.0, 0.1, 0.0001, 6.256]
        },
        {
            'name': 'Multi-Env Champion (Robust)',
            'genome': [5.0, 0.1, 0.0001, 6.283]
        }
    ]
    
    # Add original 8 champions if you want
    # genomes.append({
    #     'name': 'Original Champion',
    #     'genome': [5.0, 0.1, 0.005, 6.28]  # Example
    # })
    
    # Test all genomes
    results_list = []
    for genome_info in genomes:
        results = test_genome_all_environments(
            genome_info['genome'],
            genome_info['name']
        )
        results_list.append(results)
    
    # Save results
    output_path = Path(__file__).parent / "cross_environment_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Visualize
    viz_path = Path(__file__).parent / "cross_environment_comparison.png"
    visualize_comparison(results_list, viz_path)
    
    # Final comparison
    print("\n" + "="*70)
    print("🏆 WINNER ANALYSIS")
    print("="*70)
    
    for results in results_list:
        print(f"\n{results['name']}:")
        print(f"   Worst-case (Min): {results['stats']['min']:.2f}")
        print(f"   Best-case (Max):  {results['stats']['max']:.2f}")
        print(f"   Average:          {results['stats']['mean']:.2f}")
        print(f"   Consistency (Std): {results['stats']['std']:.2f} (lower is better)")
    
    # Determine best overall
    best_worst_case = max(results_list, key=lambda x: x['stats']['min'])
    best_average = max(results_list, key=lambda x: x['stats']['mean'])
    most_consistent = min(results_list, key=lambda x: x['stats']['std'])
    
    print("\n🎯 CONCLUSIONS:")
    print(f"   • Best Worst-Case (Robust): {best_worst_case['name']}")
    print(f"   • Best Average Performance: {best_average['name']}")
    print(f"   • Most Consistent: {most_consistent['name']}")
    
    print("\n✅ Cross-environment testing complete!")


if __name__ == "__main__":
    main()
