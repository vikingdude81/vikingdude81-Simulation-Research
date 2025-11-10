"""
Multi-Environment Evolution and Testing
Test champion across all environments and evolve environment-specific specialists
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from quantum_genetic_agents import QuantumAgent
from adaptive_mutation_gpu_ml import AdaptiveMutationEvolution
import json
from datetime import datetime
import time

# Setup
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (20, 12)

# Environments to test
ENVIRONMENTS = ['standard', 'harsh', 'gentle', 'chaotic', 'oscillating']

# Champion from ultra-scale run
CHAMPION_GENOME = [3.0, 0.1, 0.005, 0.1842]

def evaluate_genome_in_environment(genome, environment, num_trials=50):
    """Evaluate a genome in a specific environment"""
    results = []
    
    for trial in range(num_trials):
        agent = QuantumAgent(trial, genome, environment)
        
        # Run for 100 timesteps
        for t in range(100):
            agent.evolve(t)
        
        fitness = agent.get_final_fitness()
        results.append(fitness)
    
    return {
        'mean': np.mean(results),
        'std': np.std(results),
        'median': np.median(results),
        'min': np.min(results),
        'max': np.max(results),
        'trials': results
    }

def test_champion_all_environments():
    """Test champion genome across all environments"""
    print("\n" + "="*80)
    print("🏆 TESTING CHAMPION ACROSS ALL ENVIRONMENTS")
    print("="*80)
    print(f"Champion Genome: μ={CHAMPION_GENOME[0]}, ω={CHAMPION_GENOME[1]}, d={CHAMPION_GENOME[2]}, φ={CHAMPION_GENOME[3]:.4f}")
    print()
    
    results = {}
    
    for env in ENVIRONMENTS:
        print(f"Testing in {env.upper()} environment... ", end='', flush=True)
        start = time.time()
        
        env_results = evaluate_genome_in_environment(CHAMPION_GENOME, env, num_trials=50)
        results[env] = env_results
        
        elapsed = time.time() - start
        print(f"✓ Mean: {env_results['mean']:.6f} ± {env_results['std']:.6f} ({elapsed:.1f}s)")
    
    return results

def evolve_environment_specialist(environment, population_size=200, generations=100):
    """Evolve a specialist genome for a specific environment"""
    print(f"\n🧬 EVOLVING SPECIALIST FOR {environment.upper()} ENVIRONMENT")
    print(f"Population: {population_size} agents, Generations: {generations}")
    print("-" * 80)
    
    evolution = AdaptiveMutationEvolution(
        population_size=population_size,
        strategy='ml_adaptive',
        use_gpu=True
    )
    
    # Run evolution with environment-specific evaluation
    results = evolution.run(generations=generations, environment=environment, train_predictor=False)
    
    # Get best genome
    best_agent = evolution.population[0]  # Sorted by fitness
    best_genome = best_agent[1]
    
    print(f"Final: μ={best_genome[0]:.2f}, ω={best_genome[1]:.2f}, d={best_genome[2]:.4f}, φ={best_genome[3]:.2f}")
    
    return {
        'best_genome': best_genome,
        'best_fitness': best_agent[0],
        'fitness_history': results['best_fitness'],
        'mean_history': results['avg_fitness']
    }

def cross_test_specialists(specialists):
    """Test each specialist in all environments"""
    print("\n" + "="*80)
    print("🔬 CROSS-TESTING SPECIALISTS IN ALL ENVIRONMENTS")
    print("="*80)
    
    results = {}
    
    for specialist_env, data in specialists.items():
        genome = data['best_genome']
        print(f"\n{specialist_env.upper()} Specialist: μ={genome[0]:.2f}, ω={genome[1]:.2f}, d={genome[2]:.4f}, φ={genome[3]:.2f}")
        
        env_results = {}
        for test_env in ENVIRONMENTS:
            env_result = evaluate_genome_in_environment(genome, test_env, num_trials=30)
            env_results[test_env] = env_result
            
            marker = "⭐" if test_env == specialist_env else "  "
            print(f"  {marker} {test_env:12s}: {env_result['mean']:12.6f} ± {env_result['std']:.6f}")
        
        results[specialist_env] = env_results
    
    return results

def visualize_results(champion_results, specialists, cross_test_results):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Champion performance across environments
    ax1 = fig.add_subplot(gs[0, :])
    envs = list(champion_results.keys())
    means = [champion_results[e]['mean'] for e in envs]
    stds = [champion_results[e]['std'] for e in envs]
    
    bars = ax1.bar(envs, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
    ax1.set_title('🏆 Champion Performance Across All Environments', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Fitness (Mean ± Std)', fontsize=12)
    ax1.set_xlabel('Environment', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2-6. Specialist evolution histories
    colors_env = {'standard': '#3498db', 'harsh': '#e74c3c', 'gentle': '#2ecc71', 
                  'chaotic': '#f39c12', 'oscillating': '#9b59b6'}
    
    for idx, (env, data) in enumerate(specialists.items()):
        row = 1 + (idx // 3)
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        gens = range(len(data['fitness_history']))
        ax.plot(gens, data['fitness_history'], label='Best', color=colors_env[env], linewidth=2)
        ax.plot(gens, data['mean_history'], label='Mean', color=colors_env[env], 
                linewidth=1.5, alpha=0.6, linestyle='--')
        
        ax.set_title(f'{env.upper()} Specialist Evolution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation', fontsize=10)
        ax.set_ylabel('Fitness', fontsize=10)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Add final genome info
        genome = data['best_genome']
        info_text = f"Final: μ={genome[0]:.2f}, ω={genome[1]:.2f}\nd={genome[2]:.4f}, φ={genome[3]:.2f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 7. Cross-test heatmap
    ax7 = fig.add_subplot(gs[3, :2])
    
    # Build matrix: rows=specialists, cols=test_environments
    specialist_names = list(cross_test_results.keys())
    matrix = []
    for specialist_env in specialist_names:
        row = [cross_test_results[specialist_env][test_env]['mean'] 
               for test_env in ENVIRONMENTS]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Normalize for visualization (log scale if needed)
    if np.max(matrix) > 1000:
        matrix_vis = np.log10(matrix + 1)
        label = "Fitness (log10 scale)"
    else:
        matrix_vis = matrix
        label = "Fitness"
    
    im = ax7.imshow(matrix_vis, cmap='YlOrRd', aspect='auto')
    ax7.set_xticks(range(len(ENVIRONMENTS)))
    ax7.set_yticks(range(len(specialist_names)))
    ax7.set_xticklabels(ENVIRONMENTS, rotation=45, ha='right')
    ax7.set_yticklabels([f"{s.upper()} Specialist" for s in specialist_names])
    ax7.set_title('🔬 Cross-Environment Performance Matrix', fontsize=14, fontweight='bold', pad=15)
    ax7.set_xlabel('Test Environment', fontsize=12)
    ax7.set_ylabel('Specialist Type', fontsize=12)
    
    # Add values to heatmap
    for i in range(len(specialist_names)):
        for j in range(len(ENVIRONMENTS)):
            text = ax7.text(j, i, f'{matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9,
                           fontweight='bold' if i == j else 'normal')
    
    plt.colorbar(im, ax=ax7, label=label)
    
    # 8. Genome parameter comparison
    ax8 = fig.add_subplot(gs[3, 2])
    
    # Compare genome parameters
    param_names = ['μ', 'ω', 'd', 'φ']
    x = np.arange(len(param_names))
    width = 0.15
    
    # Champion
    champion_params = CHAMPION_GENOME
    ax8.bar(x - 2*width, champion_params, width, label='Champion', alpha=0.8, color='gold')
    
    # Specialists
    for idx, (env, data) in enumerate(specialists.items()):
        genome = data['best_genome']
        ax8.bar(x + (idx - 1)*width, genome, width, label=f'{env}', alpha=0.7, color=colors_env[env])
    
    ax8.set_ylabel('Parameter Value', fontsize=10)
    ax8.set_title('Genome Parameter Comparison', fontsize=12, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(param_names, fontsize=10)
    ax8.legend(fontsize=8, loc='upper right')
    ax8.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Multi-Environment Evolution Analysis', fontsize=20, fontweight='bold', y=0.995)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multi_environment_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {filename}")
    
    return filename

def main():
    start_time = time.time()
    
    print("\n" + "="*80)
    print("🌍 MULTI-ENVIRONMENT EVOLUTION EXPERIMENT")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Phase 1: Test champion in all environments
    champion_results = test_champion_all_environments()
    
    # Phase 2: Evolve specialist for each environment
    print("\n" + "="*80)
    print("🧬 EVOLVING ENVIRONMENT-SPECIFIC SPECIALISTS")
    print("="*80)
    
    specialists = {}
    for env in ENVIRONMENTS:
        specialist_data = evolve_environment_specialist(env, population_size=200, generations=100)
        specialists[env] = specialist_data
    
    # Phase 3: Cross-test all specialists
    cross_test_results = cross_test_specialists(specialists)
    
    # Phase 4: Find generalist
    print("\n" + "="*80)
    print("🌟 FINDING BEST GENERALIST")
    print("="*80)
    
    # Test champion + all specialists for average performance
    all_genomes = {
        'champion': {
            'genome': CHAMPION_GENOME,
            'source': 'ultra-scale evolution'
        }
    }
    
    for env, data in specialists.items():
        all_genomes[f'{env}_specialist'] = {
            'genome': data['best_genome'],
            'source': f'{env} evolution'
        }
    
    generalist_scores = {}
    
    for name, data in all_genomes.items():
        genome = data['genome']
        env_scores = []
        
        print(f"\nTesting {name}...")
        for env in ENVIRONMENTS:
            result = evaluate_genome_in_environment(genome, env, num_trials=30)
            env_scores.append(result['mean'])
            print(f"  {env:12s}: {result['mean']:12.6f}")
        
        avg_score = np.mean(env_scores)
        min_score = np.min(env_scores)
        std_score = np.std(env_scores)
        
        generalist_scores[name] = {
            'average': avg_score,
            'minimum': min_score,
            'std': std_score,
            'scores': env_scores,
            'genome': genome
        }
        
        print(f"  → Average: {avg_score:.6f}, Min: {min_score:.6f}, Std: {std_score:.6f}")
    
    # Rank generalists
    print("\n" + "="*80)
    print("🏆 GENERALIST RANKINGS")
    print("="*80)
    
    # Sort by average performance
    ranked = sorted(generalist_scores.items(), key=lambda x: x[1]['average'], reverse=True)
    
    for rank, (name, data) in enumerate(ranked, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        genome = data['genome']
        print(f"{emoji} {name:20s}: Avg={data['average']:12.6f}, Min={data['minimum']:12.6f}")
        print(f"   Genome: μ={genome[0]:.2f}, ω={genome[1]:.2f}, d={genome[2]:.4f}, φ={genome[3]:.2f}")
    
    # Visualization
    viz_file = visualize_results(champion_results, specialists, cross_test_results)
    
    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"multi_environment_results_{timestamp}.json"
    
    export_data = {
        'timestamp': timestamp,
        'champion_genome': CHAMPION_GENOME,
        'champion_results': {k: {
            'mean': float(v['mean']),
            'std': float(v['std']),
            'median': float(v['median'])
        } for k, v in champion_results.items()},
        'specialists': {k: {
            'best_genome': [float(x) for x in v['best_genome']],
            'best_fitness': float(v['best_fitness'])
        } for k, v in specialists.items()},
        'generalist_rankings': {k: {
            'average': float(v['average']),
            'minimum': float(v['minimum']),
            'genome': [float(x) for x in v['genome']]
        } for k, v in generalist_scores.items()}
    }
    
    with open(results_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n📁 Results saved: {results_file}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
    print("\n" + "="*80)
    print("✅ MULTI-ENVIRONMENT EXPERIMENT COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
