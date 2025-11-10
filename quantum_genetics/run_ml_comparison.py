"""
🚀⚛️ LARGE-SCALE ML MUTATION COMPARISON
========================================

Comprehensive comparison with larger populations where ML + GPU excel.
Tests all three strategies with proper training data collection.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import sys

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    class Style:
        BRIGHT = RESET_ALL = ''

from adaptive_mutation_gpu_ml import AdaptiveMutationEvolution


def run_large_comparison():
    """Run comprehensive comparison with optimal settings"""
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}🚀⚛️ LARGE-SCALE ML MUTATION STRATEGY COMPARISON{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    # Optimal configuration for fair comparison
    POPULATION_SIZE = 100  # Sweet spot for GPU
    GENERATIONS = 150      # Enough for ML to learn
    ENVIRONMENT = 'standard'
    
    print(f'{Fore.CYAN}Configuration:{Style.RESET_ALL}')
    print(f'  Population: {Fore.YELLOW}{POPULATION_SIZE}{Style.RESET_ALL} agents (GPU advantage)')
    print(f'  Generations: {Fore.YELLOW}{GENERATIONS}{Style.RESET_ALL}')
    print(f'  Environment: {Fore.YELLOW}{ENVIRONMENT}{Style.RESET_ALL}')
    print(f'  ML Training: {Fore.YELLOW}Every 50 generations{Style.RESET_ALL}')
    print()
    
    strategies = ['fixed', 'simple_adaptive', 'ml_adaptive']
    results = {}
    
    for i, strategy in enumerate(strategies, 1):
        print(f'\n{Fore.CYAN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
        print(f'{Fore.CYAN}{Style.BRIGHT}[{i}/3] Testing Strategy: {strategy.upper()}{Style.RESET_ALL}')
        print(f'{Fore.CYAN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
        
        # Decide GPU usage
        use_gpu = (strategy == 'ml_adaptive' or POPULATION_SIZE >= 100)
        
        print(f'{Fore.YELLOW}Initializing...{Style.RESET_ALL}')
        evo = AdaptiveMutationEvolution(
            population_size=POPULATION_SIZE,
            strategy=strategy,
            use_gpu=use_gpu
        )
        
        print(f'{Fore.GREEN}✅ Ready! Running {GENERATIONS} generations...{Style.RESET_ALL}\n')
        
        start_time = time.time()
        
        # Custom evolution loop with progress updates
        for gen in range(GENERATIONS):
            # Adapt mutation
            evo.adapt_mutation_rate(gen)
            
            # Evaluate
            evo.evaluate_population(ENVIRONMENT)
            
            # Track metrics
            best_fitness = evo.population[0][0]
            avg_fitness = np.mean([agent[0] for agent in evo.population])
            diversity = evo._calculate_diversity()
            
            evo.best_fitness_history.append(best_fitness)
            evo.avg_fitness_history.append(avg_fitness)
            evo.diversity_history.append(diversity)
            
            # Collect ML training data
            if strategy == 'ml_adaptive' and gen > 0:
                features = evo.predictor.extract_features(
                    evo.population, gen, evo.best_fitness_history
                )
                evo.predictor.add_training_data(
                    features, 
                    evo.current_mutation_rate,
                    best_fitness
                )
            
            # Evolve
            if gen < GENERATIONS - 1:
                evo.evolve_generation()
            
            # Progress update every 10 generations
            if (gen + 1) % 10 == 0:
                sys.stdout.write(f'\r{Fore.CYAN}Gen {gen+1:3d}/{GENERATIONS}{Style.RESET_ALL} | '
                                f'Best: {Fore.GREEN}{best_fitness:.6f}{Style.RESET_ALL} | '
                                f'Avg: {Fore.YELLOW}{avg_fitness:.6f}{Style.RESET_ALL} | '
                                f'Mutation: {Fore.MAGENTA}{evo.current_mutation_rate:.4f}{Style.RESET_ALL} | '
                                f'Diversity: {Fore.CYAN}{diversity:.4f}{Style.RESET_ALL}')
                sys.stdout.flush()
            
            # Train ML predictor periodically
            if strategy == 'ml_adaptive' and (gen + 1) % 50 == 0:
                print(f'\n{Fore.MAGENTA}🎓 Training ML predictor...{Style.RESET_ALL}')
                evo.predictor.train(epochs=30)
                print(f'{Fore.GREEN}✅ Training complete!{Style.RESET_ALL}')
        
        total_time = time.time() - start_time
        final_best = evo.population[0][0]
        
        # Store results
        results[strategy] = {
            'evo': evo,
            'time': total_time,
            'final_best': final_best,
            'avg_per_gen': total_time / GENERATIONS
        }
        
        print(f'\n\n{Fore.GREEN}✅ Complete!{Style.RESET_ALL}')
        print(f'{Fore.WHITE}Time: {Fore.GREEN}{total_time:.2f}s{Style.RESET_ALL} '
              f'({total_time/GENERATIONS:.3f}s/gen)')
        print(f'{Fore.WHITE}Final Best: {Fore.GREEN}{Style.BRIGHT}{final_best:.6f}{Style.RESET_ALL}')
    
    # Generate comparison
    print(f'\n{Fore.CYAN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.CYAN}{Style.BRIGHT}📊 GENERATING COMPARISON VISUALIZATIONS{Style.RESET_ALL}')
    print(f'{Fore.CYAN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🚀⚛️ ML Mutation Strategy Comparison - Large Scale', 
                 fontsize=16, fontweight='bold')
    
    colors = {
        'fixed': '#FF6B6B',
        'simple_adaptive': '#4ECDC4', 
        'ml_adaptive': '#95E1D3'
    }
    
    labels = {
        'fixed': '📌 Fixed (μ=0.3)',
        'simple_adaptive': '⭐ Simple Adaptive',
        'ml_adaptive': '🤖 ML Adaptive (GPU)'
    }
    
    # Plot 1: Best Fitness Evolution
    ax = axes[0, 0]
    for strategy in strategies:
        evo = results[strategy]['evo']
        ax.plot(evo.best_fitness_history, 
               label=labels[strategy],
               color=colors[strategy],
               linewidth=2,
               alpha=0.8)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.set_title('🏆 Best Fitness Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Average Fitness Evolution
    ax = axes[0, 1]
    for strategy in strategies:
        evo = results[strategy]['evo']
        ax.plot(evo.avg_fitness_history,
               label=labels[strategy],
               color=colors[strategy],
               linewidth=2,
               alpha=0.8)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Average Fitness', fontsize=12)
    ax.set_title('📊 Average Fitness Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Mutation Rate Adaptation
    ax = axes[1, 0]
    for strategy in strategies:
        evo = results[strategy]['evo']
        ax.plot(evo.mutation_rate_history,
               label=labels[strategy],
               color=colors[strategy],
               linewidth=2,
               alpha=0.8)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Mutation Rate', fontsize=12)
    ax.set_title('🧬 Mutation Rate Dynamics', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Performance Summary Bar Chart
    ax = axes[1, 1]
    strategy_names = [labels[s] for s in strategies]
    final_fitness = [results[s]['final_best'] for s in strategies]
    bar_colors = [colors[s] for s in strategies]
    
    bars = ax.bar(range(len(strategies)), final_fitness, color=bar_colors, alpha=0.8)
    ax.set_ylabel('Final Best Fitness', fontsize=12)
    ax.set_title('🏅 Final Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategy_names, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, final_fitness)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.6f}',
               ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'large_scale_ml_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'{Fore.GREEN}✅ Saved: {filename}{Style.RESET_ALL}')
    
    # Final summary
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}🏆 FINAL RESULTS SUMMARY{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    # Sort by fitness
    sorted_strategies = sorted(strategies, 
                               key=lambda s: results[s]['final_best'],
                               reverse=True)
    
    for rank, strategy in enumerate(sorted_strategies, 1):
        res = results[strategy]
        
        if rank == 1:
            rank_color = Fore.YELLOW + Style.BRIGHT
            medal = '🥇'
        elif rank == 2:
            rank_color = Fore.WHITE + Style.BRIGHT
            medal = '🥈'
        else:
            rank_color = Fore.RED
            medal = '🥉'
        
        print(f'{rank_color}{medal} #{rank} {strategy.upper()}{Style.RESET_ALL}')
        print(f'   Final Fitness: {Fore.GREEN}{Style.BRIGHT}{res["final_best"]:.6f}{Style.RESET_ALL}')
        print(f'   Total Time:    {Fore.CYAN}{res["time"]:.2f}s{Style.RESET_ALL}')
        print(f'   Avg per Gen:   {Fore.YELLOW}{res["avg_per_gen"]:.3f}s{Style.RESET_ALL}')
        
        # Calculate improvement over fixed baseline
        if strategy != 'fixed':
            baseline = results['fixed']['final_best']
            improvement = ((res['final_best'] - baseline) / baseline * 100)
            if improvement > 0:
                print(f'   vs Fixed:      {Fore.GREEN}+{improvement:.1f}%{Style.RESET_ALL} better')
            else:
                print(f'   vs Fixed:      {Fore.RED}{improvement:.1f}%{Style.RESET_ALL} worse')
        
        print()
    
    # Key insights
    print(f'{Fore.CYAN}{Style.BRIGHT}📈 KEY INSIGHTS:{Style.RESET_ALL}\n')
    
    ml_result = results['ml_adaptive']
    simple_result = results['simple_adaptive']
    fixed_result = results['fixed']
    
    print(f'  • Population size: {Fore.YELLOW}{POPULATION_SIZE}{Style.RESET_ALL} agents '
          f'(optimal for GPU)')
    print(f'  • ML strategy collected {Fore.CYAN}{len(ml_result["evo"].predictor.training_data)}{Style.RESET_ALL} '
          f'training samples')
    print(f'  • GPU acceleration: {Fore.GREEN}Enabled{Style.RESET_ALL} for populations ≥100')
    
    if ml_result['final_best'] > simple_result['final_best']:
        ratio = ml_result['final_best'] / simple_result['final_best']
        print(f'  • {Fore.GREEN}{Style.BRIGHT}🤖 ML strategy won!{Style.RESET_ALL} '
              f'{ratio:.2f}× better than simple adaptive')
    else:
        ratio = simple_result['final_best'] / ml_result['final_best']
        print(f'  • {Fore.YELLOW}⭐ Simple adaptive won!{Style.RESET_ALL} '
              f'{ratio:.2f}× better than ML')
        print(f'    (ML may need more training data or different hyperparameters)')
    
    best_vs_worst = max([r['final_best'] for r in results.values()]) / \
                    min([r['final_best'] for r in results.values()])
    print(f'  • Best vs worst strategy: {Fore.CYAN}{best_vs_worst:.1f}×{Style.RESET_ALL} difference')
    
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    
    return results


if __name__ == "__main__":
    print(f'\n{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}🚀⚛️ LARGE-SCALE ML COMPARISON{Style.RESET_ALL}')
    print(f'{Fore.GREEN}{Style.BRIGHT}{"="*80}{Style.RESET_ALL}\n')
    print(f'{Fore.YELLOW}This will run a comprehensive comparison with optimal settings:{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}  • 100 agents (GPU advantage){Style.RESET_ALL}')
    print(f'{Fore.YELLOW}  • 150 generations (ML learning time){Style.RESET_ALL}')
    print(f'{Fore.YELLOW}  • All 3 strategies compared{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}  • Estimated time: 3-5 minutes{Style.RESET_ALL}\n')
    
    try:
        results = run_large_comparison()
        print(f'\n{Fore.GREEN}{Style.BRIGHT}✅ Comparison complete!{Style.RESET_ALL}')
        print(f'{Fore.CYAN}Check the generated PNG for detailed visualizations.{Style.RESET_ALL}\n')
        
    except KeyboardInterrupt:
        print(f'\n\n{Fore.YELLOW}⚠️  Stopped by user.{Style.RESET_ALL}\n')
    except Exception as e:
        print(f'\n\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}\n')
        raise
