"""
🔬 Deep Analysis - Evolution Dynamics & Convergence

Analyzes evolution history, convergence patterns, and ML prediction accuracy.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


def load_evolution_results():
    """Load all available evolution result files."""
    results_dir = Path(__file__).parent
    
    results = {
        'hybrid': None,
        'ultra_scale': None,
        'multi_env': None
    }
    
    # Find most recent files
    for file in results_dir.glob("*.json"):
        if 'hybrid_evolution_results' in file.name:
            with open(file, 'r') as f:
                results['hybrid'] = json.load(f)
                print(f"✅ Loaded: {file.name}")
        elif 'ultra_scale_ml_evolution' in file.name:
            with open(file, 'r') as f:
                results['ultra_scale'] = json.load(f)
                print(f"✅ Loaded: {file.name}")
        elif 'multi_env_ml_evolution' in file.name:
            with open(file, 'r') as f:
                results['multi_env'] = json.load(f)
                print(f"✅ Loaded: {file.name}")
    
    return results


def analyze_convergence_patterns(results_dict, output_dir):
    """Analyze and visualize convergence patterns across all evolution runs."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    colors = {'hybrid': '#3498db', 'ultra_scale': '#e74c3c', 'multi_env': '#2ecc71'}
    labels = {
        'hybrid': 'Hybrid (300 pop, 50 gen)',
        'ultra_scale': 'Ultra-Scale (1000 pop, 200 gen)',
        'multi_env': 'Multi-Env (1000 pop, 200 gen)'
    }
    
    # 1. Best fitness evolution
    ax1 = axes[0, 0]
    for name, results in results_dict.items():
        if results and 'history' in results:
            history = results['history']
            if 'best_fitness' in history:
                ax1.plot(history['generations'], history['best_fitness'],
                        linewidth=2, label=labels[name], color=colors[name])
            elif 'best_overall_fitness' in history:
                ax1.plot(history['generations'], history['best_overall_fitness'],
                        linewidth=2, label=labels[name], color=colors[name])
    
    ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
    ax1.set_title('Convergence: Best Fitness Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Log-scale convergence
    ax2 = axes[0, 1]
    for name, results in results_dict.items():
        if results and 'history' in results:
            history = results['history']
            if 'best_fitness' in history:
                fitness = np.array(history['best_fitness'])
                # Calculate improvement from initial
                initial = fitness[0]
                improvement = (fitness - initial) / initial * 100
                ax2.plot(history['generations'], improvement,
                        linewidth=2, label=labels[name], color=colors[name])
            elif 'best_overall_fitness' in history:
                fitness = np.array(history['best_overall_fitness'])
                initial = fitness[0]
                improvement = (fitness - initial) / initial * 100
                ax2.plot(history['generations'], improvement,
                        linewidth=2, label=labels[name], color=colors[name])
    
    ax2.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement from Initial (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Improvement Over Time', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Diversity evolution
    ax3 = axes[1, 0]
    for name, results in results_dict.items():
        if results and 'history' in results:
            history = results['history']
            if 'diversity' in history:
                ax3.semilogy(history['generations'], history['diversity'],
                           linewidth=2, label=labels[name], color=colors[name])
    
    ax3.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Population Diversity (log scale)', fontsize=12, fontweight='bold')
    ax3.set_title('Population Diversity Over Time', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Time per generation comparison
    ax4 = axes[1, 1]
    
    time_data = []
    labels_list = []
    
    for name, results in results_dict.items():
        if results and 'history' in results:
            history = results['history']
            if 'time_per_generation' in history:
                time_data.append(history['time_per_generation'])
                labels_list.append(labels[name])
    
    if time_data:
        positions = np.arange(len(labels_list))
        bp = ax4.boxplot(time_data, positions=positions, labels=labels_list,
                        patch_artist=True, widths=0.6)
        
        # Color boxes
        for patch, name in zip(bp['boxes'], results_dict.keys()):
            if results_dict[name]:
                patch.set_facecolor(colors[name])
                patch.set_alpha(0.7)
        
        ax4.set_ylabel('Time per Generation (s)', fontsize=12, fontweight='bold')
        ax4.set_title('Time per Generation Distribution', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
    print(f"📊 Saved: convergence_analysis.png")
    plt.close()


def analyze_ml_efficiency(results_dict, output_dir):
    """Analyze ML prediction efficiency and speedup factors."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    colors = {'hybrid': '#3498db', 'ultra_scale': '#e74c3c', 'multi_env': '#2ecc71'}
    labels = {
        'hybrid': 'Hybrid',
        'ultra_scale': 'Ultra-Scale',
        'multi_env': 'Multi-Env'
    }
    
    # 1. ML vs Simulation time per generation
    ax1 = axes[0, 0]
    for name, results in results_dict.items():
        if results and 'history' in results:
            history = results['history']
            if 'ml_time' in history and 'sim_time' in history:
                generations = history['generations']
                ax1.plot(generations, history['ml_time'], '--', 
                        label=f'{labels[name]} ML', color=colors[name], alpha=0.7)
                ax1.plot(generations, history['sim_time'], '-', 
                        label=f'{labels[name]} Sim', color=colors[name], linewidth=2)
    
    ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('ML Prediction vs Simulation Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Speedup factors
    ax2 = axes[0, 1]
    
    speedup_data = []
    speedup_labels = []
    
    for name, results in results_dict.items():
        if results and 'performance' in results:
            perf = results['performance']
            if 'speedup_factor' in perf:
                speedup_data.append(perf['speedup_factor'])
                speedup_labels.append(labels[name])
    
    if speedup_data:
        bars = ax2.bar(range(len(speedup_labels)), speedup_data, 
                      color=[colors[k] for k in results_dict.keys() if results_dict[k]],
                      alpha=0.8)
        ax2.set_xticks(range(len(speedup_labels)))
        ax2.set_xticklabels(speedup_labels)
        ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
        ax2.set_title('ML-Guided Evolution Speedup', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}x',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 3. Cumulative time comparison
    ax3 = axes[1, 0]
    for name, results in results_dict.items():
        if results and 'history' in results:
            history = results['history']
            if 'time_per_generation' in history:
                cumulative_time = np.cumsum(history['time_per_generation'])
                ax3.plot(history['generations'], cumulative_time,
                        linewidth=2, label=labels[name], color=colors[name])
    
    ax3.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Computation Time', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Simulations avoided (efficiency)
    ax4 = axes[1, 1]
    
    bar_positions = []
    bar_labels = []
    bar_data = []
    bar_colors = []
    
    for i, (name, results) in enumerate(results_dict.items()):
        if results and 'performance' in results:
            perf = results['performance']
            if 'total_simulations' in perf:
                # Calculate what traditional would have needed
                config = results.get('config', {})
                pop = config.get('population_size', 0)
                gens = config.get('generations', 0)
                
                if pop and gens:
                    traditional_sims = pop * gens
                    actual_sims = perf['total_simulations']
                    avoided = traditional_sims - actual_sims
                    
                    # Add avoided bar
                    bar_positions.append(i * 2)
                    bar_labels.append(f'{labels[name]}\nAvoided')
                    bar_data.append(avoided)
                    bar_colors.append(colors[name])
                    
                    # Add actual bar
                    bar_positions.append(i * 2 + 0.8)
                    bar_labels.append(f'{labels[name]}\nActual')
                    bar_data.append(actual_sims)
                    bar_colors.append(colors[name])
    
    if bar_positions:
        bars = ax4.bar(bar_positions, bar_data, width=0.7, color=bar_colors, alpha=0.8)
        # Make avoided bars lighter
        for i in range(0, len(bars), 2):
            bars[i].set_alpha(0.4)
    
        ax4.set_xticks(bar_positions)
        ax4.set_xticklabels(bar_labels, fontsize=9)
    
    ax4.set_ylabel('Number of Simulations', fontsize=12, fontweight='bold')
    ax4.set_title('Simulations: Avoided vs Actually Run', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "ml_efficiency_analysis.png", dpi=300, bbox_inches='tight')
    print(f"📊 Saved: ml_efficiency_analysis.png")
    plt.close()


def analyze_multi_environment_performance(results, output_dir):
    """Detailed analysis of multi-environment evolution."""
    if not results or 'history' not in results:
        print("⚠️  No multi-environment results available")
        return
    
    history = results['history']
    
    if 'best_fitness_per_env' not in history:
        print("⚠️  No per-environment fitness data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    environments = list(history['best_fitness_per_env'].keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(environments)))
    
    # 1. Best fitness per environment over time
    ax1 = axes[0, 0]
    for env, color in zip(environments, colors):
        ax1.plot(history['generations'], history['best_fitness_per_env'][env],
                linewidth=2, label=env, color=color)
    
    ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
    ax1.set_title('Best Fitness per Environment', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Average fitness per environment over time
    ax2 = axes[0, 1]
    for env, color in zip(environments, colors):
        if env in history.get('avg_fitness_per_env', {}):
            ax2.plot(history['generations'], history['avg_fitness_per_env'][env],
                    linewidth=2, label=env, color=color, alpha=0.7)
    
    ax2.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Fitness', fontsize=12, fontweight='bold')
    ax2.set_title('Average Population Fitness per Environment', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Final fitness comparison across environments
    ax3 = axes[1, 0]
    final_fitness = [history['best_fitness_per_env'][env][-1] for env in environments]
    bars = ax3.bar(range(len(environments)), final_fitness, color=colors, alpha=0.8)
    ax3.set_xticks(range(len(environments)))
    ax3.set_xticklabels(environments, rotation=45, ha='right')
    ax3.set_ylabel('Final Best Fitness', fontsize=12, fontweight='bold')
    ax3.set_title('Final Performance Across Environments', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, final_fitness):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Fitness improvement per environment
    ax4 = axes[1, 1]
    improvements = []
    for env in environments:
        initial = history['best_fitness_per_env'][env][0]
        final = history['best_fitness_per_env'][env][-1]
        improvement = ((final - initial) / initial * 100) if initial > 0 else 0
        improvements.append(improvement)
    
    bars = ax4.bar(range(len(environments)), improvements, color=colors, alpha=0.8)
    ax4.set_xticks(range(len(environments)))
    ax4.set_xticklabels(environments, rotation=45, ha='right')
    ax4.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Fitness Improvement by Environment', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.1f}%', ha='center', 
                va='bottom' if val >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "multi_environment_detailed.png", dpi=300, bbox_inches='tight')
    print(f"📊 Saved: multi_environment_detailed.png")
    plt.close()


def create_comparison_summary(results_dict, output_dir):
    """Create comprehensive comparison summary visualization."""
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Summary data
    summary_data = {}
    for name, results in results_dict.items():
        if results:
            summary_data[name] = {
                'best_fitness': results.get('best_fitness', 0) or results.get('best_overall_fitness', 0),
                'time': results.get('performance', {}).get('total_time', 0),
                'speedup': results.get('performance', {}).get('speedup_factor', 0),
                'simulations': results.get('performance', {}).get('total_simulations', 0),
                'ml_predictions': results.get('performance', {}).get('total_ml_predictions', 0),
                'generations': results.get('config', {}).get('generations', 0),
                'population': results.get('config', {}).get('populations_size', 0)
            }
    
    labels_map = {
        'hybrid': 'Hybrid\n(300 pop, 50 gen)',
        'ultra_scale': 'Ultra-Scale\n(1000 pop, 200 gen)',
        'multi_env': 'Multi-Env\n(1000 pop, 200 gen)'
    }
    
    colors = {'hybrid': '#3498db', 'ultra_scale': '#e74c3c', 'multi_env': '#2ecc71'}
    
    # 1. Best Fitness Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    names = list(summary_data.keys())
    fitness_vals = [summary_data[n]['best_fitness'] for n in names]
    ax1.bar(range(len(names)), fitness_vals, 
           color=[colors[n] for n in names], alpha=0.8)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([labels_map[n] for n in names], fontsize=9)
    ax1.set_ylabel('Best Fitness', fontweight='bold')
    ax1.set_title('Best Fitness Achieved', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(fitness_vals):
        ax1.text(i, val, f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Total Time Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    time_vals = [summary_data[n]['time'] for n in names]
    ax2.bar(range(len(names)), time_vals,
           color=[colors[n] for n in names], alpha=0.8)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([labels_map[n] for n in names], fontsize=9)
    ax2.set_ylabel('Time (seconds)', fontweight='bold')
    ax2.set_title('Total Computation Time', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(time_vals):
        ax2.text(i, val, f'{val:.1f}s', ha='center', va='bottom', fontsize=10)
    
    # 3. Speedup Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    speedup_vals = [summary_data[n]['speedup'] for n in names]
    ax3.bar(range(len(names)), speedup_vals,
           color=[colors[n] for n in names], alpha=0.8)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([labels_map[n] for n in names], fontsize=9)
    ax3.set_ylabel('Speedup Factor', fontweight='bold')
    ax3.set_title('ML-Guided Speedup', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(speedup_vals):
        ax3.text(i, val, f'{val:.1f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4-6. Evolution curves (using full width)
    ax4 = fig.add_subplot(gs[1, :])
    for name, results in results_dict.items():
        if results and 'history' in results:
            history = results['history']
            if 'best_fitness' in history:
                ax4.plot(history['generations'], history['best_fitness'],
                        linewidth=2, label=labels_map[name], color=colors[name])
            elif 'best_overall_fitness' in history:
                ax4.plot(history['generations'], history['best_overall_fitness'],
                        linewidth=2, label=labels_map[name], color=colors[name])
    ax4.set_xlabel('Generation', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Best Fitness', fontweight='bold', fontsize=11)
    ax4.set_title('Fitness Evolution Comparison', fontweight='bold', fontsize=13)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 7. Simulations Comparison
    ax7 = fig.add_subplot(gs[2, 0])
    sim_vals = [summary_data[n]['simulations'] for n in names]
    ax7.bar(range(len(names)), sim_vals,
           color=[colors[n] for n in names], alpha=0.8)
    ax7.set_xticks(range(len(names)))
    ax7.set_xticklabels([labels_map[n] for n in names], fontsize=9)
    ax7.set_ylabel('Simulations', fontweight='bold')
    ax7.set_title('Total Simulations Run', fontweight='bold', fontsize=12)
    ax7.grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(sim_vals):
        ax7.text(i, val, f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    # 8. ML Predictions Comparison
    ax8 = fig.add_subplot(gs[2, 1])
    ml_vals = [summary_data[n]['ml_predictions'] for n in names]
    ax8.bar(range(len(names)), ml_vals,
           color=[colors[n] for n in names], alpha=0.8)
    ax8.set_xticks(range(len(names)))
    ax8.set_xticklabels([labels_map[n] for n in names], fontsize=9)
    ax8.set_ylabel('ML Predictions', fontweight='bold')
    ax8.set_title('Total ML Predictions', fontweight='bold', fontsize=12)
    ax8.grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(ml_vals):
        ax8.text(i, val, f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    # 9. Efficiency (fitness per second)
    ax9 = fig.add_subplot(gs[2, 2])
    efficiency_vals = [summary_data[n]['best_fitness'] / summary_data[n]['time'] 
                      if summary_data[n]['time'] > 0 else 0 for n in names]
    ax9.bar(range(len(names)), efficiency_vals,
           color=[colors[n] for n in names], alpha=0.8)
    ax9.set_xticks(range(len(names)))
    ax9.set_xticklabels([labels_map[n] for n in names], fontsize=9)
    ax9.set_ylabel('Fitness / Second', fontweight='bold')
    ax9.set_title('Efficiency (Fitness per Second)', fontweight='bold', fontsize=12)
    ax9.grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(efficiency_vals):
        ax9.text(i, val, f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.savefig(output_dir / "comprehensive_comparison.png", dpi=300, bbox_inches='tight')
    print(f"📊 Saved: comprehensive_comparison.png")
    plt.close()


def main():
    """Run evolution dynamics analysis."""
    print("\n" + "="*70)
    print("🔬 DEEP ANALYSIS - EVOLUTION DYNAMICS & CONVERGENCE")
    print("="*70)
    
    output_dir = Path(__file__).parent / "deep_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    print("\n📂 Loading evolution results...")
    results_dict = load_evolution_results()
    
    # Check what we have
    available = [k for k, v in results_dict.items() if v is not None]
    print(f"\n✅ Available results: {', '.join(available)}")
    
    if not available:
        print("❌ No evolution results found!")
        return
    
    # Run analyses
    print("\n" + "="*70)
    print("📊 Analyzing Convergence Patterns...")
    print("="*70)
    analyze_convergence_patterns(results_dict, output_dir)
    
    print("\n" + "="*70)
    print("📊 Analyzing ML Efficiency...")
    print("="*70)
    analyze_ml_efficiency(results_dict, output_dir)
    
    if results_dict['multi_env']:
        print("\n" + "="*70)
        print("📊 Analyzing Multi-Environment Performance...")
        print("="*70)
        analyze_multi_environment_performance(results_dict['multi_env'], output_dir)
    
    print("\n" + "="*70)
    print("📊 Creating Comprehensive Comparison...")
    print("="*70)
    create_comparison_summary(results_dict, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ EVOLUTION DYNAMICS ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📊 Generated files:")
    print(f"   1. convergence_analysis.png")
    print(f"   2. ml_efficiency_analysis.png")
    if results_dict['multi_env']:
        print(f"   3. multi_environment_detailed.png")
    print(f"   4. comprehensive_comparison.png")
    print("\n✨ Analysis complete!")


if __name__ == "__main__":
    main()
