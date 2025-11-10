"""
Compare Baseline vs Conductor-Enhanced Training Results

Analyzes performance differences between:
1. Baseline specialist (300 gens, fixed mutation=0.1, crossover=0.7)
2. Conductor-enhanced specialist (300 gens, adaptive all parameters)

Shows:
- Convergence speed comparison
- Final performance metrics
- Parameter adaptation traces
- Population dynamics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def load_results():
    """Load baseline and conductor-enhanced results"""
    
    # Find latest baseline volatile specialist
    baseline_files = list(Path('outputs').glob('specialist_volatile_*.json'))
    if not baseline_files:
        print("❌ No baseline volatile specialist results found!")
        return None, None
    
    baseline_file = max(baseline_files, key=lambda p: p.stat().st_mtime)
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    print(f"✓ Loaded baseline: {baseline_file.name}")
    
    # Find latest conductor-enhanced volatile specialist
    conductor_files = list(Path('outputs').glob('conductor_enhanced_volatile_*.json'))
    if not conductor_files:
        print("❌ No conductor-enhanced results found yet!")
        print("   Still training? Check back when training completes.")
        return baseline, None
    
    conductor_file = max(conductor_files, key=lambda p: p.stat().st_mtime)
    with open(conductor_file, 'r') as f:
        conductor = json.load(f)
    
    print(f"✓ Loaded conductor-enhanced: {conductor_file.name}")
    
    return baseline, conductor


def print_performance_comparison(baseline, conductor):
    """Print side-by-side performance metrics"""
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'Baseline':<20} {'Conductor-Enhanced':<20} {'Improvement':<15}")
    print("-"*80)
    
    # Handle different formats
    if 'best_agent' in baseline:
        b_agent = baseline['best_agent']
        b_fitness = b_agent['fitness']
        b_return = b_agent['total_return']
        b_sharpe = b_agent['sharpe_ratio']
        b_dd = b_agent['max_drawdown']
        b_trades = b_agent['num_trades']
        b_winrate = b_agent.get('win_rate', 0)
    else:
        # Baseline format
        b_fitness = baseline['best_fitness']
        b_metrics = baseline['final_metrics']
        b_return = b_metrics['total_return']
        b_sharpe = b_metrics['sharpe_ratio']
        b_dd = b_metrics['max_drawdown']
        b_trades = b_metrics['num_trades']
        b_winrate = b_metrics.get('win_rate', 0)
    
    c_agent = conductor['best_agent']
    c_fitness = c_agent['fitness']
    c_return = c_agent['total_return']
    c_sharpe = c_agent['sharpe_ratio']
    c_dd = c_agent['max_drawdown']
    c_trades = c_agent['num_trades']
    c_winrate = c_agent.get('win_rate', 0)
    
    # Fitness
    fitness_imp = ((c_fitness - b_fitness) / abs(b_fitness)) * 100
    print(f"{'Fitness':<25} {b_fitness:<20.2f} {c_fitness:<20.2f} {fitness_imp:+.1f}%")
    
    # Total Return
    return_imp = c_return - b_return
    print(f"{'Total Return':<25} {b_return:<20.2f}% {c_return:<20.2f}% {return_imp:+.2f}pp")
    
    # Sharpe Ratio
    sharpe_imp = ((c_sharpe - b_sharpe) / abs(b_sharpe)) * 100 if b_sharpe != 0 else 0
    print(f"{'Sharpe Ratio':<25} {b_sharpe:<20.2f} {c_sharpe:<20.2f} {sharpe_imp:+.1f}%")
    
    # Max Drawdown
    dd_imp = ((c_dd - b_dd) / abs(b_dd)) * 100 if b_dd != 0 else 0
    print(f"{'Max Drawdown':<25} {b_dd:<20.2f}% {c_dd:<20.2f}% {dd_imp:+.1f}%")
    
    # Num Trades
    trades_imp = c_trades - b_trades
    print(f"{'Number of Trades':<25} {b_trades:<20} {c_trades:<20} {trades_imp:+d}")
    
    # Win Rate
    wr_imp = c_winrate - b_winrate
    print(f"{'Win Rate':<25} {b_winrate:<20.1f}% {c_winrate:<20.1f}% {wr_imp:+.1f}pp")
    
    print("\n" + "="*80)


def plot_convergence_comparison(baseline, conductor):
    """Plot convergence curves side by side"""
    
    b_history = baseline['training_history']
    c_history = conductor['training_history']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Convergence Comparison: Baseline vs Conductor-Enhanced', fontsize=16, fontweight='bold')
    
    # 1. Best Fitness Over Time
    ax = axes[0, 0]
    ax.plot(b_history['generation'], b_history['best_fitness'], 
            label='Baseline (fixed params)', linewidth=2, alpha=0.8)
    ax.plot(c_history['generation'], c_history['best_fitness'], 
            label='Conductor-Enhanced (adaptive)', linewidth=2, alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Best Fitness Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Average Fitness Over Time
    ax = axes[0, 1]
    ax.plot(b_history['generation'], b_history['avg_fitness'], 
            label='Baseline', linewidth=2, alpha=0.8)
    ax.plot(c_history['generation'], c_history['avg_fitness'], 
            label='Conductor-Enhanced', linewidth=2, alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Fitness')
    ax.set_title('Population Average Fitness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Diversity Over Time
    ax = axes[1, 0]
    ax.plot(b_history['generation'], b_history['diversity'], 
            label='Baseline', linewidth=2, alpha=0.8)
    ax.plot(c_history['generation'], c_history['diversity'], 
            label='Conductor-Enhanced', linewidth=2, alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Genetic Diversity')
    ax.set_title('Population Diversity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Time Since Improvement
    ax = axes[1, 1]
    ax.plot(b_history['generation'], b_history['time_since_improvement'], 
            label='Baseline', linewidth=2, alpha=0.8)
    ax.plot(c_history['generation'], c_history['time_since_improvement'], 
            label='Conductor-Enhanced', linewidth=2, alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Generations Since Improvement')
    ax.set_title('Stagnation Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'outputs/convergence_comparison_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Convergence plots saved to {output_file}")
    
    plt.close()


def plot_parameter_adaptation(conductor):
    """Plot how conductor adapted parameters over time"""
    
    history = conductor['training_history']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('GA Conductor Parameter Adaptation Over Time', fontsize=16, fontweight='bold')
    
    # 1. Mutation Rate
    ax = axes[0, 0]
    ax.plot(history['generation'], history['mutation_rate'], linewidth=2, color='blue')
    ax.axhline(y=0.1, color='red', linestyle='--', label='Baseline Fixed (0.1)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mutation Rate')
    ax.set_title('Adaptive Mutation Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Crossover Rate
    ax = axes[0, 1]
    ax.plot(history['generation'], history['crossover_rate'], linewidth=2, color='green')
    ax.axhline(y=0.7, color='red', linestyle='--', label='Baseline Fixed (0.7)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Crossover Rate')
    ax.set_title('Adaptive Crossover Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Selection Pressure
    ax = axes[1, 0]
    ax.plot(history['generation'], history['selection_pressure'], linewidth=2, color='purple')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Selection Pressure')
    ax.set_title('Adaptive Selection Pressure')
    ax.grid(True, alpha=0.3)
    
    # 4. Population Size
    ax = axes[1, 1]
    ax.plot(history['generation'], history['population_size'], linewidth=2, color='orange')
    ax.axhline(y=200, color='red', linestyle='--', label='Baseline Fixed (200)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Population Size')
    ax.set_title('Dynamic Population Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Diversity Injection
    ax = axes[2, 0]
    ax.plot(history['generation'], history['diversity_injection'], linewidth=2, color='cyan')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity Injection Rate')
    ax.set_title('Diversity Injection Events')
    ax.grid(True, alpha=0.3)
    
    # 6. Elite Preservation
    ax = axes[2, 1]
    ax.plot(history['generation'], history['elite_preservation'], linewidth=2, color='magenta')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Elite Preservation Rate')
    ax.set_title('Elite Preservation Strategy')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'outputs/parameter_adaptation_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Parameter adaptation plots saved to {output_file}")
    
    plt.close()


def plot_population_dynamics(conductor):
    """Plot population dynamics unique to conductor"""
    
    history = conductor['training_history']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Population Dynamics (Conductor-Enhanced)', fontsize=16, fontweight='bold')
    
    # 1. Age Distribution
    ax = axes[0, 0]
    ax.plot(history['generation'], history['avg_age'], label='Average Age', linewidth=2)
    ax.plot(history['generation'], history['oldest_age'], label='Oldest Agent', linewidth=2, alpha=0.6)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Age (generations)')
    ax.set_title('Agent Age Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Wealth Distribution (Gini coefficient)
    ax = axes[0, 1]
    ax.plot(history['generation'], history['wealth_gini'], linewidth=2, color='gold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('Wealth Inequality (Fitness Distribution)')
    ax.grid(True, alpha=0.3)
    
    # 3. Immigration Events
    ax = axes[1, 0]
    ax.plot(history['generation'], history['immigration'], linewidth=2, color='green')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Immigration Count')
    ax.set_title('Diversity Injection via Immigration')
    ax.grid(True, alpha=0.3)
    
    # 4. Crisis Management
    ax = axes[1, 1]
    ax.plot(history['generation'], history['extinction_trigger'], 
            label='Extinction Trigger', linewidth=2, color='red')
    ax.plot(history['generation'], history['restart_signal'], 
            label='Restart Signal', linewidth=2, color='orange', alpha=0.6)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Signal Strength')
    ax.set_title('Crisis Management Signals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'outputs/population_dynamics_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Population dynamics plots saved to {output_file}")
    
    plt.close()


def analyze_convergence_speed(baseline, conductor):
    """Analyze which approach converged faster"""
    
    print("\n" + "="*80)
    print("CONVERGENCE SPEED ANALYSIS")
    print("="*80)
    
    b_history = baseline['training_history']
    c_history = conductor['training_history']
    
    # Find when each reached 90% of final fitness
    b_final = b_history['best_fitness'][-1]
    c_final = c_history['best_fitness'][-1]
    
    b_target = b_final * 0.9
    c_target = c_final * 0.9
    
    b_gen_90 = next((i for i, f in enumerate(b_history['best_fitness']) if f >= b_target), len(b_history['generation']))
    c_gen_90 = next((i for i, f in enumerate(c_history['best_fitness']) if f >= c_target), len(c_history['generation']))
    
    print(f"\nGenerations to reach 90% of final fitness:")
    print(f"  Baseline:            {b_gen_90} generations")
    print(f"  Conductor-Enhanced:  {c_gen_90} generations")
    print(f"  Speedup:             {(b_gen_90 - c_gen_90) / b_gen_90 * 100:+.1f}%")
    
    # Average improvement per generation
    b_improvements = [b_history['best_fitness'][i+1] - b_history['best_fitness'][i] 
                      for i in range(len(b_history['best_fitness'])-1)]
    c_improvements = [c_history['best_fitness'][i+1] - c_history['best_fitness'][i] 
                      for i in range(len(c_history['best_fitness'])-1)]
    
    b_avg_improve = np.mean([x for x in b_improvements if x > 0])
    c_avg_improve = np.mean([x for x in c_improvements if x > 0])
    
    print(f"\nAverage improvement per generation (when improving):")
    print(f"  Baseline:            {b_avg_improve:.4f}")
    print(f"  Conductor-Enhanced:  {c_avg_improve:.4f}")
    print(f"  Difference:          {(c_avg_improve - b_avg_improve) / b_avg_improve * 100:+.1f}%")
    
    # Stagnation episodes
    b_stagnation = max(b_history['time_since_improvement'])
    c_stagnation = max(c_history['time_since_improvement'])
    
    print(f"\nMaximum stagnation period:")
    print(f"  Baseline:            {b_stagnation} generations")
    print(f"  Conductor-Enhanced:  {c_stagnation} generations")
    
    print("\n" + "="*80)


def main():
    """Main comparison analysis"""
    
    print("\n" + "="*80)
    print("📊 BASELINE vs CONDUCTOR-ENHANCED COMPARISON")
    print("="*80)
    
    baseline, conductor = load_results()
    
    if baseline is None:
        print("❌ Cannot proceed without baseline results!")
        return
    
    if conductor is None:
        print("⏳ Conductor-enhanced training not complete yet.")
        print("   Run this script again when training finishes!")
        return
    
    # Performance comparison
    print_performance_comparison(baseline, conductor)
    
    # Convergence speed analysis
    analyze_convergence_speed(baseline, conductor)
    
    # Generate plots
    print("\n📈 Generating comparison plots...")
    plot_convergence_comparison(baseline, conductor)
    plot_parameter_adaptation(conductor)
    plot_population_dynamics(conductor)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nAll plots saved to outputs/ directory.")
    print("Review the convergence comparison to see if conductor-enhanced")
    print("training converged faster and achieved better final fitness!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
