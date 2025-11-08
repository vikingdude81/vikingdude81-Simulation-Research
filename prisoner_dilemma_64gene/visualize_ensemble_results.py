"""
Visualize Multi-Quantum Ensemble Results
Shows performance comparison and strategy switching behavior
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results():
    """Load the latest ensemble test results"""
    results_dir = Path("outputs/god_ai")
    result_files = list(results_dir.glob("multi_quantum_ensemble_*.json"))
    
    if not result_files:
        raise FileNotFoundError("No ensemble results found!")
    
    # Get most recent file
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"📊 Loading results from: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def create_performance_comparison(results):
    """Create bar chart comparing all strategies"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate scores by strategy
    strategy_scores = {}
    for test in results['results']:
        strategy = test['strategy']
        score = test['total_score']
        if strategy not in strategy_scores:
            strategy_scores[strategy] = []
        strategy_scores[strategy].append(score)
    
    # Strategy scores
    strategies = []
    scores = []
    colors = []
    
    # Ensemble strategies
    for strategy in ['phase_based', 'adaptive']:
        if strategy in strategy_scores:
            strategies.append(strategy.replace('_', '\n').title())
            scores.append(sum(strategy_scores[strategy]))
            colors.append('#2ecc71' if 'phase' in strategy else '#3498db')
    
    # Main comparison
    bars = ax1.bar(strategies, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Total Score', fontsize=12, fontweight='bold')
    ax1.set_title('🏆 Multi-Quantum Ensemble Performance', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')
    
    # Performance improvement percentages
    phase_based_score = sum(strategy_scores.get('phase_based', [0]))
    adaptive_score = sum(strategy_scores.get('adaptive', [0]))
    
    # For comparison, use average scores
    avg_phase = np.mean(strategy_scores.get('phase_based', [0]))
    avg_adaptive = np.mean(strategy_scores.get('adaptive', [0]))
    
    improvements = {
        'Phase-Based\nvs Adaptive': ((phase_based_score / adaptive_score) - 1) * 100 if adaptive_score > 0 else 0,
        'Total Improvement': ((phase_based_score / avg_phase) - 1) * 100 if avg_phase > 0 else 0,
    }
    
    comparison_names = list(improvements.keys())
    improvement_values = list(improvements.values())
    improvement_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in improvement_values]
    
    bars2 = ax2.bar(comparison_names, improvement_values, color=improvement_colors, 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('📈 Performance Improvements', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = Path("outputs/god_ai/ensemble_performance_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved performance comparison: {output_path}")
    
    return fig

def create_specialist_performance(results):
    """Show individual specialist genome performance"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    specialists = results['ensemble_config']
    
    for idx, specialist in enumerate(specialists):
        ax = axes[idx // 2, idx % 2]
        
        name = specialist['name'].replace('_', ' ')
        history = specialist['performance_history']
        
        if not history:
            ax.text(0.5, 0.5, f'{name}\n(No data)', 
                   ha='center', va='center', fontsize=14)
            ax.set_title(name, fontweight='bold')
            continue
        
        # Plot performance over runs
        runs = range(1, len(history) + 1)
        ax.plot(runs, history, marker='o', linewidth=2, markersize=8, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(runs, history, 1)
        p = np.poly1d(z)
        ax.plot(runs, p(runs), "--", alpha=0.5, linewidth=2, color='red', 
               label=f'Trend ({"↗" if z[0] > 0 else "↘"})')
        
        # Stats
        avg_score = np.mean(history)
        std_score = np.std(history)
        
        ax.axhline(y=avg_score, color='green', linestyle=':', linewidth=2, 
                  label=f'Avg: {avg_score:,.0f}')
        ax.fill_between(runs, avg_score - std_score, avg_score + std_score, 
                        alpha=0.2, color='green')
        
        ax.set_xlabel('Run Number', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_title(f'🎯 {name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add phase info
        phase = specialist.get('optimal_phase', 'unknown')
        ax.text(0.02, 0.98, f'Optimal Phase: {phase.upper()}',
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9, fontweight='bold')
    
    plt.suptitle('🧬 Individual Specialist Performance', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save
    output_path = Path("outputs/god_ai/specialist_performance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved specialist performance: {output_path}")
    
    return fig

def create_genome_comparison(results):
    """Visualize genome parameter differences across specialists"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    specialists = results['ensemble_config']
    param_names = [
        'Threshold',
        'Tax Rate',
        'Welfare $',
        'Stimulus $',
        'Cooperation\nWeight',
        'Wealth\nWeight',
        'Diversity\nWeight',
        'Cooldown'
    ]
    
    # Normalize genomes for visualization
    all_genomes = []
    names = []
    for specialist in specialists:
        all_genomes.append(specialist['genome'])
        names.append(specialist['name'].replace('_', '\n'))
    
    # Create heatmap data
    genome_array = np.array(all_genomes)
    
    # Log scale for large values
    genome_array_scaled = np.log10(genome_array + 1e-6)
    
    im = ax.imshow(genome_array_scaled, cmap='viridis', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(param_names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(param_names, fontsize=10, fontweight='bold')
    ax.set_yticklabels(names, fontsize=10, fontweight='bold')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(value)', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations with actual values
    for i in range(len(names)):
        for j in range(len(param_names)):
            value = genome_array[i, j]
            if value < 0.01:
                text = f'{value:.4f}'
            elif value < 1:
                text = f'{value:.3f}'
            elif value < 10:
                text = f'{value:.2f}'
            else:
                text = f'{value:.1f}'
            
            ax.text(j, i, text,
                   ha="center", va="center", color="white",
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    ax.set_title('🧬 Genome Parameter Comparison Across Specialists', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_path = Path("outputs/god_ai/genome_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved genome comparison: {output_path}")
    
    return fig

def print_summary(results):
    """Print text summary of results"""
    print("\n" + "="*80)
    print("🏆 MULTI-QUANTUM ENSEMBLE RESULTS SUMMARY")
    print("="*80)
    
    # Calculate scores by strategy
    strategy_scores = {}
    for test in results['results']:
        strategy = test['strategy']
        score = test['total_score']
        if strategy not in strategy_scores:
            strategy_scores[strategy] = []
        strategy_scores[strategy].append(score)
    
    # Total scores
    phase_score = sum(strategy_scores.get('phase_based', [0]))
    adaptive_score = sum(strategy_scores.get('adaptive', [0]))
    
    winner = "Phase-Based" if phase_score > adaptive_score else "Adaptive"
    winner_score = max(phase_score, adaptive_score)
    
    print(f"\n🥇 CHAMPION: {winner}")
    print(f"   Total Score: {winner_score:,}")
    
    print(f"\n📊 Strategy Totals:")
    print(f"   Phase-Based: {phase_score:,}")
    print(f"   Adaptive:    {adaptive_score:,}")
    
    # Average per run
    if strategy_scores.get('phase_based'):
        avg_phase = np.mean(strategy_scores['phase_based'])
        print(f"\n� Average per Run:")
        print(f"   Phase-Based: {avg_phase:,.0f}")
    if strategy_scores.get('adaptive'):
        avg_adaptive = np.mean(strategy_scores['adaptive'])
        print(f"   Adaptive:    {avg_adaptive:,.0f}")
    
    # Specialist stats
    print(f"\n🧬 Specialist Summary:")
    for specialist in results['ensemble_config']:
        name = specialist['name']
        history = specialist.get('performance_history', [])
        if history:
            avg = np.mean(history)
            print(f"   {name:25s}: {len(history):2d} runs, avg {avg:,.0f}")
        else:
            print(f"   {name:25s}: No runs")
    
    print("\n" + "="*80)

def main():
    print("🎨 Multi-Quantum Ensemble Results Visualization")
    print("=" * 60)
    
    # Load results
    results = load_results()
    
    # Print summary
    print_summary(results)
    
    print("\n📊 Creating visualizations...")
    
    # Create all visualizations
    create_performance_comparison(results)
    create_specialist_performance(results)
    create_genome_comparison(results)
    
    print("\n✅ All visualizations complete!")
    print("📁 Check outputs/god_ai/ folder for PNG files")

if __name__ == "__main__":
    main()
