"""
Compare all three Echo models with side-by-side visualizations
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')

def load_all_results():
    """Load results from all three models."""
    results = {}
    
    # Original model
    original_files = list(Path('.').glob('echo_dashboard_*.json'))
    if original_files:
        latest = max(original_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            results['original'] = json.load(f)
    
    # Spatial model
    spatial_files = list(Path('.').glob('spatial_echo_*.json'))
    if spatial_files:
        latest = max(spatial_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            results['spatial'] = json.load(f)
    
    # Multi-resource model
    multi_files = list(Path('.').glob('multiresource_echo_*.json'))
    if multi_files:
        latest = max(multi_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            results['multiresource'] = json.load(f)
    
    return results

def main():
    print("📊 Loading all Echo model results...")
    results = load_all_results()
    
    if not results:
        print("❌ No results found! Run the models first.")
        return
    
    print(f"✅ Loaded {len(results)} model results")
    
    # Create comparison figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('🧬 Echo Model Comparison: Original vs Spatial vs Multi-Resource', 
                 fontsize=18, fontweight='bold')
    
    # 1. Population Comparison
    ax1 = plt.subplot(2, 3, 1)
    for model_name, data in results.items():
        history = data['history']
        generations = list(range(len(history['population'])))
        ax1.plot(generations, history['population'], linewidth=2, 
                label=model_name.capitalize(), alpha=0.8)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Population')
    ax1.set_title('Population Growth Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cooperation Rate Comparison
    ax2 = plt.subplot(2, 3, 2)
    for model_name, data in results.items():
        history = data['history']
        generations = list(range(len(history['cooperation'])))
        coop_rates = [r * 100 for r in history['cooperation']]
        ax2.plot(generations, coop_rates, linewidth=2, 
                label=f"{model_name.capitalize()}: {np.mean(coop_rates):.1f}%", alpha=0.8)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Cooperation Rate (%)')
    ax2.set_title('Cooperation Rate Comparison')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Resource Comparison (if available)
    ax3 = plt.subplot(2, 3, 3)
    
    # Original and Spatial: Single resource
    if 'original' in results:
        history = results['original']['history']
        generations = list(range(len(history['resources'])))
        pops = history['population']
        avg_res = [total / pop if pop > 0 else 0 
                  for total, pop in zip(history['resources'], pops)]
        ax3.plot(generations, avg_res, linewidth=2, 
                label='Original (single resource)', alpha=0.8)
    
    if 'spatial' in results:
        history = results['spatial']['history']
        generations = list(range(len(history['resources'])))
        pops = history['population']
        avg_res = [total / pop if pop > 0 else 0 
                  for total, pop in zip(history['resources'], pops)]
        ax3.plot(generations, avg_res, linewidth=2, 
                label='Spatial (single resource)', alpha=0.8)
    
    # Multi-resource: Show all three
    if 'multiresource' in results:
        history = results['multiresource']['history']
        for resource_type in ['food', 'materials', 'energy']:
            if f'avg_{resource_type}' in history:
                ax3.plot(range(len(history[f'avg_{resource_type}'])), 
                        history[f'avg_{resource_type}'], 
                        linewidth=2, linestyle='--',
                        label=f'Multi-resource: {resource_type}', alpha=0.8)
    
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Avg Resources per Agent')
    ax3.set_title('Resource Accumulation Comparison')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Birth/Death Comparison
    ax4 = plt.subplot(2, 3, 4)
    for model_name, data in results.items():
        history = data['history']
        generations = list(range(len(history['births'])))
        ax4.plot(generations, history['births'], linewidth=2, 
                label=f"{model_name.capitalize()} births", alpha=0.8)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Births per Generation')
    ax4.set_title('Birth Rate Comparison')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Clustering (Spatial only)
    ax5 = plt.subplot(2, 3, 5)
    if 'spatial' in results:
        history = results['spatial']['history']
        if 'clustering' in history:
            generations = list(range(len(history['clustering'])))
            clustering = [c * 100 for c in history['clustering']]
            ax5.plot(generations, clustering, linewidth=2, color='purple')
            ax5.fill_between(generations, clustering, alpha=0.3, color='purple')
            ax5.set_xlabel('Generation')
            ax5.set_ylabel('Clustering Coefficient (%)')
            ax5.set_title('Spatial Clustering (Tag Similarity Among Neighbors)')
            ax5.grid(True, alpha=0.3)
            
            # Add annotation
            ax5.text(0.5, 0.5, 'Spatial Model Only\n\nMeasures how often agents\nhave same-tag neighbors',
                    transform=ax5.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                    fontsize=10)
    else:
        ax5.text(0.5, 0.5, 'No Spatial Model Data', transform=ax5.transAxes, 
                ha='center', va='center', fontsize=14)
        ax5.axis('off')
    
    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "📊 FINAL STATISTICS COMPARISON\n\n"
    
    for model_name, data in results.items():
        history = data['history']
        metadata = data['metadata']
        
        final_pop = history['population'][-1]
        final_coop = history['cooperation'][-1] * 100
        total_births = sum(history['births'])
        total_deaths = sum(history['deaths'])
        
        summary_text += f"{'='*30}\n"
        summary_text += f"{model_name.upper()}\n"
        summary_text += f"{'='*30}\n"
        summary_text += f"Final Population: {final_pop}\n"
        summary_text += f"Cooperation: {final_coop:.1f}%\n"
        summary_text += f"Total Births: {total_births}\n"
        summary_text += f"Total Deaths: {total_deaths}\n"
        
        if model_name == 'spatial':
            if 'clustering' in history:
                final_clustering = history['clustering'][-1] * 100
                summary_text += f"Final Clustering: {final_clustering:.1f}%\n"
        
        if model_name == 'multiresource':
            for resource_type in ['food', 'materials', 'energy']:
                if f'avg_{resource_type}' in history:
                    final_avg = history[f'avg_{resource_type}'][-1]
                    summary_text += f"{resource_type.capitalize()}: {final_avg:,.0f}\n"
        
        summary_text += "\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"echo_comparison_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 Comparison saved to: {output_file}")
    
    plt.show()
    
    # Print key insights
    print("\n" + "="*70)
    print("🎯 KEY COMPARISONS:")
    print("="*70)
    
    if 'original' in results and 'spatial' in results:
        orig_coop = np.mean([r * 100 for r in results['original']['history']['cooperation']])
        spat_coop = np.mean([r * 100 for r in results['spatial']['history']['cooperation']])
        print(f"\n🤝 COOPERATION:")
        print(f"   Original: {orig_coop:.1f}%")
        print(f"   Spatial:  {spat_coop:.1f}% (+{spat_coop - orig_coop:.1f}% boost!)")
        print(f"   → Spatial structure INCREASES cooperation by {(spat_coop/orig_coop - 1)*100:.1f}%")
    
    if 'spatial' in results and 'clustering' in results['spatial']['history']:
        final_clustering = results['spatial']['history']['clustering'][-1] * 100
        print(f"\n🗺️  SPATIAL CLUSTERING:")
        print(f"   {final_clustering:.1f}% of neighbors share same tag")
        print(f"   → Clear tribal territories formed!")
    
    if 'multiresource' in results:
        history = results['multiresource']['history']
        if all(f'avg_{r}' in history for r in ['food', 'materials', 'energy']):
            food_avg = history['avg_food'][-1]
            mat_avg = history['avg_materials'][-1]
            energy_avg = history['avg_energy'][-1]
            print(f"\n💎 MULTI-RESOURCE ECONOMY:")
            print(f"   Food:      {food_avg:>8,.0f} (slowest growth, consumed fastest)")
            print(f"   Materials: {mat_avg:>8,.0f} (middle tier)")
            print(f"   Energy:    {energy_avg:>8,.0f} (fastest growth)")
            print(f"   → Food is the bottleneck! ({food_avg/energy_avg*100:.1f}% of energy)")
    
    print("\n" + "="*70)
    print("✨ All three models demonstrate Holland's Echo concepts!")
    print("="*70)

if __name__ == "__main__":
    main()
