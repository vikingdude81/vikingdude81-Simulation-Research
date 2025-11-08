"""
Progressive Fine-Tuning with Diminishing Returns Analysis
===========================================================

Fine-tune specialists with progressively longer runs to find
the optimal training duration where returns diminish.

Strategy:
1. Start with 50 generations
2. Double each round: 50, 100, 200, 400, 800
3. Track improvement rate per generation
4. Identify where returns slow down
5. Generate detailed analysis and recommendations
"""

import numpy as np
import torch
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from adaptive_mutation_gpu_ml import AdaptiveMutationEvolution
from quantum_genetic_agents import QuantumAgent

class ProgressiveFineTuner:
    """Progressive fine-tuning with diminishing returns analysis"""
    
    def __init__(self):
        self.specialists = {
            'standard': np.array([2.9460, 0.1269, 0.0050, 0.2996]),
            'harsh': np.array([3.0000, 2.0000, 0.0050, 0.5713]),
            'gentle': np.array([2.7668, 0.1853, 0.0050, 0.6798]),
            'chaotic': np.array([3.0000, 0.5045, 0.0050, 0.4108]),
            'oscillating': np.array([3.0000, 1.8126, 0.0050, 0.0000])
        }
        self.tuning_rounds = [50, 100, 200, 400, 800]  # Progressive generation counts
        self.results = {}
        
    def fine_tune_progressive(self, env_name, environment, population_size=300):
        """Fine-tune with progressively longer runs"""
        print(f"\n{'='*80}")
        print(f"🎯 PROGRESSIVE FINE-TUNING: {env_name.upper()}")
        print(f"{'='*80}")
        print(f"Rounds: {self.tuning_rounds}")
        print(f"Population: {population_size}")
        print(f"Starting genome: {self.specialists[env_name]}")
        
        # Initialize evolution
        evolution = AdaptiveMutationEvolution(population_size=population_size)
        
        # Seed population with specialist variations (overwrite ALL genomes)
        specialist_genome = self.specialists[env_name]
        for i in range(population_size):
            if i < population_size // 2:
                # First half: variations of specialist
                mutated = specialist_genome + np.random.randn(4) * np.array([0.15, 0.08, 0.0015, 0.15])
            else:
                # Second half: more random variations
                mutated = specialist_genome + np.random.randn(4) * np.array([0.3, 0.15, 0.003, 0.3])
            
            # Clip to valid ranges [μ, ω, d, φ]
            mutated[0] = np.clip(mutated[0], 0.1, 3.0)
            mutated[1] = np.clip(mutated[1], 0.01, 2.0)
            mutated[2] = np.clip(mutated[2], 0.001, 0.02)
            mutated[3] = np.clip(mutated[3], 0.0, 2*np.pi)
            evolution.population[i] = [evolution.population[i][0], mutated, evolution.population[i][2]]
        
        # Track cumulative progress
        round_results = []
        cumulative_gens = 0
        cumulative_time = 0
        best_ever_fitness = -float('inf')
        best_ever_genome = None
        
        for round_idx, gens in enumerate(self.tuning_rounds):
            print(f"\n{'─'*80}")
            print(f"📊 ROUND {round_idx + 1}: {gens} generations")
            print(f"{'─'*80}")
            
            round_start = time.time()
            history = {
                'generation': [],
                'best_fitness': [],
                'mean_fitness': [],
                'best_genome': [],
                'improvement_rate': []
            }
            
            # Run this round
            for gen in range(gens):
                evolution.adapt_mutation_rate(cumulative_gens + gen)
                evolution.evaluate_population(environment=environment)
                
                best_agent = evolution.population[0]
                mean_fitness = np.mean([agent[0] for agent in evolution.population])
                
                history['generation'].append(cumulative_gens + gen)
                history['best_fitness'].append(best_agent[0])
                history['mean_fitness'].append(mean_fitness)
                history['best_genome'].append(best_agent[1].copy())
                
                # Calculate improvement rate (fitness gain per generation)
                if len(history['best_fitness']) > 10:
                    recent_improvement = (history['best_fitness'][-1] - 
                                        history['best_fitness'][-11]) / 10
                    history['improvement_rate'].append(recent_improvement)
                else:
                    history['improvement_rate'].append(0)
                
                evolution.evolve_generation()
                
                # Progress updates every 50 gens
                if (gen + 1) % 50 == 0 or gen == gens - 1:
                    elapsed = time.time() - round_start
                    throughput = (gen + 1) * population_size / elapsed
                    improvement_rate = history['improvement_rate'][-1] if history['improvement_rate'] else 0
                    print(f"   Gen {cumulative_gens + gen + 1:4d} | "
                          f"Best: {best_agent[0]:.8f} | "
                          f"Mean: {mean_fitness:.8f} | "
                          f"Δ/gen: {improvement_rate:+.2e} | "
                          f"{throughput:.0f} agents/s")
            
            round_time = time.time() - round_start
            cumulative_gens += gens
            cumulative_time += round_time
            
            # Round summary
            final_best = evolution.population[0]
            if final_best[0] > best_ever_fitness:
                best_ever_fitness = final_best[0]
                best_ever_genome = final_best[1].copy()
            
            round_improvement = history['best_fitness'][-1] - history['best_fitness'][0]
            improvement_per_gen = round_improvement / gens
            
            # Calculate diminishing returns metrics
            if round_idx > 0:
                prev_improvement_per_gen = round_results[-1]['improvement_per_gen']
                diminishing_rate = (improvement_per_gen / prev_improvement_per_gen - 1) * 100
            else:
                diminishing_rate = 0
            
            round_data = {
                'round': round_idx + 1,
                'generations': gens,
                'cumulative_generations': cumulative_gens,
                'time_seconds': round_time,
                'cumulative_time': cumulative_time,
                'start_fitness': history['best_fitness'][0],
                'end_fitness': history['best_fitness'][-1],
                'improvement': round_improvement,
                'improvement_per_gen': improvement_per_gen,
                'diminishing_rate': diminishing_rate,
                'final_genome': final_best[1].tolist(),
                'history': history,
                'throughput': gens * population_size / round_time
            }
            
            round_results.append(round_data)
            
            print(f"\n   ✅ Round {round_idx + 1} Complete:")
            print(f"      Fitness: {history['best_fitness'][0]:.8f} → {history['best_fitness'][-1]:.8f}")
            print(f"      Improvement: +{round_improvement:.8f} ({improvement_per_gen:.2e}/gen)")
            print(f"      Time: {round_time/60:.2f} minutes ({round_data['throughput']:.0f} agents/s)")
            if round_idx > 0:
                print(f"      Diminishing rate: {diminishing_rate:+.1f}%")
        
        # Final analysis
        print(f"\n{'='*80}")
        print(f"📈 PROGRESSIVE TUNING COMPLETE: {env_name.upper()}")
        print(f"{'='*80}")
        print(f"Total generations: {cumulative_gens}")
        print(f"Total time: {cumulative_time/60:.2f} minutes")
        print(f"Best ever fitness: {best_ever_fitness:.8f}")
        print(f"Best ever genome: {best_ever_genome}")
        
        # Find optimal stopping point
        optimal_round = self._find_optimal_stopping_point(round_results)
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Optimal stopping point: Round {optimal_round['round']}")
        print(f"   Generations: {optimal_round['cumulative_generations']}")
        print(f"   Reason: {optimal_round['reason']}")
        
        self.results[env_name] = {
            'rounds': round_results,
            'best_fitness': best_ever_fitness,
            'best_genome': best_ever_genome.tolist(),
            'optimal_stopping_point': optimal_round
        }
        
        return round_results, best_ever_genome
    
    def _find_optimal_stopping_point(self, round_results):
        """Identify where diminishing returns make further training inefficient"""
        # Calculate efficiency score for each round
        # Efficiency = improvement per generation / time cost
        
        best_score = -float('inf')
        optimal_round = None
        
        for i, round_data in enumerate(round_results):
            # Skip first round (no baseline)
            if i == 0:
                efficiency = round_data['improvement_per_gen']
                score = efficiency
            else:
                # Calculate return on investment
                improvement_per_gen = round_data['improvement_per_gen']
                prev_improvement = round_results[i-1]['improvement_per_gen']
                
                # Diminishing returns indicator
                diminishing_rate = round_data['diminishing_rate']
                
                # Efficiency: improvement per second
                efficiency = round_data['improvement'] / round_data['time_seconds']
                
                # Score combines efficiency and diminishing rate
                # Penalize if diminishing rate is very negative
                score = efficiency * (1 + min(diminishing_rate/100, 0))
            
            if score > best_score:
                best_score = score
                optimal_round = {
                    'round': round_data['round'],
                    'cumulative_generations': round_data['cumulative_generations'],
                    'fitness': round_data['end_fitness'],
                    'efficiency': efficiency,
                    'reason': self._get_stopping_reason(i, round_results)
                }
        
        return optimal_round
    
    def _get_stopping_reason(self, round_idx, round_results):
        """Generate human-readable reason for optimal stopping point"""
        if round_idx == 0:
            return "First round baseline"
        
        round_data = round_results[round_idx]
        
        if round_data['diminishing_rate'] < -50:
            return f"Severe diminishing returns ({round_data['diminishing_rate']:.1f}%)"
        elif round_data['diminishing_rate'] < -20:
            return f"Moderate diminishing returns ({round_data['diminishing_rate']:.1f}%)"
        elif round_idx == len(round_results) - 1:
            return "Best efficiency maintained through final round"
        else:
            return f"Optimal balance of improvement ({round_data['improvement_per_gen']:.2e}/gen) and time"
    
    def fine_tune_all_progressive(self, population_size=300):
        """Fine-tune all specialists with progressive approach"""
        print("\n" + "="*80)
        print("🚀 PROGRESSIVE FINE-TUNING - ALL SPECIALISTS")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Rounds: {len(self.tuning_rounds)}")
        print(f"  Generations per specialist: {sum(self.tuning_rounds)}")
        print(f"  Total evaluations per specialist: {sum(self.tuning_rounds) * population_size:,}")
        print(f"  Population size: {population_size}")
        
        environments = {
            'standard': 'standard',
            'harsh': 'harsh',
            'gentle': 'gentle',
            'chaotic': 'chaotic',
            'oscillating': 'oscillating'
        }
        
        overall_start = time.time()
        all_results = {}
        
        for env_name, env_type in environments.items():
            round_results, best_genome = self.fine_tune_progressive(
                env_name, env_type, population_size
            )
            all_results[env_name] = {
                'rounds': round_results,
                'best_genome': best_genome.tolist()
            }
        
        total_time = time.time() - overall_start
        
        print(f"\n{'='*80}")
        print(f"✅ ALL SPECIALISTS FINE-TUNED")
        print(f"{'='*80}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Total generations: {sum(self.tuning_rounds) * len(environments):,}")
        print(f"Total evaluations: {sum(self.tuning_rounds) * population_size * len(environments):,}")
        
        return all_results
    
    def visualize_diminishing_returns(self, save_path=None):
        """Create comprehensive visualization of diminishing returns"""
        n_specialists = len(self.results)
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Progressive Fine-Tuning: Diminishing Returns Analysis', 
                     fontsize=18, fontweight='bold')
        
        specialist_names = list(self.results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, n_specialists))
        
        # Plot 1: Fitness evolution for all specialists
        ax1 = fig.add_subplot(gs[0, :])
        for idx, (name, data) in enumerate(self.results.items()):
            all_gens = []
            all_fitness = []
            for round_data in data['rounds']:
                all_gens.extend(round_data['history']['generation'])
                all_fitness.extend(round_data['history']['best_fitness'])
            ax1.plot(all_gens, all_fitness, label=name.capitalize(), 
                    color=colors[idx], linewidth=2, alpha=0.8)
            
            # Mark optimal stopping point
            optimal = data['optimal_stopping_point']
            ax1.axvline(optimal['cumulative_generations'], 
                       color=colors[idx], linestyle='--', alpha=0.3)
        
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Best Fitness', fontsize=12)
        ax1.set_title('Fitness Evolution Across All Rounds', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement per generation by round
        ax2 = fig.add_subplot(gs[1, 0])
        for idx, (name, data) in enumerate(self.results.items()):
            rounds = [r['round'] for r in data['rounds']]
            improvements = [r['improvement_per_gen'] for r in data['rounds']]
            ax2.plot(rounds, improvements, marker='o', label=name.capitalize(),
                    color=colors[idx], linewidth=2, markersize=8)
        ax2.set_xlabel('Round', fontsize=11)
        ax2.set_ylabel('Improvement per Generation', fontsize=11)
        ax2.set_title('Improvement Rate by Round', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Diminishing returns rate
        ax3 = fig.add_subplot(gs[1, 1])
        for idx, (name, data) in enumerate(self.results.items()):
            rounds = [r['round'] for r in data['rounds'][1:]]  # Skip first
            dim_rates = [r['diminishing_rate'] for r in data['rounds'][1:]]
            ax3.plot(rounds, dim_rates, marker='s', label=name.capitalize(),
                    color=colors[idx], linewidth=2, markersize=8)
        ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax3.axhline(-20, color='orange', linestyle='--', linewidth=1, alpha=0.3, 
                   label='Moderate threshold')
        ax3.axhline(-50, color='red', linestyle='--', linewidth=1, alpha=0.3,
                   label='Severe threshold')
        ax3.set_xlabel('Round', fontsize=11)
        ax3.set_ylabel('Diminishing Rate (%)', fontsize=11)
        ax3.set_title('Diminishing Returns Rate', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Time efficiency
        ax4 = fig.add_subplot(gs[1, 2])
        for idx, (name, data) in enumerate(self.results.items()):
            rounds = [r['round'] for r in data['rounds']]
            efficiency = [r['improvement'] / r['time_seconds'] * 1000 for r in data['rounds']]
            ax4.plot(rounds, efficiency, marker='^', label=name.capitalize(),
                    color=colors[idx], linewidth=2, markersize=8)
        ax4.set_xlabel('Round', fontsize=11)
        ax4.set_ylabel('Efficiency (improvement/1000s)', fontsize=11)
        ax4.set_title('Time Efficiency by Round', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5-9: Individual specialist detailed views
        for idx, (name, data) in enumerate(self.results.items()):
            ax = fig.add_subplot(gs[2 + idx//3, idx%3])
            
            # Plot fitness with round boundaries
            all_gens = []
            all_fitness = []
            round_boundaries = [0]
            
            for round_data in data['rounds']:
                all_gens.extend(round_data['history']['generation'])
                all_fitness.extend(round_data['history']['best_fitness'])
                round_boundaries.append(round_data['cumulative_generations'])
            
            ax.plot(all_gens, all_fitness, color=colors[idx], linewidth=2)
            
            # Mark round boundaries
            for boundary in round_boundaries[1:-1]:
                ax.axvline(boundary, color='gray', linestyle=':', alpha=0.5)
            
            # Mark optimal stopping point
            optimal = data['optimal_stopping_point']
            ax.axvline(optimal['cumulative_generations'], 
                      color='red', linestyle='--', linewidth=2, alpha=0.7,
                      label=f"Optimal: {optimal['cumulative_generations']} gens")
            
            ax.set_xlabel('Generation', fontsize=10)
            ax.set_ylabel('Fitness', fontsize=10)
            ax.set_title(f"{name.capitalize()}: Best={data['best_fitness']:.6f}", 
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Plot 10: Optimal stopping points comparison
        ax10 = fig.add_subplot(gs[3, :])
        optimal_gens = [data['optimal_stopping_point']['cumulative_generations'] 
                       for data in self.results.values()]
        optimal_fitness = [data['optimal_stopping_point']['fitness']
                          for data in self.results.values()]
        
        scatter = ax10.scatter(optimal_gens, optimal_fitness, 
                             s=300, c=range(len(specialist_names)), 
                             cmap='tab10', alpha=0.7, edgecolors='black', linewidths=2)
        for i, (name, gens, fit) in enumerate(zip(specialist_names, optimal_gens, optimal_fitness)):
            ax10.annotate(f"{name}\n({gens} gens)", 
                         (gens, fit), fontsize=9, ha='center',
                         xytext=(0, -30), textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', fc=colors[i], alpha=0.3))
        
        ax10.set_xlabel('Optimal Generations', fontsize=12)
        ax10.set_ylabel('Fitness at Optimal Point', fontsize=12)
        ax10.set_title('Optimal Stopping Points: Efficiency vs Performance', 
                      fontsize=14, fontweight='bold')
        ax10.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Visualization saved: {save_path}")
        
        return fig


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("🔬 PROGRESSIVE FINE-TUNING WITH DIMINISHING RETURNS ANALYSIS")
    print("="*80)
    
    print("\nThis will fine-tune specialists progressively:")
    print("  Round 1:   50 generations")
    print("  Round 2:  100 generations")
    print("  Round 3:  200 generations")
    print("  Round 4:  400 generations")
    print("  Round 5:  800 generations")
    print("  ─────────────────────────")
    print("  Total:  1,650 generations per specialist")
    
    population = int(input("\nPopulation size [default: 300]: ").strip() or "300")
    
    print(f"\nEstimated time per specialist: ~30-40 minutes")
    print(f"Total estimated time: ~2.5-3.5 hours")
    print(f"Total evaluations per specialist: {1650 * population:,}")
    
    proceed = input("\nProceed? (y/n) [default: y]: ").strip().lower() or "y"
    if proceed != 'y':
        print("Cancelled.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create fine-tuner
    tuner = ProgressiveFineTuner()
    
    # Run progressive tuning
    results = tuner.fine_tune_all_progressive(population_size=population)
    
    # Save results (convert numpy arrays to lists)
    def convert_to_serializable(obj):
        """Recursively convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    output_file = f'progressive_tuning_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(tuner.results), f, indent=2)
    print(f"\n💾 Results saved: {output_file}")
    
    # Generate visualization
    viz_file = f'progressive_tuning_analysis_{timestamp}.png'
    tuner.visualize_diminishing_returns(save_path=viz_file)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 SUMMARY: OPTIMAL STOPPING POINTS")
    print("="*80)
    
    for name, data in tuner.results.items():
        optimal = data['optimal_stopping_point']
        print(f"\n{name.upper()}:")
        print(f"  Optimal generations: {optimal['cumulative_generations']}")
        print(f"  Fitness at optimal: {optimal['fitness']:.8f}")
        print(f"  Best ever fitness: {data['best_fitness']:.8f}")
        print(f"  Reason: {optimal['reason']}")
    
    print("\n✅ Progressive fine-tuning complete!")
    print(f"\nFiles generated:")
    print(f"  • {output_file}")
    print(f"  • {viz_file}")


if __name__ == "__main__":
    main()
