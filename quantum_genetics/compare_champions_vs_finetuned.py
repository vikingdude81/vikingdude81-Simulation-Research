"""
🏆 HEAD-TO-HEAD COMPARISON: Original Champions vs Progressive Fine-Tuned Genomes
===============================================================================

This script evaluates both sets of genomes using identical conditions to determine
which actually performs better and identify where the fitness discrepancies come from.

Comparison includes:
- Direct fitness evaluation (same environment, same timesteps)
- Detailed trait evolution tracking
- Stability and longevity analysis
- Statistical significance testing
- Visualization of differences
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from quantum_genetic_agents import QuantumAgent
import json
from datetime import datetime
from scipy import stats

class GenomeComparator:
    """Compare two sets of genomes head-to-head"""
    
    def __init__(self):
        # Original champions (from mega-long 1000-gen evolution)
        self.original_champions = {
            'gentle': {
                'genome': np.array([2.7668, 0.1853, 0.0050, 0.6798]),
                'reported_fitness': 0.013622,
                'description': 'Best overall from mega-long evolution'
            },
            'standard': {
                'genome': np.array([2.9460, 0.1269, 0.0050, 0.2996]),
                'reported_fitness': 0.009678,
                'description': 'Standard environment specialist'
            },
            'chaotic': {
                'genome': np.array([3.0000, 0.5045, 0.0050, 0.4108]),
                'reported_fitness': 0.008864,
                'description': 'Best generalist'
            },
            'oscillating': {
                'genome': np.array([3.0000, 1.8126, 0.0050, 0.0000]),
                'reported_fitness': 0.008222,
                'description': 'Oscillating environment specialist'
            },
            'harsh': {
                'genome': np.array([3.0000, 2.0000, 0.0050, 0.5713]),
                'reported_fitness': 0.004644,
                'description': 'Harsh environment specialist'
            }
        }
        
        # Progressive fine-tuned genomes (from recent 36-min run)
        self.finetuned_genomes = {
            'gentle': {
                'genome': np.array([3.0000, 0.0100, 0.0010, 0.9238]),
                'reported_fitness': 0.16154188,
                'description': 'Fine-tuned with d=0.001, ω=0.01'
            },
            'standard': {
                'genome': np.array([2.9460, 0.1269, 0.0050, 0.2996]),  # Same as original!
                'reported_fitness': 0.16154188,
                'description': 'Unchanged from original'
            },
            'chaotic': {
                'genome': np.array([2.9441, 0.0100, 0.0010, 0.0295]),
                'reported_fitness': 0.09826174,
                'description': 'Fine-tuned with d=0.001, ω=0.01'
            },
            'oscillating': {
                'genome': np.array([2.9279, 1.7890, 0.0010, 0.3342]),
                'reported_fitness': 0.04095594,
                'description': 'Fine-tuned with d=0.001'
            }
        }
        
        self.results = {}
        
    def evaluate_genome(self, genome, environment='standard', timesteps=80, trials=10):
        """
        Evaluate genome performance with multiple trials
        
        Returns detailed metrics:
        - fitness (mean, std, min, max)
        - trait evolution (energy, coherence, phase)
        - stability metrics
        - longevity metrics
        """
        trial_results = []
        
        for trial in range(trials):
            agent = QuantumAgent(agent_id=trial, genome=genome, environment=environment)
            
            # Evolve agent
            for t in range(1, timesteps):
                agent.evolve(t)
            
            # Get final fitness
            fitness = agent.get_final_fitness()
            
            # Extract trait evolution
            history = np.array(agent.history)
            energy_evolution = history[:, 0]
            coherence_evolution = history[:, 1]
            phase_evolution = history[:, 2]
            fitness_evolution = history[:, 3]
            
            # Calculate stability
            fitness_std = np.std(fitness_evolution)
            stability = 1.0 / (1.0 + fitness_std)
            
            # Calculate longevity penalty
            coherence_decay = coherence_evolution[0] - coherence_evolution[-1]
            longevity_penalty = np.exp(-coherence_decay * 2)
            
            trial_results.append({
                'fitness': fitness,
                'avg_fitness': np.mean(fitness_evolution),
                'fitness_std': fitness_std,
                'stability': stability,
                'coherence_decay': coherence_decay,
                'longevity_penalty': longevity_penalty,
                'final_coherence': coherence_evolution[-1],
                'energy_range': np.max(energy_evolution) - np.min(energy_evolution),
                'fitness_evolution': fitness_evolution,
                'coherence_evolution': coherence_evolution
            })
        
        # Aggregate results
        return {
            'fitness_mean': np.mean([r['fitness'] for r in trial_results]),
            'fitness_std': np.std([r['fitness'] for r in trial_results]),
            'fitness_min': np.min([r['fitness'] for r in trial_results]),
            'fitness_max': np.max([r['fitness'] for r in trial_results]),
            'avg_fitness_mean': np.mean([r['avg_fitness'] for r in trial_results]),
            'stability_mean': np.mean([r['stability'] for r in trial_results]),
            'coherence_decay_mean': np.mean([r['coherence_decay'] for r in trial_results]),
            'longevity_penalty_mean': np.mean([r['longevity_penalty'] for r in trial_results]),
            'final_coherence_mean': np.mean([r['final_coherence'] for r in trial_results]),
            'trial_results': trial_results
        }
    
    def compare_all(self, environments=['standard', 'harsh', 'gentle', 'chaotic', 'oscillating'], 
                    timesteps=80, trials=10):
        """Compare all genomes across all environments"""
        print(f"\n{'='*80}")
        print(f"🏆 GENOME COMPARISON: Original Champions vs Fine-Tuned")
        print(f"{'='*80}")
        print(f"Evaluation settings:")
        print(f"  Timesteps: {timesteps}")
        print(f"  Trials per genome: {trials}")
        print(f"  Environments: {environments}")
        print(f"{'='*80}\n")
        
        for env in environments:
            print(f"\n{'─'*80}")
            print(f"📊 ENVIRONMENT: {env.upper()}")
            print(f"{'─'*80}\n")
            
            self.results[env] = {
                'original': {},
                'finetuned': {},
                'comparison': {}
            }
            
            # Evaluate original champions
            for name, data in self.original_champions.items():
                print(f"  Evaluating ORIGINAL {name}...", end=' ')
                results = self.evaluate_genome(data['genome'], env, timesteps, trials)
                self.results[env]['original'][name] = results
                print(f"✓ Fitness: {results['fitness_mean']:.8f} ± {results['fitness_std']:.8f}")
            
            # Evaluate fine-tuned genomes
            for name, data in self.finetuned_genomes.items():
                if name in self.finetuned_genomes:
                    print(f"  Evaluating FINETUNED {name}...", end=' ')
                    results = self.evaluate_genome(data['genome'], env, timesteps, trials)
                    self.results[env]['finetuned'][name] = results
                    print(f"✓ Fitness: {results['fitness_mean']:.8f} ± {results['fitness_std']:.8f}")
            
            # Statistical comparison
            print(f"\n  📈 Statistical Comparison ({env}):")
            for name in self.original_champions.keys():
                if name in self.results[env]['finetuned']:
                    orig_fitness = [r['fitness'] for r in self.results[env]['original'][name]['trial_results']]
                    fine_fitness = [r['fitness'] for r in self.results[env]['finetuned'][name]['trial_results']]
                    
                    # T-test
                    t_stat, p_value = stats.ttest_ind(orig_fitness, fine_fitness)
                    
                    orig_mean = self.results[env]['original'][name]['fitness_mean']
                    fine_mean = self.results[env]['finetuned'][name]['fitness_mean']
                    improvement = ((fine_mean - orig_mean) / orig_mean) * 100
                    
                    winner = "FINETUNED" if fine_mean > orig_mean else "ORIGINAL"
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
                    
                    print(f"    {name:12s}: Original={orig_mean:.8f}, Finetuned={fine_mean:.8f}")
                    print(f"                 Δ={improvement:+.2f}%, Winner={winner} {significance} (p={p_value:.4f})")
    
    def visualize_comparison(self):
        """Create comprehensive comparison visualizations"""
        # Use standard environment for main comparison
        env = 'standard'
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Fitness comparison bar chart
        ax1 = fig.add_subplot(gs[0, :2])
        names = list(self.original_champions.keys())
        orig_fitness = [self.results[env]['original'][n]['fitness_mean'] for n in names if n in self.results[env]['original']]
        fine_fitness = [self.results[env]['finetuned'][n]['fitness_mean'] for n in names if n in self.results[env]['finetuned']]
        
        x = np.arange(len(names[:len(orig_fitness)]))
        width = 0.35
        
        ax1.bar(x - width/2, orig_fitness, width, label='Original Champions', alpha=0.8, color='steelblue')
        ax1.bar(x + width/2, fine_fitness, width, label='Fine-Tuned', alpha=0.8, color='coral')
        ax1.set_xlabel('Genome Type')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Comparison: Original vs Fine-Tuned (Standard Environment)', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names[:len(orig_fitness)], rotation=45)
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')
        
        # 2. Reported vs Actual fitness scatter
        ax2 = fig.add_subplot(gs[0, 2:])
        for name in names:
            if name in self.results[env]['original']:
                reported = self.original_champions[name]['reported_fitness']
                actual = self.results[env]['original'][name]['fitness_mean']
                ax2.scatter(reported, actual, s=100, label=f'Original {name}', alpha=0.7)
        
        for name in names:
            if name in self.results[env]['finetuned']:
                reported = self.finetuned_genomes[name]['reported_fitness']
                actual = self.results[env]['finetuned'][name]['fitness_mean']
                ax2.scatter(reported, actual, s=100, marker='^', label=f'Finetuned {name}', alpha=0.7)
        
        # Add diagonal line
        max_val = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Perfect match')
        
        ax2.set_xlabel('Reported Fitness')
        ax2.set_ylabel('Actual Fitness (Re-evaluated)')
        ax2.set_title('Reported vs Actual Fitness', fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(alpha=0.3)
        
        # 3. Stability comparison
        ax3 = fig.add_subplot(gs[1, 0])
        orig_stability = [self.results[env]['original'][n]['stability_mean'] for n in names if n in self.results[env]['original']]
        fine_stability = [self.results[env]['finetuned'][n]['stability_mean'] for n in names if n in self.results[env]['finetuned']]
        
        ax3.bar(x - width/2, orig_stability, width, label='Original', alpha=0.8, color='green')
        ax3.bar(x + width/2, fine_stability, width, label='Fine-Tuned', alpha=0.8, color='orange')
        ax3.set_xlabel('Genome Type')
        ax3.set_ylabel('Stability')
        ax3.set_title('Stability Comparison', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(names[:len(orig_stability)], rotation=45, fontsize=8)
        ax3.legend()
        ax3.grid(alpha=0.3, axis='y')
        
        # 4. Coherence decay comparison
        ax4 = fig.add_subplot(gs[1, 1])
        orig_decay = [self.results[env]['original'][n]['coherence_decay_mean'] for n in names if n in self.results[env]['original']]
        fine_decay = [self.results[env]['finetuned'][n]['coherence_decay_mean'] for n in names if n in self.results[env]['finetuned']]
        
        ax4.bar(x - width/2, orig_decay, width, label='Original', alpha=0.8, color='purple')
        ax4.bar(x + width/2, fine_decay, width, label='Fine-Tuned', alpha=0.8, color='pink')
        ax4.set_xlabel('Genome Type')
        ax4.set_ylabel('Coherence Decay')
        ax4.set_title('Coherence Decay (Lower = Better Longevity)', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(names[:len(orig_decay)], rotation=45, fontsize=8)
        ax4.legend()
        ax4.grid(alpha=0.3, axis='y')
        
        # 5. Longevity penalty comparison
        ax5 = fig.add_subplot(gs[1, 2])
        orig_longevity = [self.results[env]['original'][n]['longevity_penalty_mean'] for n in names if n in self.results[env]['original']]
        fine_longevity = [self.results[env]['finetuned'][n]['longevity_penalty_mean'] for n in names if n in self.results[env]['finetuned']]
        
        ax5.bar(x - width/2, orig_longevity, width, label='Original', alpha=0.8, color='teal')
        ax5.bar(x + width/2, fine_longevity, width, label='Fine-Tuned', alpha=0.8, color='salmon')
        ax5.set_xlabel('Genome Type')
        ax5.set_ylabel('Longevity Penalty')
        ax5.set_title('Longevity Penalty (Higher = Better)', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(names[:len(orig_longevity)], rotation=45, fontsize=8)
        ax5.legend()
        ax5.grid(alpha=0.3, axis='y')
        
        # 6. Genome parameter comparison heatmap
        ax6 = fig.add_subplot(gs[1, 3])
        param_names = ['μ', 'ω', 'd', 'φ']
        
        # Create matrix of parameter differences
        diff_matrix = []
        for name in names[:4]:  # Limit to 4 for clarity
            if name in self.results[env]['finetuned']:
                orig_genome = self.original_champions[name]['genome']
                fine_genome = self.finetuned_genomes[name]['genome']
                diff = ((fine_genome - orig_genome) / orig_genome) * 100  # Percent change
                diff_matrix.append(diff)
        
        if diff_matrix:
            sns.heatmap(diff_matrix, annot=True, fmt='.1f', cmap='RdYlGn_r', center=0,
                       xticklabels=param_names, yticklabels=names[:len(diff_matrix)],
                       cbar_kws={'label': 'Change (%)'}, ax=ax6)
            ax6.set_title('Parameter Changes: Fine-Tuned vs Original', fontweight='bold')
        
        # 7-8. Sample fitness evolution curves
        ax7 = fig.add_subplot(gs[2, :2])
        ax8 = fig.add_subplot(gs[2, 2:])
        
        # Plot gentle genome evolution (best original)
        name = 'gentle'
        if name in self.results[env]['original'] and name in self.results[env]['finetuned']:
            orig_trials = self.results[env]['original'][name]['trial_results']
            fine_trials = self.results[env]['finetuned'][name]['trial_results']
            
            # Plot first 3 trials
            for i in range(min(3, len(orig_trials))):
                ax7.plot(orig_trials[i]['fitness_evolution'], alpha=0.5, color='blue', 
                        label='Original' if i == 0 else '')
                ax8.plot(orig_trials[i]['coherence_evolution'], alpha=0.5, color='blue',
                        label='Original' if i == 0 else '')
            
            for i in range(min(3, len(fine_trials))):
                ax7.plot(fine_trials[i]['fitness_evolution'], alpha=0.5, color='red',
                        label='Fine-Tuned' if i == 0 else '')
                ax8.plot(fine_trials[i]['coherence_evolution'], alpha=0.5, color='red',
                        label='Fine-Tuned' if i == 0 else '')
            
            ax7.set_xlabel('Timestep')
            ax7.set_ylabel('Fitness')
            ax7.set_title(f'Fitness Evolution: {name.upper()} (Sample Trials)', fontweight='bold')
            ax7.legend()
            ax7.grid(alpha=0.3)
            
            ax8.set_xlabel('Timestep')
            ax8.set_ylabel('Coherence')
            ax8.set_title(f'Coherence Evolution: {name.upper()} (Sample Trials)', fontweight='bold')
            ax8.legend()
            ax8.grid(alpha=0.3)
        
        plt.suptitle('COMPREHENSIVE GENOME COMPARISON ANALYSIS', fontsize=16, fontweight='bold', y=0.995)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'genome_comparison_analysis_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved: {filename}")
        
        return filename

def main():
    comparator = GenomeComparator()
    
    # Run comparison
    comparator.compare_all(
        environments=['standard', 'gentle', 'harsh', 'chaotic', 'oscillating'],
        timesteps=80,
        trials=10
    )
    
    # Visualize
    comparator.visualize_comparison()
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'genome_comparison_results_{timestamp}.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(comparator.results), f, indent=2)
    print(f"\n💾 Detailed results saved: {output_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY CONCLUSIONS")
    print(f"{'='*80}\n")
    
    env = 'standard'
    total_orig_wins = 0
    total_fine_wins = 0
    
    for name in comparator.original_champions.keys():
        if name in comparator.results[env]['finetuned']:
            orig_mean = comparator.results[env]['original'][name]['fitness_mean']
            fine_mean = comparator.results[env]['finetuned'][name]['fitness_mean']
            
            if fine_mean > orig_mean:
                winner = "FINE-TUNED"
                total_fine_wins += 1
            else:
                winner = "ORIGINAL"
                total_orig_wins += 1
            
            improvement = ((fine_mean - orig_mean) / orig_mean) * 100
            
            print(f"{name.upper():12s}: {winner} wins by {abs(improvement):.2f}%")
    
    print(f"\n{'─'*80}")
    print(f"OVERALL: Original Champions: {total_orig_wins} | Fine-Tuned: {total_fine_wins}")
    print(f"{'─'*80}\n")

if __name__ == "__main__":
    main()
