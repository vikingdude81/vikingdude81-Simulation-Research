"""
Improve and Deploy Specialist Genomes
=====================================

This script fine-tunes the 5 environment specialists and creates
an intelligent deployment system.

Features:
1. Fine-tune each specialist with additional targeted evolution
2. Create ensemble predictor that selects best specialist
3. Build adaptive switcher that detects environment changes
4. Test real-world deployment scenarios
"""

import numpy as np
import torch
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from adaptive_mutation_evolution import AdaptiveMutationEvolution
from quantum_genetic_agent import QuantumGeneticAgent

class SpecialistImprover:
    """Fine-tune specialist genomes with targeted evolution"""
    
    def __init__(self):
        self.specialists = {
            'standard': np.array([2.9460, 0.1269, 0.0050, 0.2996]),
            'harsh': np.array([3.0000, 2.0000, 0.0050, 0.5713]),
            'gentle': np.array([2.7668, 0.1853, 0.0050, 0.6798]),
            'chaotic': np.array([3.0000, 0.5045, 0.0050, 0.4108]),
            'oscillating': np.array([3.0000, 1.8126, 0.0050, 0.0000])
        }
        self.improved_specialists = {}
        self.improvement_history = {}
        
    def fine_tune_specialist(self, env_name, environment, generations=100, population_size=200):
        """Fine-tune a single specialist with focused evolution"""
        print(f"\n🔧 Fine-tuning {env_name} specialist...")
        print(f"   Starting genome: μ={self.specialists[env_name][0]:.4f}, "
              f"ω={self.specialists[env_name][1]:.4f}, "
              f"d={self.specialists[env_name][2]:.6f}, "
              f"φ={self.specialists[env_name][3]:.4f}")
        
        # Initialize evolution with specialist as seed
        evolution = AdaptiveMutationEvolution(population_size=population_size)
        
        # Replace half the population with variations of the specialist
        specialist_genome = self.specialists[env_name]
        for i in range(population_size // 2):
            # Add small mutations around the specialist
            mutated = specialist_genome + np.random.randn(4) * np.array([0.1, 0.05, 0.001, 0.1])
            mutated[0] = np.clip(mutated[0], 0.1, 3.0)  # μ
            mutated[1] = np.clip(mutated[1], 0.01, 2.0)  # ω
            mutated[2] = np.clip(mutated[2], 0.001, 0.02)  # d
            mutated[3] = np.clip(mutated[3], 0.0, 2*np.pi)  # φ
            
            evolution.population[i] = [
                evolution.population[i][0],  # Keep initial fitness
                mutated,
                evolution.population[i][2]   # Keep ID
            ]
        
        # Evolve with progress tracking
        history = {'generation': [], 'best_fitness': [], 'best_genome': []}
        start_time = time.time()
        
        for gen in range(generations):
            # Adapt mutation rate
            evolution.adapt_mutation_rate()
            
            # Evaluate in target environment
            evolution.evaluate_population(environment=environment)
            
            # Track progress
            best_agent = evolution.population[0]
            history['generation'].append(gen)
            history['best_fitness'].append(best_agent[0])
            history['best_genome'].append(best_agent[1].copy())
            
            # Evolve next generation
            evolution.evolve_generation()
            
            # Progress update every 20 generations
            if (gen + 1) % 20 == 0:
                elapsed = time.time() - start_time
                throughput = (gen + 1) * population_size / elapsed
                print(f"   Gen {gen+1}/{generations} | Best: {best_agent[0]:.6f} | "
                      f"{throughput:.0f} agents/s")
        
        # Get final champion
        final_champion = evolution.population[0]
        improvement = (final_champion[0] / history['best_fitness'][0] - 1) * 100
        
        print(f"   ✅ Improved fitness: {history['best_fitness'][0]:.6f} → "
              f"{final_champion[0]:.6f} (+{improvement:.1f}%)")
        print(f"   Final genome: μ={final_champion[1][0]:.4f}, "
              f"ω={final_champion[1][1]:.4f}, "
              f"d={final_champion[1][2]:.6f}, "
              f"φ={final_champion[1][3]:.4f}")
        
        self.improved_specialists[env_name] = final_champion[1]
        self.improvement_history[env_name] = history
        
        return final_champion[1], history
    
    def fine_tune_all(self, generations=100, population_size=200):
        """Fine-tune all 5 specialists"""
        print("\n" + "="*80)
        print("🚀 FINE-TUNING ALL 5 SPECIALISTS")
        print("="*80)
        
        environments = {
            'standard': 'standard',
            'harsh': 'harsh',
            'gentle': 'gentle',
            'chaotic': 'chaotic',
            'oscillating': 'oscillating'
        }
        
        overall_start = time.time()
        
        for env_name, env_type in environments.items():
            self.fine_tune_specialist(env_name, env_type, generations, population_size)
        
        total_time = time.time() - overall_start
        print(f"\n✅ All specialists fine-tuned in {total_time/60:.1f} minutes")
        
        return self.improved_specialists


class EnsembleDeployer:
    """Intelligent system for deploying specialists in production"""
    
    def __init__(self, specialists):
        self.specialists = specialists
        self.performance_history = {name: [] for name in specialists.keys()}
        self.environment_detector = EnvironmentDetector()
        
    def detect_environment(self, recent_data, window=50):
        """Detect current environment type from recent performance data"""
        return self.environment_detector.classify(recent_data, window)
    
    def select_specialist(self, environment_type=None, recent_data=None):
        """Select best specialist for current conditions"""
        if environment_type is None and recent_data is not None:
            environment_type = self.detect_environment(recent_data)
        
        if environment_type in self.specialists:
            return self.specialists[environment_type], environment_type
        else:
            # Default to chaotic (best generalist)
            return self.specialists['chaotic'], 'chaotic'
    
    def adaptive_deployment(self, test_episodes=100, trials_per_episode=20):
        """Test adaptive specialist switching"""
        print("\n" + "="*80)
        print("🎯 ADAPTIVE DEPLOYMENT TESTING")
        print("="*80)
        
        environments = ['standard', 'harsh', 'gentle', 'chaotic', 'oscillating']
        results = {
            'fixed_specialist': {},
            'adaptive_ensemble': {},
            'best_generalist': {}
        }
        
        print("\nTesting 3 deployment strategies:")
        print("1. Fixed specialist (uses only environment-specific agent)")
        print("2. Adaptive ensemble (switches based on performance)")
        print("3. Best generalist (uses chaotic specialist for all)")
        
        for env_name in environments:
            print(f"\n📊 Testing in {env_name} environment...")
            
            # Strategy 1: Fixed specialist
            specialist_genome = self.specialists[env_name]
            agent = QuantumGeneticAgent(specialist_genome)
            fixed_fitness = []
            for _ in range(trials_per_episode):
                fitness = agent.evaluate_fitness(environment=env_name)
                fixed_fitness.append(fitness)
            results['fixed_specialist'][env_name] = {
                'mean': np.mean(fixed_fitness),
                'std': np.std(fixed_fitness),
                'genome': specialist_genome.tolist()
            }
            print(f"   Fixed specialist: {np.mean(fixed_fitness):.6f} ± {np.std(fixed_fitness):.6f}")
            
            # Strategy 2: Adaptive (test all, use best performer)
            best_fitness_adaptive = -float('inf')
            best_specialist_name = None
            for spec_name, spec_genome in self.specialists.items():
                agent = QuantumGeneticAgent(spec_genome)
                fitness_samples = [agent.evaluate_fitness(environment=env_name) 
                                 for _ in range(10)]
                mean_fitness = np.mean(fitness_samples)
                if mean_fitness > best_fitness_adaptive:
                    best_fitness_adaptive = mean_fitness
                    best_specialist_name = spec_name
            
            # Run full trials with best
            agent = QuantumGeneticAgent(self.specialists[best_specialist_name])
            adaptive_fitness = [agent.evaluate_fitness(environment=env_name) 
                              for _ in range(trials_per_episode)]
            results['adaptive_ensemble'][env_name] = {
                'mean': np.mean(adaptive_fitness),
                'std': np.std(adaptive_fitness),
                'selected': best_specialist_name,
                'genome': self.specialists[best_specialist_name].tolist()
            }
            print(f"   Adaptive (selected {best_specialist_name}): "
                  f"{np.mean(adaptive_fitness):.6f} ± {np.std(adaptive_fitness):.6f}")
            
            # Strategy 3: Best generalist (chaotic)
            agent = QuantumGeneticAgent(self.specialists['chaotic'])
            generalist_fitness = [agent.evaluate_fitness(environment=env_name) 
                                for _ in range(trials_per_episode)]
            results['best_generalist'][env_name] = {
                'mean': np.mean(generalist_fitness),
                'std': np.std(generalist_fitness),
                'genome': self.specialists['chaotic'].tolist()
            }
            print(f"   Best generalist: {np.mean(generalist_fitness):.6f} ± "
                  f"{np.std(generalist_fitness):.6f}")
        
        return results
    
    def visualize_deployment_comparison(self, results, save_path=None):
        """Visualize deployment strategy comparison"""
        environments = list(results['fixed_specialist'].keys())
        strategies = ['fixed_specialist', 'adaptive_ensemble', 'best_generalist']
        strategy_names = ['Fixed Specialist', 'Adaptive Ensemble', 'Best Generalist']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Specialist Deployment Strategy Comparison', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Performance by environment
        ax = axes[0, 0]
        x = np.arange(len(environments))
        width = 0.25
        for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
            means = [results[strategy][env]['mean'] for env in environments]
            ax.bar(x + i*width, means, width, label=name)
        ax.set_xlabel('Environment')
        ax.set_ylabel('Mean Fitness')
        ax.set_title('Performance by Environment')
        ax.set_xticks(x + width)
        ax.set_xticklabels(environments, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Relative performance
        ax = axes[0, 1]
        for strategy, name in zip(strategies, strategy_names):
            means = [results[strategy][env]['mean'] for env in environments]
            ax.plot(environments, means, marker='o', label=name, linewidth=2)
        ax.set_xlabel('Environment')
        ax.set_ylabel('Mean Fitness')
        ax.set_title('Performance Trends')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Adaptive selections
        ax = axes[0, 2]
        if 'adaptive_ensemble' in results:
            selections = [results['adaptive_ensemble'][env].get('selected', env) 
                         for env in environments]
            selection_counts = {}
            for sel in selections:
                selection_counts[sel] = selection_counts.get(sel, 0) + 1
            ax.bar(selection_counts.keys(), selection_counts.values())
            ax.set_xlabel('Selected Specialist')
            ax.set_ylabel('Count')
            ax.set_title('Adaptive Ensemble Selections')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Variance comparison
        ax = axes[1, 0]
        x = np.arange(len(environments))
        for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
            stds = [results[strategy][env]['std'] for env in environments]
            ax.bar(x + i*width, stds, width, label=name)
        ax.set_xlabel('Environment')
        ax.set_ylabel('Std Dev')
        ax.set_title('Performance Stability')
        ax.set_xticks(x + width)
        ax.set_xticklabels(environments, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Win rate
        ax = axes[1, 1]
        win_rates = {name: 0 for name in strategy_names}
        for env in environments:
            best_mean = max(results[strategy][env]['mean'] for strategy in strategies)
            for strategy, name in zip(strategies, strategy_names):
                if results[strategy][env]['mean'] == best_mean:
                    win_rates[name] += 1
        ax.bar(win_rates.keys(), win_rates.values())
        ax.set_ylabel('Environments Won')
        ax.set_title('Strategy Win Rate')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Overall ranking
        ax = axes[1, 2]
        overall_means = []
        for strategy, name in zip(strategies, strategy_names):
            all_means = [results[strategy][env]['mean'] for env in environments]
            overall_means.append(np.mean(all_means))
        colors = ['gold', 'silver', 'chocolate']
        sorted_indices = np.argsort(overall_means)[::-1]
        sorted_names = [strategy_names[i] for i in sorted_indices]
        sorted_means = [overall_means[i] for i in sorted_indices]
        ax.barh(sorted_names, sorted_means, color=colors)
        ax.set_xlabel('Average Fitness')
        ax.set_title('Overall Performance Ranking')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Visualization saved: {save_path}")
        
        return fig


class EnvironmentDetector:
    """Detect environment type from recent performance patterns"""
    
    def __init__(self):
        self.patterns = {
            'standard': {'volatility': 'medium', 'trend': 'neutral'},
            'harsh': {'volatility': 'high', 'trend': 'negative'},
            'gentle': {'volatility': 'low', 'trend': 'positive'},
            'chaotic': {'volatility': 'extreme', 'trend': 'random'},
            'oscillating': {'volatility': 'cyclic', 'trend': 'oscillating'}
        }
    
    def classify(self, data, window=50):
        """Classify environment from recent data"""
        if len(data) < window:
            return 'standard'
        
        recent = data[-window:]
        volatility = np.std(recent)
        trend = np.mean(np.diff(recent))
        
        # Simple classification rules
        if volatility > 2.0:
            return 'chaotic'
        elif volatility < 0.5:
            return 'gentle'
        elif abs(trend) < 0.01:
            return 'standard'
        elif trend < -0.05:
            return 'harsh'
        else:
            return 'oscillating'


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("🔬 SPECIALIST GENOME IMPROVEMENT & DEPLOYMENT")
    print("="*80)
    print("\nOptions:")
    print("1. Fine-tune specialists (improve existing genomes)")
    print("2. Test deployment strategies (compare approaches)")
    print("3. Both (full analysis)")
    
    choice = input("\nEnter choice (1-3) [default: 3]: ").strip() or "3"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load original specialists
    improver = SpecialistImprover()
    
    if choice in ['1', '3']:
        # Fine-tune specialists
        print("\n🎯 Starting fine-tuning process...")
        generations = int(input("Generations per specialist [default: 100]: ").strip() or "100")
        population = int(input("Population size [default: 200]: ").strip() or "200")
        
        improved = improver.fine_tune_all(generations=generations, 
                                          population_size=population)
        
        # Save improved specialists
        improved_data = {
            'timestamp': timestamp,
            'original_specialists': {k: v.tolist() for k, v in improver.specialists.items()},
            'improved_specialists': {k: v.tolist() for k, v in improved.items()},
            'improvement_history': {
                k: {
                    'generations': v['generation'],
                    'best_fitness': v['best_fitness']
                }
                for k, v in improver.improvement_history.items()
            }
        }
        
        with open(f'improved_specialists_{timestamp}.json', 'w') as f:
            json.dump(improved_data, f, indent=2)
        print(f"\n💾 Improved specialists saved: improved_specialists_{timestamp}.json")
        
        specialists_to_use = improved
    else:
        specialists_to_use = improver.specialists
    
    if choice in ['2', '3']:
        # Test deployment strategies
        print("\n🚀 Testing deployment strategies...")
        deployer = EnsembleDeployer(specialists_to_use)
        
        results = deployer.adaptive_deployment(test_episodes=100, trials_per_episode=20)
        
        # Visualize results
        deployer.visualize_deployment_comparison(
            results, 
            save_path=f'deployment_comparison_{timestamp}.png'
        )
        
        # Save results
        with open(f'deployment_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Deployment results saved: deployment_results_{timestamp}.json")
        
        # Print summary
        print("\n" + "="*80)
        print("📊 DEPLOYMENT SUMMARY")
        print("="*80)
        
        for env in results['fixed_specialist'].keys():
            print(f"\n{env.upper()} Environment:")
            for strategy in ['fixed_specialist', 'adaptive_ensemble', 'best_generalist']:
                mean = results[strategy][env]['mean']
                std = results[strategy][env]['std']
                name = strategy.replace('_', ' ').title()
                print(f"  {name:20s}: {mean:10.6f} ± {std:.6f}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
