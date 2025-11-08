"""
Advanced Specialist Ensemble Techniques
========================================

Advanced methods for combining and deploying specialists:
1. Weighted ensemble (blend multiple specialists)
2. Meta-learner (learn when to use which specialist)
3. Hybrid genomes (combine traits from multiple specialists)
4. Dynamic portfolio (maintain multiple agents, switch based on performance)
"""

import numpy as np
import torch
import torch.nn as nn
import json
from datetime import datetime
import matplotlib.pyplot as plt
from quantum_genetic_agent import QuantumGeneticAgent

class WeightedEnsemble:
    """Blend multiple specialists with learned weights"""
    
    def __init__(self, specialists):
        self.specialists = specialists
        self.weights = {name: 1.0/len(specialists) for name in specialists.keys()}
        self.performance_window = 50
        self.performance_history = {name: [] for name in specialists.keys()}
    
    def update_weights(self, recent_performance):
        """Update weights based on recent performance"""
        # Softmax over recent average performance
        performances = {}
        for name in self.specialists.keys():
            if len(self.performance_history[name]) > 0:
                recent = self.performance_history[name][-self.performance_window:]
                performances[name] = np.mean(recent)
            else:
                performances[name] = 0.0
        
        # Softmax to get weights
        values = np.array(list(performances.values()))
        if np.max(values) > 0:
            exp_values = np.exp(values - np.max(values))
            weights_array = exp_values / np.sum(exp_values)
            
            for i, name in enumerate(self.specialists.keys()):
                self.weights[name] = weights_array[i]
    
    def get_blended_genome(self):
        """Create weighted average genome"""
        blended = np.zeros(4)
        for name, genome in self.specialists.items():
            blended += genome * self.weights[name]
        return blended
    
    def record_performance(self, specialist_name, fitness):
        """Record performance for weight updates"""
        self.performance_history[specialist_name].append(fitness)
        
        # Update weights periodically
        if sum(len(h) for h in self.performance_history.values()) % 10 == 0:
            self.update_weights(None)


class MetaLearner(nn.Module):
    """Neural network that learns when to use which specialist"""
    
    def __init__(self, n_specialists=5, state_dim=10):
        super().__init__()
        self.specialist_names = None
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_specialists),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss_history = []
    
    def forward(self, state):
        """Predict specialist weights from environment state"""
        return self.network(state)
    
    def extract_state_features(self, recent_fitnesses, window=10):
        """Extract state features from recent performance"""
        if len(recent_fitnesses) < window:
            # Pad with zeros
            padded = [0.0] * (window - len(recent_fitnesses)) + list(recent_fitnesses)
        else:
            padded = recent_fitnesses[-window:]
        
        features = [
            np.mean(padded),           # Mean fitness
            np.std(padded),            # Volatility
            np.max(padded),            # Peak
            np.min(padded),            # Trough
            padded[-1],                # Current
            np.mean(np.diff(padded)),  # Trend
            np.std(np.diff(padded)),   # Trend volatility
            len([x for x in padded if x > np.mean(padded)]) / window,  # % above mean
            np.percentile(padded, 75) - np.percentile(padded, 25),     # IQR
            np.corrcoef(np.arange(window), padded)[0, 1] if window > 1 else 0  # Autocorr
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def train_episode(self, state, actual_performances):
        """Train on one episode"""
        self.optimizer.zero_grad()
        
        # Predict weights
        predicted_weights = self.forward(state)
        
        # Loss: negative weighted performance
        # (we want to maximize weighted performance)
        performances_tensor = torch.tensor(actual_performances, dtype=torch.float32)
        weighted_performance = torch.sum(predicted_weights * performances_tensor)
        loss = -weighted_performance  # Negative because we're maximizing
        
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        
        return predicted_weights.detach().numpy()


class HybridGenomeCreator:
    """Create hybrid genomes by combining specialist traits"""
    
    def __init__(self, specialists):
        self.specialists = specialists
        self.hybrids = {}
    
    def create_hybrid(self, name1, name2, blend_ratio=0.5):
        """Create hybrid between two specialists"""
        genome1 = self.specialists[name1]
        genome2 = self.specialists[name2]
        
        # Weighted blend
        hybrid = genome1 * blend_ratio + genome2 * (1 - blend_ratio)
        
        # Ensure d stays close to 0.005 (our universal constant)
        hybrid[2] = 0.005
        
        return hybrid
    
    def create_all_hybrids(self):
        """Create all pairwise hybrids"""
        names = list(self.specialists.keys())
        
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                hybrid_name = f"{name1}_{name2}_hybrid"
                self.hybrids[hybrid_name] = self.create_hybrid(name1, name2)
                print(f"Created {hybrid_name}")
        
        return self.hybrids
    
    def create_super_hybrid(self):
        """Create super hybrid from all specialists"""
        # Average all specialists
        all_genomes = np.array(list(self.specialists.values()))
        super_hybrid = np.mean(all_genomes, axis=0)
        super_hybrid[2] = 0.005  # Maintain optimal d
        
        self.hybrids['super_hybrid'] = super_hybrid
        return super_hybrid
    
    def test_hybrids(self, environments, trials_per_env=20):
        """Test all hybrids across environments"""
        print("\n" + "="*80)
        print("🧬 TESTING HYBRID GENOMES")
        print("="*80)
        
        results = {}
        
        # Test each hybrid
        for hybrid_name, hybrid_genome in self.hybrids.items():
            print(f"\nTesting {hybrid_name}...")
            agent = QuantumGeneticAgent(hybrid_genome)
            
            results[hybrid_name] = {}
            for env in environments:
                fitnesses = [agent.evaluate_fitness(environment=env) 
                           for _ in range(trials_per_env)]
                results[hybrid_name][env] = {
                    'mean': np.mean(fitnesses),
                    'std': np.std(fitnesses),
                    'genome': hybrid_genome.tolist()
                }
                print(f"  {env}: {np.mean(fitnesses):.6f} ± {np.std(fitnesses):.6f}")
        
        return results


class DynamicPortfolio:
    """Maintain portfolio of agents, dynamically allocate based on performance"""
    
    def __init__(self, specialists):
        self.specialists = specialists
        self.agents = {name: QuantumGeneticAgent(genome) 
                      for name, genome in specialists.items()}
        self.allocations = {name: 1.0/len(specialists) for name in specialists.keys()}
        self.performance_history = {name: [] for name in specialists.keys()}
        self.rebalance_frequency = 10
        self.step_count = 0
    
    def step(self, environment):
        """Execute one step with current allocations"""
        results = {}
        
        for name, agent in self.agents.items():
            fitness = agent.evaluate_fitness(environment=environment)
            self.performance_history[name].append(fitness)
            results[name] = fitness
        
        self.step_count += 1
        
        # Rebalance periodically
        if self.step_count % self.rebalance_frequency == 0:
            self.rebalance()
        
        # Return weighted portfolio performance
        portfolio_fitness = sum(results[name] * self.allocations[name] 
                               for name in self.specialists.keys())
        
        return portfolio_fitness, results
    
    def rebalance(self, lookback=50):
        """Rebalance allocations based on recent performance"""
        # Calculate Sharpe ratios (return/risk)
        sharpe_ratios = {}
        
        for name in self.specialists.keys():
            recent = self.performance_history[name][-lookback:]
            if len(recent) > 1:
                mean_return = np.mean(recent)
                std_return = np.std(recent)
                sharpe = mean_return / (std_return + 1e-8)
                sharpe_ratios[name] = sharpe
            else:
                sharpe_ratios[name] = 0.0
        
        # Convert to allocations (softmax over positive Sharpe ratios)
        sharpe_values = np.array(list(sharpe_ratios.values()))
        sharpe_values = np.maximum(sharpe_values, 0)  # Only positive
        
        if np.sum(sharpe_values) > 0:
            allocations_array = sharpe_values / np.sum(sharpe_values)
        else:
            allocations_array = np.ones(len(sharpe_values)) / len(sharpe_values)
        
        for i, name in enumerate(self.specialists.keys()):
            self.allocations[name] = allocations_array[i]
        
        print(f"\n📊 Rebalanced allocations (step {self.step_count}):")
        for name, alloc in self.allocations.items():
            print(f"   {name:15s}: {alloc*100:.1f}%")


def visualize_ensemble_comparison(results, save_path=None):
    """Visualize comparison of ensemble methods"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Advanced Ensemble Methods Comparison', 
                 fontsize=16, fontweight='bold')
    
    methods = list(results.keys())
    environments = list(results[methods[0]].keys())
    
    # Plot 1: Performance heatmap
    ax = axes[0, 0]
    perf_matrix = []
    for method in methods:
        row = [results[method][env]['mean'] for env in environments]
        perf_matrix.append(row)
    
    im = ax.imshow(perf_matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(environments)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(environments, rotation=45)
    ax.set_yticklabels(methods)
    ax.set_title('Performance Heatmap')
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(environments)):
            text = ax.text(j, i, f'{perf_matrix[i][j]:.4f}',
                         ha="center", va="center", color="w", fontsize=8)
    
    # Plot 2: Average performance by method
    ax = axes[0, 1]
    avg_perfs = []
    for method in methods:
        avg = np.mean([results[method][env]['mean'] for env in environments])
        avg_perfs.append(avg)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    ax.barh(methods, avg_perfs, color=colors)
    ax.set_xlabel('Average Fitness')
    ax.set_title('Overall Performance Ranking')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Stability comparison
    ax = axes[1, 0]
    stability_scores = []
    for method in methods:
        stds = [results[method][env]['std'] for env in environments]
        avg_std = np.mean(stds)
        stability_scores.append(1.0 / (avg_std + 1e-8))  # Higher is more stable
    
    ax.bar(methods, stability_scores, color=colors)
    ax.set_ylabel('Stability Score (1/std)')
    ax.set_title('Performance Stability')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Best method per environment
    ax = axes[1, 1]
    best_methods = []
    for env in environments:
        best_perf = -float('inf')
        best_method = None
        for method in methods:
            if results[method][env]['mean'] > best_perf:
                best_perf = results[method][env]['mean']
                best_method = method
        best_methods.append(best_method)
    
    # Count wins per method
    method_wins = {method: best_methods.count(method) for method in methods}
    ax.bar(method_wins.keys(), method_wins.values(), color=colors)
    ax.set_ylabel('Environments Won')
    ax.set_title('Win Rate by Method')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved: {save_path}")
    
    return fig


def main():
    """Main execution for advanced ensemble techniques"""
    print("\n" + "="*80)
    print("🚀 ADVANCED SPECIALIST ENSEMBLE TECHNIQUES")
    print("="*80)
    
    # Load specialists
    specialists = {
        'standard': np.array([2.9460, 0.1269, 0.0050, 0.2996]),
        'harsh': np.array([3.0000, 2.0000, 0.0050, 0.5713]),
        'gentle': np.array([2.7668, 0.1853, 0.0050, 0.6798]),
        'chaotic': np.array([3.0000, 0.5045, 0.0050, 0.4108]),
        'oscillating': np.array([3.0000, 1.8126, 0.0050, 0.0000])
    }
    
    environments = ['standard', 'harsh', 'gentle', 'chaotic', 'oscillating']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nTesting 4 advanced ensemble methods:")
    print("1. Weighted Ensemble (adaptive blending)")
    print("2. Meta-Learner (neural network selection)")
    print("3. Hybrid Genomes (trait combination)")
    print("4. Dynamic Portfolio (allocation optimization)")
    
    all_results = {}
    
    # 1. Test Hybrid Genomes
    print("\n" + "="*80)
    print("🧬 HYBRID GENOME CREATION")
    print("="*80)
    
    hybrid_creator = HybridGenomeCreator(specialists)
    hybrid_creator.create_all_hybrids()
    super_hybrid = hybrid_creator.create_super_hybrid()
    
    print(f"\nCreated {len(hybrid_creator.hybrids)} hybrid genomes")
    hybrid_results = hybrid_creator.test_hybrids(environments, trials_per_env=20)
    
    # Add best hybrid to results
    best_hybrid_name = None
    best_hybrid_avg = -float('inf')
    for name, data in hybrid_results.items():
        avg = np.mean([data[env]['mean'] for env in environments])
        if avg > best_hybrid_avg:
            best_hybrid_avg = avg
            best_hybrid_name = name
    
    all_results[f'hybrid_{best_hybrid_name}'] = hybrid_results[best_hybrid_name]
    
    # 2. Weighted Ensemble
    print("\n" + "="*80)
    print("⚖️  WEIGHTED ENSEMBLE")
    print("="*80)
    
    weighted_ensemble = WeightedEnsemble(specialists)
    weighted_results = {}
    
    for env in environments:
        fitnesses = []
        for trial in range(20):
            # Get blended genome
            blended_genome = weighted_ensemble.get_blended_genome()
            agent = QuantumGeneticAgent(blended_genome)
            fitness = agent.evaluate_fitness(environment=env)
            fitnesses.append(fitness)
            
            # Update weights based on specialist performance
            for spec_name in specialists.keys():
                spec_agent = QuantumGeneticAgent(specialists[spec_name])
                spec_fitness = spec_agent.evaluate_fitness(environment=env)
                weighted_ensemble.record_performance(spec_name, spec_fitness)
        
        weighted_results[env] = {
            'mean': np.mean(fitnesses),
            'std': np.std(fitnesses),
            'genome': blended_genome.tolist()
        }
        print(f"{env}: {np.mean(fitnesses):.6f} ± {np.std(fitnesses):.6f}")
    
    all_results['weighted_ensemble'] = weighted_results
    
    # 3. Dynamic Portfolio
    print("\n" + "="*80)
    print("💼 DYNAMIC PORTFOLIO")
    print("="*80)
    
    portfolio_results = {}
    
    for env in environments:
        print(f"\nTesting in {env} environment...")
        portfolio = DynamicPortfolio(specialists)
        
        fitnesses = []
        for step in range(100):
            port_fitness, _ = portfolio.step(env)
            fitnesses.append(port_fitness)
        
        portfolio_results[env] = {
            'mean': np.mean(fitnesses),
            'std': np.std(fitnesses),
            'final_allocations': portfolio.allocations
        }
        print(f"Final performance: {np.mean(fitnesses):.6f} ± {np.std(fitnesses):.6f}")
    
    all_results['dynamic_portfolio'] = portfolio_results
    
    # Add original specialists for comparison
    for spec_name, spec_genome in specialists.items():
        spec_results = {}
        agent = QuantumGeneticAgent(spec_genome)
        for env in environments:
            fitnesses = [agent.evaluate_fitness(environment=env) for _ in range(20)]
            spec_results[env] = {
                'mean': np.mean(fitnesses),
                'std': np.std(fitnesses),
                'genome': spec_genome.tolist()
            }
        all_results[f'specialist_{spec_name}'] = spec_results
    
    # Visualize
    visualize_ensemble_comparison(all_results, 
                                  save_path=f'ensemble_comparison_{timestamp}.png')
    
    # Save results
    with open(f'ensemble_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    print(f"\n💾 Results saved: ensemble_results_{timestamp}.json")
    print("\n✅ Advanced ensemble analysis complete!")


if __name__ == "__main__":
    main()
