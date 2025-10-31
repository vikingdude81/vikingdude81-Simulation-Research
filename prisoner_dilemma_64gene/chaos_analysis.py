"""
🌀 Chaos Theory Analysis Pipeline
=================================

Analyzes evolutionary dynamics data for chaotic behavior.

Implements:
1. Lyapunov Exponent Calculation (divergence of nearby trajectories)
2. Attractor Reconstruction (delay embedding)
3. Correlation Dimension (fractal structure)
4. Entropy Measures (predictability)
5. Bifurcation Detection (phase transitions)
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from datetime import datetime

class ChaosAnalyzer:
    """Complete chaos theory analysis toolkit"""
    
    def __init__(self, data_file):
        """Load chaos dataset"""
        print(f"\n📂 Loading data from: {data_file}")
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.num_runs = len(self.data['runs'])
        self.generations = len(self.data['runs'][0]['fitness_trajectory'])
        print(f"✅ Loaded: {self.num_runs} runs × {self.generations} generations = {self.num_runs * self.generations:,} points")
    
    def calculate_lyapunov_exponent(self, time_series, max_lag=50):
        """
        Calculate largest Lyapunov exponent.
        
        Positive λ → chaos
        Zero λ → periodic/quasiperiodic
        Negative λ → convergence to fixed point
        """
        n = len(time_series)
        series = np.array(time_series)
        
        # Normalize to unit variance
        series = (series - np.mean(series)) / (np.std(series) + 1e-10)
        
        divergences = []
        
        for lag in range(1, min(max_lag, n // 4)):
            # Find pairs of points that are close initially
            distances = []
            
            for i in range(n - 2 * lag):
                # Initial distance
                d0 = abs(series[i] - series[i + 1])
                
                # Distance after lag
                d_lag = abs(series[i + lag] - series[i + 1 + lag])
                
                if d0 > 0 and d_lag > 0:
                    distances.append(np.log(d_lag / d0))
            
            if distances:
                divergences.append(np.mean(distances) / lag)
        
        lyapunov = np.mean(divergences) if divergences else 0
        return lyapunov
    
    def reconstruct_attractor(self, time_series, embedding_dim=3, delay=1):
        """
        Delay embedding reconstruction (Takens' theorem).
        
        Creates embedding_dim-dimensional vectors from time series:
        X(t) = [x(t), x(t+delay), x(t+2*delay), ...]
        """
        n = len(time_series)
        num_vectors = n - (embedding_dim - 1) * delay
        
        if num_vectors < 1:
            return np.array([])
        
        attractor = np.zeros((num_vectors, embedding_dim))
        
        for i in range(num_vectors):
            for j in range(embedding_dim):
                attractor[i, j] = time_series[i + j * delay]
        
        return attractor
    
    def correlation_dimension(self, attractor, max_r=None, num_points=20):
        """
        Calculate correlation dimension (Grassberger-Procaccia algorithm).
        
        D2 = fractal dimension of attractor
        D2 ≈ 2.0 → likely chaotic
        D2 = integer → periodic
        """
        if len(attractor) < 100:
            return 0.0
        
        # Sample if too large
        if len(attractor) > 1000:
            indices = np.random.choice(len(attractor), 1000, replace=False)
            attractor = attractor[indices]
        
        # Calculate all pairwise distances
        distances = pdist(attractor, metric='euclidean')
        
        if max_r is None:
            max_r = np.percentile(distances, 50)
        
        # Calculate correlation integral C(r) for different radii
        radii = np.logspace(np.log10(max_r/100), np.log10(max_r), num_points)
        correlations = []
        
        for r in radii:
            count = np.sum(distances < r)
            total_pairs = len(distances)
            correlations.append(count / total_pairs if total_pairs > 0 else 0)
        
        # Fit log(C(r)) vs log(r) to get slope = dimension
        log_r = np.log(radii)
        log_c = np.log(np.array(correlations) + 1e-10)
        
        # Linear regression on middle region
        valid = np.isfinite(log_c) & (log_c > -10)
        if np.sum(valid) > 5:
            coeffs = np.polyfit(log_r[valid], log_c[valid], 1)
            dimension = coeffs[0]
            return max(0, dimension)
        else:
            return 0.0
    
    def calculate_entropy_rate(self, time_series, bins=20):
        """
        Estimate entropy rate (bits of unpredictability per step).
        
        High entropy → unpredictable/chaotic
        Low entropy → predictable/periodic
        """
        # Discretize time series into bins
        hist, _ = np.histogram(time_series, bins=bins)
        
        # Calculate Shannon entropy
        probabilities = hist / np.sum(hist)
        h = entropy(probabilities, base=2)
        
        return h
    
    def detect_bifurcations(self, parameter_values, attractors):
        """
        Detect bifurcation points (sudden changes in attractor structure).
        
        Returns indices where major transitions occur.
        """
        bifurcation_points = []
        
        for i in range(1, len(attractors)):
            # Measure change in attractor size/spread
            std_prev = np.std(attractors[i-1])
            std_curr = np.std(attractors[i])
            
            relative_change = abs(std_curr - std_prev) / (std_prev + 1e-10)
            
            if relative_change > 0.5:  # 50% change threshold
                bifurcation_points.append(i)
        
        return bifurcation_points
    
    def analyze_all_runs(self):
        """Comprehensive chaos analysis on all runs"""
        print("\n" + "="*70)
        print("🌀 CHAOS ANALYSIS")
        print("="*70)
        
        results = {
            'lyapunov_exponents': [],
            'correlation_dimensions': [],
            'entropy_rates': [],
            'attractor_info': [],
            'chaos_indicators': []
        }
        
        print(f"\nAnalyzing {self.num_runs} evolution runs...")
        
        for run_idx, run in enumerate(self.data['runs']):
            if (run_idx + 1) % 10 == 0:
                print(f"  Progress: {run_idx + 1}/{self.num_runs} runs")
            
            # Extract fitness trajectory
            fitness = run['fitness_trajectory']
            
            # 1. Lyapunov exponent
            lyap = self.calculate_lyapunov_exponent(fitness)
            results['lyapunov_exponents'].append(lyap)
            
            # 2. Attractor reconstruction
            attractor = self.reconstruct_attractor(fitness, embedding_dim=3, delay=5)
            
            # 3. Correlation dimension
            if len(attractor) > 0:
                corr_dim = self.correlation_dimension(attractor)
                results['correlation_dimensions'].append(corr_dim)
            else:
                results['correlation_dimensions'].append(0.0)
            
            # 4. Entropy rate
            ent = self.calculate_entropy_rate(fitness)
            results['entropy_rates'].append(ent)
            
            # 5. Classify behavior
            is_chaotic = (lyap > 0.01 and corr_dim > 1.5 and ent > 2.0)
            is_periodic = (abs(lyap) < 0.01 and corr_dim < 1.2)
            is_convergent = (lyap < -0.01)
            
            behavior = 'unknown'
            if is_chaotic:
                behavior = 'chaotic'
            elif is_periodic:
                behavior = 'periodic'
            elif is_convergent:
                behavior = 'convergent'
            
            results['chaos_indicators'].append(behavior)
            
            results['attractor_info'].append({
                'run': run_idx,
                'attractor_points': len(attractor),
                'attractor_spread': float(np.std(attractor)) if len(attractor) > 0 else 0.0
            })
        
        return results
    
    def visualize_results(self, results):
        """Create comprehensive visualization of chaos analysis"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Lyapunov Exponents Distribution
        ax1 = plt.subplot(3, 3, 1)
        lyaps = results['lyapunov_exponents']
        ax1.hist(lyaps, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', label='λ = 0 (boundary)')
        ax1.set_xlabel('Lyapunov Exponent (λ)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Lyapunov Exponent Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Correlation Dimensions
        ax2 = plt.subplot(3, 3, 2)
        dims = results['correlation_dimensions']
        ax2.hist(dims, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax2.set_xlabel('Correlation Dimension (D2)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Correlation Dimension Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Entropy Rates
        ax3 = plt.subplot(3, 3, 3)
        entropies = results['entropy_rates']
        ax3.hist(entropies, bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax3.set_xlabel('Entropy Rate (bits)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Entropy Rate Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Lyapunov vs Correlation Dimension
        ax4 = plt.subplot(3, 3, 4)
        ax4.scatter(lyaps, dims, alpha=0.5)
        ax4.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax4.axhline(2.0, color='blue', linestyle='--', alpha=0.5, label='D2 = 2 (chaos threshold)')
        ax4.set_xlabel('Lyapunov Exponent')
        ax4.set_ylabel('Correlation Dimension')
        ax4.set_title('Chaos Indicators Scatter')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Behavior Classification
        ax5 = plt.subplot(3, 3, 5)
        behaviors = results['chaos_indicators']
        behavior_counts = {
            'chaotic': behaviors.count('chaotic'),
            'periodic': behaviors.count('periodic'),
            'convergent': behaviors.count('convergent'),
            'unknown': behaviors.count('unknown')
        }
        colors = ['red', 'blue', 'green', 'gray']
        ax5.bar(behavior_counts.keys(), behavior_counts.values(), color=colors, alpha=0.7, edgecolor='black')
        ax5.set_ylabel('Number of Runs')
        ax5.set_title('Evolutionary Behavior Classification')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Sample Attractor (3D projection of first run)
        ax6 = plt.subplot(3, 3, 6, projection='3d')
        first_run = self.data['runs'][0]['fitness_trajectory']
        attractor = self.reconstruct_attractor(first_run, embedding_dim=3, delay=5)
        if len(attractor) > 0:
            ax6.plot(attractor[:, 0], attractor[:, 1], attractor[:, 2], alpha=0.5, linewidth=0.5)
            ax6.scatter(attractor[0, 0], attractor[0, 1], attractor[0, 2], color='green', s=100, label='Start')
            ax6.scatter(attractor[-1, 0], attractor[-1, 1], attractor[-1, 2], color='red', s=100, label='End')
            ax6.set_title('Sample Attractor (Run 1)')
            ax6.legend()
        
        # 7. Fitness Trajectories Sample
        ax7 = plt.subplot(3, 3, 7)
        for i in range(min(10, len(self.data['runs']))):
            fitness = self.data['runs'][i]['fitness_trajectory']
            ax7.plot(fitness, alpha=0.5, linewidth=1)
        ax7.set_xlabel('Generation')
        ax7.set_ylabel('Best Fitness')
        ax7.set_title('Sample Fitness Trajectories (10 runs)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Entropy vs Lyapunov
        ax8 = plt.subplot(3, 3, 8)
        ax8.scatter(lyaps, entropies, alpha=0.5, color='purple')
        ax8.set_xlabel('Lyapunov Exponent')
        ax8.set_ylabel('Entropy Rate')
        ax8.set_title('Predictability Analysis')
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary_text = f"""
        CHAOS ANALYSIS SUMMARY
        {'='*30}
        
        Total Runs: {len(lyaps)}
        
        Lyapunov Exponents:
          Mean: {np.mean(lyaps):.4f}
          Std: {np.std(lyaps):.4f}
          Chaotic (λ>0): {sum(1 for l in lyaps if l > 0.01)}
          
        Correlation Dimension:
          Mean: {np.mean(dims):.3f}
          Std: {np.std(dims):.3f}
          
        Entropy Rate:
          Mean: {np.mean(entropies):.3f}
          Std: {np.std(entropies):.3f}
          
        Behavior Types:
          Chaotic: {behavior_counts['chaotic']}
          Periodic: {behavior_counts['periodic']}
          Convergent: {behavior_counts['convergent']}
          Unknown: {behavior_counts['unknown']}
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        
        filename = f"chaos_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved: {filename}")
        
        return filename

def analyze_chaos_dataset(data_file):
    """Main analysis function"""
    print("\n" + "="*70)
    print("🌀 CHAOS THEORY ANALYSIS PIPELINE")
    print("="*70)
    
    # Initialize analyzer
    analyzer = ChaosAnalyzer(data_file)
    
    # Run analysis
    results = analyzer.analyze_all_runs()
    
    # Print results
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    lyaps = results['lyapunov_exponents']
    dims = results['correlation_dimensions']
    entropies = results['entropy_rates']
    behaviors = results['chaos_indicators']
    
    print(f"\n🔥 Lyapunov Exponents:")
    print(f"   Mean: {np.mean(lyaps):.4f}")
    print(f"   Std: {np.std(lyaps):.4f}")
    print(f"   Min: {np.min(lyaps):.4f}")
    print(f"   Max: {np.max(lyaps):.4f}")
    print(f"   Chaotic (λ>0.01): {sum(1 for l in lyaps if l > 0.01)}/{len(lyaps)}")
    
    print(f"\n📐 Correlation Dimensions:")
    print(f"   Mean: {np.mean(dims):.3f}")
    print(f"   Std: {np.std(dims):.3f}")
    print(f"   Min: {np.min(dims):.3f}")
    print(f"   Max: {np.max(dims):.3f}")
    
    print(f"\n🎲 Entropy Rates:")
    print(f"   Mean: {np.mean(entropies):.3f} bits")
    print(f"   Std: {np.std(entropies):.3f} bits")
    
    print(f"\n🎯 Behavior Classification:")
    behavior_counts = {
        'chaotic': behaviors.count('chaotic'),
        'periodic': behaviors.count('periodic'),
        'convergent': behaviors.count('convergent'),
        'unknown': behaviors.count('unknown')
    }
    for behavior, count in behavior_counts.items():
        percentage = count / len(behaviors) * 100
        print(f"   {behavior.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Visualize
    print("\n" + "="*70)
    print("📊 Generating visualizations...")
    print("="*70)
    viz_file = analyzer.visualize_results(results)
    
    # Save results
    results_file = f"chaos_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_data = {
        'lyapunov_exponents': lyaps,
        'correlation_dimensions': dims,
        'entropy_rates': entropies,
        'behaviors': behaviors,
        'behavior_counts': behavior_counts,
        'summary': {
            'mean_lyapunov': float(np.mean(lyaps)),
            'mean_dimension': float(np.mean(dims)),
            'mean_entropy': float(np.mean(entropies)),
            'chaos_percentage': float(behavior_counts['chaotic'] / len(behaviors) * 100)
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"✅ Results saved: {results_file}")
    
    print("\n" + "="*70)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*70)
    
    return results, viz_file

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage: python chaos_analysis.py <data_file.json>")
        print("\nExample: python chaos_analysis.py chaos_dataset_100runs_20251030_120000.json")
        sys.exit(1)
    
    data_file = sys.argv[1]
    results, viz_file = analyze_chaos_dataset(data_file)
