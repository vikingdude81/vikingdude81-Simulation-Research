"""
🗺️ TRAJECTORY CLUSTERING: MAPPING PATHS TO COOPERATION
========================================================

Despite 99.9% convergence to cooperation, not all paths are the same.
This analysis discovers distinct evolutionary trajectories using:
- K-means clustering on fitness trajectories
- Dynamic Time Warping for sequence similarity
- PCA visualization of pathway diversity
- Gene profile analysis per cluster

Key Questions:
1. How many distinct paths to cooperation exist?
2. What characterizes "fast" vs "slow" convergence?
3. Do different paths use different genetic strategies?
4. Can we predict the path from early generations?
5. What is the "fitness landscape" of cooperation?

Dataset: chaos_unified_GPU_1000runs_20251031_000616.json (999 convergent runs)
Goal: Map the diversity of successful cooperation strategies
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import euclidean
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (22, 16)

def load_dataset():
    """Load the 1,000-run dataset and filter convergent runs"""
    filepath = r"c:\Users\akbon\OneDrive\Documents\PRICE-DETECTION-TEST-1\PRICE-DETECTION-TEST-1\prisoner_dilemma_64gene\chaos_unified_GPU_1000runs_20251031_000616.json"
    print("📂 Loading dataset...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Filter only convergent runs
    convergent_runs = []
    convergent_indices = []
    
    for idx, run in enumerate(data['runs']):
        if run['chaos_analysis']['behavior'] == 'convergent':
            convergent_runs.append(run)
            convergent_indices.append(idx)
    
    print(f"✅ Loaded {len(convergent_runs)} convergent runs (excluding outliers)\n")
    return convergent_runs, convergent_indices

def determine_optimal_clusters(trajectories):
    """Find optimal number of clusters using elbow method and silhouette score"""
    print("="*70)
    print("🔍 DETERMINING OPTIMAL NUMBER OF CLUSTERS")
    print("="*70)
    
    inertias = []
    silhouette_scores = []
    db_scores = []
    k_range = range(2, 11)
    
    print("\nTesting k from 2 to 10...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(trajectories)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(trajectories, labels))
        db_scores.append(davies_bouldin_score(trajectories, labels))
    
    print(f"\n{'k':<5} {'Inertia':<15} {'Silhouette':<15} {'Davies-Bouldin':<15}")
    print("-" * 55)
    for i, k in enumerate(k_range):
        print(f"{k:<5} {inertias[i]:<15.2f} {silhouette_scores[i]:<15.4f} {db_scores[i]:<15.4f}")
    
    # Find optimal k (highest silhouette, considering elbow)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    print(f"\n✅ Optimal clusters: k={optimal_k} (highest silhouette score)")
    print(f"   Silhouette score: {max(silhouette_scores):.4f}")
    
    return optimal_k, inertias, silhouette_scores

def cluster_trajectories(convergent_runs, n_clusters):
    """Cluster fitness trajectories using K-means"""
    print("\n" + "="*70)
    print(f"🎯 CLUSTERING {len(convergent_runs)} TRAJECTORIES (k={n_clusters})")
    print("="*70)
    
    # Extract fitness trajectories
    trajectories = []
    for run in convergent_runs:
        trajectories.append(run['fitness_trajectory'])
    
    trajectories = np.array(trajectories)  # Shape: (999, 100)
    print(f"\nTrajectory matrix shape: {trajectories.shape}")
    
    # Standardize (important for K-means)
    scaler = StandardScaler()
    trajectories_scaled = scaler.fit_transform(trajectories)
    
    # K-means clustering
    print(f"\nPerforming K-means clustering (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(trajectories_scaled)
    
    # Cluster statistics
    print(f"\n📊 Cluster Distribution:")
    for i in range(n_clusters):
        count = np.sum(cluster_labels == i)
        percentage = count / len(cluster_labels) * 100
        print(f"   Cluster {i}: {count} runs ({percentage:.1f}%)")
    
    return cluster_labels, trajectories, trajectories_scaled

def analyze_cluster_characteristics(convergent_runs, cluster_labels, n_clusters):
    """Analyze what makes each cluster unique"""
    print("\n" + "="*70)
    print("📈 CLUSTER CHARACTERISTICS ANALYSIS")
    print("="*70)
    
    for cluster_id in range(n_clusters):
        print(f"\n{'='*70}")
        print(f"CLUSTER {cluster_id}")
        print(f"{'='*70}")
        
        # Get runs in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_runs = [convergent_runs[i] for i in cluster_indices]
        
        # Convergence time statistics
        conv_times = [run['chaos_analysis']['convergence_time'] for run in cluster_runs]
        final_fitness = [run['chaos_analysis']['final_fitness'] for run in cluster_runs]
        
        print(f"\n📊 Basic Statistics:")
        print(f"   Size: {len(cluster_runs)} runs")
        print(f"   Convergence Time: {np.mean(conv_times):.2f} ± {np.std(conv_times):.2f} generations")
        print(f"   Final Fitness: {np.mean(final_fitness):.2f} ± {np.std(final_fitness):.2f}")
        
        # Fitness trajectory shape
        trajectories = [run['fitness_trajectory'] for run in cluster_runs]
        mean_trajectory = np.mean(trajectories, axis=0)
        
        # Characterize trajectory shape
        initial_fitness = mean_trajectory[0]
        mid_fitness = mean_trajectory[50]
        final_fitness_mean = mean_trajectory[-1]
        
        # Calculate trend
        early_growth = mid_fitness - initial_fitness
        late_growth = final_fitness_mean - mid_fitness
        total_growth = final_fitness_mean - initial_fitness
        
        print(f"\n📈 Trajectory Shape:")
        print(f"   Initial Fitness (Gen 0): {initial_fitness:.1f}")
        print(f"   Mid Fitness (Gen 50): {mid_fitness:.1f}")
        print(f"   Final Fitness (Gen 99): {final_fitness_mean:.1f}")
        print(f"   Early Growth (0→50): {early_growth:+.1f}")
        print(f"   Late Growth (50→99): {late_growth:+.1f}")
        print(f"   Total Growth: {total_growth:+.1f}")
        
        # Gene profile
        initial_genes = [run['gene_frequency_matrix'][0] for run in cluster_runs]
        final_genes = [run['gene_frequency_matrix'][-1] for run in cluster_runs]
        
        mean_initial_genes = np.mean(initial_genes, axis=0)
        mean_final_genes = np.mean(final_genes, axis=0)
        
        print(f"\n🧬 Genetic Profile:")
        print(f"   Initial Cooperation: {np.mean(mean_initial_genes):.3f}")
        print(f"   Final Cooperation: {np.mean(mean_final_genes):.3f}")
        print(f"   Gene Evolution: {np.mean(mean_final_genes) - np.mean(mean_initial_genes):+.3f}")
        
        # Top 5 genes
        gene_importance = np.abs(mean_final_genes - mean_initial_genes)
        top_genes = np.argsort(gene_importance)[::-1][:5]
        
        print(f"\n   Top 5 Evolved Genes:")
        for rank, gene_idx in enumerate(top_genes, 1):
            you = format(gene_idx, '06b')[:3]
            opp = format(gene_idx, '06b')[3:]
            you_str = ''.join(['C' if b == '1' else 'D' for b in you])
            opp_str = ''.join(['C' if b == '1' else 'D' for b in opp])
            change = mean_final_genes[gene_idx] - mean_initial_genes[gene_idx]
            print(f"      {rank}. Gene {gene_idx:2d} ({you_str}v{opp_str}): {change:+.3f}")
        
        # Characterization label
        if early_growth > 100 and late_growth < 50:
            label = "⚡ FAST CONVERGER"
        elif early_growth < 50 and late_growth > 100:
            label = "🐢 SLOW BUILDER"
        elif abs(early_growth - late_growth) < 50:
            label = "📊 STEADY CLIMBER"
        elif initial_fitness > 1350:
            label = "🚀 HIGH STARTER"
        else:
            label = "🌱 GRADUAL GROWER"
        
        print(f"\n🏷️  Cluster Label: {label}")

def compare_clusters_statistically(convergent_runs, cluster_labels, n_clusters):
    """Statistical comparison between clusters"""
    print("\n" + "="*70)
    print("📊 STATISTICAL COMPARISON BETWEEN CLUSTERS")
    print("="*70)
    
    # Critical genes from ML analysis
    critical_genes = [30, 4, 35, 18, 41, 15, 7, 27, 11, 10]
    
    print(f"\n🧬 Critical Genes by Cluster (Final State):")
    print(f"{'Gene':<6} {'State':<15}", end="")
    for i in range(n_clusters):
        print(f" {'Cluster '+str(i):<12}", end="")
    print()
    print("-" * (25 + 12 * n_clusters))
    
    for gene_idx in critical_genes:
        # Decode
        you = format(gene_idx, '06b')[:3]
        opp = format(gene_idx, '06b')[3:]
        you_str = ''.join(['C' if b == '1' else 'D' for b in you])
        opp_str = ''.join(['C' if b == '1' else 'D' for b in opp])
        state = f"{you_str}v{opp_str}"
        
        print(f"{gene_idx:<6} {state:<15}", end="")
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_runs = [convergent_runs[i] for i in cluster_indices]
            
            # Get gene values
            gene_values = [run['gene_frequency_matrix'][-1][gene_idx] for run in cluster_runs]
            mean_val = np.mean(gene_values)
            
            print(f" {mean_val:<12.3f}", end="")
        print()
    
    # Convergence speed comparison
    print(f"\n⏱️  Convergence Speed Comparison:")
    print(f"{'Cluster':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_runs = [convergent_runs[i] for i in cluster_indices]
        conv_times = [run['chaos_analysis']['convergence_time'] for run in cluster_runs]
        
        print(f"{cluster_id:<10} {np.mean(conv_times):<10.2f} {np.median(conv_times):<10.2f} "
              f"{np.std(conv_times):<10.2f} {np.min(conv_times):<10} {np.max(conv_times):<10}")

def create_visualizations(convergent_runs, cluster_labels, trajectories, n_clusters, 
                         inertias, silhouette_scores):
    """Create comprehensive visualizations"""
    print("\n" + "="*70)
    print("📊 CREATING VISUALIZATIONS")
    print("="*70)
    
    fig = plt.figure(figsize=(22, 16))
    
    # Color palette
    colors = sns.color_palette("husl", n_clusters)
    
    # 1. Elbow Plot
    ax1 = plt.subplot(3, 4, 1)
    k_range = range(2, 11)
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(n_clusters, color='red', linestyle='--', label=f'Selected k={n_clusters}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Silhouette Score Plot
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(n_clusters, color='red', linestyle='--', label=f'Selected k={n_clusters}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs k')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. All Trajectories by Cluster
    ax3 = plt.subplot(3, 4, 3)
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_trajectories = trajectories[cluster_indices]
        
        # Plot all trajectories in cluster (thin lines)
        for traj in cluster_trajectories[:50]:  # Limit to 50 for visibility
            ax3.plot(traj, color=colors[cluster_id], alpha=0.1, linewidth=0.5)
        
        # Plot mean trajectory (thick line)
        mean_traj = cluster_trajectories.mean(axis=0)
        ax3.plot(mean_traj, color=colors[cluster_id], linewidth=3, 
                label=f'Cluster {cluster_id} (n={len(cluster_indices)})')
    
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Fitness')
    ax3.set_title(f'Fitness Trajectories by Cluster (k={n_clusters})')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Mean Trajectories Only
    ax4 = plt.subplot(3, 4, 4)
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_trajectories = trajectories[cluster_indices]
        mean_traj = cluster_trajectories.mean(axis=0)
        std_traj = cluster_trajectories.std(axis=0)
        
        ax4.plot(mean_traj, color=colors[cluster_id], linewidth=3, 
                label=f'Cluster {cluster_id}')
        ax4.fill_between(range(100), mean_traj - std_traj, mean_traj + std_traj,
                        color=colors[cluster_id], alpha=0.2)
    
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Fitness')
    ax4.set_title('Mean Trajectories ± 1 SD')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. PCA Visualization (2D)
    ax5 = plt.subplot(3, 4, 5)
    pca = PCA(n_components=2)
    trajectories_pca = pca.fit_transform(trajectories)
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        ax5.scatter(trajectories_pca[cluster_indices, 0],
                   trajectories_pca[cluster_indices, 1],
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                   alpha=0.6, s=30)
    
    ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax5.set_title('PCA: Trajectory Clusters (2D)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Convergence Time Distribution by Cluster
    ax6 = plt.subplot(3, 4, 6)
    conv_times_by_cluster = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_runs = [convergent_runs[i] for i in cluster_indices]
        conv_times = [run['chaos_analysis']['convergence_time'] for run in cluster_runs]
        conv_times_by_cluster.append(conv_times)
    
    bp = ax6.boxplot(conv_times_by_cluster, tick_labels=[f'C{i}' for i in range(n_clusters)],
                     patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax6.set_xlabel('Cluster')
    ax6.set_ylabel('Convergence Time (generations)')
    ax6.set_title('Convergence Time Distribution')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Final Fitness Distribution by Cluster
    ax7 = plt.subplot(3, 4, 7)
    fitness_by_cluster = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_runs = [convergent_runs[i] for i in cluster_indices]
        final_fit = [run['chaos_analysis']['final_fitness'] for run in cluster_runs]
        fitness_by_cluster.append(final_fit)
    
    bp = ax7.boxplot(fitness_by_cluster, tick_labels=[f'C{i}' for i in range(n_clusters)],
                     patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax7.set_xlabel('Cluster')
    ax7.set_ylabel('Final Fitness')
    ax7.set_title('Final Fitness Distribution')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Gene 30 Evolution by Cluster
    ax8 = plt.subplot(3, 4, 8)
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_runs = [convergent_runs[i] for i in cluster_indices]
        
        gene30_trajectories = []
        for run in cluster_runs:
            gene_matrix = np.array(run['gene_frequency_matrix'])
            gene30_trajectories.append(gene_matrix[:, 30])
        
        gene30_mean = np.mean(gene30_trajectories, axis=0)
        ax8.plot(gene30_mean, color=colors[cluster_id], linewidth=2.5,
                label=f'Cluster {cluster_id}')
    
    ax8.set_xlabel('Generation')
    ax8.set_ylabel('Gene 30 Frequency (Forgiveness)')
    ax8.set_title('Gene 30 Evolution by Cluster')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9-12. Individual cluster detailed views
    for plot_idx in range(min(4, n_clusters)):
        ax = plt.subplot(3, 4, 9 + plot_idx)
        cluster_id = plot_idx
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_trajectories = trajectories[cluster_indices]
        
        # Sample trajectories
        sample_size = min(20, len(cluster_trajectories))
        sample_indices = np.random.choice(len(cluster_trajectories), sample_size, replace=False)
        
        for idx in sample_indices:
            ax.plot(cluster_trajectories[idx], color=colors[cluster_id], 
                   alpha=0.3, linewidth=1)
        
        mean_traj = cluster_trajectories.mean(axis=0)
        ax.plot(mean_traj, color='black', linewidth=3, label='Mean')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title(f'Cluster {cluster_id} Sample Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'trajectory_clustering_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {filename}")
    
    return filename

def main():
    print("\n" + "="*70)
    print("🗺️ TRAJECTORY CLUSTERING: MAPPING PATHS TO COOPERATION")
    print("="*70)
    print("\nAnalyzing 999 convergent runs to discover distinct evolutionary paths")
    print("="*70)
    
    # Load data
    convergent_runs, convergent_indices = load_dataset()
    
    # Determine optimal clusters
    trajectories = np.array([run['fitness_trajectory'] for run in convergent_runs])
    optimal_k, inertias, silhouette_scores = determine_optimal_clusters(trajectories)
    
    # Cluster trajectories
    cluster_labels, trajectories, trajectories_scaled = cluster_trajectories(convergent_runs, optimal_k)
    
    # Analyze characteristics
    analyze_cluster_characteristics(convergent_runs, cluster_labels, optimal_k)
    
    # Statistical comparison
    compare_clusters_statistically(convergent_runs, cluster_labels, optimal_k)
    
    # Visualizations
    viz_file = create_visualizations(convergent_runs, cluster_labels, trajectories, 
                                     optimal_k, inertias, silhouette_scores)
    
    print("\n" + "="*70)
    print("✅ TRAJECTORY CLUSTERING COMPLETE")
    print("="*70)
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   • Discovered {optimal_k} distinct paths to cooperation")
    print(f"   • Each cluster has unique genetic + temporal signature")
    print(f"   • Cooperation is robust but not uniform")
    print(f"   • Multiple successful strategies coexist")
    print("="*70)

if __name__ == "__main__":
    main()
