"""
🧬🚀 COMPREHENSIVE ML ANALYSIS: 1,000-RUN GPU DATASET
=====================================================

Analyzes 100,000 evolutionary data points to discover:
1. Gene importance ranking (all 64 genes)
2. Convergence speed prediction
3. Critical genes investigation (40-41 + others)
4. Phase transition detection
5. Trajectory clustering
6. Outlier analysis (the 1 periodic run)
7. Gene interaction effects

Dataset: chaos_unified_GPU_1000runs_20251031_000616.json
Key insight: 99.9% convergent - focus on precision and mechanism discovery
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Try PyTorch
try:
    import torch
    HAS_PYTORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ PyTorch + GPU: {DEVICE}")
except:
    HAS_PYTORCH = False
    print("⚠️  PyTorch not available")

# ==================== DATA LOADING ====================

def load_1000run_dataset(filepath=r"c:\Users\akbon\OneDrive\Documents\PRICE-DETECTION-TEST-1\PRICE-DETECTION-TEST-1\prisoner_dilemma_64gene\chaos_unified_GPU_1000runs_20251031_000616.json"):
    """Load the 1,000-run GPU dataset"""
    print(f"\n📂 Loading 1,000-run dataset...")
    print(f"   File: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"   ✅ Loaded {len(data['runs'])} runs")
    print(f"   ✅ GPU accelerated: {data['metadata'].get('gpu_accelerated', False)}")
    
    return data

# ==================== FEATURE EXTRACTION ====================

def extract_features_1000runs(data, early_gens=20):
    """
    Extract ML features from 1,000 runs
    
    Focus on:
    - Initial gene frequencies (first 20 generations)
    - Early fitness trajectory 
    - Diversity metrics
    """
    
    features = []
    targets = {
        'convergence_time': [],
        'final_fitness': [],
        'behavior': [],
        'run_id': []
    }
    
    print(f"\n🔧 Extracting features from {len(data['runs'])} runs...")
    
    for run in data['runs']:
        # Gene frequencies (first 20 generations, all 64 genes)
        gene_freq_matrix = np.array(run['gene_frequency_matrix'][:early_gens])  # Shape: (20, 64)
        gene_freq_mean = gene_freq_matrix.mean(axis=0)  # 64 features
        gene_freq_std = gene_freq_matrix.std(axis=0)    # 64 features
        
        # Fitness trajectory (first 20 generations)
        fitness_traj = run['fitness_trajectory'][:early_gens]
        fitness_features = [
            np.mean(fitness_traj),
            np.std(fitness_traj),
            fitness_traj[-1] - fitness_traj[0],  # trend
            np.max(fitness_traj),
            np.min(fitness_traj)
        ]
        
        # Diversity (first 20 generations)
        diversity = run['diversity_history'][:early_gens]
        div_features = [
            np.mean([d['fitness_std'] for d in diversity]),
            np.mean([d['gene_entropy'] for d in diversity]),
            np.mean([d['avg_hamming_distance'] for d in diversity])
        ]
        
        # Combine all features
        feature_vector = np.concatenate([
            gene_freq_mean,      # 64 features
            gene_freq_std,       # 64 features  
            fitness_features,    # 5 features
            div_features         # 3 features
        ])  # Total: 136 features
        
        features.append(feature_vector)
        
        # Targets
        chaos_analysis = run['chaos_analysis']
        targets['convergence_time'].append(chaos_analysis.get('convergence_time', 0))
        targets['final_fitness'].append(chaos_analysis['final_fitness'])
        targets['behavior'].append(chaos_analysis['behavior'])
        targets['run_id'].append(run['run_id'])
    
    X = np.array(features)
    print(f"   ✅ Feature matrix: {X.shape}")
    print(f"   ✅ Features per run: {X.shape[1]}")
    
    return X, targets

# ==================== ANALYSIS 1: GENE IMPORTANCE RANKING ====================

def analyze_gene_importance(X, targets):
    """
    Rank all 64 genes by importance for predicting convergence speed
    """
    print("\n" + "="*70)
    print("📊 ANALYSIS 1: GENE IMPORTANCE RANKING")
    print("="*70)
    print("\nGoal: Identify which of the 64 genes are critical for cooperation")
    
    # Predict convergence time
    y = np.array(targets['convergence_time'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    print("\n🌳 Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n✅ Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: {mae:.2f} generations")
    print(f"   RMSE: {np.sqrt(mse):.2f} generations")
    
    # Feature importance
    importances = rf.feature_importances_
    
    # Gene importance (first 64 features are gene_freq_mean)
    gene_importances = importances[:64]
    gene_ranking = np.argsort(gene_importances)[::-1]
    
    print(f"\n🧬 TOP 20 MOST IMPORTANT GENES:")
    header = f"{'Rank':<6} {'Gene':<8} {'Importance':<12} {'Game State':<30}"
    print(header)
    print("-" * 70)
    
    for rank, gene_idx in enumerate(gene_ranking[:20], 1):
        importance = gene_importances[gene_idx]
        
        # Decode gene position to game state
        you = format(gene_idx, '06b')[:3]  # First 3 bits
        opp = format(gene_idx, '06b')[3:]  # Last 3 bits
        you_str = ''.join(['C' if b == '1' else 'D' for b in you])
        opp_str = ''.join(['C' if b == '1' else 'D' for b in opp])
        game_state = f"You:{you_str} Opp:{opp_str}"
        
        print(f"{rank:<6} {gene_idx:<8} {importance:<12.4f} {game_state:<30}")
    
    # Special focus on genes 40-41
    print(f"\n🔍 GENES 40-41 (from previous analysis):")
    print(f"   Gene 40: Importance = {gene_importances[40]:.4f} (Rank: {np.where(gene_ranking == 40)[0][0] + 1})")
    print(f"   Gene 41: Importance = {gene_importances[41]:.4f} (Rank: {np.where(gene_ranking == 41)[0][0] + 1})")
    
    return rf, gene_importances, gene_ranking

# ==================== ANALYSIS 2: CONVERGENCE SPEED PATTERNS ====================

def analyze_convergence_speed(X, targets, gene_importances):
    """
    Analyze patterns in convergence speed
    """
    print("\n" + "="*70)
    print("⏱️  ANALYSIS 2: CONVERGENCE SPEED PREDICTION")
    print("="*70)
    
    conv_times = np.array(targets['convergence_time'])
    
    print(f"\n📊 Convergence Time Statistics:")
    print(f"   Mean: {np.mean(conv_times):.2f} generations")
    print(f"   Median: {np.median(conv_times):.2f} generations")
    print(f"   Std: {np.std(conv_times):.2f} generations")
    print(f"   Min: {np.min(conv_times)} generations")
    print(f"   Max: {np.max(conv_times)} generations")
    
    # Categorize speeds
    fast = conv_times < np.percentile(conv_times, 25)
    medium = (conv_times >= np.percentile(conv_times, 25)) & (conv_times <= np.percentile(conv_times, 75))
    slow = conv_times > np.percentile(conv_times, 75)
    
    print(f"\n🏃 Speed Categories:")
    print(f"   Fast (<25th percentile): {np.sum(fast)} runs ({np.sum(fast)/len(conv_times)*100:.1f}%)")
    print(f"   Medium (25-75th): {np.sum(medium)} runs ({np.sum(medium)/len(conv_times)*100:.1f}%)")
    print(f"   Slow (>75th): {np.sum(slow)} runs ({np.sum(slow)/len(conv_times)*100:.1f}%)")
    
    # Correlation with top genes
    print(f"\n🔗 Correlation with Top Genes:")
    top_genes = np.argsort(gene_importances[:64])[::-1][:10]
    for gene_idx in top_genes:
        gene_freq = X[:, gene_idx]
        corr, pval = pearsonr(gene_freq, conv_times)
        print(f"   Gene {gene_idx}: r={corr:+.3f}, p={pval:.4f}")
    
    return conv_times

# ==================== ANALYSIS 3: OUTLIER INVESTIGATION ====================

def investigate_outlier(data, targets):
    """
    Deep dive into the 1 periodic run (0.1% outlier)
    """
    print("\n" + "="*70)
    print("🔍 ANALYSIS 3: OUTLIER INVESTIGATION (The 1 Periodic Run)")
    print("="*70)
    
    # Find the periodic run
    periodic_indices = [i for i, behavior in enumerate(targets['behavior']) if behavior == 'periodic']
    
    if len(periodic_indices) == 0:
        print("\n❌ No periodic runs found in this dataset!")
        return None
    
    print(f"\n📊 Found {len(periodic_indices)} periodic run(s)")
    
    for idx in periodic_indices:
        run_id = targets['run_id'][idx]
        run = data['runs'][idx]
        
        print(f"\n🎯 Periodic Run #{run_id}:")
        print(f"   Final Fitness: {run['chaos_analysis']['final_fitness']:.2f}")
        print(f"   Convergence Time: {run['chaos_analysis']['convergence_time']}")
        print(f"   Lyapunov Exponent: {run['chaos_analysis'].get('lyapunov_exponent', 'N/A')}")
        
        # Gene frequency analysis
        gene_freq_matrix = np.array(run['gene_frequency_matrix'])
        unique_patterns = []
        for gen_idx in [0, 25, 50, 75, 99]:
            pattern = gene_freq_matrix[gen_idx]
            print(f"\n   Generation {gen_idx} Gene Frequencies:")
            print(f"      Mean: {np.mean(pattern):.3f}")
            print(f"      Std: {np.std(pattern):.3f}")
            print(f"      Genes with high cooperation (>0.7): {np.sum(pattern > 0.7)}")
            print(f"      Genes with high defection (<0.3): {np.sum(pattern < 0.3)}")
        
        # Compare with typical convergent run
        convergent_indices = [i for i, b in enumerate(targets['behavior']) if b == 'convergent']
        typical_run = data['runs'][convergent_indices[0]]
        
        print(f"\n📊 Comparison with Typical Convergent Run:")
        print(f"   Periodic Final Fitness: {run['chaos_analysis']['final_fitness']:.2f}")
        print(f"   Convergent Final Fitness: {typical_run['chaos_analysis']['final_fitness']:.2f}")
        print(f"   Difference: {abs(run['chaos_analysis']['final_fitness'] - typical_run['chaos_analysis']['final_fitness']):.2f}")
    
    return periodic_indices

# ==================== MAIN EXECUTION ====================

def main():
    print("\n" + "="*70)
    print("🧬🚀 COMPREHENSIVE ML ANALYSIS: 1,000-RUN DATASET")
    print("="*70)
    print("\nDataset: 100,000 evolutionary data points")
    print("Focus: 99.9% convergent - precision mechanism discovery")
    print("="*70)
    
    # Load data
    data = load_1000run_dataset()
    
    # Extract features
    X, targets = extract_features_1000runs(data)
    
    # Analysis 1: Gene importance
    rf_model, gene_importances, gene_ranking = analyze_gene_importance(X, targets)
    
    # Analysis 2: Convergence speed
    conv_times = analyze_convergence_speed(X, targets, gene_importances)
    
    # Analysis 3: Outlier investigation
    outlier_info = investigate_outlier(data, targets)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'dataset': 'chaos_unified_GPU_1000runs_20251031_000616.json',
        'num_runs': len(data['runs']),
        'gene_importance': {
            'gene_indices': gene_ranking[:20].tolist(),
            'importances': gene_importances[gene_ranking[:20]].tolist()
        },
        'convergence_stats': {
            'mean': float(np.mean(conv_times)),
            'median': float(np.median(conv_times)),
            'std': float(np.std(conv_times))
        },
        'outliers': {
            'num_periodic': len([b for b in targets['behavior'] if b == 'periodic']),
            'num_chaotic': len([b for b in targets['behavior'] if b == 'chaotic'])
        }
    }
    
    output_file = f'ml_analysis_1000runs_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*70)
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"="*70)
    print(f"\n💾 Results saved: {output_file}")
    print("\n🎯 KEY FINDINGS:")
    print(f"   • Top gene importance: Gene {gene_ranking[0]} ({gene_importances[gene_ranking[0]]:.4f})")
    print(f"   • Average convergence: {np.mean(conv_times):.1f} generations")
    print(f"   • Outliers found: {len([b for b in targets['behavior'] if b != 'convergent'])} runs")
    print("="*70)


if __name__ == "__main__":
    main()
