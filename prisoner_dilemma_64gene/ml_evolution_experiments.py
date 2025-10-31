"""
🧬🤖 ML-POWERED EVOLUTIONARY ANALYSIS
====================================

Applies deep learning to evolutionary game theory data to discover patterns
beyond traditional chaos analysis.

EXPERIMENTS:
1. **Outcome Prediction**: Predict final fitness from initial conditions (LSTM/Transformer)
2. **Regime Classification**: Classify trajectories as convergent/periodic/chaotic (RF/XGBoost)
3. **Trajectory Forecasting**: Predict evolution path from early generations (LSTM)
4. **Gene Importance**: Discover critical genes via attention & feature importance
5. **Latent Space Analysis**: Find hidden structure in gene evolution (Autoencoder)

INPUT: chaos_dataset_100runs_*.json (10,000 evolutionary data points)
OUTPUT: ml_results_*.json with predictions, classifications, visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import PyTorch for neural networks
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 PyTorch available! Using device: {DEVICE}")
except ImportError:
    HAS_PYTORCH = False
    print("⚠️  PyTorch not available. Will use only classical ML models.")

# ==================== DATA LOADING & PREPROCESSING ====================

def load_chaos_dataset(filepath):
    """Load the evolutionary chaos dataset."""
    print(f"\n📂 Loading dataset from: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"   ✅ Loaded {data['metadata']['num_runs']} runs")
    print(f"   ✅ {data['metadata']['generations_per_run']} generations per run")
    print(f"   ✅ Total: {data['metadata']['total_generations']} data points")
    
    return data

def load_chaos_results(filepath):
    """Load the chaos analysis results."""
    print(f"\n📂 Loading chaos results from: {filepath}")
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"   ✅ {len(results['lyapunov_exponents'])} Lyapunov exponents")
    
    # Calculate percentages from behavior_counts
    total = sum(results['behavior_counts'].values())
    convergent_pct = (results['behavior_counts']['convergent'] / total) * 100
    print(f"   ✅ Classification: {convergent_pct:.1f}% convergent")
    
    return results

def extract_features_from_run(run, early_gens=20):
    """Extract features from a single evolutionary run.
    
    Features include:
    - Initial fitness and diversity
    - Fitness statistics from first N generations (mean, std, trend)
    - Diversity metrics from first N generations
    - Gene frequency statistics
    
    Args:
        run: Single run from chaos dataset
        early_gens: Number of early generations to extract features from
    
    Returns:
        dict: Feature dictionary
    """
    fitness_trajectory = np.array(run['fitness_trajectory'][:early_gens])
    diversity_history = run['diversity_history'][:early_gens]
    
    # Extract diversity metrics as arrays
    gene_entropy = np.array([d['gene_entropy'] for d in diversity_history])
    hamming_dist = np.array([d['avg_hamming_distance'] for d in diversity_history])
    unique_strat = np.array([d['unique_strategies'] for d in diversity_history])
    
    # Gene frequency features (average over early generations)
    gene_freq_matrix = run['gene_frequency_matrix'][:early_gens]
    gene_freq_array = np.array(gene_freq_matrix)  # Shape: (early_gens, 64)
    gene_freq_mean = gene_freq_array.mean(axis=0)  # Average frequency per gene
    gene_freq_std = gene_freq_array.std(axis=0)    # Variability per gene
    
    features = {
        # Initial state
        'initial_fitness': fitness_trajectory[0],
        'initial_gene_entropy': gene_entropy[0],
        'initial_hamming_distance': hamming_dist[0],
        'initial_unique_strategies': unique_strat[0],
        
        # Fitness statistics (early generations)
        'fitness_mean': fitness_trajectory.mean(),
        'fitness_std': fitness_trajectory.std(),
        'fitness_min': fitness_trajectory.min(),
        'fitness_max': fitness_trajectory.max(),
        'fitness_trend': np.polyfit(range(len(fitness_trajectory)), fitness_trajectory, 1)[0],
        
        # Diversity statistics (gene entropy)
        'gene_entropy_mean': gene_entropy.mean(),
        'gene_entropy_std': gene_entropy.std(),
        'gene_entropy_min': gene_entropy.min(),
        'gene_entropy_max': gene_entropy.max(),
        'gene_entropy_trend': np.polyfit(range(len(gene_entropy)), gene_entropy, 1)[0],
        
        # Hamming distance statistics
        'hamming_mean': hamming_dist.mean(),
        'hamming_std': hamming_dist.std(),
        
        # Unique strategies statistics
        'unique_strat_mean': unique_strat.mean(),
        'unique_strat_std': unique_strat.std(),
        
        # Gene frequency statistics (64 features each)
        **{f'gene_freq_mean_{i}': gene_freq_mean[i] for i in range(64)},
        **{f'gene_freq_std_{i}': gene_freq_std[i] for i in range(64)},
    }
    
    return features

def extract_targets_from_run(run):
    """Extract prediction targets from a run.
    
    Targets:
    - Final fitness (regression)
    - Convergence time (regression)
    - Final diversity (regression)
    """
    fitness_trajectory = np.array(run['fitness_trajectory'])
    diversity_history = run['diversity_history']
    
    # Final diversity metrics
    final_gene_entropy = diversity_history[-1]['gene_entropy']
    
    # Find convergence time (when fitness stabilizes)
    convergence_time = len(fitness_trajectory)
    for i in range(10, len(fitness_trajectory)):
        window = fitness_trajectory[i-10:i]
        if window.std() < 10:  # Fitness variance < 10 = converged
            convergence_time = i
            break
    
    return {
        'final_fitness': fitness_trajectory[-1],
        'convergence_time': convergence_time,
        'final_diversity': final_gene_entropy
    }

def prepare_ml_dataset(chaos_data, chaos_results, early_gens=20):
    """Prepare dataset for ML experiments.
    
    Returns:
        X: Feature matrix (n_runs, n_features)
        y_reg: Regression targets dict
        y_class: Classification labels (0=convergent, 1=periodic, 2=chaotic)
        feature_names: List of feature names
    """
    print(f"\n🔧 Preparing ML dataset...")
    print(f"   Using first {early_gens} generations as features")
    
    X_list = []
    y_final_fitness = []
    y_convergence_time = []
    y_final_diversity = []
    y_labels = []
    
    behaviors = chaos_results['behaviors']
    
    for i, run in enumerate(chaos_data['runs']):
        # Extract features
        features = extract_features_from_run(run, early_gens)
        X_list.append(list(features.values()))
        
        # Extract targets
        targets = extract_targets_from_run(run)
        y_final_fitness.append(targets['final_fitness'])
        y_convergence_time.append(targets['convergence_time'])
        y_final_diversity.append(targets['final_diversity'])
        
        # Classification label
        behavior = behaviors[i]
        if behavior == 'convergent':
            y_labels.append(0)
        elif behavior == 'periodic':
            y_labels.append(1)
        else:  # chaotic
            y_labels.append(2)
    
    X = np.array(X_list)
    feature_names = list(extract_features_from_run(chaos_data['runs'][0], early_gens).keys())
    
    y_reg = {
        'final_fitness': np.array(y_final_fitness),
        'convergence_time': np.array(y_convergence_time),
        'final_diversity': np.array(y_final_diversity)
    }
    
    y_class = np.array(y_labels)
    
    print(f"   ✅ X shape: {X.shape}")
    print(f"   ✅ Features: {len(feature_names)}")
    print(f"   ✅ Samples: {len(X)}")
    print(f"   ✅ Class distribution:")
    unique, counts = np.unique(y_class, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = ['convergent', 'periodic', 'chaotic'][label]
        print(f"      - {label_name}: {count} ({count/len(y_class)*100:.1f}%)")
    
    return X, y_reg, y_class, feature_names

# ==================== EXPERIMENT 1: OUTCOME PREDICTION ====================

def experiment_outcome_prediction(X, y_reg, feature_names):
    """Experiment 1: Predict final fitness from initial conditions.
    
    Uses Random Forest to predict:
    - Final fitness
    - Convergence time
    - Final diversity
    
    Returns feature importance ranking.
    """
    print("\n" + "="*70)
    print("🎯 EXPERIMENT 1: OUTCOME PREDICTION")
    print("="*70)
    print("Goal: Predict final evolutionary outcomes from initial conditions")
    print("Model: Random Forest Regressor")
    
    results = {}
    
    # Split data
    X_train, X_test, y_train_fit, y_test_fit = train_test_split(
        X, y_reg['final_fitness'], test_size=0.2, random_state=42
    )
    
    X_train, X_test, y_train_conv, y_test_conv = train_test_split(
        X, y_reg['convergence_time'], test_size=0.2, random_state=42
    )
    
    X_train, X_test, y_train_div, y_test_div = train_test_split(
        X, y_reg['final_diversity'], test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Predict Final Fitness ---
    print("\n📊 Predicting final fitness...")
    rf_fitness = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_fitness.fit(X_train_scaled, y_train_fit)
    
    y_pred_fit = rf_fitness.predict(X_test_scaled)
    mse_fit = mean_squared_error(y_test_fit, y_pred_fit)
    r2_fit = r2_score(y_test_fit, y_pred_fit)
    
    print(f"   MSE: {mse_fit:.2f}")
    print(f"   R²:  {r2_fit:.4f}")
    print(f"   RMSE: {np.sqrt(mse_fit):.2f}")
    
    results['final_fitness'] = {
        'mse': float(mse_fit),
        'r2': float(r2_fit),
        'rmse': float(np.sqrt(mse_fit)),
        'predictions': y_pred_fit.tolist(),
        'actual': y_test_fit.tolist()
    }
    
    # --- Predict Convergence Time ---
    print("\n⏱️  Predicting convergence time...")
    rf_conv = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_conv.fit(X_train_scaled, y_train_conv)
    
    y_pred_conv = rf_conv.predict(X_test_scaled)
    mse_conv = mean_squared_error(y_test_conv, y_pred_conv)
    r2_conv = r2_score(y_test_conv, y_pred_conv)
    
    print(f"   MSE: {mse_conv:.2f}")
    print(f"   R²:  {r2_conv:.4f}")
    print(f"   RMSE: {np.sqrt(mse_conv):.2f}")
    
    results['convergence_time'] = {
        'mse': float(mse_conv),
        'r2': float(r2_conv),
        'rmse': float(np.sqrt(mse_conv))
    }
    
    # --- Predict Final Diversity ---
    print("\n🌈 Predicting final diversity...")
    rf_div = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_div.fit(X_train_scaled, y_train_div)
    
    y_pred_div = rf_div.predict(X_test_scaled)
    mse_div = mean_squared_error(y_test_div, y_pred_div)
    r2_div = r2_score(y_test_div, y_pred_div)
    
    print(f"   MSE: {mse_div:.2f}")
    print(f"   R²:  {r2_div:.4f}")
    print(f"   RMSE: {np.sqrt(mse_div):.2f}")
    
    results['final_diversity'] = {
        'mse': float(mse_div),
        'r2': float(r2_div),
        'rmse': float(np.sqrt(mse_div))
    }
    
    # --- Feature Importance ---
    print("\n🔍 Feature importance analysis...")
    importance = rf_fitness.feature_importances_
    
    # Top 20 most important features
    indices = np.argsort(importance)[::-1][:20]
    top_features = [(feature_names[i], importance[i]) for i in indices]
    
    print("\n   Top 20 most important features:")
    for i, (name, imp) in enumerate(top_features, 1):
        print(f"      {i:2d}. {name:30s} = {imp:.4f}")
    
    results['feature_importance'] = {
        'all': {feature_names[i]: float(importance[i]) for i in range(len(feature_names))},
        'top_20': [{'feature': name, 'importance': float(imp)} for name, imp in top_features]
    }
    
    return results

# ==================== EXPERIMENT 2: REGIME CLASSIFICATION ====================

def experiment_regime_classification(X, y_class, feature_names):
    """Experiment 2: Classify evolutionary regimes.
    
    Classify trajectories as:
    - 0: Convergent (stable)
    - 1: Periodic (oscillating)
    - 2: Chaotic (unpredictable)
    
    Uses Random Forest Classifier.
    """
    print("\n" + "="*70)
    print("🏷️  EXPERIMENT 2: REGIME CLASSIFICATION")
    print("="*70)
    print("Goal: Classify evolutionary trajectories by behavior type")
    print("Model: Random Forest Classifier")
    print("Classes: 0=Convergent, 1=Periodic, 2=Chaotic")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    print("\n🌳 Training Random Forest classifier...")
    rf_class = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_class.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf_class.predict(X_test_scaled)
    accuracy = (y_pred == y_test).mean()
    
    print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print("\n📋 Classification Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=['Convergent', 'Periodic', 'Chaotic'],
        zero_division=0
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n🔢 Confusion Matrix:")
    print("                 Predicted")
    print("              Conv  Peri  Chao")
    print(f"   Conv       {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
    if cm.shape[0] > 1:
        print(f"   Peri       {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
    if cm.shape[0] > 2:
        print(f"   Chao       {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")
    
    # Feature importance
    importance = rf_class.feature_importances_
    indices = np.argsort(importance)[::-1][:20]
    top_features = [(feature_names[i], importance[i]) for i in indices]
    
    print("\n🔍 Top 20 discriminative features:")
    for i, (name, imp) in enumerate(top_features, 1):
        print(f"   {i:2d}. {name:30s} = {imp:.4f}")
    
    results = {
        'accuracy': float(accuracy),
        'predictions': y_pred.tolist(),
        'actual': y_test.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'feature_importance': {
            'top_20': [{'feature': name, 'importance': float(imp)} for name, imp in top_features]
        }
    }
    
    return results

# ==================== EXPERIMENT 3: TRAJECTORY FORECASTING (LSTM) ====================

class EvolutionLSTM(nn.Module):
    """LSTM for forecasting evolutionary trajectories."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size)
        
        # Use last time step
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        
        return prediction.squeeze()

def prepare_sequence_data(chaos_data, lookback=20, forecast_horizon=1):
    """Prepare sequential data for LSTM.
    
    Args:
        chaos_data: Chaos dataset
        lookback: Number of past generations to use
        forecast_horizon: How many generations ahead to predict
    
    Returns:
        X_seq: Input sequences (n_samples, lookback, features)
        y_seq: Target values (n_samples,)
    """
    X_seq_list = []
    y_seq_list = []
    
    for run in chaos_data['runs']:
        fitness_trajectory = np.array(run['fitness_trajectory'])
        diversity_history = run['diversity_history']
        
        # Extract gene entropy as diversity metric
        gene_entropy = np.array([d['gene_entropy'] for d in diversity_history])
        
        # Create sequences
        for i in range(lookback, len(fitness_trajectory) - forecast_horizon):
            # Input: fitness + diversity for past 'lookback' generations
            fitness_seq = fitness_trajectory[i-lookback:i]
            entropy_seq = gene_entropy[i-lookback:i]
            
            # Stack as features
            seq = np.column_stack([fitness_seq, entropy_seq])
            X_seq_list.append(seq)
            
            # Target: fitness at t + forecast_horizon
            y_seq_list.append(fitness_trajectory[i + forecast_horizon])
    
    X_seq = np.array(X_seq_list)
    y_seq = np.array(y_seq_list)
    
    return X_seq, y_seq

def experiment_trajectory_forecasting(chaos_data):
    """Experiment 3: Forecast evolutionary trajectories using LSTM.
    
    Predicts future fitness values from past fitness + diversity sequences.
    """
    if not HAS_PYTORCH:
        print("\n⚠️  EXPERIMENT 3 SKIPPED: PyTorch not available")
        return {'skipped': True, 'reason': 'PyTorch not available'}
    
    print("\n" + "="*70)
    print("📈 EXPERIMENT 3: TRAJECTORY FORECASTING")
    print("="*70)
    print("Goal: Predict future fitness from past trajectory")
    print("Model: LSTM Neural Network")
    
    # Prepare sequence data
    print("\n🔧 Preparing sequence data...")
    lookback = 20
    forecast_horizon = 1
    
    X_seq, y_seq = prepare_sequence_data(chaos_data, lookback, forecast_horizon)
    
    print(f"   ✅ X_seq shape: {X_seq.shape}")
    print(f"   ✅ y_seq shape: {y_seq.shape}")
    print(f"   ✅ Lookback: {lookback} generations")
    print(f"   ✅ Forecast horizon: {forecast_horizon} generations")
    
    # Split data
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # Scale targets
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(DEVICE)
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(DEVICE)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    input_size = X_seq.shape[2]  # 2 features: fitness + diversity
    model = EvolutionLSTM(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🏗️  LSTM Architecture:")
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: 64")
    print(f"   Num layers: 2")
    print(f"   Device: {DEVICE}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Training
    print("\n🚂 Training LSTM...")
    epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_tensor)
            val_loss = criterion(val_pred, y_test_tensor).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
    
    # Unscale predictions
    y_pred = y_pred_scaled * y_std + y_mean
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n✅ Final Results:")
    print(f"   MSE:  {mse:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²:   {r2:.4f}")
    
    results = {
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'lookback': lookback,
        'forecast_horizon': forecast_horizon,
        'predictions_sample': y_pred[:100].tolist(),
        'actual_sample': y_test[:100].tolist()
    }
    
    return results

# ==================== MAIN EXPERIMENT RUNNER ====================

def run_all_experiments(chaos_dataset_path, chaos_results_path, output_dir='outputs/ml_evolution'):
    """Run all ML experiments on evolutionary data."""
    
    print("\n" + "="*70)
    print("🧬🤖 ML-POWERED EVOLUTIONARY ANALYSIS")
    print("="*70)
    print(f"Chaos Dataset: {chaos_dataset_path}")
    print(f"Chaos Results: {chaos_results_path}")
    print(f"Output Directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    chaos_data = load_chaos_dataset(chaos_dataset_path)
    chaos_results = load_chaos_results(chaos_results_path)
    
    # Prepare ML dataset
    X, y_reg, y_class, feature_names = prepare_ml_dataset(chaos_data, chaos_results, early_gens=20)
    
    # Run experiments
    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'chaos_dataset': str(chaos_dataset_path),
            'chaos_results': str(chaos_results_path),
            'n_samples': len(X),
            'n_features': len(feature_names)
        },
        'experiments': {}
    }
    
    # Experiment 1: Outcome Prediction
    exp1_results = experiment_outcome_prediction(X, y_reg, feature_names)
    all_results['experiments']['outcome_prediction'] = exp1_results
    
    # Experiment 2: Regime Classification
    exp2_results = experiment_regime_classification(X, y_class, feature_names)
    all_results['experiments']['regime_classification'] = exp2_results
    
    # Experiment 3: Trajectory Forecasting (LSTM)
    exp3_results = experiment_trajectory_forecasting(chaos_data)
    all_results['experiments']['trajectory_forecasting'] = exp3_results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"ml_evolution_results_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Generate visualizations
    generate_visualizations(all_results, output_dir, timestamp)
    
    return all_results

def generate_visualizations(results, output_dir, timestamp):
    """Generate visualization plots for ML results."""
    
    print("\n📊 Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🧬 ML-Powered Evolutionary Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Feature Importance (Outcome Prediction)
    ax1 = axes[0, 0]
    exp1 = results['experiments']['outcome_prediction']
    top_features = exp1['feature_importance']['top_20'][:10]  # Top 10
    
    feature_names = [f['feature'] for f in top_features]
    importances = [f['importance'] for f in top_features]
    
    y_pos = np.arange(len(feature_names))
    ax1.barh(y_pos, importances, color='steelblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feature_names, fontsize=8)
    ax1.set_xlabel('Importance')
    ax1.set_title('Top 10 Features (Outcome Prediction)')
    ax1.invert_yaxis()
    
    # Plot 2: Prediction Accuracy (Outcome Prediction)
    ax2 = axes[0, 1]
    metrics = ['Final Fitness', 'Convergence Time', 'Final Diversity']
    r2_scores = [
        exp1['final_fitness']['r2'],
        exp1['convergence_time']['r2'],
        exp1['final_diversity']['r2']
    ]
    
    colors = ['green' if r2 > 0.5 else 'orange' if r2 > 0.3 else 'red' for r2 in r2_scores]
    bars = ax2.bar(metrics, r2_scores, color=colors, alpha=0.7)
    ax2.set_ylabel('R² Score')
    ax2.set_title('Prediction Accuracy (R² Score)')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    for bar, r2 in zip(bars, r2_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Confusion Matrix (Regime Classification)
    ax3 = axes[1, 0]
    exp2 = results['experiments']['regime_classification']
    cm = np.array(exp2['confusion_matrix'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Conv', 'Peri', 'Chao'],
                yticklabels=['Conv', 'Peri', 'Chao'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title(f'Regime Classification (Acc: {exp2["accuracy"]*100:.1f}%)')
    
    # Plot 4: Trajectory Forecasting Performance
    ax4 = axes[1, 1]
    exp3 = results['experiments']['trajectory_forecasting']
    
    if 'skipped' not in exp3:
        # Plot actual vs predicted for sample trajectories
        actual = np.array(exp3['actual_sample'][:50])
        predicted = np.array(exp3['predictions_sample'][:50])
        
        ax4.plot(actual, label='Actual', marker='o', markersize=3, alpha=0.7)
        ax4.plot(predicted, label='Predicted', marker='x', markersize=3, alpha=0.7)
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('Fitness')
        ax4.set_title(f'Trajectory Forecasting (R²: {exp3["r2"]:.3f})')
        ax4.legend()
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '⚠️ LSTM Experiment Skipped\n(PyTorch not available)',
                ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.axis('off')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"ml_evolution_visualization_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved visualization: {output_path}")
    
    plt.close()

# ==================== ENTRY POINT ====================

if __name__ == '__main__':
    import sys
    
    # Use most recent chaos dataset by default
    chaos_dir = Path(__file__).parent
    
    # Find most recent chaos dataset
    chaos_datasets = list(chaos_dir.glob('chaos_dataset_*.json'))
    if not chaos_datasets:
        print("❌ Error: No chaos datasets found!")
        print("   Run chaos_data_collection.py first to generate data.")
        sys.exit(1)
    
    chaos_dataset = sorted(chaos_datasets)[-1]
    
    # Find corresponding results (look for any matching date/time prefix)
    dataset_timestamp = chaos_dataset.stem.replace('chaos_dataset_100runs_', '')
    date_part = dataset_timestamp.split('_')[0]  # e.g., 20251030
    
    # Find closest chaos_results file by date
    chaos_results_files = list(chaos_dir.glob(f'chaos_results_{date_part}_*.json'))
    
    if not chaos_results_files:
        print(f"❌ Error: No chaos results found for date {date_part}")
        print("   Run chaos_analysis.py first to generate results.")
        sys.exit(1)
    
    chaos_results = sorted(chaos_results_files)[-1]  # Most recent for that date
    
    print(f"📂 Found chaos dataset: {chaos_dataset.name}")
    print(f"📂 Found chaos results: {chaos_results.name}")
    
    # Run all experiments
    results = run_all_experiments(
        chaos_dataset_path=chaos_dataset,
        chaos_results_path=chaos_results,
        output_dir='outputs/ml_evolution'
    )
    
    print("\n" + "="*70)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\n🎯 KEY FINDINGS:")
    print(f"   Outcome Prediction R²: {results['experiments']['outcome_prediction']['final_fitness']['r2']:.3f}")
    print(f"   Regime Classification Accuracy: {results['experiments']['regime_classification']['accuracy']*100:.1f}%")
    
    if 'skipped' not in results['experiments']['trajectory_forecasting']:
        print(f"   Trajectory Forecasting R²: {results['experiments']['trajectory_forecasting']['r2']:.3f}")
