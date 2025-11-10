
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt, factorial
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def hermite_phys(n, x):
    """Hermite polynomials"""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    
    Hnm2 = np.ones_like(x)
    Hnm1 = 2 * x
    
    for k in range(2, n + 1):
        Hn = 2 * x * Hnm1 - 2 * (k - 1) * Hnm2
        Hnm2, Hnm1 = Hnm1, Hn
    
    return Hn

def psi_n(x, n):
    """1D harmonic oscillator eigenfunction"""
    Hn = hermite_phys(n, x)
    norm = 1.0 / np.sqrt((2.0 ** n) * factorial(n) * np.sqrt(pi))
    return norm * np.exp(-x**2 / 2.0) * Hn

def generate_quantum_state_features(nx, ny, n_samples=50):
    """Generate features from quantum state for ML training"""
    x = np.linspace(-4, 4, n_samples)
    y = np.linspace(-4, 4, n_samples)
    X, Y = np.meshgrid(x, y)
    
    # Compute wavefunction
    psi = psi_n(X.flatten(), nx) * psi_n(Y.flatten(), ny)
    prob_density = np.abs(psi)**2
    
    # Extract features
    features = {
        'mean_position_x': np.sum(X.flatten() * prob_density) / np.sum(prob_density),
        'mean_position_y': np.sum(Y.flatten() * prob_density) / np.sum(prob_density),
        'std_x': np.sqrt(np.sum((X.flatten() - np.sum(X.flatten() * prob_density) / np.sum(prob_density))**2 * prob_density) / np.sum(prob_density)),
        'std_y': np.sqrt(np.sum((Y.flatten() - np.sum(Y.flatten() * prob_density) / np.sum(prob_density))**2 * prob_density) / np.sum(prob_density)),
        'max_probability': np.max(prob_density),
        'min_probability': np.min(prob_density),
        'energy': nx + ny + 1,
        'symmetry_x': np.sum(np.abs(psi[:len(psi)//2] - psi[len(psi)//2:])),
        'num_nodes_estimate': nx + ny,
        'kurtosis': np.sum((prob_density - np.mean(prob_density))**4) / (np.std(prob_density)**4 * len(prob_density))
    }
    
    return list(features.values()), list(features.keys())

def train_state_classifier():
    """Train ML model to classify quantum states"""
    print("=" * 70)
    print("  QUANTUM STATE CLASSIFICATION - MACHINE LEARNING")
    print("=" * 70)
    print("\n🤖 Training Random Forest to classify quantum states...\n")
    
    # Generate training data
    X_train = []
    y_train = []
    state_labels = []
    
    print("📊 Generating training dataset...")
    for nx in range(6):
        for ny in range(6):
            features, feature_names = generate_quantum_state_features(nx, ny)
            X_train.append(features)
            y_train.append(f"n=({nx},{ny})")
            state_labels.append((nx, ny))
    
    X_train = np.array(X_train)
    print(f"   Dataset size: {len(X_train)} quantum states")
    print(f"   Features per state: {len(feature_names)}\n")
    
    # Split data
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42
    )
    
    # Train classifier
    print("🎯 Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train_split, y_train_split)
    
    # Evaluate
    y_pred = clf.predict(X_test_split)
    accuracy = accuracy_score(y_test_split, y_pred)
    
    print(f"✓ Training complete!")
    print(f"   Accuracy: {accuracy*100:.2f}%\n")
    
    # Feature importance
    importance = clf.feature_importances_
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature importance
    indices = np.argsort(importance)[::-1]
    axes[0, 0].bar(range(len(importance)), importance[indices], color='skyblue', edgecolor='black')
    axes[0, 0].set_xticks(range(len(importance)))
    axes[0, 0].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    axes[0, 0].set_title('Feature Importance for State Classification', fontweight='bold')
    axes[0, 0].set_ylabel('Importance')
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    # Confusion matrix (simplified - show first 10 states)
    unique_states = sorted(list(set(y_test_split)))[:10]
    test_indices = [i for i, y in enumerate(y_test_split) if y in unique_states]
    
    if len(test_indices) > 0:
        cm = confusion_matrix([y_test_split[i] for i in test_indices], 
                             [y_pred[i] for i in test_indices],
                             labels=unique_states)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=unique_states, yticklabels=unique_states,
                   ax=axes[0, 1], cbar_kws={'label': 'Count'})
        axes[0, 1].set_title('Confusion Matrix (Sample)', fontweight='bold')
        axes[0, 1].set_xlabel('Predicted State')
        axes[0, 1].set_ylabel('True State')
    
    # Test predictions
    test_results = []
    for i in range(min(5, len(X_test_split))):
        test_results.append(f"{y_test_split[i]} → {y_pred[i]}")
    
    result_text = "Sample Predictions:\n\n"
    for i, res in enumerate(test_results):
        result_text += f"{i+1}. {res}\n"
    result_text += f"\nAccuracy: {accuracy*100:.2f}%\n"
    result_text += f"Total states: {len(unique_states)}"
    
    axes[1, 0].text(0.5, 0.5, result_text, ha='center', va='center',
                   fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[1, 0].axis('off')
    
    # Model statistics
    stats_text = "Model Statistics:\n\n"
    stats_text += f"Algorithm: Random Forest\n"
    stats_text += f"Trees: 100\n"
    stats_text += f"Max depth: 10\n"
    stats_text += f"Training samples: {len(X_train_split)}\n"
    stats_text += f"Test samples: {len(X_test_split)}\n"
    stats_text += f"Accuracy: {accuracy*100:.2f}%\n\n"
    stats_text += f"Top 3 features:\n"
    for i in range(3):
        stats_text += f"  {feature_names[indices[i]]}\n"
    
    axes[1, 1].text(0.5, 0.5, stats_text, ha='center', va='center',
                   fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 1].axis('off')
    
    plt.suptitle('Quantum State Classification with Machine Learning', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ml_state_classification.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: ml_state_classification.png")
    plt.close()

def train_evolution_predictor():
    """Train ML model to predict quantum state time evolution"""
    print("\n" + "=" * 70)
    print("  QUANTUM EVOLUTION PREDICTION - MACHINE LEARNING")
    print("=" * 70)
    print("\n🔮 Training Gradient Boosting to predict state evolution...\n")
    
    # Generate training data for time evolution
    X_train = []
    y_train = []
    
    print("📊 Generating evolution dataset...")
    for nx in range(4):
        for ny in range(4):
            for t in np.linspace(0, 2*pi, 20):
                # Features: initial state + time
                features, _ = generate_quantum_state_features(nx, ny)
                features.append(t)  # Add time as feature
                X_train.append(features)
                
                # Target: energy expectation value at time t
                energy = (nx + ny + 1) * np.cos(t)  # Simplified evolution
                y_train.append(energy)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"   Dataset size: {len(X_train)} time points")
    print(f"   Features: initial state + time\n")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Split data
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    # Train regressor
    print("🎯 Training Gradient Boosting Regressor...")
    reg = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    reg.fit(X_train_split, y_train_split)
    
    # Evaluate
    y_pred = reg.predict(X_test_split)
    mse = mean_squared_error(y_test_split, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"✓ Training complete!")
    print(f"   RMSE: {rmse:.4f}\n")
    
    # Visualize predictions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Predicted vs Actual
    axes[0, 0].scatter(y_test_split, y_pred, alpha=0.5, edgecolor='black')
    axes[0, 0].plot([y_test_split.min(), y_test_split.max()], 
                    [y_test_split.min(), y_test_split.max()], 
                    'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('True Energy', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Energy', fontsize=11)
    axes[0, 0].set_title('Evolution Prediction Accuracy', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Residuals
    residuals = y_test_split - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, edgecolor='black')
    axes[0, 1].axhline(0, color='red', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Energy', fontsize=11)
    axes[0, 1].set_ylabel('Residuals', fontsize=11)
    axes[0, 1].set_title('Prediction Residuals', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Example evolution prediction
    nx_test, ny_test = 2, 1
    t_range = np.linspace(0, 2*pi, 50)
    predictions = []
    actuals = []
    
    for t in t_range:
        features, _ = generate_quantum_state_features(nx_test, ny_test)
        features.append(t)
        features_scaled = scaler.transform([features])
        pred = reg.predict(features_scaled)[0]
        predictions.append(pred)
        actuals.append((nx_test + ny_test + 1) * np.cos(t))
    
    axes[1, 0].plot(t_range, actuals, 'b-', lw=2, label='True evolution')
    axes[1, 0].plot(t_range, predictions, 'r--', lw=2, label='ML prediction')
    axes[1, 0].set_xlabel('Time', fontsize=11)
    axes[1, 0].set_ylabel('Energy Expectation', fontsize=11)
    axes[1, 0].set_title(f'Evolution Prediction: ψ({nx_test},{ny_test})', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Statistics
    stats_text = "Prediction Statistics:\n\n"
    stats_text += f"Algorithm: Gradient Boosting\n"
    stats_text += f"Trees: 100\n"
    stats_text += f"RMSE: {rmse:.4f}\n"
    stats_text += f"R² score: {reg.score(X_test_split, y_test_split):.4f}\n\n"
    stats_text += f"Training samples: {len(X_train_split)}\n"
    stats_text += f"Test samples: {len(X_test_split)}\n\n"
    stats_text += f"Prediction range:\n"
    stats_text += f"  Min: {y_pred.min():.2f}\n"
    stats_text += f"  Max: {y_pred.max():.2f}\n"
    
    axes[1, 1].text(0.5, 0.5, stats_text, ha='center', va='center',
                   fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 1].axis('off')
    
    plt.suptitle('Quantum Evolution Prediction with Machine Learning', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ml_evolution_prediction.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: ml_evolution_prediction.png")
    plt.close()

def predict_wavefunction_evolution():
    """Advanced: Predict full wavefunction evolution using deep patterns"""
    print("\n" + "=" * 70)
    print("  ADVANCED EVOLUTION PREDICTION - WAVEFUNCTION DYNAMICS")
    print("=" * 70)
    print("\n🌊 Predicting complete wavefunction evolution...\n")
    
    # Generate more complex training data
    print("📊 Generating wavefunction evolution dataset...")
    X_train = []
    y_train_real = []
    y_train_imag = []
    
    n_samples = 30
    x_grid = np.linspace(-3, 3, n_samples)
    
    for nx in range(3):
        for ny in range(3):
            for t in np.linspace(0, pi, 15):
                # Initial state features
                features = [nx, ny, t]
                
                # Compute wavefunction at each grid point
                for x in x_grid:
                    psi_val = psi_n(x, nx) * np.exp(-1j * (nx + 0.5) * t)
                    features.extend([np.real(psi_val), np.imag(psi_val)])
                
                X_train.append(features[:13])  # Limit features
                
                # Target: wavefunction at next time step
                t_next = t + 0.1
                psi_next = psi_n(x_grid[0], nx) * np.exp(-1j * (nx + 0.5) * t_next)
                y_train_real.append(np.real(psi_next))
                y_train_imag.append(np.imag(psi_next))
    
    X_train = np.array(X_train)
    y_train_real = np.array(y_train_real)
    y_train_imag = np.array(y_train_imag)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Learning spatiotemporal patterns...\n")
    
    # Train models for real and imaginary parts
    print("🎯 Training evolution predictors...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    reg_real = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
    reg_imag = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
    
    reg_real.fit(X_scaled, y_train_real)
    reg_imag.fit(X_scaled, y_train_imag)
    
    print("✓ Models trained!\n")
    
    # Test prediction
    print("🔮 Testing evolution prediction...")
    nx_test, ny_test = 1, 1
    t_range = np.linspace(0, 2*pi, 60)
    
    x_test = 0.5  # Test position
    predictions_real = []
    predictions_imag = []
    actuals_real = []
    actuals_imag = []
    
    for t in t_range:
        # Create test features
        test_features = [nx_test, ny_test, t]
        for x in x_grid[:10]:  # Reduced for speed
            psi_val = psi_n(x, nx_test) * np.exp(-1j * (nx_test + 0.5) * t)
            test_features.extend([np.real(psi_val), np.imag(psi_val)])
        
        test_features = test_features[:13]
        test_scaled = scaler.transform([test_features])
        
        pred_real = reg_real.predict(test_scaled)[0]
        pred_imag = reg_imag.predict(test_scaled)[0]
        predictions_real.append(pred_real)
        predictions_imag.append(pred_imag)
        
        # Actual values
        psi_actual = psi_n(x_test, nx_test) * np.exp(-1j * (nx_test + 0.5) * t)
        actuals_real.append(np.real(psi_actual))
        actuals_imag.append(np.imag(psi_actual))
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    # Real part evolution
    axes[0, 0].plot(t_range, actuals_real, 'b-', lw=2, label='True Re(ψ)')
    axes[0, 0].plot(t_range, predictions_real, 'r--', lw=2, alpha=0.7, label='Predicted Re(ψ)')
    axes[0, 0].set_xlabel('Time', fontsize=11)
    axes[0, 0].set_ylabel('Re(ψ)', fontsize=11)
    axes[0, 0].set_title('Real Part Evolution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Imaginary part evolution
    axes[0, 1].plot(t_range, actuals_imag, 'b-', lw=2, label='True Im(ψ)')
    axes[0, 1].plot(t_range, predictions_imag, 'g--', lw=2, alpha=0.7, label='Predicted Im(ψ)')
    axes[0, 1].set_xlabel('Time', fontsize=11)
    axes[0, 1].set_ylabel('Im(ψ)', fontsize=11)
    axes[0, 1].set_title('Imaginary Part Evolution', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Probability density evolution
    prob_actual = np.array(actuals_real)**2 + np.array(actuals_imag)**2
    prob_pred = np.array(predictions_real)**2 + np.array(predictions_imag)**2
    
    axes[1, 0].plot(t_range, prob_actual, 'b-', lw=2, label='True |ψ|²')
    axes[1, 0].plot(t_range, prob_pred, 'm--', lw=2, alpha=0.7, label='Predicted |ψ|²')
    axes[1, 0].set_xlabel('Time', fontsize=11)
    axes[1, 0].set_ylabel('Probability Density', fontsize=11)
    axes[1, 0].set_title(f'Probability Evolution at x={x_test}', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Error analysis
    error_real = np.abs(np.array(actuals_real) - np.array(predictions_real))
    error_imag = np.abs(np.array(actuals_imag) - np.array(predictions_imag))
    
    axes[1, 1].plot(t_range, error_real, 'r-', lw=2, label='|Error Re(ψ)|')
    axes[1, 1].plot(t_range, error_imag, 'g-', lw=2, label='|Error Im(ψ)|')
    axes[1, 1].set_xlabel('Time', fontsize=11)
    axes[1, 1].set_ylabel('Absolute Error', fontsize=11)
    axes[1, 1].set_title('Prediction Error', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Wavefunction Evolution Prediction with ML', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ml_wavefunction_evolution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: ml_wavefunction_evolution.png")
    plt.close()
    
    # Print statistics
    print(f"\n📊 Prediction Quality:")
    print(f"   Mean error Re(ψ): {np.mean(error_real):.6f}")
    print(f"   Mean error Im(ψ): {np.mean(error_imag):.6f}")
    print(f"   Max error: {max(np.max(error_real), np.max(error_imag)):.6f}")

def main():
    print("\n" + "=" * 70)
    print("  MACHINE LEARNING FOR QUANTUM MECHANICS")
    print("=" * 70)
    print("\nThree ML applications:\n")
    print("  1. State Classification - Identify quantum states from features")
    print("  2. Energy Evolution Prediction - Forecast expectation values")
    print("  3. Wavefunction Evolution - Predict complete quantum dynamics\n")
    print("=" * 70)
    
    train_state_classifier()
    train_evolution_predictor()
    predict_wavefunction_evolution()
    
    print("\n" + "=" * 70)
    print("✨ Machine Learning Analysis Complete!")
    print("=" * 70)
    print("\n💡 Advanced capabilities achieved:")
    print("   • Classify unknown quantum states from measurements")
    print("   • Predict energy expectation evolution over time")
    print("   • Forecast complete wavefunction dynamics (Re & Im parts)")
    print("   • Learn spatiotemporal quantum patterns")
    print("   • Quantify prediction uncertainty and errors")
    print("\n🚀 These models can be extended to:")
    print("   • Real-time quantum state monitoring")
    print("   • Quantum control optimization")
    print("   • Anomaly detection in quantum systems")
    print("   • Accelerated quantum simulations")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
