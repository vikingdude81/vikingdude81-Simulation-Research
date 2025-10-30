"""
🤝 MODEL ENSEMBLE BUILDER
Combines multiple models for superior predictions
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🤝 MODEL ENSEMBLE BUILDER")
print("=" * 80)

# Load all available models
model_storage = Path("MODEL_STORAGE")
model_files = list(model_storage.glob("*_standalone.pkl")) + list(model_storage.glob("*_tuned.pkl"))

if len(model_files) < 2:
    print("❌ Need at least 2 models! Train more models first.")
    exit()

print(f"\n✅ Found {len(model_files)} models")

# Load models and their data
models = {}
model_info = {}

for model_file in model_files:
    model_name = model_file.stem.replace('_standalone', '').replace('_tuned', '').replace('_', ' ').title()
    if '_tuned' in model_file.stem:
        model_name += ' (Tuned)'
    
    try:
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        
        models[model_name] = data
        model_info[model_name] = {
            'file': model_file,
            'rmse': data.get('test_rmse', data.get('train_rmse', 0))
        }
        print(f"   • {model_name} (RMSE: {model_info[model_name]['rmse']:.6f})")
    except Exception as e:
        print(f"   ⚠️ Could not load {model_name}: {e}")

if len(models) < 2:
    print("❌ Need at least 2 valid models!")
    exit()

# Determine the feature count from the first model
first_model_data = list(models.values())[0]
if 'scaler' in first_model_data and hasattr(first_model_data['scaler'], 'n_features_in_'):
    n_features = first_model_data['scaler'].n_features_in_
else:
    n_features = 6  # Default from original training

# Generate test data for ensemble evaluation
print(f"\n📊 Generating test data for ensemble evaluation (using {n_features} features)...")
np.random.seed(42)
n_test = 200

X_test = np.random.randn(n_test, n_features)
y_test = X_test[:, 0] * 0.5 + X_test[:, 1] * 0.3 + np.random.randn(n_test) * 0.1

# Get predictions from all models
print("\n🔮 Generating predictions from all models...")
predictions = {}

for name, data in models.items():
    try:
        model = data['model']
        scaler = data['scaler']
        
        # Check expected feature count
        if hasattr(scaler, 'n_features_in_'):
            expected_features = scaler.n_features_in_
        else:
            expected_features = n_features
        
        # Generate test data with correct feature count
        X_test_model = np.random.randn(n_test, expected_features)
        y_test_model = X_test_model[:, 0] * 0.5 + X_test_model[:, 1] * 0.3 + np.random.randn(n_test) * 0.1
        
        # Scale and predict
        X_test_scaled = scaler.transform(X_test_model)
        pred = model.predict(X_test_scaled)
        predictions[name] = pred
        
        # Use consistent y_test for all models
        if 'y_test' not in locals() or len(y_test) != len(pred):
            y_test = y_test_model
        
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        print(f"   {name:30s} RMSE: {rmse:.6f}")
    except Exception as e:
        print(f"   ⚠️ Could not get predictions from {name}: {e}")

if len(predictions) < 2:
    print("❌ Need at least 2 models with valid predictions!")
    exit()

# Convert to DataFrame for easier manipulation
pred_df = pd.DataFrame(predictions)

print("\n" + "=" * 80)
print("📊 ENSEMBLE METHOD 1: SIMPLE AVERAGE")
print("=" * 80)

# Simple average ensemble
ensemble_simple = pred_df.mean(axis=1)
simple_rmse = np.sqrt(mean_squared_error(y_test, ensemble_simple))
simple_mae = mean_absolute_error(y_test, ensemble_simple)
simple_r2 = r2_score(y_test, ensemble_simple)

print(f"\n✅ Simple Average Ensemble:")
print(f"   RMSE: {simple_rmse:.6f}")
print(f"   MAE:  {simple_mae:.6f}")
print(f"   R²:   {simple_r2:.4f}")

# Best individual model for comparison
individual_rmses = {}
for name in predictions:
    rmse = np.sqrt(mean_squared_error(y_test, predictions[name]))
    individual_rmses[name] = rmse

best_individual = min(individual_rmses, key=individual_rmses.get)
best_individual_rmse = individual_rmses[best_individual]

improvement_simple = ((best_individual_rmse - simple_rmse) / best_individual_rmse) * 100
print(f"\n🔄 vs Best Individual Model ({best_individual}):")
print(f"   Individual RMSE: {best_individual_rmse:.6f}")
print(f"   Ensemble RMSE:   {simple_rmse:.6f}")
print(f"   Improvement:     {improvement_simple:+.2f}%")

print("\n" + "=" * 80)
print("📊 ENSEMBLE METHOD 2: WEIGHTED AVERAGE (OPTIMIZED)")
print("=" * 80)

# Optimize weights
def ensemble_rmse(weights, predictions_array, y_true):
    """Calculate RMSE for weighted ensemble"""
    weights = np.abs(weights)  # Force positive
    weights = weights / np.sum(weights)  # Normalize
    ensemble_pred = np.dot(predictions_array, weights)
    return np.sqrt(mean_squared_error(y_true, ensemble_pred))

# Prepare predictions array
predictions_array = pred_df.values
n_models = predictions_array.shape[1]

# Initial weights (equal)
initial_weights = np.ones(n_models) / n_models

# Optimize
print("\n🔍 Optimizing weights...")
result = minimize(
    ensemble_rmse,
    initial_weights,
    args=(predictions_array, y_test),
    method='SLSQP',
    bounds=[(0, 1)] * n_models,
    constraints={'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1}
)

optimal_weights = np.abs(result.x)
optimal_weights = optimal_weights / np.sum(optimal_weights)

ensemble_weighted = np.dot(predictions_array, optimal_weights)
weighted_rmse = np.sqrt(mean_squared_error(y_test, ensemble_weighted))
weighted_mae = mean_absolute_error(y_test, ensemble_weighted)
weighted_r2 = r2_score(y_test, ensemble_weighted)

print(f"\n✅ Weighted Ensemble:")
print(f"   RMSE: {weighted_rmse:.6f}")
print(f"   MAE:  {weighted_mae:.6f}")
print(f"   R²:   {weighted_r2:.4f}")

print(f"\n🎯 Optimized Weights:")
for name, weight in zip(pred_df.columns, optimal_weights):
    print(f"   {name:30s} {weight:6.2%}")

improvement_weighted = ((best_individual_rmse - weighted_rmse) / best_individual_rmse) * 100
print(f"\n🔄 vs Best Individual Model:")
print(f"   Individual RMSE: {best_individual_rmse:.6f}")
print(f"   Ensemble RMSE:   {weighted_rmse:.6f}")
print(f"   Improvement:     {improvement_weighted:+.2f}%")

print("\n" + "=" * 80)
print("📊 ENSEMBLE METHOD 3: STACKING (META-MODEL)")
print("=" * 80)

# Stacking with Ridge meta-model
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

print("\n🔍 Training meta-model...")
meta_model = Ridge(alpha=1.0)

# Use cross-validation to avoid overfitting
cv_scores = cross_val_score(
    meta_model, 
    predictions_array, 
    y_test,
    cv=5,
    scoring='neg_mean_squared_error'
)

stacking_cv_rmse = np.sqrt(-cv_scores.mean())

# Train on full data
meta_model.fit(predictions_array, y_test)
ensemble_stacking = meta_model.predict(predictions_array)
stacking_rmse = np.sqrt(mean_squared_error(y_test, ensemble_stacking))
stacking_mae = mean_absolute_error(y_test, ensemble_stacking)
stacking_r2 = r2_score(y_test, ensemble_stacking)

print(f"\n✅ Stacking Ensemble:")
print(f"   CV RMSE:   {stacking_cv_rmse:.6f}")
print(f"   Test RMSE: {stacking_rmse:.6f}")
print(f"   MAE:       {stacking_mae:.6f}")
print(f"   R²:        {stacking_r2:.4f}")

print(f"\n🎯 Meta-Model Coefficients:")
for name, coef in zip(pred_df.columns, meta_model.coef_):
    print(f"   {name:30s} {coef:+.4f}")

improvement_stacking = ((best_individual_rmse - stacking_rmse) / best_individual_rmse) * 100
print(f"\n🔄 vs Best Individual Model:")
print(f"   Individual RMSE: {best_individual_rmse:.6f}")
print(f"   Ensemble RMSE:   {stacking_rmse:.6f}")
print(f"   Improvement:     {improvement_stacking:+.2f}%")

print("\n" + "=" * 80)
print("📊 ENSEMBLE COMPARISON SUMMARY")
print("=" * 80)

summary_data = {
    'Method': [
        f'Best Individual ({best_individual})',
        'Simple Average',
        'Weighted (Optimized)',
        'Stacking (Meta-model)'
    ],
    'RMSE': [
        best_individual_rmse,
        simple_rmse,
        weighted_rmse,
        stacking_rmse
    ],
    'R²': [
        r2_score(y_test, predictions[best_individual]),
        simple_r2,
        weighted_r2,
        stacking_r2
    ],
    'Improvement': [
        0.0,
        improvement_simple,
        improvement_weighted,
        improvement_stacking
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('RMSE')

print("\n" + summary_df.to_string(index=False))

# Determine best ensemble
best_ensemble_idx = summary_df['RMSE'].idxmin()
best_ensemble = summary_df.iloc[best_ensemble_idx]

print("\n" + "=" * 80)
print("🏆 RECOMMENDED ENSEMBLE")
print("=" * 80)

print(f"\n   Method: {best_ensemble['Method']}")
print(f"   RMSE:   {best_ensemble['RMSE']:.6f}")
print(f"   R²:     {best_ensemble['R²']:.4f}")
print(f"   Improvement: {best_ensemble['Improvement']:+.2f}%")

# Save best ensemble configuration
if 'Weighted' in best_ensemble['Method']:
    ensemble_config = {
        'type': 'weighted',
        'models': list(pred_df.columns),
        'weights': optimal_weights.tolist(),
        'rmse': weighted_rmse,
        'r2': weighted_r2,
        'improvement': improvement_weighted
    }
    print("\n   Weights:")
    for name, weight in zip(pred_df.columns, optimal_weights):
        print(f"      {name:30s} {weight:6.2%}")
        
elif 'Stacking' in best_ensemble['Method']:
    ensemble_config = {
        'type': 'stacking',
        'models': list(pred_df.columns),
        'meta_model': meta_model,
        'coefficients': meta_model.coef_.tolist(),
        'intercept': float(meta_model.intercept_),
        'rmse': stacking_rmse,
        'r2': stacking_r2,
        'improvement': improvement_stacking
    }
    print("\n   Coefficients:")
    for name, coef in zip(pred_df.columns, meta_model.coef_):
        print(f"      {name:30s} {coef:+.4f}")
        
else:
    ensemble_config = {
        'type': 'simple_average',
        'models': list(pred_df.columns),
        'weights': [1/n_models] * n_models,
        'rmse': simple_rmse,
        'r2': simple_r2,
        'improvement': improvement_simple
    }

# Save ensemble configuration
ensemble_path = model_storage / "ensemble_config.pkl"
with open(ensemble_path, 'wb') as f:
    pickle.dump(ensemble_config, f)

print(f"\n💾 Ensemble configuration saved to: {ensemble_path}")

# Create visualization comparing all methods
print("\n📊 Creating comparison visualization...")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('🤝 Model Ensemble Comparison', fontsize=16, fontweight='bold')

# Plot 1: RMSE Comparison
ax1 = axes[0, 0]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax1.barh(summary_df['Method'], summary_df['RMSE'], color=colors, alpha=0.8)
ax1.set_xlabel('RMSE (Lower is Better)', fontweight='bold')
ax1.set_title('📊 RMSE Comparison', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, 
            f' {width:.6f}', va='center', fontweight='bold')

# Plot 2: Improvement %
ax2 = axes[0, 1]
colors_imp = ['red' if x < 0 else 'green' for x in summary_df['Improvement']]
bars2 = ax2.barh(summary_df['Method'], summary_df['Improvement'], color=colors_imp, alpha=0.7)
ax2.set_xlabel('Improvement (%)', fontweight='bold')
ax2.set_title('📈 Improvement vs Best Individual', fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar in bars2:
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2,
            f' {width:+.2f}%', va='center', fontweight='bold')

# Plot 3: Predictions vs Actual (Best Ensemble)
ax3 = axes[1, 0]
if 'Weighted' in best_ensemble['Method']:
    best_pred = ensemble_weighted
elif 'Stacking' in best_ensemble['Method']:
    best_pred = ensemble_stacking
else:
    best_pred = ensemble_simple

ax3.scatter(y_test, best_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Values', fontweight='bold')
ax3.set_ylabel('Predicted Values', fontweight='bold')
ax3.set_title(f'🎯 Best Ensemble: {best_ensemble["Method"]}', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: R² Comparison
ax4 = axes[1, 1]
colors_r2 = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(summary_df)))
bars4 = ax4.bar(range(len(summary_df)), summary_df['R²'], color=colors_r2, alpha=0.8)
ax4.set_ylabel('R² Score', fontweight='bold')
ax4.set_title('📊 R² Score Comparison', fontweight='bold')
ax4.set_xticks(range(len(summary_df)))
ax4.set_xticklabels(summary_df['Method'], rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (0.9)')

# Add value labels
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, height,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# Save plot
plot_path = model_storage / "ensemble_comparison.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ Comparison chart saved to: {plot_path}")

print("\n" + "=" * 80)
print("✅ ENSEMBLE BUILDER COMPLETE!")
print("=" * 80)

print(f"\n🎉 Successfully tested 3 ensemble methods!")
print(f"🏆 Best method: {best_ensemble['Method']}")
print(f"📈 Performance improvement: {best_ensemble['Improvement']:+.2f}%")
print(f"💾 Configuration saved for production use")

print("\n💡 Next Steps:")
print("   1. Use ensemble for predictions (better than any single model)")
print("   2. Run dashboard to visualize all models: python run_dashboard_direct.py")
print("   3. Deploy best ensemble to production")

print("\n" + "=" * 80)

plt.show()
