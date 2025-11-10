"""
🔧 HYPERPARAMETER TUNING DEMO
Quick demonstration of hyperparameter optimization
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import pickle
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔧 HYPERPARAMETER TUNING - QUICK DEMO")
print("=" * 80)

# Generate synthetic data
print("\n📊 Generating demo data...")
np.random.seed(42)
n_samples = 500
n_features = 15

X_train = np.random.randn(n_samples, n_features)
y_train = X_train[:, 0] * 0.5 + X_train[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1

X_test = np.random.randn(100, n_features)
y_test = X_test[:, 0] * 0.5 + X_test[:, 1] * 0.3 + np.random.randn(100) * 0.1

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Training samples: {n_samples}")
print(f"✅ Test samples: {len(X_test)}")
print(f"✅ Features: {n_features}")

# Test 3 models with quick parameter search
models_to_tune = {
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5]
        }
    },
    'XGBoost': {
        'model': xgb.XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    },
    'LightGBM': {
        'model': lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    }
}

results = []

for model_name, config in models_to_tune.items():
    print("\n" + "=" * 80)
    print(f"🎯 Tuning: {model_name}")
    print("=" * 80)
    
    # Calculate combinations
    param_grid = config['params']
    total_combos = 1
    for values in param_grid.values():
        total_combos *= len(values)
    
    print(f"\n📊 Testing {total_combos} parameter combinations...")
    print(f"   Parameters: {list(param_grid.keys())}")
    
    # Time the search
    start_time = time.time()
    
    # Setup GridSearch with 3-fold CV
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit
    print(f"   🔍 Searching... ", end='', flush=True)
    grid_search.fit(X_train_scaled, y_train)
    
    elapsed = time.time() - start_time
    print(f"✅ Done in {elapsed:.1f}s")
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate
    train_pred = best_model.predict(X_train_scaled)
    test_pred = best_model.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    cv_rmse = np.sqrt(-grid_search.best_score_)
    
    # Test default parameters for comparison
    default_model = config['model']
    default_model.fit(X_train_scaled, y_train)
    default_pred = default_model.predict(X_test_scaled)
    default_rmse = np.sqrt(mean_squared_error(y_test, default_pred))
    
    improvement = ((default_rmse - test_rmse) / default_rmse) * 100
    
    print(f"\n   🏆 Best Parameters:")
    for param, value in best_params.items():
        print(f"      {param:20s} = {value}")
    
    print(f"\n   📊 Performance:")
    print(f"      CV RMSE:       {cv_rmse:.6f}")
    print(f"      Test RMSE:     {test_rmse:.6f}")
    print(f"      Test R²:       {test_r2:.4f}")
    
    print(f"\n   🔄 Comparison:")
    print(f"      Default RMSE:  {default_rmse:.6f}")
    print(f"      Tuned RMSE:    {test_rmse:.6f}")
    print(f"      Improvement:   {improvement:+.2f}%")
    
    results.append({
        'Model': model_name,
        'Default RMSE': default_rmse,
        'Tuned RMSE': test_rmse,
        'Improvement (%)': improvement,
        'R²': test_r2,
        'CV RMSE': cv_rmse,
        'Best Params': best_params,
        'Time (s)': elapsed
    })

# Summary comparison
print("\n" + "=" * 80)
print("📊 TUNING RESULTS SUMMARY")
print("=" * 80)

df_results = pd.DataFrame(results)
print("\n" + df_results[['Model', 'Default RMSE', 'Tuned RMSE', 'Improvement (%)', 'R²']].to_string(index=False))

print("\n" + "=" * 80)
print("🏆 BEST MODEL AFTER TUNING")
print("=" * 80)

best_idx = df_results['Tuned RMSE'].idxmin()
best = df_results.iloc[best_idx]

print(f"\n   Model: {best['Model']}")
print(f"   Test RMSE: {best['Tuned RMSE']:.6f}")
print(f"   R² Score: {best['R²']:.4f}")
print(f"   Improvement: {best['Improvement (%)']:+.2f}%")

print(f"\n   Best Parameters:")
for param, value in best['Best Params'].items():
    print(f"      {param:20s} = {value}")

# Save tuned models
print("\n" + "=" * 80)
print("💾 SAVING TUNED MODELS")
print("=" * 80)

model_storage = Path("MODEL_STORAGE")
model_storage.mkdir(exist_ok=True)

for result in results:
    model_name = result['Model']
    
    # Re-train best model on full data
    if model_name == 'Random Forest':
        model = RandomForestRegressor(**result['Best Params'], random_state=42, n_jobs=-1)
    elif model_name == 'XGBoost':
        model = xgb.XGBRegressor(**result['Best Params'], random_state=42, tree_method='hist', n_jobs=-1)
    else:
        model = lgb.LGBMRegressor(**result['Best Params'], random_state=42, verbose=-1, n_jobs=-1)
    
    model.fit(X_train_scaled, y_train)
    
    # Save
    filename = model_name.lower().replace(' ', '_') + '_tuned.pkl'
    filepath = model_storage / filename
    
    with open(filepath, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'best_params': result['Best Params'],
            'test_rmse': result['Tuned RMSE'],
            'test_r2': result['R²'],
            'improvement': result['Improvement (%)']
        }, f)
    
    print(f"   ✅ {filename}")

print("\n" + "=" * 80)
print("💡 WHAT HYPERPARAMETER TUNING DOES")
print("=" * 80)
print("""
Hyperparameter tuning systematically searches for the best model settings:

🎯 What it tests:
   • n_estimators: Number of trees (more = better but slower)
   • max_depth: How deep trees grow (controls overfitting)
   • learning_rate: How fast the model learns (lower = more stable)
   • min_samples_split: Minimum samples to split a node
   • subsample: Fraction of data used per tree (prevents overfitting)
   • And many more...

⚙️ How it works:
   1. Creates a grid of all parameter combinations
   2. Tests each combination with cross-validation
   3. Measures performance (RMSE) for each
   4. Selects the best performing combination

✅ Benefits:
   • Automatically finds optimal settings
   • Improves model accuracy by 5-20% typically
   • Uses cross-validation to prevent overfitting
   • Saves you from manual trial-and-error

🚀 Search Modes:
   • Quick Search:  ~3-5 min, tests 8-27 combinations
   • Deep Search:   ~15-30 min, tests 100-500 combinations
   • Ultra Deep:    ~30-60+ min, tests 1000+ combinations

💡 When to use:
   • After initial model training
   • When you need maximum accuracy
   • Before deploying to production
   • When comparing different algorithms
""")

print("\n" + "=" * 80)
print("✅ DEMO COMPLETE!")
print("=" * 80)
print(f"\n🎉 All 3 models tuned successfully!")
print(f"💾 Tuned models saved to: MODEL_STORAGE/")
print(f"🏆 Best model: {best['Model']} (RMSE: {best['Tuned RMSE']:.6f})")
print(f"\n💡 Run the dashboard again to see the tuned models!")
print("   python run_dashboard_direct.py")
print("\n" + "=" * 80)
