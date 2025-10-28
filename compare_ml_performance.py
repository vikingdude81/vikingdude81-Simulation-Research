"""
Compare ML Model Performance: Before vs After GMA Features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)

print("\n" + "="*70)
print("📊 ML MODEL COMPARISON: BEFORE vs AFTER GMA")
print("="*70)

# Load data
df = pd.read_csv('DATA/yf_btc_1h.csv')
df['time'] = pd.to_datetime(df['time'])

print(f"\n✅ Loaded {len(df)} rows")

# Add enhanced features
from enhanced_features import add_all_enhanced_features
df = add_all_enhanced_features(df.copy())

# Clean data
df = df.dropna()
df['target_return'] = df['close'].pct_change().shift(-1)
df = df.dropna()

print(f"🧹 Clean dataset: {len(df)} rows")

# Use last 2000 rows for faster testing
df_test = df.tail(2000).copy()

# Prepare target
y = df_test['target_return']

# Split timestamp
split_idx = int(len(df_test) * 0.8)
train_idx = df_test.index[:split_idx]
test_idx = df_test.index[split_idx:]

print(f"\n📊 Dataset split:")
print(f"   Train: {len(train_idx)} samples")
print(f"   Test: {len(test_idx)} samples")

# ============================================================================
# TEST 1: OLD FEATURE SET (no GMA)
# ============================================================================
print("\n" + "="*70)
print("🔵 TEST 1: OLD FEATURE SET (Without GMA)")
print("="*70)

# Load old selected features
try:
    with open('MODEL_STORAGE/feature_data/selected_features_with_interactions.txt', 'r') as f:
        old_features = [line.strip() for line in f.readlines()]
    
    # Remove GMA features if any sneaked in
    old_features = [f for f in old_features if 'gma' not in f.lower() and f in df_test.columns]
    
    print(f"\n📋 Old features: {len(old_features)}")
    
    X_old = df_test[old_features]
    X_train_old = X_old.loc[train_idx]
    X_test_old = X_old.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    
    # Train models
    models_old = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    results_old = {}
    for name, model in models_old.items():
        print(f"\n🌲 Training {name}...")
        model.fit(X_train_old, y_train)
        
        # Predictions
        train_pred = model.predict(X_train_old)
        test_pred = model.predict(X_test_old)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        results_old[name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        print(f"   Train RMSE: {train_rmse:.6f}, R²: {train_r2:.6f}")
        print(f"   Test RMSE:  {test_rmse:.6f}, R²: {test_r2:.6f}")

except Exception as e:
    print(f"⚠️  Could not load old features: {e}")
    results_old = None

# ============================================================================
# TEST 2: NEW FEATURE SET (with GMA + interactions)
# ============================================================================
print("\n" + "="*70)
print("🚀 TEST 2: NEW FEATURE SET (With GMA + Interactions)")
print("="*70)

# Load new selected features
with open('MODEL_STORAGE/feature_data/selected_features_with_gma.txt', 'r') as f:
    new_features = [line.strip() for line in f.readlines()]

new_features = [f for f in new_features if f in df_test.columns]

print(f"\n📋 New features: {len(new_features)}")
gma_count = len([f for f in new_features if 'gma' in f.lower()])
print(f"   Including {gma_count} GMA features")

X_new = df_test[new_features]
X_train_new = X_new.loc[train_idx]
X_test_new = X_new.loc[test_idx]

# Train models
models_new = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results_new = {}
for name, model in models_new.items():
    print(f"\n🌲 Training {name}...")
    model.fit(X_train_new, y_train)
    
    # Predictions
    train_pred = model.predict(X_train_new)
    test_pred = model.predict(X_test_new)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    results_new[name] = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    
    print(f"   Train RMSE: {train_rmse:.6f}, R²: {train_r2:.6f}")
    print(f"   Test RMSE:  {test_rmse:.6f}, R²: {test_r2:.6f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*70)
print("⚖️  PERFORMANCE COMPARISON")
print("="*70)

if results_old:
    for model_name in results_new.keys():
        print(f"\n🤖 {model_name}:")
        print("-" * 70)
        
        old_r = results_old[model_name]
        new_r = results_new[model_name]
        
        # Test RMSE comparison
        rmse_improvement = ((old_r['test_rmse'] - new_r['test_rmse']) / old_r['test_rmse']) * 100
        r2_improvement = ((new_r['test_r2'] - old_r['test_r2']) / abs(old_r['test_r2'])) * 100 if old_r['test_r2'] != 0 else 0
        
        print(f"Test RMSE:")
        print(f"   OLD: {old_r['test_rmse']:.6f}")
        print(f"   NEW: {new_r['test_rmse']:.6f}")
        if rmse_improvement > 0:
            print(f"   ✅ IMPROVEMENT: {rmse_improvement:+.2f}% (lower RMSE is better)")
        else:
            print(f"   ⚠️  DECLINE: {rmse_improvement:+.2f}%")
        
        print(f"\nTest R²:")
        print(f"   OLD: {old_r['test_r2']:.6f}")
        print(f"   NEW: {new_r['test_r2']:.6f}")
        if r2_improvement > 0:
            print(f"   ✅ IMPROVEMENT: {r2_improvement:+.2f}% (higher R² is better)")
        else:
            print(f"   ⚠️  DECLINE: {r2_improvement:+.2f}%")

# ============================================================================
# GMA FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("🚀 GMA FEATURE IMPORTANCE (Random Forest)")
print("="*70)

rf_model = models_new['Random Forest']
feature_importance = pd.DataFrame({
    'feature': new_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

gma_importance = feature_importance[feature_importance['feature'].str.contains('gma', case=False)]

print(f"\nTop 10 GMA features:")
for idx, row in gma_importance.head(10).iterrows():
    overall_rank = feature_importance[feature_importance['feature'] == row['feature']].index[0] + 1
    print(f"   #{overall_rank:3d}. {row['feature']:30s} {row['importance']:.6f}")

print("\n" + "="*70)
print("✅ COMPARISON COMPLETE!")
print("="*70)
