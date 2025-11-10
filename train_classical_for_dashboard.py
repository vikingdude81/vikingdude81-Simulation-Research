"""
Quick Training Script - Classical Models for Dashboard
Trains RF, XGBoost, LightGBM and saves as *_standalone.pkl
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 TRAINING CLASSICAL MODELS FOR DASHBOARD")
print("=" * 80)

# Create MODEL_STORAGE if doesn't exist
model_storage = Path("MODEL_STORAGE")
model_storage.mkdir(exist_ok=True)

print("\n📊 Step 1: Loading and preparing data...")

# Try to load data from existing CSV files
try:
    # Look for recent prediction files with data
    data_files = list(Path(".").glob("*_predictions_*.csv"))
    if data_files:
        # Use most recent file
        latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
        print(f"   Loading from: {latest_file}")
        df = pd.read_csv(latest_file)
    else:
        # Generate synthetic data for demo
        print("   No data files found - generating synthetic demo data...")
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features (like price data)
        df = pd.DataFrame({
            'price': np.cumsum(np.random.randn(n_samples) * 0.01) + 100,
            'volume': np.random.lognormal(10, 1, n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'macd': np.random.randn(n_samples) * 2,
            'bb_upper': np.cumsum(np.random.randn(n_samples) * 0.01) + 105,
            'bb_lower': np.cumsum(np.random.randn(n_samples) * 0.01) + 95,
        })
        
        # Calculate returns as target
        df['returns'] = df['price'].pct_change().fillna(0)
        df = df.fillna(method='ffill').fillna(0)
        
        print("   ✅ Generated 1000 samples with 6 features")

    # Prepare features and target
    # Drop any columns that look like targets or dates
    exclude_cols = ['returns', 'target', 'date', 'timestamp', 'Unnamed: 0']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # If we have a 'returns' column, use it as target, otherwise use price changes
    if 'returns' in df.columns:
        target_col = 'returns'
    elif 'price' in df.columns:
        df['returns'] = df['price'].pct_change().fillna(0)
        target_col = 'returns'
    else:
        # Use first numeric column
        target_col = df.select_dtypes(include=[np.number]).columns[0]
    
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Remove any infinite or NaN values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"   ✅ Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Features: {list(X.columns[:5])}{'...' if len(X.columns) > 5 else ''}")
    
except Exception as e:
    print(f"   ⚠️ Error loading data: {e}")
    print("   Generating synthetic data instead...")
    
    # Fallback: Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(
        X.iloc[:, 0] * 0.5 + X.iloc[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1,
        name='target'
    )
    
    print(f"   ✅ Generated synthetic data: {X.shape[0]} samples, {X.shape[1]} features")

# Split data
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"   Train: {len(X_train)} samples")
print(f"   Test:  {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

feature_names = list(X.columns)

print("\n" + "=" * 80)
print("🌲 Model 1/3: Random Forest")
print("=" * 80)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42
)

print("   Training...")
rf_model.fit(X_train_scaled, y_train)

# Evaluate
rf_train_pred = rf_model.predict(X_train_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
train_r2 = r2_score(y_train, rf_train_pred)
test_r2 = r2_score(y_test, rf_test_pred)

print(f"   Train RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}")
print(f"   Test RMSE:  {test_rmse:.6f}, R²: {test_r2:.4f}")

# Save model
model_path = model_storage / "random_forest_standalone.pkl"
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': rf_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }, f)

print(f"   ✅ Saved to: {model_path}")

print("\n" + "=" * 80)
print("⚡ Model 2/3: XGBoost")
print("=" * 80)

try:
    import xgboost as xgb
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
    
    print("   Training...")
    xgb_model.fit(X_train_scaled, y_train, verbose=False)
    
    # Evaluate
    xgb_train_pred = xgb_model.predict(X_train_scaled)
    xgb_test_pred = xgb_model.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, xgb_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
    train_r2 = r2_score(y_train, xgb_train_pred)
    test_r2 = r2_score(y_test, xgb_test_pred)
    
    print(f"   Train RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}")
    print(f"   Test RMSE:  {test_rmse:.6f}, R²: {test_r2:.4f}")
    
    # Save model
    model_path = model_storage / "xgboost_standalone.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': xgb_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }, f)
    
    print(f"   ✅ Saved to: {model_path}")

except ImportError:
    print("   ⚠️ XGBoost not installed - skipping")

print("\n" + "=" * 80)
print("💡 Model 3/3: LightGBM")
print("=" * 80)

try:
    import lightgbm as lgb
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    
    print("   Training...")
    lgb_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    lgb_train_pred = lgb_model.predict(X_train_scaled)
    lgb_test_pred = lgb_model.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, lgb_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, lgb_test_pred))
    train_r2 = r2_score(y_train, lgb_train_pred)
    test_r2 = r2_score(y_test, lgb_test_pred)
    
    print(f"   Train RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}")
    print(f"   Test RMSE:  {test_rmse:.6f}, R²: {test_r2:.4f}")
    
    # Save model
    model_path = model_storage / "lightgbm_standalone.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': lgb_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }, f)
    
    print(f"   ✅ Saved to: {model_path}")

except ImportError:
    print("   ⚠️ LightGBM not installed - skipping")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)

# List saved models
saved_models = list(model_storage.glob("*_standalone.pkl"))
print(f"\n📦 Saved {len(saved_models)} models:")
for model_file in saved_models:
    print(f"   • {model_file.name}")

print("\n🎯 Next step: Run 'python ml_models_menu.py' and select Option 18 (Dashboard)")
print("=" * 80)
