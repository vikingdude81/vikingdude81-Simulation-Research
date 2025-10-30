"""
Quick Training Script for Performance Dashboard Demo
Trains 3 classical models quickly for dashboard visualization
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("QUICK TRAINING - 3 Classical Models")
print("=" * 80)

# Import data pipeline
print("\nLoading data pipeline...")
from main import load_data, engineer_features

raw_data = load_data()
X_train, X_test, y_train, y_test, feature_names = engineer_features(raw_data)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {len(feature_names)}")

model_storage = Path("MODEL_STORAGE")
model_storage.mkdir(exist_ok=True)

# 1. Random Forest
print("\n" + "=" * 80)
print("1/3 RANDOM FOREST")
print("=" * 80)

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
])

print("Training...")
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"RMSE: {rmse_rf:.6f}")

# Save
rf_path = model_storage / "random_forest_standalone.pkl"
with open(rf_path, 'wb') as f:
    pickle.dump(rf_pipeline, f)
print(f"Saved: {rf_path}")

# 2. XGBoost
print("\n" + "=" * 80)
print("2/3 XGBOOST")
print("=" * 80)

xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.01, 
                               random_state=42, tree_method='hist', n_jobs=-1))
])

print("Training...")
xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"RMSE: {rmse_xgb:.6f}")

# Save
xgb_path = model_storage / "xgboost_standalone.pkl"
with open(xgb_path, 'wb') as f:
    pickle.dump(xgb_pipeline, f)
print(f"Saved: {xgb_path}")

# 3. LightGBM
print("\n" + "=" * 80)
print("3/3 LIGHTGBM")
print("=" * 80)

lgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.01,
                                random_state=42, verbose=-1, n_jobs=-1))
])

print("Training...")
lgb_pipeline.fit(X_train, y_train)
y_pred_lgb = lgb_pipeline.predict(X_test)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
print(f"RMSE: {rmse_lgb:.6f}")

# Save
lgb_path = model_storage / "lightgbm_standalone.pkl"
with open(lgb_path, 'wb') as f:
    pickle.dump(lgb_pipeline, f)
print(f"Saved: {lgb_path}")

# Summary
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nRandom Forest RMSE: {rmse_rf:.6f}")
print(f"XGBoost RMSE:       {rmse_xgb:.6f}")
print(f"LightGBM RMSE:      {rmse_lgb:.6f}")
print(f"\nAll models saved to: {model_storage}/")
print("\nReady for Performance Dashboard (Option 18)!")
print("=" * 80)
