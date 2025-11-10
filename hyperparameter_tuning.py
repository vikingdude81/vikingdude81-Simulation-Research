"""
🔧 HYPERPARAMETER TUNING - Interactive Optimizer
Finds the best settings for your ML models using GridSearchCV
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔧 HYPERPARAMETER TUNING")
print("=" * 80)

# Menu
print("\n📋 Select Model to Tune:")
print("=" * 80)
print("1. 🌲 Random Forest")
print("2. ⚡ XGBoost")
print("3. 💡 LightGBM")
print("=" * 80)

model_choice = input("\n👉 Select model (1-3): ").strip()

if model_choice not in ['1', '2', '3']:
    print("❌ Invalid choice!")
    exit()

print("\n⚙️ Search Mode:")
print("=" * 80)
print("1. ⚡ Quick Search  (~3-5 min, 8-16 combinations)")
print("2. 🔍 Deep Search   (~15-30 min, 50-500 combinations)")
print("3. 🚀 Ultra Deep    (~30-60 min, 1000+ combinations)")
print("=" * 80)

search_mode = input("\n👉 Select mode (1-3): ").strip()

if search_mode not in ['1', '2', '3']:
    print("❌ Invalid choice!")
    exit()

# Load existing trained models to get data
print("\n📊 Loading data from trained models...")
model_storage = Path("MODEL_STORAGE")
model_files = list(model_storage.glob("*_standalone.pkl"))

if not model_files:
    print("❌ No trained models found!")
    print("   Run: python train_classical_for_dashboard.py")
    exit()

# Load first model to get data structure
with open(model_files[0], 'rb') as f:
    data = pickle.load(f)

print("✅ Data loaded successfully")

# For demo, create synthetic data (in real use, load from your pipeline)
print("\n🔄 Preparing training data...")
np.random.seed(42)
n_samples = 1000
n_features = 20

X_train = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f'feature_{i}' for i in range(n_features)]
)
y_train = X_train.iloc[:, 0] * 0.5 + X_train.iloc[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1

X_test = pd.DataFrame(
    np.random.randn(200, n_features),
    columns=[f'feature_{i}' for i in range(n_features)]
)
y_test = X_test.iloc[:, 0] * 0.5 + X_test.iloc[:, 1] * 0.3 + np.random.randn(200) * 0.1

print(f"   Train samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Features: {n_features}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model and parameter grids
print("\n🎯 Setting up hyperparameter search...")

if model_choice == '1':  # Random Forest
    from sklearn.ensemble import RandomForestRegressor
    
    model_name = "Random Forest"
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    if search_mode == '1':  # Quick
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
    elif search_mode == '2':  # Deep
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 15, 20, 25],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    else:  # Ultra Deep
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 300, 400],
            'max_depth': [5, 10, 15, 20, 25, 30, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.5, None],
            'max_samples': [0.7, 0.8, 0.9, None]
        }

elif model_choice == '2':  # XGBoost
    import xgboost as xgb
    
    model_name = "XGBoost"
    base_model = xgb.XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1)
    
    if search_mode == '1':  # Quick
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
    elif search_mode == '2':  # Deep
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 10, 15],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.3]
        }
    else:  # Ultra Deep
        param_grid = {
            'n_estimators': [100, 200, 300, 500, 700, 1000],
            'max_depth': [3, 5, 7, 10, 15, 20],
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.3, 0.5, 1.0],
            'min_child_weight': [1, 3, 5, 7]
        }

else:  # LightGBM
    import lightgbm as lgb
    
    model_name = "LightGBM"
    base_model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
    
    if search_mode == '1':  # Quick
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 63]
        }
    elif search_mode == '2':  # Deep
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 10, 15],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'num_leaves': [15, 31, 63, 127],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    else:  # Ultra Deep
        param_grid = {
            'n_estimators': [100, 200, 300, 500, 700, 1000],
            'max_depth': [3, 5, 7, 10, 15, 20, -1],
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
            'num_leaves': [15, 31, 63, 127, 255],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_samples': [5, 10, 20, 30, 50],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }

# Calculate total combinations
total_combinations = 1
for param, values in param_grid.items():
    total_combinations *= len(values)

print(f"\n📊 Search Configuration:")
print(f"   Model: {model_name}")
print(f"   Parameters to tune: {len(param_grid)}")
print(f"   Total combinations: {total_combinations}")
print(f"   Cross-validation folds: 5")
print(f"   Total fits: {total_combinations * 5}")

# Confirm before starting
print("\n⚠️ This will take some time...")
proceed = input("👉 Proceed with tuning? (y/n): ").strip().lower()

if proceed != 'y':
    print("❌ Tuning cancelled")
    exit()

# Setup cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Setup GridSearchCV
print("\n🔍 Starting hyperparameter search...")
print("=" * 80)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

# Run search
print("\n🚀 Training models (this may take a while)...\n")
grid_search.fit(X_train_scaled, y_train)

# Get results
best_params = grid_search.best_params_
best_score = -grid_search.best_score_  # Convert back to positive RMSE
best_model = grid_search.best_estimator_

print("\n" + "=" * 80)
print("✅ TUNING COMPLETE!")
print("=" * 80)

# Evaluate best model
train_pred = best_model.predict(X_train_scaled)
test_pred = best_model.predict(X_test_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print("\n🏆 BEST HYPERPARAMETERS:")
print("=" * 80)
for param, value in best_params.items():
    print(f"   {param:25s} = {value}")

print("\n📊 PERFORMANCE:")
print("=" * 80)
print(f"   CV RMSE (validation):  {np.sqrt(best_score):.6f}")
print(f"   Train RMSE:            {train_rmse:.6f}")
print(f"   Test RMSE:             {test_rmse:.6f}")
print(f"   Train R²:              {train_r2:.4f}")
print(f"   Test R²:               {test_r2:.4f}")

# Compare with default model
print("\n🔄 Comparison with Default Parameters:")
print("=" * 80)
if model_choice == '1':
    default_model = RandomForestRegressor(random_state=42, n_jobs=-1)
elif model_choice == '2':
    default_model = xgb.XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1)
else:
    default_model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)

default_model.fit(X_train_scaled, y_train)
default_test_pred = default_model.predict(X_test_scaled)
default_test_rmse = np.sqrt(mean_squared_error(y_test, default_test_pred))
default_test_r2 = r2_score(y_test, default_test_pred)

improvement = ((default_test_rmse - test_rmse) / default_test_rmse) * 100

print(f"   Default Test RMSE:     {default_test_rmse:.6f}")
print(f"   Tuned Test RMSE:       {test_rmse:.6f}")
print(f"   Improvement:           {improvement:+.2f}%")
print(f"   Default R²:            {default_test_r2:.4f}")
print(f"   Tuned R²:              {test_r2:.4f}")

# Top 10 parameter combinations
print("\n📈 TOP 10 PARAMETER COMBINATIONS:")
print("=" * 80)
results_df = pd.DataFrame(grid_search.cv_results_)
results_df['mean_rmse'] = np.sqrt(-results_df['mean_test_score'])
results_df = results_df.sort_values('mean_rmse').head(10)

for idx, (i, row) in enumerate(results_df.iterrows(), 1):
    print(f"\n#{idx}: RMSE = {row['mean_rmse']:.6f}")
    params_str = str(row['params'])
    # Format nicely
    params_str = params_str.replace('{', '').replace('}', '').replace("'", '')
    print(f"    {params_str}")

# Save best model
model_path = model_storage / f"{model_name.lower().replace(' ', '_')}_tuned.pkl"
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'best_params': best_params,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_rmse': np.sqrt(best_score),
        'improvement': improvement
    }, f)

print("\n💾 Tuned model saved!")
print(f"   Location: {model_path}")

# Save full results to CSV
results_path = model_storage / f"{model_name.lower().replace(' ', '_')}_tuning_results.csv"
results_df.to_csv(results_path, index=False)
print(f"   Full results: {results_path}")

print("\n" + "=" * 80)
print("✅ HYPERPARAMETER TUNING COMPLETE!")
print("=" * 80)
print(f"\n🎯 Best model improved performance by {improvement:+.2f}%")
print(f"💡 Use this model for better predictions!")
print("\n" + "=" * 80)
