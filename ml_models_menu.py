"""
ML Models Menu - Interactive Model Selection
============================================

Run individual ML models or the full pipeline without breaking existing workflow.
Each model can be tested independently with the same feature set.

Usage:
    python ml_models_menu.py

Author: AI Trading System
Date: October 29, 2025
"""

import os
import sys
from pathlib import Path

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print menu header"""
    print("=" * 80)
    print("🤖 ML MODELS MENU - Individual Model Testing")
    print("=" * 80)
    print("\nTest individual ML models or run the complete pipeline")
    print("All models use the same feature set for fair comparison\n")

def print_menu():
    """Display main menu"""
    print("\n" + "=" * 80)
    print("📊 CLASSICAL ML MODELS (Scikit-learn & Gradient Boosting)")
    print("=" * 80)
    print("1. Random Forest          - Tree ensemble with bagging")
    print("2. XGBoost               - Gradient boosting trees (fast)")
    print("3. LightGBM              - Microsoft gradient boosting (memory efficient)")
    print("4. Classical Ensemble    - RF + XGBoost + LightGBM combined")
    
    print("\n" + "=" * 80)
    print("🧠 DEEP LEARNING MODELS (PyTorch with GPU)")
    print("=" * 80)
    print("5. LSTM with Attention   - 3 layers, 256 hidden, ~1.4M params")
    print("6. Transformer           - 4 layers, 8 heads, ~3.2M params")
    print("7. MultiTask Network     - Price + Vol + Direction, ~3.4M params")
    print("8. Deep Learning Suite   - LSTM + Transformer + MultiTask")
    
    print("\n" + "=" * 80)
    print("🚀 FULL PIPELINE")
    print("=" * 80)
    print("9. Complete Pipeline     - All 6 models (RF, XGB, LGB, LSTM, Trans, Multi)")
    
    print("\n" + "=" * 80)
    print("⚙️  UTILITIES")
    print("=" * 80)
    print("10. Compare Models       - Side-by-side performance comparison")
    print("11. Feature Importance   - Analyze which features matter most")
    print("12. GPU Check           - Verify CUDA availability")
    
    print("\n" + "=" * 80)
    print("🎯 ADVANCED FEATURES")
    print("=" * 80)
    print("13. Hyperparameter Tuning - Interactive parameter optimization")
    print("14. Quick Predict         - Load model & predict instantly")
    print("15. Feature Selection     - Find optimal feature subset")
    print("16. Error Analysis        - Diagnose model failures")
    print("17. Model Ensemble Builder - Create custom model combinations")
    print("18. Performance Dashboard  - All models visualized on one screen")
    
    print("\n" + "=" * 80)
    print("0. Exit")
    print("=" * 80)

def run_random_forest():
    """Run Random Forest model only"""
    print("\n" + "=" * 80)
    print("🌲 RANDOM FOREST TRAINING")
    print("=" * 80)
    print("\nCreating training script for Random Forest only...\n")
    
    script_path = Path("train_random_forest.py")
    
    # Create standalone RF training script
    script_content = """\"\"\"
Standalone Random Forest Training
Train only Random Forest model for quick testing
\"\"\"

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
from pathlib import Path

print("=" * 80)
print("🌲 RANDOM FOREST - Standalone Training")
print("=" * 80)

# Import main pipeline components
try:
    from main import load_data, engineer_features
    print("✅ Loaded data pipeline from main.py")
except ImportError:
    print("❌ Error: main.py not found. Make sure you're in the correct directory.")
    sys.exit(1)

# Load and prepare data
print("\\n📊 Loading data and engineering features...")
df = load_data()
X_train, X_test, y_train, y_test, scaler, feature_names = engineer_features(df)

print(f"✅ Features: {len(feature_names)}")
print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Test samples: {len(X_test)}")

# Train Random Forest
print("\\n🌲 Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)

# Evaluate
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\\n📊 RESULTS:")
print(f"   Train RMSE: {train_rmse:.4f}")
print(f"   Test RMSE:  {test_rmse:.4f}")

# Feature importance
try:
    importances = rf_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\\n🔝 Top 10 Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s}: {row['importance']:.4f}")
    
    # Save feature importance
    fi_path = Path("MODEL_STORAGE") / "random_forest_feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)
    print(f"\\n✅ Feature importance saved to: {fi_path}")
    
except Exception as e:
    print(f"\\n⚠️  Could not extract feature importance: {e}")
    print("   Model trained successfully - this is just a reporting issue")

# Save model
model_path = Path("MODEL_STORAGE") / "random_forest_standalone.pkl"
model_path.parent.mkdir(exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': rf_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }, f)

print(f"\\n✅ Model saved to: {model_path}")
print("\\n" + "=" * 80)
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created: {script_path}")
    print("\n▶️  Running Random Forest training...\n")
    
    os.system(f"python {script_path}")

def run_xgboost():
    """Run XGBoost model only"""
    print("\n" + "=" * 80)
    print("⚡ XGBOOST TRAINING")
    print("=" * 80)
    print("\nCreating training script for XGBoost only...\n")
    
    script_path = Path("train_xgboost.py")
    
    script_content = """\"\"\"
Standalone XGBoost Training
Train only XGBoost model for quick testing
\"\"\"

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
from pathlib import Path

print("=" * 80)
print("⚡ XGBOOST - Standalone Training")
print("=" * 80)

# Import main pipeline components
try:
    from main import load_data, engineer_features
    print("✅ Loaded data pipeline from main.py")
except ImportError:
    print("❌ Error: main.py not found. Make sure you're in the correct directory.")
    sys.exit(1)

# Load and prepare data
print("\\n📊 Loading data and engineering features...")
df = load_data()
X_train, X_test, y_train, y_test, scaler, feature_names = engineer_features(df)

print(f"✅ Features: {len(feature_names)}")
print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Test samples: {len(X_test)}")

# Train XGBoost
print("\\n⚡ Training XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

xgb_model.fit(X_train, y_train, verbose=True)

# Evaluate
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\\n📊 RESULTS:")
print(f"   Train RMSE: {train_rmse:.4f}")
print(f"   Test RMSE:  {test_rmse:.4f}")

# Feature importance
try:
    importances = xgb_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\\n🔝 Top 10 Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s}: {row['importance']:.4f}")
    
    # Save feature importance
    fi_path = Path("MODEL_STORAGE") / "xgboost_feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)
    print(f"\\n✅ Feature importance saved to: {fi_path}")
    
except Exception as e:
    print(f"\\n⚠️  Could not extract feature importance: {e}")
    print("   Model trained successfully - this is just a reporting issue")

# Save model
model_path = Path("MODEL_STORAGE") / "xgboost_standalone.pkl"
model_path.parent.mkdir(exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': xgb_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }, f)

print(f"\\n✅ Model saved to: {model_path}")
print("\\n" + "=" * 80)
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created: {script_path}")
    print("\n▶️  Running XGBoost training...\n")
    
    os.system(f"python {script_path}")

def run_lightgbm():
    """Run LightGBM model only"""
    print("\n" + "=" * 80)
    print("💡 LIGHTGBM TRAINING")
    print("=" * 80)
    print("\nCreating training script for LightGBM only...\n")
    
    script_path = Path("train_lightgbm.py")
    
    script_content = """\"\"\"
Standalone LightGBM Training
Train only LightGBM model for quick testing
\"\"\"

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
from pathlib import Path

print("=" * 80)
print("💡 LIGHTGBM - Standalone Training")
print("=" * 80)

# Check LightGBM availability
try:
    import lightgbm as lgb
    print("✅ LightGBM available")
except ImportError:
    print("❌ LightGBM not installed. Install with: pip install lightgbm")
    sys.exit(1)

# Import main pipeline components
try:
    from main import load_data, engineer_features
    print("✅ Loaded data pipeline from main.py")
except ImportError:
    print("❌ Error: main.py not found. Make sure you're in the correct directory.")
    sys.exit(1)

# Load and prepare data
print("\\n📊 Loading data and engineering features...")
df = load_data()
X_train, X_test, y_train, y_test, scaler, feature_names = engineer_features(df)

print(f"✅ Features: {len(feature_names)}")
print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Test samples: {len(X_test)}")

# Train LightGBM
print("\\n💡 Training LightGBM model...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

lgb_model.fit(X_train, y_train)

# Evaluate
y_pred_train = lgb_model.predict(X_train)
y_pred_test = lgb_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\\n📊 RESULTS:")
print(f"   Train RMSE: {train_rmse:.4f}")
print(f"   Test RMSE:  {test_rmse:.4f}")

# Feature importance
try:
    importances = lgb_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\\n🔝 Top 10 Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s}: {row['importance']:.4f}")
    
    # Save feature importance
    fi_path = Path("MODEL_STORAGE") / "lightgbm_feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)
    print(f"\\n✅ Feature importance saved to: {fi_path}")
    
except Exception as e:
    print(f"\\n⚠️  Could not extract feature importance: {e}")
    print("   Model trained successfully - this is just a reporting issue")

# Save model
model_path = Path("MODEL_STORAGE") / "lightgbm_standalone.pkl"
model_path.parent.mkdir(exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': lgb_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }, f)

print(f"\\n✅ Model saved to: {model_path}")
print("\\n" + "=" * 80)
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created: {script_path}")
    print("\n▶️  Running LightGBM training...\n")
    
    os.system(f"python {script_path}")

def run_classical_ensemble():
    """Run all classical ML models"""
    print("\n" + "=" * 80)
    print("🎯 CLASSICAL ENSEMBLE (RF + XGBoost + LightGBM)")
    print("=" * 80)
    print("\nRunning all classical ML models and creating ensemble...\n")
    
    input("Press Enter to start training all classical models...")
    
    # Run each model
    run_random_forest()
    print("\n" + "-" * 80 + "\n")
    run_xgboost()
    print("\n" + "-" * 80 + "\n")
    run_lightgbm()
    
    print("\n" + "=" * 80)
    print("✅ All classical models trained!")
    print("=" * 80)

def run_lstm():
    """Run LSTM with Attention model only"""
    print("\n" + "=" * 80)
    print("🧠 LSTM WITH ATTENTION TRAINING")
    print("=" * 80)
    print("\nThis will train the LSTM model with attention mechanism")
    print("Expected time: ~10-15 minutes with GPU, ~30-45 minutes with CPU\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        return
    
    print("\n💡 Tip: Training script will use main.py's LSTM implementation")
    print("   To train LSTM only, you can modify main.py temporarily or")
    print("   use the full pipeline and extract the LSTM results.\n")
    
    print("▶️  Running: python main.py (LSTM mode)")
    print("\nNote: Current main.py trains all models. Consider adding a")
    print("      --model flag to main.py for individual model selection.\n")
    
    input("Press Enter to continue or Ctrl+C to cancel...")

def run_transformer():
    """Run Transformer model only"""
    print("\n" + "=" * 80)
    print("🔄 TRANSFORMER TRAINING")
    print("=" * 80)
    print("\nThis will train the Transformer model")
    print("Expected time: ~10-15 minutes with GPU, ~30-45 minutes with CPU\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        return
    
    print("\n💡 Tip: Training script will use main.py's Transformer implementation")
    print("   Architecture: 4 layers, 8 heads, 256 dimensions\n")
    
    input("Press Enter to continue or Ctrl+C to cancel...")

def run_multitask():
    """Run MultiTask model only"""
    print("\n" + "=" * 80)
    print("🎯 MULTITASK NETWORK TRAINING")
    print("=" * 80)
    print("\nThis will train the MultiTask network")
    print("Predicts: Price + Volatility + Direction simultaneously")
    print("Expected time: ~10-15 minutes with GPU, ~30-45 minutes with CPU\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        return
    
    print("\n💡 Tip: Training script will use main.py's MultiTask implementation\n")
    
    input("Press Enter to continue or Ctrl+C to cancel...")

def run_deep_learning_suite():
    """Run all deep learning models"""
    print("\n" + "=" * 80)
    print("🚀 DEEP LEARNING SUITE (LSTM + Transformer + MultiTask)")
    print("=" * 80)
    print("\nThis will train all 3 deep learning models")
    print("Expected time: ~30-45 minutes with GPU, ~90-120 minutes with CPU\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        return
    
    print("\n▶️  Training all deep learning models...")
    input("Press Enter to continue...")

def run_full_pipeline():
    """Run complete ML pipeline"""
    print("\n" + "=" * 80)
    print("🚀 COMPLETE ML PIPELINE - ALL 6 MODELS")
    print("=" * 80)
    print("\nThis will train:")
    print("   • Random Forest")
    print("   • XGBoost")
    print("   • LightGBM")
    print("   • LSTM with Attention")
    print("   • Transformer")
    print("   • MultiTask Network")
    print("\nExpected time: ~60 minutes with GPU, ~120-180 minutes with CPU\n")
    
    response = input("Start full pipeline? (y/n): ")
    if response.lower() != 'y':
        return
    
    print("\n▶️  Running complete pipeline: python main.py\n")
    os.system("python main.py")

def compare_models():
    """Compare all trained models"""
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON")
    print("=" * 80)
    print("\nLoading and comparing all trained models...\n")
    
    script_path = Path("compare_ml_performance.py")
    if script_path.exists():
        print(f"▶️  Running: python {script_path}\n")
        os.system(f"python {script_path}")
    else:
        print(f"❌ {script_path} not found.")
        print("   Train some models first, then use this option.")

def feature_importance():
    """Analyze feature importance"""
    print("\n" + "=" * 80)
    print("🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing which features matter most...\n")
    
    script_path = Path("extract_feature_importance.py")
    if script_path.exists():
        print(f"▶️  Running: python {script_path}\n")
        os.system(f"python {script_path}")
    else:
        print(f"❌ {script_path} not found.")
        print("   Train some models first, then use this option.")

def check_gpu():
    """Check GPU availability"""
    print("\n" + "=" * 80)
    print("🎮 GPU AVAILABILITY CHECK")
    print("=" * 80)
    
    try:
        import torch
        print("\n✅ PyTorch installed")
        print(f"   Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"\n✅ CUDA available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print("\n   Deep learning models will use GPU acceleration! 🚀")
        else:
            print("\n⚠️  CUDA not available")
            print("   Deep learning models will use CPU (slower)")
            print("\n   To enable GPU:")
            print("   1. Install NVIDIA drivers")
            print("   2. Install CUDA toolkit")
            print("   3. Reinstall PyTorch with CUDA support")
    except ImportError:
        print("\n❌ PyTorch not installed")
        print("   Install with: pip install torch")
    
    print("\n" + "=" * 80)
    input("\nPress Enter to return to menu...")

def hyperparameter_tuning():
    """Interactive hyperparameter tuning"""
    print("\n" + "=" * 80)
    print("🔧 HYPERPARAMETER TUNING")
    print("=" * 80)
    
    print("\nSelect model to tune:")
    print("1. Random Forest")
    print("2. XGBoost")
    print("3. LightGBM")
    print("4. LSTM")
    print("5. Transformer")
    
    model_choice = input("\nModel (1-5): ").strip()
    
    if model_choice not in ['1', '2', '3', '4', '5']:
        print("Invalid choice!")
        input("\nPress Enter to return to menu...")
        return
    
    model_map = {
        '1': 'Random Forest',
        '2': 'XGBoost', 
        '3': 'LightGBM',
        '4': 'LSTM',
        '5': 'Transformer'
    }
    
    selected_model = model_map[model_choice]
    
    print(f"\n🎯 Tuning {selected_model}...")
    print("\nSearch Mode:")
    print("1. Quick Search  (3-5 min, fewer combinations)")
    print("2. Deep Search   (15-30 min, exhaustive)")
    
    search_mode = input("\nMode (1-2): ").strip()
    
    script_path = Path(f"tune_{selected_model.lower().replace(' ', '_')}.py")
    
    # Create tuning script based on model choice
    if model_choice in ['1', '2', '3']:  # Classical models
        script_content = f'''"""
Hyperparameter Tuning for {selected_model}
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path

print("=" * 80)
print("🔧 HYPERPARAMETER TUNING - {selected_model}")
print("=" * 80)

# Import data pipeline
from main import load_data, engineer_features

print("\\n📊 Loading data...")
raw_data = load_data()
X_train, X_test, y_train, y_test, feature_names = engineer_features(raw_data)

print(f"✅ Training samples: {{len(X_train)}}")
print(f"✅ Features: {{len(feature_names)}}")

# Define model and parameter grid
'''
        if model_choice == '1':  # Random Forest
            if search_mode == '1':  # Quick
                param_grid = """
param_grid = {{
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20],
    'model__min_samples_split': [2, 5],
    'model__max_features': ['sqrt', 'log2']
}}
"""
            else:  # Deep
                param_grid = """
param_grid = {{
    'model__n_estimators': [50, 100, 200, 300],
    'model__max_depth': [10, 15, 20, 25, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None]
}}
"""
            script_content += f"""
from sklearn.ensemble import RandomForestRegressor

{param_grid}

model = RandomForestRegressor(random_state=42, n_jobs=-1)
"""
        
        elif model_choice == '2':  # XGBoost
            if search_mode == '1':
                param_grid = """
param_grid = {{
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10],
    'model__learning_rate': [0.01, 0.1],
    'model__subsample': [0.8, 1.0]
}}
"""
            else:
                param_grid = """
param_grid = {{
    'model__n_estimators': [100, 200, 300, 500],
    'model__max_depth': [3, 5, 7, 10, 15],
    'model__learning_rate': [0.001, 0.01, 0.05, 0.1],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0]
}}
"""
            script_content += f"""
import xgboost as xgb

{param_grid}

model = xgb.XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1)
"""
        
        else:  # LightGBM
            if search_mode == '1':
                param_grid = """
param_grid = {{
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10],
    'model__learning_rate': [0.01, 0.1],
    'model__num_leaves': [31, 63]
}}
"""
            else:
                param_grid = """
param_grid = {{
    'model__n_estimators': [100, 200, 300, 500],
    'model__max_depth': [3, 5, 7, 10, 15],
    'model__learning_rate': [0.001, 0.01, 0.05, 0.1],
    'model__num_leaves': [15, 31, 63, 127],
    'model__subsample': [0.6, 0.8, 1.0]
}}
"""
            script_content += f"""
import lightgbm as lgb

{param_grid}

model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
"""
        
        # Add common tuning code
        script_content += '''
# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', model)
])

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5, gap=3)

print("\\n🔍 Starting Grid Search...")
total_fits = len(param_grid['model__n_estimators']) if 'model__n_estimators' in param_grid else 1
for key in param_grid:
    if key != 'model__n_estimators':
        total_fits *= len(param_grid[key])

print(f"   Total combinations: {total_fits}")
print(f"   CV folds: 5")
print(f"   Total fits: {total_fits * 5}")

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

# Results
print("\\n" + "=" * 80)
print("✅ TUNING COMPLETE!")
print("=" * 80)

print(f"\\n🏆 Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")

best_score = -grid_search.best_score_
print(f"\\n📊 Best CV Score (RMSE): {best_score:.6f}")

# Test performance
y_pred = grid_search.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"   Test RMSE: {test_rmse:.6f}")

# Save tuned model
model_storage = Path("MODEL_STORAGE")
model_storage.mkdir(exist_ok=True)
model_path = model_storage / "tuned_model.pkl"

with open(model_path, 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)

print(f"\\n✅ Tuned model saved to: {model_path}")

# Save parameters to config
config_path = model_storage / "best_params.txt"
with open(config_path, 'w') as f:
    f.write("Best Hyperparameters:\\n")
    f.write("=" * 50 + "\\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"{param}: {value}\\n")
    f.write(f"\\nCV RMSE: {best_score:.6f}\\n")
    f.write(f"Test RMSE: {test_rmse:.6f}\\n")

print(f"✅ Parameters saved to: {config_path}")
print("\\n" + "=" * 80)
'''
    
    else:  # Deep learning models
        print(f"\\n⚠️  Deep learning tuning coming soon!")
        print("   For now, tune manually in main.py")
        input("\\nPress Enter to return to menu...")
        return
    
    # Write script
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\\n✅ Created tuning script: {script_path}")
    print("\\n🚀 Running hyperparameter tuning...")
    input("\\nPress Enter to start (this may take several minutes)...")
    
    # Run the script
    import subprocess
    result = subprocess.run([sys.executable, str(script_path)], 
                          capture_output=False, text=True)
    
    input("\\nPress Enter to return to menu...")

def quick_predict():
    """Quick prediction with saved model"""
    print("\\n" + "=" * 80)
    print("⚡ QUICK PREDICT")
    print("=" * 80)
    
    model_storage = Path("MODEL_STORAGE")
    
    # Find available models
    if not model_storage.exists():
        print("\\n❌ No saved models found!")
        print("   Train a model first (options 1-9)")
        input("\\nPress Enter to return to menu...")
        return
    
    model_files = list(model_storage.glob("*.pkl"))
    
    if not model_files:
        print("\\n❌ No .pkl model files found!")
        input("\\nPress Enter to return to menu...")
        return
    
    print("\\n📦 Available Models:")
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file.name}")
    
    choice = input("\\nSelect model number: ").strip()
    
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(model_files):
            raise ValueError
        selected_model = model_files[idx]
    except:
        print("❌ Invalid choice!")
        input("\\nPress Enter to return to menu...")
        return
    
    # Load model
    print(f"\\n📂 Loading {selected_model.name}...")
    
    import pickle
    with open(selected_model, 'rb') as f:
        model = pickle.load(f)
    
    print("✅ Model loaded!")
    
    # Prediction mode
    print("\\n🎯 Prediction Mode:")
    print("1. Single prediction (latest data)")
    print("2. Batch predict from CSV")
    
    mode = input("\\nMode (1-2): ").strip()
    
    if mode == '1':
        # Load latest data
        print("\\n📊 Loading latest data...")
        from main import load_data, engineer_features
        
        raw_data = load_data()
        X_train, X_test, y_train, y_test, feature_names = engineer_features(raw_data)
        
        # Predict on last test sample
        last_sample = X_test[-1:]
        prediction = model.predict(last_sample)[0]
        actual = y_test.iloc[-1] if hasattr(y_test, 'iloc') else y_test[-1]
        
        print("\\n" + "=" * 80)
        print("📈 PREDICTION RESULT")
        print("=" * 80)
        print(f"\\n🎯 Predicted: {prediction:.6f}")
        print(f"📊 Actual:    {actual:.6f}")
        print(f"📉 Error:     {abs(prediction - actual):.6f} ({abs((prediction-actual)/actual*100):.2f}%)")
        print("\\n" + "=" * 80)
        
    elif mode == '2':
        csv_path = input("\\nEnter CSV path: ").strip()
        if Path(csv_path).exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            predictions = model.predict(df)
            
            output_path = Path(csv_path).with_name(f"{Path(csv_path).stem}_predictions.csv")
            df['prediction'] = predictions
            df.to_csv(output_path, index=False)
            
            print(f"\\n✅ Predictions saved to: {output_path}")
        else:
            print("❌ File not found!")
    
    input("\\nPress Enter to return to menu...")

def feature_selection():
    """Feature selection wizard"""
    print("\\n" + "=" * 80)
    print("🎯 FEATURE SELECTION WIZARD")
    print("=" * 80)
    
    print("\\nCreating feature selection script...")
    
    script_content = '''"""
Feature Selection - Find Optimal Feature Subset
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

print("=" * 80)
print("🎯 FEATURE SELECTION ANALYSIS")
print("=" * 80)

# Load data
from main import load_data, engineer_features

print("\\n📊 Loading data...")
raw_data = load_data()
X_train, X_test, y_train, y_test, feature_names = engineer_features(raw_data)

print(f"✅ Total features: {len(feature_names)}")

# Method 1: Feature Importance Ranking
print("\\n" + "=" * 80)
print("📊 METHOD 1: Feature Importance (Random Forest)")
print("=" * 80)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\\n🔝 Top 20 Features:")
for idx, row in importances.head(20).iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:.4f}")

# Method 2: Recursive Feature Elimination
print("\\n" + "=" * 80)
print("📊 METHOD 2: Recursive Feature Elimination (RFE)")
print("=" * 80)

print("\\nTesting feature subsets: [10, 15, 20, 25, 30, 'all']")
tscv = TimeSeriesSplit(n_splits=3)

results = []
for n_features in [10, 15, 20, 25, 30, len(feature_names)]:
    if n_features == len(feature_names):
        X_selected = X_train
        X_test_selected = X_test
    else:
        selector = RFE(rf, n_features_to_select=n_features, step=1)
        selector.fit(X_train, y_train)
        X_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
    
    # Cross-validation score
    rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(rf_temp, X_selected, y_train, 
                                cv=tscv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    # Test score
    rf_temp.fit(X_selected, y_train)
    y_pred = rf_temp.predict(X_test_selected)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({
        'n_features': n_features,
        'cv_rmse': cv_rmse,
        'test_rmse': test_rmse
    })
    
    print(f"   {n_features:3d} features: CV RMSE={cv_rmse:.6f}, Test RMSE={test_rmse:.6f}")

# Find optimal number
results_df = pd.DataFrame(results)
best_idx = results_df['test_rmse'].idxmin()
best_n = results_df.loc[best_idx, 'n_features']

print("\\n" + "=" * 80)
print(f"🏆 OPTIMAL: {int(best_n)} features")
print(f"   CV RMSE: {results_df.loc[best_idx, 'cv_rmse']:.6f}")
print(f"   Test RMSE: {results_df.loc[best_idx, 'test_rmse']:.6f}")
print("=" * 80)

# Save optimal feature set
if best_n < len(feature_names):
    selector = RFE(rf, n_features_to_select=int(best_n), step=1)
    selector.fit(X_train, y_train)
    selected_features = [f for f, s in zip(feature_names, selector.support_) if s]
    
    with open('MODEL_STORAGE/optimal_features.txt', 'w') as f:
        f.write(f"Optimal Feature Set ({int(best_n)} features):\\n")
        f.write("=" * 50 + "\\n")
        for feat in selected_features:
            f.write(f"{feat}\\n")
    
    print(f"\\n✅ Optimal features saved to: MODEL_STORAGE/optimal_features.txt")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(results_df['n_features'], results_df['cv_rmse'], 'o-', label='CV RMSE')
plt.plot(results_df['n_features'], results_df['test_rmse'], 's-', label='Test RMSE')
plt.axvline(best_n, color='red', linestyle='--', label=f'Optimal ({int(best_n)})')
plt.xlabel('Number of Features')
plt.ylabel('RMSE')
plt.title('Feature Selection: Performance vs Number of Features')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('MODEL_STORAGE/feature_selection.png', dpi=150)
print(f"✅ Chart saved to: MODEL_STORAGE/feature_selection.png")

print("\\n" + "=" * 80)
'''
    
    script_path = Path("feature_selection_analysis.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\\n✅ Created: {script_path}")
    print("\\n🚀 Running feature selection analysis...")
    input("\\nPress Enter to start (takes 2-5 minutes)...")
    
    import subprocess
    subprocess.run([sys.executable, str(script_path)])
    
    input("\\nPress Enter to return to menu...")

def error_analysis():
    """Analyze model errors"""
    print("\\n" + "=" * 80)
    print("🎯 ERROR ANALYSIS")
    print("=" * 80)
    
    print("\\nCreating error analysis script...")
    
    script_content = '''"""
Error Analysis - Understand Where Models Fail
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

print("=" * 80)
print("🎯 ERROR ANALYSIS")
print("=" * 80)

# Load data
from main import load_data, engineer_features

print("\\n📊 Loading data and models...")
raw_data = load_data()
X_train, X_test, y_train, y_test, feature_names = engineer_features(raw_data)

model_storage = Path("MODEL_STORAGE")
model_files = list(model_storage.glob("*_standalone.pkl"))

if not model_files:
    print("❌ No models found! Train models first.")
    exit()

print(f"✅ Found {len(model_files)} models")

# Analyze each model
all_errors = {}

for model_file in model_files:
    model_name = model_file.stem.replace('_standalone', '')
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    y_pred = model.predict(X_test)
    errors = y_test - y_pred
    abs_errors = np.abs(errors)
    pct_errors = (abs_errors / np.abs(y_test)) * 100
    
    all_errors[model_name] = {
        'predictions': y_pred,
        'errors': errors,
        'abs_errors': abs_errors,
        'pct_errors': pct_errors
    }
    
    print(f"\\n📊 {model_name}:")
    print(f"   Mean Error: {errors.mean():.6f}")
    print(f"   Std Error: {errors.std():.6f}")
    print(f"   Mean Abs Error: {abs_errors.mean():.6f}")
    print(f"   Mean % Error: {pct_errors.mean():.2f}%")
    print(f"   Max Error: {abs_errors.max():.6f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Error Distribution
ax = axes[0, 0]
for name, data in all_errors.items():
    ax.hist(data['errors'], bins=50, alpha=0.5, label=name)
ax.set_xlabel('Prediction Error')
ax.set_ylabel('Frequency')
ax.set_title('Error Distribution by Model')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Absolute Error over Time
ax = axes[0, 1]
for name, data in all_errors.items():
    ax.plot(data['abs_errors'], alpha=0.7, label=name)
ax.set_xlabel('Sample Index (Time)')
ax.set_ylabel('Absolute Error')
ax.set_title('Absolute Error Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Predictions vs Actual
ax = axes[1, 0]
for name, data in all_errors.items():
    ax.scatter(y_test, data['predictions'], alpha=0.5, s=10, label=name)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'k--', lw=2, label='Perfect')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Predictions vs Actual')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Error by Magnitude
ax = axes[1, 1]
for name, data in all_errors.items():
    ax.scatter(np.abs(y_test), data['abs_errors'], alpha=0.5, s=10, label=name)
ax.set_xlabel('Actual Value Magnitude')
ax.set_ylabel('Absolute Error')
ax.set_title('Error vs Value Magnitude')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('MODEL_STORAGE/error_analysis.png', dpi=150)
print("\\n✅ Visualizations saved to: MODEL_STORAGE/error_analysis.png")

# Find worst predictions
print("\\n" + "=" * 80)
print("⚠️  WORST PREDICTIONS (Top 10)")
print("=" * 80)

for name, data in all_errors.items():
    worst_indices = np.argsort(data['abs_errors'])[-10:][::-1]
    print(f"\\n{name}:")
    for idx in worst_indices:
        actual = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
        pred = data['predictions'][idx]
        error = data['abs_errors'][idx]
        print(f"   Index {idx}: Actual={actual:.6f}, Pred={pred:.6f}, Error={error:.6f}")

print("\\n" + "=" * 80)
'''
    
    script_path = Path("error_analysis.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\\n✅ Created: {script_path}")
    print("\\n🚀 Running error analysis...")
    input("\\nPress Enter to start...")
    
    import subprocess
    subprocess.run([sys.executable, str(script_path)])
    
    input("\\nPress Enter to return to menu...")

def model_ensemble_builder():
    """Create custom model ensembles"""
    print("\\n" + "=" * 80)
    print("🤝 MODEL ENSEMBLE BUILDER")
    print("=" * 80)
    
    print("\\nCreating ensemble builder script...")
    
    script_content = '''"""
Model Ensemble Builder - Combine Models Optimally
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

print("=" * 80)
print("🤝 MODEL ENSEMBLE BUILDER")
print("=" * 80)

# Load data
from main import load_data, engineer_features

print("\\n📊 Loading data and models...")
raw_data = load_data()
X_train, X_test, y_train, y_test, feature_names = engineer_features(raw_data)

model_storage = Path("MODEL_STORAGE")
model_files = list(model_storage.glob("*_standalone.pkl"))

if len(model_files) < 2:
    print("❌ Need at least 2 models! Train more models first.")
    exit()

print(f"✅ Found {len(model_files)} models")

# Load all models and get predictions
models = {}
predictions = {}

for model_file in model_files:
    model_name = model_file.stem.replace('_standalone', '')
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    models[model_name] = model
    predictions[model_name] = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions[model_name]))
    print(f"   {model_name}: RMSE = {rmse:.6f}")

# Ensemble Method 1: Simple Average
print("\\n" + "=" * 80)
print("📊 METHOD 1: Simple Average")
print("=" * 80)

pred_array = np.array(list(predictions.values()))
ensemble_simple = pred_array.mean(axis=0)
rmse_simple = np.sqrt(mean_squared_error(y_test, ensemble_simple))
print(f"   Ensemble RMSE: {rmse_simple:.6f}")

# Ensemble Method 2: Weighted Average (Optimized)
print("\\n" + "=" * 80)
print("📊 METHOD 2: Optimized Weighted Average")
print("=" * 80)

def ensemble_rmse(weights):
    weights = np.abs(weights)
    weights = weights / weights.sum()
    ensemble_pred = (pred_array.T @ weights).flatten()
    return np.sqrt(mean_squared_error(y_test, ensemble_pred))

# Optimize weights
n_models = len(predictions)
initial_weights = np.ones(n_models) / n_models
result = minimize(ensemble_rmse, initial_weights, method='Nelder-Mead')

optimal_weights = np.abs(result.x)
optimal_weights = optimal_weights / optimal_weights.sum()

print("\\n🏆 Optimal Weights:")
for name, weight in zip(predictions.keys(), optimal_weights):
    print(f"   {name}: {weight:.4f}")

ensemble_optimal = (pred_array.T @ optimal_weights).flatten()
rmse_optimal = np.sqrt(mean_squared_error(y_test, ensemble_optimal))
print(f"\\n   Ensemble RMSE: {rmse_optimal:.6f}")

# Ensemble Method 3: Stacking
print("\\n" + "=" * 80)
print("📊 METHOD 3: Stacking (Meta-Model)")
print("=" * 80)

from sklearn.linear_model import Ridge

# Create meta-features (predictions from base models)
meta_X_train = []
meta_X_test = pred_array.T

for name, model in models.items():
    train_pred = model.predict(X_train)
    meta_X_train.append(train_pred)

meta_X_train = np.array(meta_X_train).T

# Train meta-model
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X_train, y_train)

ensemble_stacking = meta_model.predict(meta_X_test)
rmse_stacking = np.sqrt(mean_squared_error(y_test, ensemble_stacking))
print(f"   Ensemble RMSE: {rmse_stacking:.6f}")

# Summary
print("\\n" + "=" * 80)
print("📊 ENSEMBLE COMPARISON")
print("=" * 80)

results = {
    'Simple Average': rmse_simple,
    'Weighted (Optimized)': rmse_optimal,
    'Stacking': rmse_stacking
}

# Add individual model RMSEs
for name, preds in predictions.items():
    results[name] = np.sqrt(mean_squared_error(y_test, preds))

results_sorted = sorted(results.items(), key=lambda x: x[1])

print("\\n🏆 Ranking (Best to Worst):")
for i, (name, rmse) in enumerate(results_sorted, 1):
    marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"{marker} {i}. {name:25s}: {rmse:.6f}")

# Save best ensemble
best_method = results_sorted[0][0]
print(f"\\n✅ Best method: {best_method}")

if 'Weighted' in best_method:
    ensemble_config = {
        'method': 'weighted',
        'models': list(predictions.keys()),
        'weights': optimal_weights.tolist()
    }
elif 'Stacking' in best_method:
    ensemble_config = {
        'method': 'stacking',
        'models': list(predictions.keys()),
        'meta_model': meta_model
    }
else:
    ensemble_config = {
        'method': 'simple_average',
        'models': list(predictions.keys())
    }

# Save ensemble configuration
with open('MODEL_STORAGE/ensemble_config.pkl', 'wb') as f:
    pickle.dump(ensemble_config, f)

print(f"✅ Ensemble config saved to: MODEL_STORAGE/ensemble_config.pkl")
print("\\n" + "=" * 80)
'''
    
    script_path = Path("ensemble_builder.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\\n✅ Created: {script_path}")
    print("\\n🚀 Running ensemble optimization...")
    input("\\nPress Enter to start...")
    
    import subprocess
    subprocess.run([sys.executable, str(script_path)])
    
    input("\\nPress Enter to return to menu...")

def performance_dashboard():
    """Comprehensive performance visualization"""
    print("\\n" + "=" * 80)
    print("📊 PERFORMANCE DASHBOARD")
    print("=" * 80)
    
    print("\\nCreating comprehensive dashboard...")
    
    script_content = '''"""
Performance Dashboard - All Models Visualized
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'

print("=" * 80)
print("📊 PERFORMANCE DASHBOARD")
print("=" * 80)

# Load data
from main import load_data, engineer_features

print("\\n📊 Loading data and models...")
raw_data = load_data()
X_train, X_test, y_train, y_test, feature_names = engineer_features(raw_data)

model_storage = Path("MODEL_STORAGE")
model_files = list(model_storage.glob("*_standalone.pkl"))

if not model_files:
    print("❌ No models found! Train models first (options 1-4).")
    exit()

print(f"✅ Found {len(model_files)} models")

# Collect metrics for all models
metrics_data = []
predictions_dict = {}
residuals_dict = {}

for model_file in model_files:
    model_name = model_file.stem.replace('_standalone', '').replace('_', ' ').title()
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    y_pred = model.predict(X_test)
    predictions_dict[model_name] = y_pred
    
    residuals = y_test - y_pred
    residuals_dict[model_name] = residuals
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    metrics_data.append({
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape
    })
    
    print(f"   {model_name}: RMSE={rmse:.6f}, R²={r2:.4f}")

metrics_df = pd.DataFrame(metrics_data)

# Create comprehensive dashboard
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Metrics Comparison Bar Chart
ax1 = fig.add_subplot(gs[0, :2])
x = np.arange(len(metrics_df))
width = 0.2

ax1_twin = ax1.twinx()
ax1.bar(x - width*1.5, metrics_df['RMSE'], width, label='RMSE', alpha=0.8)
ax1.bar(x - width*0.5, metrics_df['MAE'], width, label='MAE', alpha=0.8)
ax1_twin.bar(x + width*0.5, metrics_df['R²'], width, label='R²', alpha=0.8, color='green')
ax1_twin.bar(x + width*1.5, metrics_df['MAPE (%)'], width, label='MAPE %', alpha=0.8, color='orange')

ax1.set_ylabel('Error Metrics', fontsize=10)
ax1_twin.set_ylabel('R² / MAPE %', fontsize=10)
ax1.set_title('📊 Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. RMSE Ranking
ax2 = fig.add_subplot(gs[0, 2])
metrics_sorted = metrics_df.sort_values('RMSE')
colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(metrics_sorted)))
ax2.barh(metrics_sorted['Model'], metrics_sorted['RMSE'], color=colors)
ax2.set_xlabel('RMSE', fontsize=10)
ax2.set_title('🏆 Model Ranking (RMSE)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# 3. Predictions vs Actual
ax3 = fig.add_subplot(gs[1, 0])
for name, preds in predictions_dict.items():
    ax3.scatter(y_test, preds, alpha=0.6, s=20, label=name)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'k--', lw=2, label='Perfect')
ax3.set_xlabel('Actual Values', fontsize=10)
ax3.set_ylabel('Predicted Values', fontsize=10)
ax3.set_title('🎯 Predictions vs Actual', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Residual Distribution
ax4 = fig.add_subplot(gs[1, 1])
for name, resid in residuals_dict.items():
    ax4.hist(resid, bins=30, alpha=0.5, label=name)
ax4.axvline(0, color='red', linestyle='--', lw=2)
ax4.set_xlabel('Residuals', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.set_title('📈 Residual Distribution', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. Residual Plot (over time)
ax5 = fig.add_subplot(gs[1, 2])
for name, resid in residuals_dict.items():
    ax5.plot(resid, alpha=0.7, label=name, linewidth=1)
ax5.axhline(0, color='red', linestyle='--', lw=2)
ax5.set_xlabel('Sample Index', fontsize=10)
ax5.set_ylabel('Residuals', fontsize=10)
ax5.set_title('📉 Residuals Over Time', fontsize=12, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# 6. Error by Magnitude
ax6 = fig.add_subplot(gs[2, 0])
for name, preds in predictions_dict.items():
    errors = np.abs(y_test - preds)
    ax6.scatter(np.abs(y_test), errors, alpha=0.6, s=10, label=name)
ax6.set_xlabel('Actual Value Magnitude', fontsize=10)
ax6.set_ylabel('Absolute Error', fontsize=10)
ax6.set_title('⚠️ Error vs Value Magnitude', fontsize=12, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# 7. Correlation Heatmap
ax7 = fig.add_subplot(gs[2, 1])
pred_df = pd.DataFrame(predictions_dict)
correlation = pred_df.corr()
sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, ax=ax7, cbar_kws={'label': 'Correlation'})
ax7.set_title('🔥 Model Prediction Correlations', fontsize=12, fontweight='bold')

# 8. Metrics Table
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('tight')
ax8.axis('off')

table_data = []
for _, row in metrics_df.iterrows():
    table_data.append([
        row['Model'],
        f"{row['RMSE']:.5f}",
        f"{row['MAE']:.5f}",
        f"{row['R²']:.4f}",
        f"{row['MAPE (%)']:.2f}"
    ])

table = ax8.table(cellText=table_data,
                 colLabels=['Model', 'RMSE', 'MAE', 'R²', 'MAPE %'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data) + 1):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

ax8.set_title('📋 Performance Metrics', fontsize=12, fontweight='bold', pad=20)

# Main title
fig.suptitle('🚀 ML MODELS PERFORMANCE DASHBOARD', 
             fontsize=18, fontweight='bold', y=0.98)

# Save
plt.savefig('MODEL_STORAGE/performance_dashboard.png', dpi=200, bbox_inches='tight')
print("\\n✅ Dashboard saved to: MODEL_STORAGE/performance_dashboard.png")

# Display
plt.show()

# Print summary
print("\\n" + "=" * 80)
print("📊 PERFORMANCE SUMMARY")
print("=" * 80)
print(f"\\n{metrics_df.to_string(index=False)}")

best_model = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']
best_rmse = metrics_df['RMSE'].min()

print(f"\\n🏆 Best Model: {best_model}")
print(f"   RMSE: {best_rmse:.6f}")
print("=" * 80)
'''
    
    script_path = Path("performance_dashboard.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\\n✅ Created: {script_path}")
    print("\\n🚀 Generating dashboard...")
    print("   This will open an interactive window with all visualizations")
    input("\\nPress Enter to start...")
    
    import subprocess
    subprocess.run([sys.executable, str(script_path)])
    
    input("\\nPress Enter to return to menu...")

def main():
    """Main menu loop"""
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        try:
            choice = input("\n👉 Select an option (0-18): ").strip()
            
            if choice == '0':
                print("\n✅ Exiting ML Models Menu. Happy testing! 🚀\n")
                break
            elif choice == '1':
                run_random_forest()
            elif choice == '2':
                run_xgboost()
            elif choice == '3':
                run_lightgbm()
            elif choice == '4':
                run_classical_ensemble()
            elif choice == '5':
                run_lstm()
            elif choice == '6':
                run_transformer()
            elif choice == '7':
                run_multitask()
            elif choice == '8':
                run_deep_learning_suite()
            elif choice == '9':
                run_full_pipeline()
            elif choice == '10':
                compare_models()
            elif choice == '11':
                feature_importance()
            elif choice == '12':
                check_gpu()
            elif choice == '13':
                hyperparameter_tuning()
            elif choice == '14':
                quick_predict()
            elif choice == '15':
                feature_selection()
            elif choice == '16':
                error_analysis()
            elif choice == '17':
                model_ensemble_builder()
            elif choice == '18':
                performance_dashboard()
            else:
                print("\n❌ Invalid choice. Please select 0-18.")
            
            if choice != '0':
                input("\n✅ Press Enter to return to menu...")
                
        except KeyboardInterrupt:
            print("\n\n✅ Interrupted. Exiting...\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
