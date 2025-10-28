"""
Local Storage Manager for Bitcoin Price Prediction System
===========================================================
Handles saving and loading:
- Training runs and metadata
- Model predictions
- Performance metrics
- External data snapshots
- Feature importance
"""

import pandas as pd
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get script directory
SCRIPT_DIR = Path(__file__).parent.absolute()

# Create storage directories
STORAGE_ROOT = SCRIPT_DIR / 'MODEL_STORAGE'
STORAGE_ROOT.mkdir(exist_ok=True)

RUNS_DIR = STORAGE_ROOT / 'training_runs'
PREDICTIONS_DIR = STORAGE_ROOT / 'predictions'
METRICS_DIR = STORAGE_ROOT / 'metrics'
MODELS_DIR = STORAGE_ROOT / 'saved_models'
EXTERNAL_DATA_DIR = STORAGE_ROOT / 'external_data'
FEATURES_DIR = STORAGE_ROOT / 'feature_data'

for dir_path in [RUNS_DIR, PREDICTIONS_DIR, METRICS_DIR, MODELS_DIR, EXTERNAL_DATA_DIR, FEATURES_DIR]:
    dir_path.mkdir(exist_ok=True)


class ModelStorageManager:
    """Manages saving and loading of all model-related data"""
    
    def __init__(self):
        self.storage_root = STORAGE_ROOT
        logging.info(f"📁 Storage Manager initialized")
        logging.info(f"   Root directory: {STORAGE_ROOT}")
    
    # =========================================================================
    # TRAINING RUN MANAGEMENT
    # =========================================================================
    
    def save_training_run(self, run_data):
        """
        Save complete training run information
        
        Args:
            run_data: dict with keys:
                - timestamp: run timestamp
                - config: configuration dict
                - metrics: performance metrics
                - duration: training duration
                - models_used: list of model names
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            'run_id': run_id,
            'timestamp': timestamp,
            'config': run_data.get('config', {}),
            'duration_seconds': run_data.get('duration', 0),
            'models_used': run_data.get('models_used', [])
        }
        
        with open(run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save metrics
        metrics = run_data.get('metrics', {})
        with open(run_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logging.info(f"✅ Saved training run: {run_id}")
        logging.info(f"   Location: {run_dir}")
        
        return run_id
    
    def load_training_run(self, run_id):
        """Load training run data"""
        run_dir = RUNS_DIR / run_id
        
        if not run_dir.exists():
            logging.warning(f"⚠️  Run not found: {run_id}")
            return None
        
        with open(run_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        with open(run_dir / 'metrics.json', 'r') as f:
            metrics = json.load(f)
        
        return {
            'metadata': metadata,
            'metrics': metrics
        }
    
    def list_training_runs(self, limit=10):
        """List recent training runs"""
        runs = sorted(RUNS_DIR.glob('run_*'), key=lambda x: x.stat().st_mtime, reverse=True)
        
        logging.info(f"\n📋 Recent Training Runs ({len(runs)} total):")
        for i, run_dir in enumerate(runs[:limit]):
            try:
                with open(run_dir / 'metadata.json', 'r') as f:
                    meta = json.load(f)
                with open(run_dir / 'metrics.json', 'r') as f:
                    metrics = json.load(f)
                
                logging.info(f"   {i+1}. {run_dir.name}")
                logging.info(f"      RMSE: {metrics.get('test_rmse', 'N/A')}")
                logging.info(f"      Duration: {meta.get('duration_seconds', 0):.1f}s")
                logging.info(f"      Models: {', '.join(meta.get('models_used', []))}")
            except Exception as e:
                logging.warning(f"   Error reading {run_dir.name}: {e}")
        
        return runs
    
    # =========================================================================
    # PREDICTION STORAGE
    # =========================================================================
    
    def save_predictions(self, predictions_df, run_id=None, name="predictions"):
        """
        Save prediction results
        
        Args:
            predictions_df: DataFrame with predictions
            run_id: optional run ID to associate with
            name: prediction name/type
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if run_id:
            filename = f"{run_id}_{name}_{timestamp}.csv"
        else:
            filename = f"{name}_{timestamp}.csv"
        
        filepath = PREDICTIONS_DIR / filename
        predictions_df.to_csv(filepath, index=True)
        
        logging.info(f"✅ Saved predictions: {filename}")
        return filepath
    
    def load_latest_predictions(self, name="predictions"):
        """Load most recent predictions"""
        pattern = f"*{name}*.csv"
        files = sorted(PREDICTIONS_DIR.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if files:
            df = pd.read_csv(files[0])
            logging.info(f"✅ Loaded predictions from: {files[0].name}")
            return df
        else:
            logging.warning(f"⚠️  No predictions found matching: {pattern}")
            return None
    
    # =========================================================================
    # MODEL PERSISTENCE
    # =========================================================================
    
    def save_model(self, model, model_name, run_id=None):
        """
        Save trained model
        
        Args:
            model: trained model object
            model_name: name of model (e.g., 'lstm', 'transformer')
            run_id: optional run ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if run_id:
            filename = f"{run_id}_{model_name}_{timestamp}.pth"
        else:
            filename = f"{model_name}_{timestamp}.pth"
        
        filepath = MODELS_DIR / filename
        
        # Handle PyTorch models
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), filepath)
        else:
            # Handle sklearn models
            with open(filepath.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        logging.info(f"✅ Saved model: {filename}")
        return filepath
    
    def load_model(self, model_name, model_class=None):
        """
        Load most recent model
        
        Args:
            model_name: name of model
            model_class: PyTorch model class (required for PyTorch models)
        """
        pattern = f"*{model_name}*.pth"
        pth_files = sorted(MODELS_DIR.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        
        pattern_pkl = f"*{model_name}*.pkl"
        pkl_files = sorted(MODELS_DIR.glob(pattern_pkl), key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Try PyTorch first
        if pth_files and model_class:
            model = model_class()
            model.load_state_dict(torch.load(pth_files[0]))
            logging.info(f"✅ Loaded PyTorch model: {pth_files[0].name}")
            return model
        
        # Try pickle
        elif pkl_files:
            with open(pkl_files[0], 'rb') as f:
                model = pickle.load(f)
            logging.info(f"✅ Loaded pickle model: {pkl_files[0].name}")
            return model
        
        else:
            logging.warning(f"⚠️  No model found: {model_name}")
            return None
    
    # =========================================================================
    # EXTERNAL DATA STORAGE
    # =========================================================================
    
    def save_external_data(self, data_dict, run_id=None):
        """Save external data snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if run_id:
            filename = f"{run_id}_external_{timestamp}.json"
        else:
            filename = f"external_{timestamp}.json"
        
        filepath = EXTERNAL_DATA_DIR / filename
        
        with open(filepath, 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        logging.info(f"✅ Saved external data: {filename}")
        return filepath
    
    # =========================================================================
    # FEATURE IMPORTANCE & ANALYSIS
    # =========================================================================
    
    def save_feature_importance(self, feature_names, importances, model_name, run_id=None):
        """Save feature importance scores"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if run_id:
            filename = f"{run_id}_{model_name}_features_{timestamp}.csv"
        else:
            filename = f"{model_name}_features_{timestamp}.csv"
        
        filepath = FEATURES_DIR / filename
        df.to_csv(filepath, index=False)
        
        logging.info(f"✅ Saved feature importance: {filename}")
        return filepath
    
    # =========================================================================
    # STATISTICS & SUMMARIES
    # =========================================================================
    
    def get_storage_stats(self):
        """Get statistics about stored data"""
        stats = {
            'training_runs': len(list(RUNS_DIR.glob('run_*'))),
            'predictions': len(list(PREDICTIONS_DIR.glob('*.csv'))),
            'models': len(list(MODELS_DIR.glob('*.pth'))) + len(list(MODELS_DIR.glob('*.pkl'))),
            'external_data': len(list(EXTERNAL_DATA_DIR.glob('*.json'))),
            'feature_importance': len(list(FEATURES_DIR.glob('*.csv')))
        }
        
        # Calculate total size
        total_size = 0
        for dir_path in [RUNS_DIR, PREDICTIONS_DIR, METRICS_DIR, MODELS_DIR, EXTERNAL_DATA_DIR, FEATURES_DIR]:
            for file in dir_path.rglob('*'):
                if file.is_file():
                    total_size += file.stat().st_size
        
        stats['total_size_mb'] = total_size / (1024 * 1024)
        
        return stats
    
    def print_storage_summary(self):
        """Print summary of stored data"""
        stats = self.get_storage_stats()
        
        print("\n" + "="*70)
        print("📊 STORAGE SUMMARY")
        print("="*70)
        print(f"\n📁 Storage Location: {STORAGE_ROOT}")
        print(f"\n📈 Stored Data:")
        print(f"   Training Runs: {stats['training_runs']}")
        print(f"   Predictions: {stats['predictions']}")
        print(f"   Saved Models: {stats['models']}")
        print(f"   External Data Snapshots: {stats['external_data']}")
        print(f"   Feature Importance Files: {stats['feature_importance']}")
        print(f"\n💾 Total Storage Used: {stats['total_size_mb']:.2f} MB")
        print("="*70 + "\n")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("💾 MODEL STORAGE MANAGER - TEST")
    print("="*70)
    
    manager = ModelStorageManager()
    
    # Test: Save a training run
    test_run = {
        'config': {
            'USE_LSTM': True,
            'USE_TRANSFORMER': True,
            'USE_MULTITASK': True,
            'LSTM_EPOCHS': 150
        },
        'metrics': {
            'test_rmse': 0.0066,
            'train_rmse': 0.0050,
            'rf_rmse': 0.0070,
            'xgb_rmse': 0.0068,
            'lstm_rmse': 0.0065,
            'transformer_rmse': 0.0064,
            'multitask_rmse': 0.0062
        },
        'duration': 868.9,
        'models_used': ['RandomForest', 'XGBoost', 'LightGBM', 'LSTM', 'Transformer', 'MultiTask']
    }
    
    run_id = manager.save_training_run(test_run)
    
    # Test: Save predictions
    test_predictions = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=12, freq='1H'),
        'predicted': [100000 + i*100 for i in range(12)],
        'lower_bound': [99000 + i*100 for i in range(12)],
        'upper_bound': [101000 + i*100 for i in range(12)]
    })
    
    manager.save_predictions(test_predictions, run_id=run_id, name="test_forecast")
    
    # Test: Save external data
    test_external = {
        'fear_greed': 65,
        'google_trends': 72,
        'social_sentiment': 0.3
    }
    
    manager.save_external_data(test_external, run_id=run_id)
    
    # Test: Save feature importance
    test_features = ['price', 'volume', 'rsi', 'macd', 'volatility']
    test_importance = [0.35, 0.25, 0.20, 0.12, 0.08]
    
    manager.save_feature_importance(test_features, test_importance, 'test_model', run_id=run_id)
    
    # Show storage summary
    manager.print_storage_summary()
    
    # List recent runs
    manager.list_training_runs(limit=5)
    
    print("\n✅ Storage system test complete!")
    print(f"   Test data saved under run_id: {run_id}")
