
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import xgboost as xgb
import logging
import time
from datetime import datetime
import os
from pathlib import Path

# Import our new enhancement modules
from external_data import ExternalDataCollector
from enhanced_features import add_all_enhanced_features, ENHANCED_FEATURE_GROUPS
from storage_manager import ModelStorageManager

# Try to import LightGBM, but continue without it if unavailable
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, OSError) as e:
    logging.warning(f"LightGBM not available: {e}")
    HAS_LIGHTGBM = False

# Try to import PyTorch for neural networks
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = torch.cuda.is_available()
    if HAS_PYTORCH:
        DEVICE = torch.device('cuda')
        logging.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
        logging.warning("GPU not available, using CPU for neural networks")
except (ImportError, OSError) as e:
    logging.warning(f"PyTorch not available: {e}")
    HAS_PYTORCH = False
    DEVICE = None

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# Build absolute paths relative to script location
FILE_PATH_90DAY = SCRIPT_DIR / 'DATA' / 'BTC_90day_data.csv'
FILE_PATH_1H = SCRIPT_DIR / 'attached_assets' / 'COINBASE_BTCUSD, 60_5c089_1761289206450.csv'
FILE_PATH_4H = SCRIPT_DIR / 'attached_assets' / 'COINBASE_BTCUSD, 240_739cb_1761290157184.csv'
FILE_PATH_12H = SCRIPT_DIR / 'attached_assets' / 'COINBASE_BTCUSD, 720_c69cd_1761290157184.csv'
FILE_PATH_1D = SCRIPT_DIR / 'attached_assets' / 'COINBASE_BTCUSD, 1D_87ac3_1761289206450.csv'
FILE_PATH_1W = SCRIPT_DIR / 'attached_assets' / 'COINBASE_BTCUSD, 1W_9771c_1761290157184.csv'

# Yahoo Finance data paths (fallback/supplement)
YF_FILE_PATH_1H = SCRIPT_DIR / 'DATA' / 'yf_eth_1h.csv'
YF_FILE_PATH_4H = SCRIPT_DIR / 'DATA' / 'yf_eth_4h.csv'
YF_FILE_PATH_12H = SCRIPT_DIR / 'DATA' / 'yf_eth_12h.csv'
YF_FILE_PATH_1D = SCRIPT_DIR / 'DATA' / 'yf_eth_1d.csv'
YF_FILE_PATH_1W = SCRIPT_DIR / 'DATA' / 'yf_btc_1w.csv'

USE_YAHOO_FINANCE = True  # Set to True to use YF data instead
TEST_SIZE = 0.2
PREDICT_STEPS = 12  # Predict next 12 hours
USE_ENSEMBLE = True  # Use ensemble of RF + XGBoost + LightGBM
USE_LSTM = True  # Use LSTM neural network (requires GPU)
USE_ATTENTION = True  # Use attention mechanism in LSTM (Phase 2!)
USE_TRANSFORMER = True  # 🚀 Use Transformer architecture (Phase 3!)
USE_MULTITASK = True  # 🎯 Use Multi-Task Learning (Phase 4!)

# Feature Selection (Phase 5 - Run 5 with Interactions!)
USE_FEATURE_SELECTION = True  # Use only high-importance features (39 features: 33 original + 16 interactions)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SELECTED_FEATURES_PATH = os.path.join(SCRIPT_DIR, 'MODEL_STORAGE', 'feature_data', 'selected_features_with_interactions.txt')

LSTM_SEQUENCE_LENGTH = 48  # Number of time steps to look back (48 hours - increased!)
LSTM_EPOCHS = 150  # Number of training epochs for LSTM (increased for attention)
LSTM_BATCH_SIZE = 128  # Batch size for LSTM training (increased for GPU!)
LSTM_LEARNING_RATE = 0.001  # Learning rate for LSTM
# Transformer Configuration (Phase 3)
TRANSFORMER_HEADS = 8  # Number of attention heads
TRANSFORMER_LAYERS = 4  # Number of transformer encoder layers
TRANSFORMER_DIM = 256  # Dimension of transformer embeddings
TRANSFORMER_DROPOUT = 0.1  # Dropout rate
# Multi-Task Configuration (Phase 4)
MULTITASK_EPOCHS = 150  # Training epochs for multi-task model
MULTITASK_DROPOUT_SAMPLES = 50  # Number of dropout samples for uncertainty
DIRECTION_THRESHOLD = 0.005  # 0.5% threshold for up/down/stable classification

# Load and preprocess multi-timeframe data
def load_multi_timeframe_data():
    """Loads all timeframe data and returns them as dataframes."""
    try:
        # Load 90-day baseline data (only needed for Coinbase data)
        if not USE_YAHOO_FINANCE:
            df_90day = pd.read_csv(FILE_PATH_90DAY)
            df_90day['timestamp'] = pd.to_datetime(df_90day['timestamp'])
            df_90day.set_index('timestamp', inplace=True)
        else:
            df_90day = None  # Not needed for Yahoo Finance
        
        if USE_YAHOO_FINANCE:
            logging.info("Loading Yahoo Finance data...")
            
            # Load Yahoo Finance data
            df_1h = pd.read_csv(YF_FILE_PATH_1H)
            df_1h['time'] = pd.to_datetime(df_1h['time'])
            df_1h.set_index('time', inplace=True)
            
            df_4h = pd.read_csv(YF_FILE_PATH_4H)
            df_4h['time'] = pd.to_datetime(df_4h['time'])
            df_4h.set_index('time', inplace=True)
            
            df_12h = pd.read_csv(YF_FILE_PATH_12H)
            df_12h['time'] = pd.to_datetime(df_12h['time'])
            df_12h.set_index('time', inplace=True)
            
            df_1d = pd.read_csv(YF_FILE_PATH_1D)
            df_1d['time'] = pd.to_datetime(df_1d['time'])
            df_1d.set_index('time', inplace=True)
            
            df_1w = pd.read_csv(YF_FILE_PATH_1W)
            df_1w['time'] = pd.to_datetime(df_1w['time'])
            df_1w.set_index('time', inplace=True)
            
        else:
            logging.info("Loading Coinbase indicator data...")
            
            # Load 1-hour data with indicators
            df_1h = pd.read_csv(FILE_PATH_1H)
            df_1h['time'] = pd.to_datetime(df_1h['time'], unit='s')
            df_1h.set_index('time', inplace=True)
            
            # Load 4-hour data with indicators
            df_4h = pd.read_csv(FILE_PATH_4H)
            df_4h['time'] = pd.to_datetime(df_4h['time'], unit='s')
            df_4h.set_index('time', inplace=True)
            
            # Load 12-hour data with indicators
            df_12h = pd.read_csv(FILE_PATH_12H)
            df_12h['time'] = pd.to_datetime(df_12h['time'], unit='s')
            df_12h.set_index('time', inplace=True)
            
            # Load 1-day data with indicators
            df_1d = pd.read_csv(FILE_PATH_1D)
            df_1d['time'] = pd.to_datetime(df_1d['time'], unit='s')
            df_1d.set_index('time', inplace=True)
            
            # Load 1-week data with indicators
            df_1w = pd.read_csv(FILE_PATH_1W)
            df_1w['time'] = pd.to_datetime(df_1w['time'], unit='s')
            df_1w.set_index('time', inplace=True)
        
        logging.info("All timeframe data loaded successfully")
        return df_90day, df_1h, df_4h, df_12h, df_1d, df_1w
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def calculate_rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def extract_indicator_features(df, prefix=''):
    """Extract key technical indicators from chart data."""
    features = {}
    
    # Price action
    if 'close' in df.columns:
        features[f'{prefix}close'] = df['close']
        features[f'{prefix}price_change'] = df['close'].pct_change()
        
        # RSI - Relative Strength Index (NEW!)
        features[f'{prefix}rsi_14'] = calculate_rsi(df['close'], period=14)
        features[f'{prefix}rsi_7'] = calculate_rsi(df['close'], period=7)
        
        # MACD - Moving Average Convergence Divergence (NEW!)
        macd_line, signal_line, macd_hist = calculate_macd(df['close'])
        features[f'{prefix}macd_line'] = macd_line
        features[f'{prefix}macd_signal'] = signal_line
        features[f'{prefix}macd_hist'] = macd_hist
        
        # Lagged price features for short-term memory (NEW!)
        features[f'{prefix}price_lag_1'] = df['close'].shift(1)
        features[f'{prefix}price_lag_2'] = df['close'].shift(2)
        features[f'{prefix}price_lag_3'] = df['close'].shift(3)
        features[f'{prefix}return_lag_1'] = df['close'].pct_change().shift(1)
        features[f'{prefix}return_lag_2'] = df['close'].pct_change().shift(2)
    
    # Volume-weighted indicators (NEW!)
    if 'volume' in df.columns and 'close' in df.columns:
        features[f'{prefix}volume'] = df['volume']
        features[f'{prefix}volume_ma_5'] = df['volume'].rolling(window=5).mean()
        features[f'{prefix}volume_ma_20'] = df['volume'].rolling(window=20).mean()
        features[f'{prefix}volume_price_trend'] = (df['close'].pct_change() * df['volume']).rolling(window=5).sum()
        features[f'{prefix}price_volume_ratio'] = df['close'] / (df['volume'] + 1)
    
    # Bollinger Bands
    if all(col in df.columns for col in ['BB Upper', 'BB Basis', 'BB Lower']):
        features[f'{prefix}bb_width'] = (df['BB Upper'] - df['BB Lower']) / df['BB Basis']
        features[f'{prefix}bb_position'] = (df['close'] - df['BB Lower']) / (df['BB Upper'] - df['BB Lower'])
    
    # Legacy Volume indicators (Coinbase data)
    if 'Volume Band (Close)' in df.columns:
        features[f'{prefix}volume_band'] = df['Volume Band (Close)']
    
    # Momentum indicators
    if 'QAO' in df.columns:
        features[f'{prefix}qao'] = df['QAO']
    
    # Z-scores for overbought/oversold
    if all(col in df.columns for col in ['z20', 'z50', 'z100']):
        features[f'{prefix}z20'] = df['z20']
        features[f'{prefix}z50'] = df['z50']
        features[f'{prefix}z100'] = df['z100']
    
    # Composite Z
    if 'Composite Z (smoothed)' in df.columns:
        features[f'{prefix}composite_z'] = df['Composite Z (smoothed)']
    
    # EMA trends
    if all(col in df.columns for col in ['GMA fast', 'GMA slow']):
        features[f'{prefix}ema_fast'] = df['GMA fast']
        features[f'{prefix}ema_slow'] = df['GMA slow']
        features[f'{prefix}ema_cross'] = df['GMA fast'] - df['GMA slow']
    
    # Volatility
    if 'volatility' in df.columns:
        features[f'{prefix}volatility'] = df['volatility']
    elif 'close' in df.columns:
        features[f'{prefix}volatility'] = df['close'].pct_change().rolling(window=7).std()
    
    return pd.DataFrame(features)

def combine_multi_timeframe_features(df_90day, df_1h, df_4h, df_12h, df_1d, df_1w):
    """Combine features from all timeframes using forward-fill for alignment."""
    
    if df_90day is not None:
        logging.info(f"90-day data shape: {df_90day.shape}, date range: {df_90day.index.min()} to {df_90day.index.max()}")
    logging.info(f"1h data shape: {df_1h.shape}, date range: {df_1h.index.min()} to {df_1h.index.max()}")
    logging.info(f"4h data shape: {df_4h.shape}, date range: {df_4h.index.min()} to {df_4h.index.max()}")
    logging.info(f"12h data shape: {df_12h.shape}, date range: {df_12h.index.min()} to {df_12h.index.max()}")
    logging.info(f"1d data shape: {df_1d.shape}, date range: {df_1d.index.min()} to {df_1d.index.max()}")
    logging.info(f"1w data shape: {df_1w.shape}, date range: {df_1w.index.min()} to {df_1w.index.max()}")
    
    # Use 1-hour data as the base timeline for maximum training samples
    combined = df_1h[['close']].copy()
    combined.rename(columns={'close': 'price'}, inplace=True)
    
    # Extract features from each timeframe
    features_1h = extract_indicator_features(df_1h, prefix='1h_')
    features_4h = extract_indicator_features(df_4h, prefix='4h_')
    features_12h = extract_indicator_features(df_12h, prefix='12h_')
    features_1d = extract_indicator_features(df_1d, prefix='1d_')
    features_1w = extract_indicator_features(df_1w, prefix='1w_')
    
    # 1h features already match the base timeframe
    features_1h_resampled = features_1h
    
    # Resample 4h to hourly with forward fill (limit to 4 hours)
    features_4h_resampled = features_4h.resample('1h').last()
    features_4h_resampled = features_4h_resampled.reindex(combined.index)
    features_4h_resampled = features_4h_resampled.ffill(limit=4)
    
    # Resample 12h to hourly with forward fill (limit to 12 hours)
    features_12h_resampled = features_12h.resample('1h').last()
    features_12h_resampled = features_12h_resampled.reindex(combined.index)
    features_12h_resampled = features_12h_resampled.ffill(limit=12)
    
    # Resample daily to hourly with forward fill (limit to 24 hours)
    features_1d_resampled = features_1d.resample('1h').last()
    features_1d_resampled = features_1d_resampled.reindex(combined.index)
    features_1d_resampled = features_1d_resampled.ffill(limit=24)
    
    # Resample weekly to hourly with forward fill (limit to 168 hours = 1 week)
    features_1w_resampled = features_1w.resample('1h').last()
    features_1w_resampled = features_1w_resampled.reindex(combined.index)
    features_1w_resampled = features_1w_resampled.ffill(limit=168)
    
    # Combine all features
    combined = pd.concat([combined, features_1h_resampled, features_4h_resampled, 
                         features_12h_resampled, features_1d_resampled, features_1w_resampled], axis=1)
    
    # Add original features from 90-day data
    combined['pct_change'] = combined['price'].pct_change()
    combined['rolling_mean_5'] = combined['price'].rolling(window=5).mean()
    combined['rolling_std_5'] = combined['price'].rolling(window=5).std()
    combined['rolling_mean_20'] = combined['price'].rolling(window=20).mean()
    combined['rolling_std_20'] = combined['price'].rolling(window=20).std()
    
    # Target variable: predict percentage return instead of absolute price
    combined['target_return'] = combined['price'].pct_change().shift(-1)
    combined['next_price'] = combined['price'].shift(-1)  # Keep for reference
    
    logging.info(f"Combined features before cleanup: {combined.shape}")
    logging.info(f"NaN counts per column:\n{combined.isna().sum()}")
    
    # Drop rows where target_return is NaN (last row)
    combined = combined[combined['target_return'].notna()]
    
    # Fill NaN in non-critical features with 0
    combined = combined.fillna(0)
    
    logging.info(f"Combined features shape after cleanup: {combined.shape}")
    logging.info(f"Remaining NaN counts: {combined.isna().sum().sum()}")
    
    return combined

def prepare_data(combined_df):
    """Prepare X and y from combined dataframe."""
    
    # Select features (exclude price, target, and next_price reference)
    feature_cols = [col for col in combined_df.columns if col not in ['price', 'target_return', 'next_price']]
    
    X = combined_df[feature_cols].values
    y = combined_df['target_return'].values
    
    logging.info(f"Feature count: {len(feature_cols)}")
    logging.info(f"Sample count: {len(X)}")
    
    return combined_df, X, y, feature_cols

def load_selected_features(path=SELECTED_FEATURES_PATH):
    """Load list of selected features from file"""
    from pathlib import Path
    
    feature_path = Path(path)
    if not feature_path.exists():
        logging.warning(f"⚠️  Selected features file not found: {path}")
        return None
    
    with open(feature_path, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    
    logging.info(f"📋 Loaded {len(features)} selected features from {path}")
    return features

def apply_feature_selection(df, selected_features):
    """Filter dataframe to keep only selected features"""
    # Keep non-feature columns
    keep_cols = ['price', 'target_return', 'next_price']
    
    # Get current feature columns
    current_features = [col for col in df.columns if col not in keep_cols]
    
    # Force-include K-Means cluster features (validated 2.7% improvement with 4 clusters)
    kmeans_features = ['market_cluster', 'cluster_0_ranging', 'cluster_1_trending',
                      'cluster_2_choppy', 'cluster_3_stable', 'cluster_confidence']
    
    for feat in kmeans_features:
        if feat in df.columns and feat not in selected_features:
            selected_features.append(feat)
            logging.info(f"   🔧 Force-included K-Means feature: {feat}")
    
    # Find which selected features are available
    available_features = [f for f in selected_features if f in df.columns]
    missing_features = [f for f in selected_features if f not in df.columns]
    removed_features = [f for f in current_features if f not in selected_features]
    
    if missing_features:
        logging.warning(f"⚠️  {len(missing_features)} selected features not in dataframe")
    
    # Filter dataframe
    final_cols = keep_cols + available_features
    df_filtered = df[final_cols].copy()
    
    logging.info(f"✂️  Feature selection applied:")
    logging.info(f"   Original features: {len(current_features)}")
    logging.info(f"   Selected features: {len(available_features)}")
    logging.info(f"   Features removed: {len(removed_features)} ({len(removed_features)/len(current_features)*100:.1f}%)")
    
    return df_filtered

def train_model(X_train, y_train, model_name='SVR', tscv=None):
    """Train model with GridSearchCV and Time Series Cross-Validation.
    
    Uses Pipeline to ensure StandardScaler is fit separately within each CV fold,
    preventing data leakage from future folds.
    
    Args:
        X_train: Training features (unscaled)
        y_train: Training target
        model_name: Name of the model ('SVR', 'RF', 'XGB', 'LGB')
        tscv: TimeSeriesSplit object for cross-validation (default: 5-fold)
    
    Returns:
        Trained pipeline (scaler + best estimator from GridSearchCV)
    """
    # Define model and parameter grid
    if model_name == 'SVR':
        model = SVR()
        param_grid = {
            'model__C': [0.1, 1, 10],
            'model__gamma': ['scale', 0.1, 1],
            'model__kernel': ['rbf']
        }
    elif model_name == 'RF':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [10, 20, 30]
        }
    elif model_name == 'XGB':
        model = xgb.XGBRegressor(random_state=42, tree_method='hist')
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [5, 10, 15],
            'model__learning_rate': [0.01, 0.1]
        }
    elif model_name == 'LGB':
        model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [5, 10, 15],
            'model__learning_rate': [0.01, 0.1]
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create pipeline with StandardScaler + Model
    # This ensures scaling happens INSIDE each CV fold
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Use TimeSeriesSplit if provided, otherwise default 3-fold CV
    if tscv is None:
        tscv = 3
    
    # Count total combinations for progress tracking
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)
    n_folds = tscv.n_splits if hasattr(tscv, 'n_splits') else tscv
    total_fits = total_combinations * n_folds
    
    logging.info(f"⏱️  Starting GridSearchCV for {model_name}...")
    logging.info(f"   • Testing {total_combinations} parameter combinations")
    logging.info(f"   • Using {n_folds}-fold time series cross-validation")
    logging.info(f"   • Total model fits: {total_fits}")
    logging.info(f"   • Scaling applied within each fold (no data leakage)")
    
    start_time = time.time()
    grid_search = GridSearchCV(
        pipeline,  # Use pipeline instead of raw model
        param_grid, 
        cv=tscv, 
        scoring='neg_mean_squared_error', 
        verbose=2,  # Show progress: 1=minimal, 2=detailed
        n_jobs=-1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)  # Fit on unscaled data
    elapsed_time = time.time() - start_time
    
    # Display CV results
    cv_results = grid_search.cv_results_
    best_index = grid_search.best_index_
    
    logging.info(f"✅ {model_name} GridSearchCV completed in {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)")
    logging.info(f"   Best Parameters: {grid_search.best_params_}")
    logging.info(f"   Best CV Score (RMSE): {np.sqrt(-grid_search.best_score_):.6f}")
    logging.info(f"   Std Dev across folds: ±{cv_results['std_test_score'][best_index]:.6f}")
    
    return grid_search.best_estimator_  # Returns the full pipeline

def train_ensemble(X_train, y_train, tscv=None):
    """Train ensemble of RF, XGBoost, and optionally LightGBM models with time series CV.
    
    Args:
        X_train: Training features
        y_train: Training target
        tscv: TimeSeriesSplit object for cross-validation
    
    Returns:
        Dictionary of trained models
    """
    models = {}
    ensemble_start = time.time()
    
    logging.info("\n" + "="*70)
    logging.info("🚀 STARTING ENSEMBLE TRAINING WITH TIME SERIES CROSS-VALIDATION")
    logging.info("="*70)
    
    # Train RandomForest
    logging.info("\n[1/2] Training RandomForest model...")
    rf_start = time.time()
    models['RF'] = train_model(X_train, y_train, 'RF', tscv=tscv)
    rf_time = time.time() - rf_start
    logging.info(f"   RandomForest total time: {rf_time:.1f}s ({rf_time/60:.2f} min)")
    
    # Train XGBoost
    logging.info("\n[2/2] Training XGBoost model...")
    xgb_start = time.time()
    models['XGB'] = train_model(X_train, y_train, 'XGB', tscv=tscv)
    xgb_time = time.time() - xgb_start
    logging.info(f"   XGBoost total time: {xgb_time:.1f}s ({xgb_time/60:.2f} min)")
    
    # Optional: Train LightGBM
    if HAS_LIGHTGBM:
        logging.info("\n[3/3] Training LightGBM model...")
        lgb_start = time.time()
        models['LGB'] = train_model(X_train, y_train, 'LGB', tscv=tscv)
        lgb_time = time.time() - lgb_start
        logging.info(f"   LightGBM total time: {lgb_time:.1f}s ({lgb_time/60:.2f} min)")
    else:
        logging.info("\n   LightGBM not available, using RF + XGBoost ensemble")
    
    total_ensemble_time = time.time() - ensemble_start
    logging.info("\n" + "="*70)
    logging.info(f"✅ ENSEMBLE TRAINING COMPLETE!")
    logging.info(f"   Total models trained: {len(models)}")
    logging.info(f"   Total training time: {total_ensemble_time:.1f}s ({total_ensemble_time/60:.2f} min)")
    logging.info("="*70 + "\n")
    
    return models

def ensemble_predict(models, X, return_std=False):
    """Make predictions using ensemble of models with equal weighting.
    
    Args:
        models: Dictionary of trained models
        X: Feature array
        return_std: If True, also return standard deviation across models
        
    Returns:
        ensemble_pred: Mean prediction
        pred_std (optional): Standard deviation of predictions
    """
    predictions = []
    
    for name, model in models.items():
        pred = model.predict(X)
        predictions.append(pred)
    
    # Average predictions from all models
    predictions = np.array(predictions)
    ensemble_pred = np.mean(predictions, axis=0)
    
    if return_std:
        pred_std = np.std(predictions, axis=0)
        return ensemble_pred, pred_std
    
    return ensemble_pred

# ==================== LSTM NEURAL NETWORK FUNCTIONS ====================

class AttentionLayer(nn.Module):
    """Attention mechanism to focus on important timesteps.
    
    This allows the model to 'pay attention' to the most relevant
    time periods when making predictions.
    """
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        
        # Calculate attention scores for each timestep
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        
        # Apply softmax to get attention weights (sum to 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # Apply attention weights to LSTM output
        # Weighted sum of all timesteps
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_size)
        
        return context_vector, attention_weights

class BitcoinLSTM(nn.Module):
    """3-layer LSTM neural network for Bitcoin price prediction.
    
    Architecture:
    - Input: Sequence of features over time (seq_length x num_features)
    - LSTM: 3 stacked layers with 256 hidden units each
    - Attention: Focus mechanism on important timesteps (PHASE 2!)
    - Dropout: 0.2 to prevent overfitting
    - Output: Single prediction (next time step return)
    """
    
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.2, use_attention=True):
        super(BitcoinLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer (PHASE 2!)
        if use_attention:
            self.attention = AttentionLayer(hidden_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, _ = self.lstm(x)
        
        # Apply attention or use last timestep
        if self.use_attention:
            # Use attention to focus on important timesteps
            out, attention_weights = self.attention(lstm_out)
            # Store attention weights for visualization (optional)
            self.last_attention_weights = attention_weights
        else:
            # Use last timestep output (original method)
            out = lstm_out[:, -1, :]
        
        # Feed-forward network with dropout
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.fc3(out)
        
        return out

# ==================== TRANSFORMER ARCHITECTURE (PHASE 3!) ====================

class PositionalEncoding(nn.Module):
    """Adds positional information to sequences.
    
    Since Transformers don't have inherent sequence order like RNNs,
    we add positional encodings to tell the model about time order.
    """
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class BitcoinTransformer(nn.Module):
    """Transformer architecture for Bitcoin price prediction.
    
    🚀 PHASE 3: Full Transformer like GPT/BERT but for time series!
    
    Architecture:
    - Input embedding: Projects features to transformer dimension
    - Positional encoding: Adds time order information
    - Multi-head self-attention: 8 heads examining different patterns
    - Transformer encoder: 4 layers of attention + feed-forward
    - Output: Single price prediction
    
    Advantages over LSTM:
    - Parallel processing (much faster!)
    - Captures long-range dependencies better
    - Multi-head attention sees multiple patterns simultaneously
    - State-of-the-art for sequence modeling
    """
    
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=1024, dropout=0.1):
        super(BitcoinTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input projection: features → transformer dimension
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Project to transformer dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Use last timestep for prediction (or could use mean pooling)
        x = x[:, -1, :]  # (batch, d_model)
        
        # Output layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class MultiTaskTransformer(nn.Module):
    """Multi-Task Transformer for Bitcoin prediction with uncertainty.
    
    🎯 PHASE 4: Multi-Task Learning + Uncertainty Quantification!
    
    This model predicts THREE outputs simultaneously:
    1. Price Movement (regression) - how much price will change
    2. Volatility/Uncertainty (regression) - how confident we are
    3. Direction (classification) - up/down/stable
    
    Architecture:
    - Shared Transformer Encoder (learns common representations)
    - Three separate output heads:
      * Price head: Predicts return (continuous)
      * Volatility head: Predicts uncertainty (continuous)
      * Direction head: Classifies movement (3 classes: down/stable/up)
    
    Advantages:
    - Multi-task learning improves all tasks (shared knowledge!)
    - Uncertainty estimates → risk-aware trading
    - Direction classification → better decision signals
    - Monte Carlo Dropout → probabilistic forecasts
    
    Training uses combined loss:
        Total = 1.0*price_loss + 0.3*volatility_loss + 0.5*direction_loss
    """
    
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=1024, dropout=0.1):
        super(MultiTaskTransformer, self).__init__()
        
        self.d_model = d_model
        self.dropout_rate = dropout
        
        # Shared transformer encoder (same as Phase 3)
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Shared feature extraction
        self.shared_fc = nn.Linear(d_model, 256)
        
        # Task 1: Price Prediction Head
        self.price_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Task 2: Volatility/Uncertainty Prediction Head
        self.volatility_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive output for volatility
        )
        
        # Task 3: Direction Classification Head (3 classes: down/stable/up)
        self.direction_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # 3 classes: down (0), stable (1), up (2)
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_features=False):
        """Forward pass with optional feature extraction.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            return_features: If True, return intermediate features for analysis
            
        Returns:
            price: Price prediction (batch, 1)
            volatility: Uncertainty estimate (batch, 1)
            direction_logits: Direction logits (batch, 3)
            features: (optional) Shared features for visualization
        """
        # Shared encoding
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Use last timestep
        
        # Shared feature extraction
        features = self.relu(self.shared_fc(x))
        features = self.dropout(features)
        
        # Three separate predictions
        price = self.price_head(features)
        volatility = self.volatility_head(features)
        direction_logits = self.direction_head(features)
        
        if return_features:
            return price, volatility, direction_logits, features
        else:
            return price, volatility, direction_logits
    
    def predict_with_uncertainty(self, x, n_samples=50):
        """Predict with uncertainty using Monte Carlo Dropout.
        
        Runs the model multiple times with dropout enabled to get
        a distribution of predictions → uncertainty estimate!
        
        Args:
            x: Input tensor (batch, seq_len, features)
            n_samples: Number of forward passes with dropout
            
        Returns:
            price_mean: Mean price prediction
            price_std: Standard deviation (uncertainty)
            volatility_mean: Mean volatility
            direction_probs: Direction probabilities (batch, 3)
        """
        self.train()  # Enable dropout
        
        price_samples = []
        volatility_samples = []
        direction_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                price, volatility, direction_logits = self.forward(x)
                price_samples.append(price)
                volatility_samples.append(volatility)
                direction_samples.append(torch.softmax(direction_logits, dim=1))
        
        # Stack and compute statistics
        price_samples = torch.stack(price_samples, dim=0)  # (n_samples, batch, 1)
        volatility_samples = torch.stack(volatility_samples, dim=0)
        direction_samples = torch.stack(direction_samples, dim=0)
        
        price_mean = price_samples.mean(dim=0)
        price_std = price_samples.std(dim=0)  # Epistemic uncertainty!
        volatility_mean = volatility_samples.mean(dim=0)  # Aleatoric uncertainty!
        direction_probs = direction_samples.mean(dim=0)
        
        self.eval()  # Disable dropout
        
        return price_mean, price_std, volatility_mean, direction_probs

def create_sequences(X, y, seq_length=24):
    """Convert tabular data to sequences for LSTM/Transformer.
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        seq_length: Number of time steps in each sequence
        
    Returns:
        X_seq: Tensor of shape (n_sequences, seq_length, n_features)
        y_seq: Tensor of shape (n_sequences,)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    
    return np.array(X_seq), np.array(y_seq)

def train_lstm(X_train, y_train, X_val=None, y_val=None, input_size=None, 
               seq_length=24, epochs=100, batch_size=32, learning_rate=0.001, use_attention=True):
    """Train LSTM model on GPU with optional attention mechanism.
    
    Args:
        X_train: Training features (already scaled)
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        input_size: Number of features (auto-detected if None)
        seq_length: Sequence length for LSTM
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        use_attention: Whether to use attention mechanism (PHASE 2!)
        
    Returns:
        model: Trained LSTM model
        scaler: StandardScaler used for sequences (if needed)
    """
    if not HAS_PYTORCH:
        logging.error("PyTorch not available - cannot train LSTM")
        return None, None
    
    logging.info("\n" + "="*70)
    logging.info("🧠 LSTM NEURAL NETWORK TRAINING")
    logging.info("="*70)
    
    # Create sequences
    logging.info(f"Creating sequences with lookback={seq_length} hours...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    
    if X_val is not None and y_val is not None:
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    else:
        X_val_seq, y_val_seq = None, None
    
    logging.info(f"   Training sequences: {len(X_train_seq):,}")
    if X_val_seq is not None:
        logging.info(f"   Validation sequences: {len(X_val_seq):,}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_seq).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train_seq).to(DEVICE)
    
    if X_val_seq is not None:
        X_val_tensor = torch.FloatTensor(X_val_seq).to(DEVICE)
        y_val_tensor = torch.FloatTensor(y_val_seq).to(DEVICE)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # Don't shuffle time series!
    
    # Initialize model
    if input_size is None:
        input_size = X_train_seq.shape[2]
    
    model = BitcoinLSTM(input_size=input_size, use_attention=use_attention).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    logging.info(f"\n📊 LSTM Architecture:")
    logging.info(f"   Input features: {input_size}")
    logging.info(f"   Sequence length: {seq_length} hours")
    logging.info(f"   Hidden units: 256")
    logging.info(f"   Layers: 3 LSTM + 3 FC")
    logging.info(f"   Attention: {'✅ ENABLED (PHASE 2!)' if use_attention else '❌ Disabled'}")
    logging.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"   Device: {DEVICE}")
    
    logging.info(f"\n⏱️  Training for {epochs} epochs...")
    start_time = time.time()
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        if X_val_seq is not None:
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val_tensor).squeeze()
                val_loss = criterion(val_predictions, y_val_tensor).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logging.info(f"   Epoch {epoch+1:3d}/{epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")
            
            if patience_counter >= patience:
                logging.info(f"   Early stopping at epoch {epoch+1} (validation loss not improving)")
                break
        else:
            if (epoch + 1) % 10 == 0:
                logging.info(f"   Epoch {epoch+1:3d}/{epochs}: Train Loss={avg_train_loss:.6f}")
    
    elapsed_time = time.time() - start_time
    
    logging.info(f"\n✅ LSTM training complete!")
    logging.info(f"   Training time: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)")
    logging.info(f"   Final train RMSE: {np.sqrt(avg_train_loss):.6f}")
    if X_val_seq is not None:
        logging.info(f"   Best val RMSE: {np.sqrt(best_val_loss):.6f}")
    logging.info("="*70 + "\n")
    
    model.eval()  # Set to evaluation mode
    return model, seq_length

def lstm_predict(model, X, seq_length):
    """Make predictions using trained LSTM model.
    
    Args:
        model: Trained LSTM model
        X: Feature array (unscaled or scaled, depending on training)
        seq_length: Sequence length used during training
        
    Returns:
        predictions: Array of predictions (same length as X minus seq_length)
    """
    if not HAS_PYTORCH or model is None:
        return None
    
    # Create sequences
    X_seq, _ = create_sequences(X, np.zeros(len(X)), seq_length)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_seq).to(DEVICE)
    
    # Predict
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).squeeze().cpu().numpy()
    
    # Pad the beginning with NaNs to match original length
    padded_predictions = np.full(len(X), np.nan)
    padded_predictions[seq_length:] = predictions
    
    return padded_predictions

# ==================== TRANSFORMER TRAINING (PHASE 3!) ====================

def train_transformer(X_train, y_train, X_val=None, y_val=None, input_size=None,
                     seq_length=48, epochs=150, batch_size=128, learning_rate=0.0005,
                     d_model=256, nhead=8, num_layers=4, dropout=0.1):
    """Train Transformer model on GPU - PHASE 3!
    
    Args:
        X_train: Training features (already scaled)
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        input_size: Number of features (auto-detected if None)
        seq_length: Sequence length for Transformer
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        d_model: Transformer embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dropout: Dropout rate
        
    Returns:
        Trained model, sequence length
    """
    if not HAS_PYTORCH:
        logging.error("PyTorch not available. Cannot train Transformer.")
        return None, None
    
    logging.info("\n" + "="*70)
    logging.info("🚀 TRANSFORMER NEURAL NETWORK TRAINING (PHASE 3!)")
    logging.info("="*70)
    
    # Create sequences
    logging.info(f"Creating sequences with lookback={seq_length} hours...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    
    if X_val is not None and y_val is not None:
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    else:
        X_val_seq, y_val_seq = None, None
    
    logging.info(f"   Training sequences: {len(X_train_seq):,}")
    if X_val_seq is not None:
        logging.info(f"   Validation sequences: {len(X_val_seq):,}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_seq).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train_seq).to(DEVICE)
    
    if X_val_seq is not None:
        X_val_tensor = torch.FloatTensor(X_val_seq).to(DEVICE)
        y_val_tensor = torch.FloatTensor(y_val_seq).to(DEVICE)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    if input_size is None:
        input_size = X_train_seq.shape[2]
    
    model = BitcoinTransformer(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        dropout=dropout
    ).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    logging.info(f"\n🏗️  Transformer Architecture:")
    logging.info(f"   Input features: {input_size}")
    logging.info(f"   Sequence length: {seq_length} hours")
    logging.info(f"   Embedding dimension: {d_model}")
    logging.info(f"   Attention heads: {nhead}")
    logging.info(f"   Encoder layers: {num_layers}")
    logging.info(f"   Feed-forward dim: {d_model * 4}")
    logging.info(f"   Dropout: {dropout}")
    logging.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"   Device: {DEVICE}")
    
    logging.info(f"\n⏱️  Training for {epochs} epochs...")
    start_time = time.time()
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20  # Early stopping patience
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        if X_val_seq is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_tensor).squeeze()
                val_loss = criterion(val_pred, y_val_tensor).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logging.info(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logging.info(f"   Epoch {epoch+1:3d}/{epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")
        else:
            # No validation set
            if (epoch + 1) % 10 == 0:
                logging.info(f"   Epoch {epoch+1:3d}/{epochs}: Train Loss={avg_train_loss:.6f}")
    
    elapsed_time = time.time() - start_time
    
    logging.info(f"\n✅ Transformer training complete!")
    logging.info(f"   Training time: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)")
    logging.info(f"   Final train RMSE: {np.sqrt(avg_train_loss):.6f}")
    if X_val_seq is not None:
        logging.info(f"   Best val RMSE: {np.sqrt(best_val_loss):.6f}")
    logging.info("="*70 + "\n")
    
    model.eval()
    return model, seq_length

def transformer_predict(model, X, seq_length):
    """Make predictions using trained Transformer model.
    
    Args:
        model: Trained Transformer model
        X: Feature array
        seq_length: Sequence length used during training
        
    Returns:
        predictions: Array of predictions
    """
    if not HAS_PYTORCH or model is None:
        return None
    
    # Create sequences
    X_seq, _ = create_sequences(X, np.zeros(len(X)), seq_length)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_seq).to(DEVICE)
    
    # Predict
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).squeeze().cpu().numpy()
    
    # Pad the beginning with NaNs to match original length
    padded_predictions = np.full(len(X), np.nan)
    padded_predictions[seq_length:] = predictions
    
    return padded_predictions

def train_multitask_transformer(X_train, y_train, X_val, y_val, seq_length=48, 
                                epochs=150, batch_size=128, learning_rate=0.0005,
                                d_model=256, nhead=8, num_layers=4, dropout=0.1,
                                direction_threshold=0.005):
    """Train Multi-Task Transformer with price + volatility + direction prediction.
    
    🎯 PHASE 4: Multi-Task Learning + Uncertainty Quantification!
    
    This function trains a transformer to predict THREE things simultaneously:
    1. Price movement (regression)
    2. Volatility/uncertainty (regression)
    3. Direction (classification: down/stable/up)
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        seq_length: Sequence length for temporal modeling
        epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Learning rate for optimizer
        d_model: Transformer embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        direction_threshold: Threshold for classifying up/down/stable (e.g., 0.005 = 0.5%)
        
    Returns:
        Trained multi-task model, sequence length
    """
    if not HAS_PYTORCH:
        logging.error("PyTorch not available. Cannot train Multi-Task Transformer.")
        return None, None
    
    logging.info("\n" + "="*70)
    logging.info("🎯 MULTI-TASK TRANSFORMER TRAINING (PHASE 4!)")
    logging.info("="*70)
    logging.info("📊 Predicting: Price + Volatility + Direction (3 tasks!)")
    
    # Create sequences
    logging.info(f"Creating sequences with lookback={seq_length} hours...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    
    logging.info(f"Training sequences: {X_train_seq.shape}")
    logging.info(f"Validation sequences: {X_val_seq.shape}")
    
    # Create direction labels based on threshold
    # 0 = down (< -threshold), 1 = stable (within threshold), 2 = up (> threshold)
    y_train_direction = np.zeros(len(y_train_seq), dtype=np.long)
    y_train_direction[y_train_seq > direction_threshold] = 2  # Up
    y_train_direction[(y_train_seq >= -direction_threshold) & (y_train_seq <= direction_threshold)] = 1  # Stable
    y_train_direction[y_train_seq < -direction_threshold] = 0  # Down
    
    y_val_direction = np.zeros(len(y_val_seq), dtype=np.long)
    y_val_direction[y_val_seq > direction_threshold] = 2
    y_val_direction[(y_val_seq >= -direction_threshold) & (y_val_seq <= direction_threshold)] = 1
    y_val_direction[y_val_seq < -direction_threshold] = 0
    
    # Create volatility targets (rolling std of recent returns as proxy)
    window = 24  # 24 hours
    y_train_volatility = []
    for i in range(len(y_train_seq)):
        if i < window:
            y_train_volatility.append(np.std(y_train_seq[:i+1]))
        else:
            y_train_volatility.append(np.std(y_train_seq[i-window:i+1]))
    y_train_volatility = np.array(y_train_volatility)
    
    y_val_volatility = []
    for i in range(len(y_val_seq)):
        if i < window:
            y_val_volatility.append(np.std(y_val_seq[:i+1]))
        else:
            y_val_volatility.append(np.std(y_val_seq[i-window:i+1]))
    y_val_volatility = np.array(y_val_volatility)
    
    logging.info(f"Direction distribution (train): Down={np.sum(y_train_direction==0)}, " +
                f"Stable={np.sum(y_train_direction==1)}, Up={np.sum(y_train_direction==2)}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_seq).to(DEVICE)
    y_train_price = torch.FloatTensor(y_train_seq).unsqueeze(1).to(DEVICE)
    y_train_vol = torch.FloatTensor(y_train_volatility).unsqueeze(1).to(DEVICE)
    y_train_dir = torch.LongTensor(y_train_direction).to(DEVICE)
    
    X_val_tensor = torch.FloatTensor(X_val_seq).to(DEVICE)
    y_val_price = torch.FloatTensor(y_val_seq).unsqueeze(1).to(DEVICE)
    y_val_vol = torch.FloatTensor(y_val_volatility).unsqueeze(1).to(DEVICE)
    y_val_dir = torch.LongTensor(y_val_direction).to(DEVICE)
    
    # Initialize model
    input_size = X_train_seq.shape[2]
    model = MultiTaskTransformer(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        dropout=dropout
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"✅ Model initialized with {trainable_params:,} trainable parameters")
    logging.info(f"   (Total: {total_params:,} parameters)")
    
    # Loss functions for each task
    price_criterion = nn.MSELoss()
    volatility_criterion = nn.MSELoss()
    direction_criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=False
    )
    
    # Training loop
    logging.info(f"\n🚀 Training for {epochs} epochs...")
    logging.info(f"   Batch size: {batch_size}")
    logging.info(f"   Learning rate: {learning_rate}")
    logging.info(f"   Device: {DEVICE}")
    
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        
        # Mini-batch training
        indices = np.random.permutation(len(X_train_tensor))
        epoch_price_loss = 0
        epoch_vol_loss = 0
        epoch_dir_loss = 0
        n_batches = 0
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:min(i+batch_size, len(indices))]
            
            X_batch = X_train_tensor[batch_idx]
            y_price_batch = y_train_price[batch_idx]
            y_vol_batch = y_train_vol[batch_idx]
            y_dir_batch = y_train_dir[batch_idx]
            
            optimizer.zero_grad()
            
            # Forward pass
            price_pred, vol_pred, dir_logits = model(X_batch)
            
            # Compute losses for each task
            price_loss = price_criterion(price_pred, y_price_batch)
            vol_loss = volatility_criterion(vol_pred, y_vol_batch)
            dir_loss = direction_criterion(dir_logits, y_dir_batch)
            
            # Combined loss with task weights
            total_loss = 1.0 * price_loss + 0.3 * vol_loss + 0.5 * dir_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_price_loss += price_loss.item()
            epoch_vol_loss += vol_loss.item()
            epoch_dir_loss += dir_loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_price_pred, val_vol_pred, val_dir_logits = model(X_val_tensor)
            
            val_price_loss = price_criterion(val_price_pred, y_val_price).item()
            val_vol_loss = volatility_criterion(val_vol_pred, y_val_vol).item()
            val_dir_loss = direction_criterion(val_dir_logits, y_val_dir).item()
            
            val_total_loss = 1.0 * val_price_loss + 0.3 * val_vol_loss + 0.5 * val_dir_loss
            
            # Direction accuracy
            val_dir_pred = torch.argmax(val_dir_logits, dim=1)
            val_dir_accuracy = (val_dir_pred == y_val_dir).float().mean().item()
        
        # Learning rate scheduling
        scheduler.step(val_total_loss)
        
        # Early stopping
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(f"Epoch {epoch+1:3d}/{epochs} | " +
                        f"Train Loss: P={epoch_price_loss/n_batches:.6f} V={epoch_vol_loss/n_batches:.6f} D={epoch_dir_loss/n_batches:.4f} | " +
                        f"Val Loss: P={val_price_loss:.6f} V={val_vol_loss:.6f} D={val_dir_loss:.4f} | " +
                        f"Dir Acc: {val_dir_accuracy:.2%}")
        
        # Early stopping check
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    logging.info(f"\n✅ Multi-Task Transformer training complete!")
    logging.info(f"   Training time: {training_time:.1f}s ({training_time/60:.2f} min)")
    logging.info(f"   Best val loss: {best_val_loss:.6f}")
    logging.info(f"   Final direction accuracy: {val_dir_accuracy:.2%}")
    
    model.eval()
    return model, seq_length

def multitask_predict(model, X, seq_length, n_samples=50):
    """Make predictions using trained Multi-Task Transformer with uncertainty.
    
    🎯 PHASE 4: Returns price + uncertainty + direction predictions!
    
    Args:
        model: Trained MultiTaskTransformer
        X: Feature array
        seq_length: Sequence length used during training
        n_samples: Number of Monte Carlo dropout samples for uncertainty
        
    Returns:
        Dictionary with:
            - price_mean: Mean price predictions
            - price_std: Uncertainty (epistemic)
            - volatility: Model's volatility predictions (aleatoric)
            - direction_probs: Direction probabilities [down, stable, up]
            - direction_class: Most likely direction (0/1/2)
            - confidence: Confidence in direction prediction
    """
    if not HAS_PYTORCH or model is None:
        return None
    
    # Create sequences
    X_seq, _ = create_sequences(X, np.zeros(len(X)), seq_length)
    X_tensor = torch.FloatTensor(X_seq).to(DEVICE)
    
    # Predict with uncertainty
    model.eval()
    price_mean, price_std, volatility, direction_probs = model.predict_with_uncertainty(X_tensor, n_samples=n_samples)
    
    # Convert to numpy
    price_mean = price_mean.squeeze().cpu().numpy()
    price_std = price_std.squeeze().cpu().numpy()
    volatility = volatility.squeeze().cpu().numpy()
    direction_probs = direction_probs.cpu().numpy()
    
    # Direction class and confidence
    direction_class = np.argmax(direction_probs, axis=1)
    confidence = np.max(direction_probs, axis=1)
    
    # Pad to match original length
    padded_price_mean = np.full(len(X), np.nan)
    padded_price_std = np.full(len(X), np.nan)
    padded_volatility = np.full(len(X), np.nan)
    padded_direction_class = np.full(len(X), -1)
    padded_confidence = np.full(len(X), np.nan)
    padded_direction_probs = np.full((len(X), 3), np.nan)
    
    padded_price_mean[seq_length:] = price_mean
    padded_price_std[seq_length:] = price_std
    padded_volatility[seq_length:] = volatility
    padded_direction_class[seq_length:] = direction_class
    padded_confidence[seq_length:] = confidence
    padded_direction_probs[seq_length:] = direction_probs
    
    return {
        'price_mean': padded_price_mean,
        'price_std': padded_price_std,
        'volatility': padded_volatility,
        'direction_probs': padded_direction_probs,
        'direction_class': padded_direction_class,
        'confidence': padded_confidence
    }

def ensemble_predict_with_lstm(models, lstm_model, X, seq_length, transformer_model=None, return_std=False):
    """Make predictions using ensemble including LSTM and optionally Transformer.
    
    Args:
        models: Dictionary of traditional ML models (RF, XGB, LGB)
        lstm_model: Trained LSTM model
        X: Feature array
        seq_length: LSTM/Transformer sequence length
        transformer_model: Trained Transformer model (optional, Phase 3!)
        return_std: If True, return std deviation
        
    Returns:
        ensemble_pred: Weighted average prediction
        pred_std (optional): Standard deviation of predictions
    """
    predictions = []
    
    # Get traditional ML predictions
    for name, model in models.items():
        pred = model.predict(X)
        predictions.append(pred)
    
    # Get LSTM predictions
    if lstm_model is not None and HAS_PYTORCH:
        lstm_pred = lstm_predict(lstm_model, X, seq_length)
        # Only use predictions where LSTM has valid values
        valid_mask = ~np.isnan(lstm_pred)
        if valid_mask.sum() > 0:
            predictions.append(lstm_pred)
    
    # Get Transformer predictions (PHASE 3!)
    if transformer_model is not None and HAS_PYTORCH:
        transformer_pred = transformer_predict(transformer_model, X, seq_length)
        valid_mask = ~np.isnan(transformer_pred)
        if valid_mask.sum() > 0:
            predictions.append(transformer_pred)
    
    # Weighted ensemble (you can tune these weights!)
    predictions = np.array(predictions)
    
    # Handle NaN values from sequence padding
    ensemble_pred = np.nanmean(predictions, axis=0)
    
    if return_std:
        pred_std = np.nanstd(predictions, axis=0)
        return ensemble_pred, pred_std
    
    return ensemble_pred

# ==================== END LSTM FUNCTIONS ====================


def predict_next_steps(model, df_last_row_full, scaler, features, steps=3, confidence_level=1.96):
    """Multi-step prediction with confidence intervals.
    
    Args:
        model: Either a single pipeline/model or dict of pipelines (ensemble)
        df_last_row_full: Last row of data with all features
        scaler: Fitted StandardScaler (can be None if model is a pipeline)
        features: List of feature names
        steps: Number of steps to predict
        confidence_level: Multiplier for std (1.96 = 95% confidence)
    
    Returns:
        predictions: List of most likely prices
        prediction_df: DataFrame with price, lower_bound, upper_bound
    """
    predictions = []
    lower_bounds = []
    upper_bounds = []
    prices = [df_last_row_full['price']]
    current_features = df_last_row_full.copy()
    
    # Check if using ensemble
    is_ensemble = isinstance(model, dict)
    
    for i in range(steps):
        # Build feature vector from current state
        feature_vector = []
        for feat in features:
            if feat in current_features.index:
                feature_vector.append(current_features[feat])
            else:
                feature_vector.append(0)
        
        X_pred = np.array(feature_vector).reshape(1, -1)
        
        # Scaling handled by pipeline if scaler is None
        if scaler is not None:
            X_pred_scaled = scaler.transform(X_pred)
        else:
            X_pred_scaled = X_pred  # Pipeline will handle scaling
        
        # Predict percentage return using ensemble or single model
        if is_ensemble:
            predicted_return, return_std = ensemble_predict(model, X_pred_scaled, return_std=True)
            predicted_return = predicted_return[0]
            return_std = return_std[0]
        else:
            predicted_return = model.predict(X_pred_scaled)[0]
            return_std = 0.005  # Default uncertainty for single model (~0.5%)
        
        # Convert return to price
        current_price = prices[-1]
        predicted_price = current_price * (1 + predicted_return)
        
        # Calculate confidence bounds using return uncertainty
        lower_return = predicted_return - (confidence_level * return_std)
        upper_return = predicted_return + (confidence_level * return_std)
        
        lower_price = current_price * (1 + lower_return)
        upper_price = current_price * (1 + upper_return)
        
        prices.append(predicted_price)
        predictions.append(predicted_price)
        lower_bounds.append(lower_price)
        upper_bounds.append(upper_price)
        
        # Update only price-dependent features for next iteration
        # Keep multi-timeframe features constant (carry forward from last actual data)
        current_features['price'] = predicted_price
        
        # Recalculate simple rolling features based on price history
        price_series = pd.Series(prices[-21:])  # Last 20 + current
        current_features['pct_change'] = predicted_return
        if len(prices) >= 5:
            current_features['rolling_mean_5'] = price_series[-5:].mean()
            current_features['rolling_std_5'] = price_series[-5:].std()
        if len(prices) >= 20:
            current_features['rolling_mean_20'] = price_series[-20:].mean()
            current_features['rolling_std_20'] = price_series[-20:].std()
    
    # Return predictions as a series with timestamps
    last_timestamp = df_last_row_full.name
    prediction_df = pd.DataFrame({
        'price': predictions,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds
    }, index=[last_timestamp + pd.Timedelta(hours=i+1) for i in range(steps)])
    
    return predictions, prediction_df

# Main execution
if __name__ == "__main__":
    try:
        # Start overall timer
        overall_start_time = time.time()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "="*70)
        print(f"🤖 BITCOIN PRICE PREDICTOR - TRAINING SESSION")
        print(f"   Started: {current_time}")
        print("="*70)
        
        # Initialize storage manager
        logging.info("\n💾 Initializing storage manager...")
        storage = ModelStorageManager()
        storage.print_storage_summary()
        
        # Load all timeframe data
        logging.info("\n📊 Loading multi-timeframe data...")
        data_start = time.time()
        df_90day, df_1h, df_4h, df_12h, df_1d, df_1w = load_multi_timeframe_data()
        data_time = time.time() - data_start
        logging.info(f"   Data loading completed in {data_time:.1f}s")
        
        # Collect external data
        logging.info("\n🌐 Collecting external data sources...")
        external_start = time.time()
        external_collector = ExternalDataCollector(cache_hours=1)
        external_data = external_collector.collect_all()
        external_time = time.time() - external_start
        logging.info(f"   External data collected in {external_time:.1f}s")
        logging.info(f"   Features: BTC.D={external_data['btc_dominance']:.2f}%, USDT.D={external_data['usdt_dominance']:.2f}%, Fear&Greed={external_data['fear_greed']}")
        
        # Combine features from all timeframes
        logging.info("\n🔧 Engineering features from multiple timeframes...")
        feature_start = time.time()
        combined_df = combine_multi_timeframe_features(df_90day, df_1h, df_4h, df_12h, df_1d, df_1w)
        feature_time = time.time() - feature_start
        logging.info(f"   Feature engineering completed in {feature_time:.1f}s")
        
        # Add enhanced features (44 advanced features)
        logging.info("\n✨ Adding enhanced features...")
        enhanced_start = time.time()
        combined_df = add_all_enhanced_features(combined_df)
        enhanced_time = time.time() - enhanced_start
        logging.info(f"   Enhanced features added in {enhanced_time:.1f}s")
        logging.info(f"   Feature categories: {', '.join(ENHANCED_FEATURE_GROUPS.keys())}")
        
        # Add external data as features (broadcast to all rows)
        logging.info("\n🌐 Adding external data features...")
        for key, value in external_data.items():
            if key != 'timestamp' and key != 'fear_greed_class':  # Skip non-numeric fields
                combined_df[f'ext_{key}'] = value
        
        # Clean up NaN values from enhanced features
        logging.info("\n🧹 Cleaning enhanced features...")
        rows_before = len(combined_df)
        combined_df = combined_df.dropna()
        rows_after = len(combined_df)
        logging.info(f"   Dropped {rows_before - rows_after} rows with NaN values")
        logging.info(f"   Clean dataset: {rows_after:,} rows")
        
        # Apply feature selection if enabled
        if USE_FEATURE_SELECTION:
            logging.info("\n✂️  Applying feature selection...")
            selected_features = load_selected_features()
            if selected_features:
                combined_df = apply_feature_selection(combined_df, selected_features)
                logging.info(f"   ✅ Feature selection complete")
            else:
                logging.warning(f"   ⚠️  Feature selection skipped (file not found)")
        
        # Prepare data
        logging.info("\n🎯 Preparing training and test datasets...")
        df_final, X, y, features = prepare_data(combined_df)
        
        # Time series split for train/test
        n_samples = len(X)
        split_point = int(n_samples * (1 - TEST_SIZE))
        
        X_train = X[:split_point]
        X_test = X[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]
        
        logging.info(f"   Training samples: {len(X_train):,}")
        logging.info(f"   Test samples: {len(X_test):,}")
        logging.info(f"   Total features: {len(features)}")
        
        # Create TimeSeriesSplit for cross-validation
        # gap=3 to prevent data leakage from lagged features (t-1, t-2, t-3)
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=3)
        
        logging.info(f"\n⏱️  Time Series Cross-Validation Configuration:")
        logging.info(f"   • Number of folds: {n_splits}")
        logging.info(f"   • Gap between train/test: 3 hours (prevents leakage from lagged features)")
        logging.info(f"   • Fold structure: Expanding window (each fold trains on all past data)")
        
        # Show fold sizes
        logging.info(f"\n   Fold breakdown:")
        for i, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
            logging.info(f"   Fold {i+1}: Train={len(train_idx):,} samples, Test={len(test_idx):,} samples")
        
        # NOTE: Scaling is now handled INSIDE the Pipeline within each CV fold
        # This prevents data leakage from future folds
        
        # Train model(s)
        lstm_model = None
        transformer_model = None
        lstm_seq_length = LSTM_SEQUENCE_LENGTH
        
        if USE_ENSEMBLE:
            model_list = "RF + XGBoost + LightGBM" if HAS_LIGHTGBM else "RF + XGBoost"
            if USE_LSTM and HAS_PYTORCH:
                model_list += " + LSTM"
            if USE_TRANSFORMER and HAS_PYTORCH:
                model_list += " + Transformer"
            logging.info(f"\n🎯 Starting Ensemble Training ({model_list})...")
            model = train_ensemble(X_train, y_train, tscv=tscv)  # Pass unscaled data
            
            # Train LSTM if enabled
            if USE_LSTM and HAS_PYTORCH:
                # Get scaled data from one of the trained models for LSTM
                # We'll use the scaler from RandomForest model
                scaler = model['RF'].named_steps['scaler']
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Use last 20% of training data as validation for LSTM
                val_split = int(len(X_train_scaled) * 0.8)
                X_train_lstm = X_train_scaled[:val_split]
                y_train_lstm = y_train[:val_split]
                X_val_lstm = X_train_scaled[val_split:]
                y_val_lstm = y_train[val_split:]
                
                lstm_model, lstm_seq_length = train_lstm(
                    X_train_lstm, y_train_lstm,
                    X_val_lstm, y_val_lstm,
                    input_size=len(features),
                    seq_length=LSTM_SEQUENCE_LENGTH,
                    epochs=LSTM_EPOCHS,
                    batch_size=LSTM_BATCH_SIZE,
                    learning_rate=LSTM_LEARNING_RATE,
                    use_attention=USE_ATTENTION
                )
            
            # Train Transformer if enabled (PHASE 3!)
            if USE_TRANSFORMER and HAS_PYTORCH:
                # Use the same scaled data
                if not USE_LSTM:
                    # Need to scale data if LSTM wasn't trained
                    scaler = model['RF'].named_steps['scaler']
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    val_split = int(len(X_train_scaled) * 0.8)
                    X_train_lstm = X_train_scaled[:val_split]
                    y_train_lstm = y_train[:val_split]
                    X_val_lstm = X_train_scaled[val_split:]
                    y_val_lstm = y_train[val_split:]
                
                transformer_model, transformer_seq_length = train_transformer(
                    X_train_lstm, y_train_lstm,
                    X_val_lstm, y_val_lstm,
                    input_size=len(features),
                    seq_length=LSTM_SEQUENCE_LENGTH,
                    epochs=LSTM_EPOCHS,
                    batch_size=LSTM_BATCH_SIZE,
                    learning_rate=LSTM_LEARNING_RATE * 0.5,  # Slightly lower LR for Transformer
                    d_model=TRANSFORMER_DIM,
                    nhead=TRANSFORMER_HEADS,
                    num_layers=TRANSFORMER_LAYERS,
                    dropout=TRANSFORMER_DROPOUT
                )
            
            # Train Multi-Task Transformer if enabled (PHASE 4!)
            multitask_model = None
            multitask_seq_length = LSTM_SEQUENCE_LENGTH
            if USE_MULTITASK and HAS_PYTORCH:
                # Use the same scaled data
                if not USE_LSTM and not USE_TRANSFORMER:
                    # Need to scale data if neither LSTM nor Transformer was trained
                    scaler = model['RF'].named_steps['scaler']
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    val_split = int(len(X_train_scaled) * 0.8)
                    X_train_lstm = X_train_scaled[:val_split]
                    y_train_lstm = y_train[:val_split]
                    X_val_lstm = X_train_scaled[val_split:]
                    y_val_lstm = y_train[val_split:]
                
                multitask_model, multitask_seq_length = train_multitask_transformer(
                    X_train_lstm, y_train_lstm,
                    X_val_lstm, y_val_lstm,
                    seq_length=LSTM_SEQUENCE_LENGTH,
                    epochs=MULTITASK_EPOCHS,
                    batch_size=LSTM_BATCH_SIZE,
                    learning_rate=LSTM_LEARNING_RATE * 0.5,  # Slightly lower LR
                    d_model=TRANSFORMER_DIM,
                    nhead=TRANSFORMER_HEADS,
                    num_layers=TRANSFORMER_LAYERS,
                    dropout=TRANSFORMER_DROPOUT,
                    direction_threshold=DIRECTION_THRESHOLD
                )
            
            # Evaluate ensemble (models are now pipelines with built-in scaling)
            if (USE_LSTM and lstm_model is not None) or (USE_TRANSFORMER and transformer_model is not None):
                # Get scaled test data for neural networks
                scaler = model['RF'].named_steps['scaler']
                X_test_scaled = scaler.transform(X_test)
                y_pred = ensemble_predict_with_lstm(
                    model, lstm_model, X_test_scaled, lstm_seq_length,
                    transformer_model=transformer_model
                )
                # Remove NaN values from sequence padding for evaluation
                valid_idx = ~np.isnan(y_pred)
                y_test_valid = y_test[valid_idx]
                y_pred_valid = y_pred[valid_idx]
            else:
                y_pred = ensemble_predict(model, X_test)
                y_test_valid = y_test
                y_pred_valid = y_pred
            
            # Individual model performance
            logging.info("\nIndividual Model Performance:")
            for name, m in model.items():
                y_pred_individual = m.predict(X_test)  # Pipeline handles scaling
                mse_individual = mean_squared_error(y_test, y_pred_individual)
                logging.info(f"  {name} RMSE: {np.sqrt(mse_individual):.6f}")
            
            # LSTM performance
            if USE_LSTM and lstm_model is not None:
                scaler = model['RF'].named_steps['scaler']
                X_test_scaled = scaler.transform(X_test)
                lstm_pred = lstm_predict(lstm_model, X_test_scaled, lstm_seq_length)
                valid_lstm = ~np.isnan(lstm_pred)
                if valid_lstm.sum() > 0:
                    mse_lstm = mean_squared_error(y_test[valid_lstm], lstm_pred[valid_lstm])
                    logging.info(f"  LSTM RMSE: {np.sqrt(mse_lstm):.6f}")
        else:
            # Train single model - RF handles complex features better than SVR
            logging.info(f"\n🎯 Training single RandomForest model...")
            tscv_single = TimeSeriesSplit(n_splits=n_splits, gap=3)
            model = train_model(X_train, y_train, model_name='RF', tscv=tscv_single)  # Unscaled data
            y_pred = model.predict(X_test)  # Pipeline handles scaling
            y_test_valid = y_test
            y_pred_valid = y_pred
        
        # Ensemble/final model evaluation
        mse = mean_squared_error(y_test_valid, y_pred_valid)
        logging.info(f"\nEnsemble Test MSE: {mse:.4f}")
        logging.info(f"Ensemble Test RMSE: {np.sqrt(mse):.4f}")
        
        # Predict next steps using the full last row with all features
        # Note: model is either a single pipeline or dict of pipelines (ensemble)
        last_row_full = df_final.iloc[-1]
        predictions, prediction_df = predict_next_steps(
            model=model,
            df_last_row_full=last_row_full,
            scaler=None,  # Scaling handled by pipeline
            features=features,
            steps=PREDICT_STEPS
        )
        
        # Convert RMSE back to price scale (since we're predicting returns)
        avg_price = df_final['price'].mean()
        price_rmse = np.sqrt(mse) * avg_price
        
        # Calculate total execution time
        total_execution_time = time.time() - overall_start_time
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Display results
        print("\n" + "="*70)
        if USE_ENSEMBLE:
            model_list = "RF + XGBoost + LightGBM" if HAS_LIGHTGBM else "RF + XGBoost"
            if USE_LSTM and lstm_model is not None:
                model_list += " + LSTM (GPU)"
            if USE_TRANSFORMER and transformer_model is not None:
                model_list += " + Transformer (GPU)"
            if USE_MULTITASK and multitask_model is not None:
                model_list += " + MultiTask (GPU)"
            print(f"🎯 ENSEMBLE MODEL SUMMARY ({model_list})")
        else:
            print("MULTI-TIMEFRAME MODEL SUMMARY")
        print("="*70)
        print(f"⏱️  TRAINING TIME")
        print(f"   Started: {current_time}")
        print(f"   Ended: {end_time}")
        print(f"   Total Duration: {total_execution_time:.1f}s ({total_execution_time/60:.2f} min)")
        print(f"\n📊 DATASET INFO")
        print(f"   Total Features Used: {len(features)}")
        print(f"   Training Samples: {len(X_train):,}")
        print(f"   Test Samples: {len(X_test):,}")
        print(f"   Cross-Validation: {n_splits}-fold Time Series CV with gap=3")
        
        print(f"\n🤖 MODEL ARCHITECTURE")
        if USE_ENSEMBLE:
            num_models = 3 if HAS_LIGHTGBM else 2
            if USE_LSTM and lstm_model is not None:
                num_models += 1
            if USE_TRANSFORMER and transformer_model is not None:
                num_models += 1
            if USE_MULTITASK and multitask_model is not None:
                num_models += 1
            print(f"   Ensemble ({num_models} models)")
            print(f"   • RandomForest (Traditional ML)")
            print(f"   • XGBoost (Gradient Boosting)")
            if HAS_LIGHTGBM:
                print(f"   • LightGBM (Fast Gradient Boosting)")
            if USE_LSTM and lstm_model is not None:
                print(f"   • LSTM with Attention (Deep Learning - GPU)")
                print(f"     - 3 layers, 256 hidden units")
                print(f"     - Attention: {'✅ ENABLED' if USE_ATTENTION else '❌ Disabled'}")
                print(f"     - Sequence length: {lstm_seq_length} hours")
                print(f"     - Trained on: {DEVICE}")
            if USE_TRANSFORMER and transformer_model is not None:
                print(f"   • 🚀 Transformer (PHASE 3 - GPU Accelerated)")
                print(f"     - {TRANSFORMER_LAYERS} encoder layers")
                print(f"     - {TRANSFORMER_HEADS} attention heads")
                print(f"     - Embedding dim: {TRANSFORMER_DIM}")
                print(f"     - Sequence length: {lstm_seq_length} hours")
                print(f"     - Trained on: {DEVICE}")
            if USE_MULTITASK and multitask_model is not None:
                print(f"   • 🎯 Multi-Task Transformer (PHASE 4 - Advanced!)")
                print(f"     - Tasks: Price + Volatility + Direction")
                print(f"     - {TRANSFORMER_LAYERS} encoder layers")
                print(f"     - {TRANSFORMER_HEADS} attention heads")
                print(f"     - Embedding dim: {TRANSFORMER_DIM}")
                print(f"     - Monte Carlo Dropout: {MULTITASK_DROPOUT_SAMPLES} samples")
                print(f"     - Uncertainty Quantification: ✅ ENABLED")
                print(f"     - Sequence length: {multitask_seq_length} hours")
                print(f"     - Trained on: {DEVICE}")
        else:
            print(f"   RandomForest")
        
        print(f"\n📈 MODEL PERFORMANCE")
        print(f"   Test Set Return RMSE: {np.sqrt(mse):.6f} ({np.sqrt(mse)*100:.2f}%)")
        print(f"   Approximate Price RMSE: ${price_rmse:.2f}")
        print(f"\n--- Last Actual Price ---")
        print(f"{last_row_full.name.strftime('%Y-%m-%d %H:%M')}: ${last_row_full['price']:.2f}")
        print(f"\n--- 12-Hour Price Forecast with 95% Confidence Intervals ---")
        print(f"{'Time':<20} {'Worst Case':<15} {'Most Likely':<15} {'Best Case':<15}")
        print("-" * 70)
        for date, row in prediction_df.iterrows():
            worst = row['lower_bound']
            likely = row['price']
            best = row['upper_bound']
            print(f"{date.strftime('%Y-%m-%d %H:%M'):<20} ${worst:>12,.2f}  ${likely:>12,.2f}  ${best:>12,.2f}")
        
        # Summary for key hours
        print(f"\n--- Key Forecast Summary ---")
        first_hour = prediction_df.iloc[0]
        sixth_hour = prediction_df.iloc[5] if len(prediction_df) >= 6 else prediction_df.iloc[-1]
        final_hour = prediction_df.iloc[-1]
        
        print(f"Hour 1:  ${first_hour['lower_bound']:,.2f} - ${first_hour['price']:,.2f} - ${first_hour['upper_bound']:,.2f}")
        print(f"Hour 6:  ${sixth_hour['lower_bound']:,.2f} - ${sixth_hour['price']:,.2f} - ${sixth_hour['upper_bound']:,.2f}")
        print(f"Hour 12: ${final_hour['lower_bound']:,.2f} - ${final_hour['price']:,.2f} - ${final_hour['upper_bound']:,.2f}")
        
        current_price = last_row_full['price']
        final_change = ((final_hour['price'] - current_price) / current_price) * 100
        final_best_change = ((final_hour['upper_bound'] - current_price) / current_price) * 100
        final_worst_change = ((final_hour['lower_bound'] - current_price) / current_price) * 100
        
        print(f"\n12-Hour Outlook: {final_worst_change:+.2f}% to {final_best_change:+.2f}% (most likely: {final_change:+.2f}%)")
        print("\n--- Last 7 Data Points (Actual) ---")
        print(df_final[['price']].tail(7))
        print("="*60)
        
        # =====================================================================
        # SAVE EVERYTHING TO STORAGE
        # =====================================================================
        logging.info("\n💾 Saving training run to storage...")
        
        # Prepare model list
        models_used = []
        if USE_ENSEMBLE:
            models_used.extend(['RandomForest', 'XGBoost'])
            if HAS_LIGHTGBM:
                models_used.append('LightGBM')
            if USE_LSTM and lstm_model is not None:
                models_used.append('LSTM')
            if USE_TRANSFORMER and transformer_model is not None:
                models_used.append('Transformer')
            if USE_MULTITASK and multitask_model is not None:
                models_used.append('MultiTask')
        
        # Save training run metadata
        run_data = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': total_execution_time,
            'models': models_used,
            'metrics': {
                'test_rmse': np.sqrt(mse),
                'test_rmse_pct': np.sqrt(mse) * 100,
                'price_rmse': price_rmse
            },
            'config': {
                'n_features': len(features),
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'cv_folds': n_splits,
                'lstm_sequence_length': lstm_seq_length if USE_LSTM else None,
                'transformer_layers': TRANSFORMER_LAYERS if USE_TRANSFORMER else None,
                'transformer_heads': TRANSFORMER_HEADS if USE_TRANSFORMER else None,
                'device': str(DEVICE) if HAS_PYTORCH else 'cpu'
            }
        }
        
        run_id = storage.save_training_run(run_data)
        logging.info(f"   ✅ Training run saved: {run_id}")
        
        # Save predictions
        storage.save_predictions(prediction_df, run_id)
        logging.info(f"   ✅ Predictions saved")
        
        # Save external data snapshot
        storage.save_external_data(external_data, run_id)
        logging.info(f"   ✅ External data saved")
        
        # Save feature importance if available
        if USE_ENSEMBLE and 'RF' in model:
            try:
                rf_pipeline = model['RF']
                rf_model = rf_pipeline.named_steps['model']
                feature_importance = pd.DataFrame({
                    'feature': features,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                storage.save_feature_importance(feature_importance, run_id, 'RandomForest')
                logging.info(f"   ✅ Feature importance saved")
            except Exception as e:
                logging.warning(f"   ⚠️  Could not save feature importance: {e}")
        
        # Save models (optional - can be large)
        try:
            if USE_LSTM and lstm_model is not None:
                storage.save_model(lstm_model, 'lstm', run_id)
                logging.info(f"   ✅ LSTM model saved")
            if USE_TRANSFORMER and transformer_model is not None:
                storage.save_model(transformer_model, 'transformer', run_id)
                logging.info(f"   ✅ Transformer model saved")
            if USE_MULTITASK and multitask_model is not None:
                storage.save_model(multitask_model, 'multitask', run_id)
                logging.info(f"   ✅ MultiTask model saved")
        except Exception as e:
            logging.warning(f"   ⚠️  Could not save neural network models: {e}")
        
        # Print final storage summary
        print("\n" + "="*70)
        storage.print_storage_summary()
        print("="*70)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
