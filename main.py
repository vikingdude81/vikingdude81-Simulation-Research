
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

# Try to import LightGBM, but continue without it if unavailable
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, OSError) as e:
    logging.warning(f"LightGBM not available: {e}")
    HAS_LIGHTGBM = False

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
FILE_PATH_90DAY = './DATA/BTC_90day_data.csv'
FILE_PATH_1H = './attached_assets/COINBASE_BTCUSD, 60_5c089_1761289206450.csv'
FILE_PATH_4H = './attached_assets/COINBASE_BTCUSD, 240_739cb_1761290157184.csv'
FILE_PATH_12H = './attached_assets/COINBASE_BTCUSD, 720_c69cd_1761290157184.csv'
FILE_PATH_1D = './attached_assets/COINBASE_BTCUSD, 1D_87ac3_1761289206450.csv'
FILE_PATH_1W = './attached_assets/COINBASE_BTCUSD, 1W_9771c_1761290157184.csv'

# Yahoo Finance data paths (fallback/supplement)
YF_FILE_PATH_1H = './DATA/yf_btc_1h.csv'
YF_FILE_PATH_4H = './DATA/yf_btc_4h.csv'
YF_FILE_PATH_12H = './DATA/yf_btc_12h.csv'
YF_FILE_PATH_1D = './DATA/yf_btc_1d.csv'
YF_FILE_PATH_1W = './DATA/yf_btc_1w.csv'

USE_YAHOO_FINANCE = True  # Set to True to use YF data instead
TEST_SIZE = 0.2
PREDICT_STEPS = 12  # Predict next 12 hours
USE_ENSEMBLE = True  # Use ensemble of RF + XGBoost + LightGBM

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
        
        # Load all timeframe data
        logging.info("\n📊 Loading multi-timeframe data...")
        data_start = time.time()
        df_90day, df_1h, df_4h, df_12h, df_1d, df_1w = load_multi_timeframe_data()
        data_time = time.time() - data_start
        logging.info(f"   Data loading completed in {data_time:.1f}s")
        
        # Combine features from all timeframes
        logging.info("\n🔧 Engineering features from multiple timeframes...")
        feature_start = time.time()
        combined_df = combine_multi_timeframe_features(df_90day, df_1h, df_4h, df_12h, df_1d, df_1w)
        feature_time = time.time() - feature_start
        logging.info(f"   Feature engineering completed in {feature_time:.1f}s")
        
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
        if USE_ENSEMBLE:
            model_list = "RF + XGBoost + LightGBM" if HAS_LIGHTGBM else "RF + XGBoost"
            logging.info(f"\n🎯 Starting Ensemble Training ({model_list})...")
            model = train_ensemble(X_train, y_train, tscv=tscv)  # Pass unscaled data
            
            # Evaluate ensemble (models are now pipelines with built-in scaling)
            y_pred = ensemble_predict(model, X_test)
            
            # Individual model performance
            logging.info("\nIndividual Model Performance:")
            for name, m in model.items():
                y_pred_individual = m.predict(X_test)  # Pipeline handles scaling
                mse_individual = mean_squared_error(y_test, y_pred_individual)
                logging.info(f"  {name} RMSE: {np.sqrt(mse_individual):.6f}")
        else:
            # Train single model - RF handles complex features better than SVR
            logging.info(f"\n🎯 Training single RandomForest model...")
            tscv_single = TimeSeriesSplit(n_splits=n_splits, gap=3)
            model = train_model(X_train, y_train, model_name='RF', tscv=tscv_single)  # Unscaled data
            y_pred = model.predict(X_test)  # Pipeline handles scaling
        
        # Ensemble/final model evaluation
        mse = mean_squared_error(y_test, y_pred)
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
        print("\n" + "="*60)
        if USE_ENSEMBLE:
            model_list = "RF + XGBoost + LightGBM" if HAS_LIGHTGBM else "RF + XGBoost"
            print(f"🎯 ENSEMBLE MODEL SUMMARY ({model_list})")
        else:
            print("MULTI-TIMEFRAME MODEL SUMMARY")
        print("="*60)
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
            print(f"   Ensemble ({num_models} models with equal weighting)")
            print(f"   • RandomForest")
            print(f"   • XGBoost")
            if HAS_LIGHTGBM:
                print(f"   • LightGBM")
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
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
