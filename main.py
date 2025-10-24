
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import logging

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

def extract_indicator_features(df, prefix=''):
    """Extract key technical indicators from chart data."""
    features = {}
    
    # Price action
    if 'close' in df.columns:
        features[f'{prefix}close'] = df['close']
        features[f'{prefix}price_change'] = df['close'].pct_change()
    
    # Bollinger Bands
    if all(col in df.columns for col in ['BB Upper', 'BB Basis', 'BB Lower']):
        features[f'{prefix}bb_width'] = (df['BB Upper'] - df['BB Lower']) / df['BB Basis']
        features[f'{prefix}bb_position'] = (df['close'] - df['BB Lower']) / (df['BB Upper'] - df['BB Lower'])
    
    # Volume indicators
    if 'Volume Band (Close)' in df.columns:
        features[f'{prefix}volume'] = df['Volume Band (Close)']
    
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

def train_model(X_train, y_train, model_name='SVR'):
    """Train model with GridSearchCV."""
    if model_name == 'SVR':
        model = SVR()
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.1, 1],
            'kernel': ['rbf']
        }
    elif model_name == 'RF':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30]
        }
    elif model_name == 'XGB':
        model = xgb.XGBRegressor(random_state=42, tree_method='hist')
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.1]
        }
    elif model_name == 'LGB':
        model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.1]
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    logging.info(f"Starting GridSearchCV for {model_name}...")
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Best {model_name} Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_ensemble(X_train, y_train):
    """Train ensemble of RF, XGBoost, and optionally LightGBM models."""
    models = {}
    
    logging.info("Training RandomForest model...")
    models['RF'] = train_model(X_train, y_train, 'RF')
    
    logging.info("Training XGBoost model...")
    models['XGB'] = train_model(X_train, y_train, 'XGB')
    
    if HAS_LIGHTGBM:
        logging.info("Training LightGBM model...")
        models['LGB'] = train_model(X_train, y_train, 'LGB')
    else:
        logging.info("LightGBM not available, using RF + XGBoost ensemble")
    
    logging.info(f"Ensemble training complete with {len(models)} models!")
    return models

def ensemble_predict(models, X):
    """Make predictions using ensemble of models with equal weighting."""
    predictions = []
    
    for name, model in models.items():
        pred = model.predict(X)
        predictions.append(pred)
    
    # Average predictions from all models
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

def predict_next_steps(model, df_last_row_full, scaler, features, steps=3):
    """Multi-step prediction carrying forward multi-timeframe features.
    
    Args:
        model: Either a single model or dict of models (ensemble)
        df_last_row_full: Last row of data with all features
        scaler: Fitted StandardScaler
        features: List of feature names
        steps: Number of steps to predict
    """
    predictions = []
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
        X_pred_scaled = scaler.transform(X_pred)
        
        # Predict percentage return using ensemble or single model
        if is_ensemble:
            predicted_return = ensemble_predict(model, X_pred_scaled)[0]
        else:
            predicted_return = model.predict(X_pred_scaled)[0]
        
        # Convert return to price
        current_price = prices[-1]
        predicted_price = current_price * (1 + predicted_return)
        prices.append(predicted_price)
        predictions.append(predicted_price)
        
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
        'price': predictions
    }, index=[last_timestamp + pd.Timedelta(hours=i+1) for i in range(steps)])
    
    return predictions, prediction_df

# Main execution
if __name__ == "__main__":
    try:
        # Load all timeframe data
        df_90day, df_1h, df_4h, df_12h, df_1d, df_1w = load_multi_timeframe_data()
        
        # Combine features from all timeframes
        combined_df = combine_multi_timeframe_features(df_90day, df_1h, df_4h, df_12h, df_1d, df_1w)
        
        # Prepare data
        df_final, X, y, features = prepare_data(combined_df)
        
        # Time series split
        n_samples = len(X)
        split_point = int(n_samples * (1 - TEST_SIZE))
        
        X_train = X[:split_point]
        X_test = X[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]
        
        logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model(s)
        if USE_ENSEMBLE:
            logging.info("="*50)
            model_list = "RF + XGBoost + LightGBM" if HAS_LIGHTGBM else "RF + XGBoost"
            logging.info(f"Training Ensemble ({model_list})")
            logging.info("="*50)
            model = train_ensemble(X_train_scaled, y_train)
            
            # Evaluate ensemble
            y_pred = ensemble_predict(model, X_test_scaled)
            
            # Individual model performance
            logging.info("\nIndividual Model Performance:")
            for name, m in model.items():
                y_pred_individual = m.predict(X_test_scaled)
                mse_individual = mean_squared_error(y_test, y_pred_individual)
                logging.info(f"  {name} RMSE: {np.sqrt(mse_individual):.6f}")
        else:
            # Train single model - RF handles complex features better than SVR
            model = train_model(X_train_scaled, y_train, model_name='RF')
            y_pred = model.predict(X_test_scaled)
        
        # Ensemble/final model evaluation
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"\nEnsemble Test MSE: {mse:.4f}")
        logging.info(f"Ensemble Test RMSE: {np.sqrt(mse):.4f}")
        
        # Predict next steps using the full last row with all features
        last_row_full = df_final.iloc[-1]
        predictions, prediction_df = predict_next_steps(
            model=model,
            df_last_row_full=last_row_full,
            scaler=scaler,
            features=features,
            steps=PREDICT_STEPS
        )
        
        # Convert RMSE back to price scale (since we're predicting returns)
        avg_price = df_final['price'].mean()
        price_rmse = np.sqrt(mse) * avg_price
        
        # Display results
        print("\n" + "="*60)
        if USE_ENSEMBLE:
            model_list = "RF + XGBoost + LightGBM" if HAS_LIGHTGBM else "RF + XGBoost"
            print(f"🎯 ENSEMBLE MODEL SUMMARY ({model_list})")
        else:
            print("MULTI-TIMEFRAME MODEL SUMMARY")
        print("="*60)
        print(f"Total Features Used: {len(features)}")
        print(f"Training Samples: {len(X_train):,}")
        print(f"Test Samples: {len(X_test):,}")
        
        if USE_ENSEMBLE:
            num_models = 3 if HAS_LIGHTGBM else 2
            print(f"Model: Ensemble ({num_models} models with equal weighting)")
            print(f"  - RandomForest")
            print(f"  - XGBoost")
            if HAS_LIGHTGBM:
                print(f"  - LightGBM")
        else:
            print(f"Model: RandomForest")
        
        print(f"\nTest Set Return RMSE: {np.sqrt(mse):.6f} ({np.sqrt(mse)*100:.2f}%)")
        print(f"Approximate Price RMSE: ${price_rmse:.2f}")
        print(f"\n--- Last Actual Price ---")
        print(f"{last_row_full.name.strftime('%Y-%m-%d %H:%M')}: ${last_row_full['price']:.2f}")
        print(f"\n--- Predicted Prices for Next {PREDICT_STEPS} Hours ---")
        for date, row in prediction_df.iterrows():
            print(f"{date.strftime('%Y-%m-%d %H:%M')}: ${row['price']:.2f}")
        print("\n--- Last 7 Data Points (Actual) ---")
        print(df_final[['price']].tail(7))
        print("="*60)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
