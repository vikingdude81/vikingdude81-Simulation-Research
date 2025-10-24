
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import logging

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
PREDICT_STEPS = 3

# Load and preprocess multi-timeframe data
def load_multi_timeframe_data():
    """Loads all timeframe data and returns them as dataframes."""
    try:
        # Load 90-day baseline data
        df_90day = pd.read_csv(FILE_PATH_90DAY)
        df_90day['timestamp'] = pd.to_datetime(df_90day['timestamp'])
        df_90day.set_index('timestamp', inplace=True)
        
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
    
    logging.info(f"90-day data shape: {df_90day.shape}, date range: {df_90day.index.min()} to {df_90day.index.max()}")
    logging.info(f"1h data shape: {df_1h.shape}, date range: {df_1h.index.min()} to {df_1h.index.max()}")
    logging.info(f"4h data shape: {df_4h.shape}, date range: {df_4h.index.min()} to {df_4h.index.max()}")
    logging.info(f"12h data shape: {df_12h.shape}, date range: {df_12h.index.min()} to {df_12h.index.max()}")
    logging.info(f"1d data shape: {df_1d.shape}, date range: {df_1d.index.min()} to {df_1d.index.max()}")
    logging.info(f"1w data shape: {df_1w.shape}, date range: {df_1w.index.min()} to {df_1w.index.max()}")
    
    # Use 1-day data as the base timeline since it has the most recent and comprehensive coverage
    combined = df_1d[['close']].copy()
    combined.rename(columns={'close': 'price'}, inplace=True)
    
    # Extract features from each timeframe
    features_1h = extract_indicator_features(df_1h, prefix='1h_')
    features_4h = extract_indicator_features(df_4h, prefix='4h_')
    features_12h = extract_indicator_features(df_12h, prefix='12h_')
    features_1d = extract_indicator_features(df_1d, prefix='1d_')
    features_1w = extract_indicator_features(df_1w, prefix='1w_')
    
    # Resample hourly to match daily frequency with forward fill (limit to 7 days)
    features_1h_resampled = features_1h.resample('1D').last()
    features_1h_resampled = features_1h_resampled.reindex(combined.index)
    features_1h_resampled = features_1h_resampled.ffill(limit=7)
    
    # Resample 4-hourly to match daily frequency (limit to 7 days)
    features_4h_resampled = features_4h.resample('1D').last()
    features_4h_resampled = features_4h_resampled.reindex(combined.index)
    features_4h_resampled = features_4h_resampled.ffill(limit=7)
    
    # Resample 12-hourly to match daily frequency (limit to 7 days)
    features_12h_resampled = features_12h.resample('1D').last()
    features_12h_resampled = features_12h_resampled.reindex(combined.index)
    features_12h_resampled = features_12h_resampled.ffill(limit=7)
    
    # Daily features already match, just use them
    features_1d_resampled = features_1d
    
    # Resample weekly to match daily frequency (limit to 14 days)
    features_1w_resampled = features_1w.resample('1D').last()
    features_1w_resampled = features_1w_resampled.reindex(combined.index)
    features_1w_resampled = features_1w_resampled.ffill(limit=14)
    
    # Combine all features
    combined = pd.concat([combined, features_1h_resampled, features_4h_resampled, 
                         features_12h_resampled, features_1d_resampled, features_1w_resampled], axis=1)
    
    # Add original features from 90-day data
    combined['pct_change'] = combined['price'].pct_change()
    combined['rolling_mean_5'] = combined['price'].rolling(window=5).mean()
    combined['rolling_std_5'] = combined['price'].rolling(window=5).std()
    combined['rolling_mean_20'] = combined['price'].rolling(window=20).mean()
    combined['rolling_std_20'] = combined['price'].rolling(window=20).std()
    
    # Target variable
    combined['target_price'] = combined['price'].shift(-1)
    
    logging.info(f"Combined features before cleanup: {combined.shape}")
    logging.info(f"NaN counts per column:\n{combined.isna().sum()}")
    
    # Fill remaining NaN values with 0 for multi-timeframe features, but drop rows where core features are missing
    # First, drop rows where target_price is NaN (last row)
    combined = combined[combined['target_price'].notna()]
    
    # Then fill NaN in non-critical features with 0
    combined = combined.fillna(0)
    
    logging.info(f"Combined features shape after cleanup: {combined.shape}")
    logging.info(f"Remaining NaN counts: {combined.isna().sum().sum()}")
    
    return combined

def prepare_data(combined_df):
    """Prepare X and y from combined dataframe."""
    
    # Select features (exclude price and target)
    feature_cols = [col for col in combined_df.columns if col not in ['price', 'target_price']]
    
    X = combined_df[feature_cols].values
    y = combined_df['target_price'].values
    
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
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    logging.info(f"Starting GridSearchCV for {model_name}...")
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Best {model_name} Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def predict_next_steps(model, df_last_rows, scaler, features, steps=3):
    """Multi-step prediction with feature recalculation."""
    predictions = []
    prediction_history = df_last_rows[['price']].copy()
    
    for i in range(steps):
        # Recalculate features for the last row
        temp_df = prediction_history.copy()
        temp_df['pct_change'] = temp_df['price'].pct_change()
        temp_df['rolling_mean_5'] = temp_df['price'].rolling(window=5).mean()
        temp_df['rolling_std_5'] = temp_df['price'].rolling(window=5).std()
        temp_df['rolling_mean_20'] = temp_df['price'].rolling(window=20).mean()
        temp_df['rolling_std_20'] = temp_df['price'].rolling(window=20).std()
        temp_df.dropna(inplace=True)
        
        if len(temp_df) == 0:
            logging.warning("Not enough data for prediction")
            break
        
        # Get the last row features that exist in the model
        current_features = []
        last_row = temp_df.iloc[-1]
        
        for feat in features:
            if feat in temp_df.columns:
                current_features.append(last_row[feat])
            else:
                # Use 0 for missing features (they'll be from other timeframes)
                current_features.append(0)
        
        X_pred = np.array(current_features).reshape(1, -1)
        X_pred_scaled = scaler.transform(X_pred)
        
        predicted_price = model.predict(X_pred_scaled)[0]
        predictions.append(predicted_price)
        
        # Add predicted price to history
        next_timestamp = temp_df.index[-1] + pd.Timedelta(hours=1)
        new_row = pd.Series({'price': predicted_price}, name=next_timestamp)
        prediction_history = pd.concat([prediction_history, new_row.to_frame().T])
    
    return predictions, prediction_history.iloc[-steps:]

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
        
        # Train model
        model = train_model(X_train_scaled, y_train, model_name='SVR')
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"Model Test MSE: {mse:.4f}")
        logging.info(f"Model Test RMSE: {np.sqrt(mse):.4f}")
        
        # Predict next steps
        last_N_rows = df_final.tail(50)[['price']].copy()
        predictions, prediction_history = predict_next_steps(
            model=model,
            df_last_rows=last_N_rows,
            scaler=scaler,
            features=features,
            steps=PREDICT_STEPS
        )
        
        # Display results
        print("\n" + "="*50)
        print("ENHANCED MULTI-TIMEFRAME MODEL SUMMARY")
        print("="*50)
        print(f"Total Features Used: {len(features)}")
        print(f"Model: SVR with params: {model.get_params()}")
        print(f"Test Set RMSE: {np.sqrt(mse):.4f}")
        print("\n--- Predicted Prices for Next 3 Periods ---")
        for date, price in prediction_history['price'].items():
            print(f"{date.strftime('%Y-%m-%d %H:%M')}: ${price:.2f}")
        print("\n--- Last 7 Data Points (Actual) ---")
        print(df_final[['price']].tail(7))
        print("="*50)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
