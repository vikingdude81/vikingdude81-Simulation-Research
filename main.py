
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
FILE_PATH = './DATA/BTC_90day_data.csv'
TEST_SIZE = 0.2  # 20% for testing (maintains time series order)
PREDICT_STEPS = 3

# Load Data from CSV
def load_data(file_path):
    """Loads data, ensures 'timestamp' is datetime index, and checks for necessary columns."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}. Please check the path.")
        raise

    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")
    if 'price' not in df.columns:
        raise ValueError("DataFrame must contain a 'price' column.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

# Calculate Features: Enhanced Indicators
def calculate_features(df):
    """Calculates necessary technical indicators for prediction."""

    df['returns'] = df['price'].pct_change()
    df['volatility'] = df['returns'].rolling(window=7).std()

    # --- ATR Calculation Note ---
    # ATR typically requires 'High' and 'Low' prices. Since they are missing,
    # we use 'price' as a proxy, which is technically inaccurate for True Range,
    # but maintains the original code's structure.
    # RECOMMENDATION: Use actual High/Low/Close data for better ATR.
    df['high'] = df.get('high', df['price'])
    df['low'] = df.get('low', df['price'])
    df['prev_close'] = df['price'].shift(1)

    df['tr'] = df[['high', 'low', 'prev_close']].apply(
        lambda x: max(x['high'] - x['low'],
                      abs(x['high'] - x['prev_close']),
                      abs(x['low'] - x['prev_close'])),
        axis=1
    )
    df['atr'] = df['tr'].rolling(window=14).mean()

    # Trend-based features (EMAs)
    df['ema_10'] = df['price'].ewm(span=10, adjust=False).mean()  # 10-period EMA
    df['ema_30'] = df['price'].ewm(span=30, adjust=False).mean()  # 30-period EMA
    df['ema_diff'] = df['ema_10'] - df['ema_30']  # Difference between short-term and long-term EMA

    # Drop intermediate columns
    df.drop(columns=['high', 'low', 'prev_close', 'tr'], errors='ignore', inplace=True)

    df.dropna(inplace=True)
    return df

# Prepare Dataset for Model
def prepare_data(df):
    """Generates final features and target variable."""
    # Generate additional features that rely on 'price'
    df['pct_change'] = df['price'].pct_change()
    df['rolling_mean'] = df['price'].rolling(window=5).mean()
    df['rolling_std'] = df['price'].rolling(window=5).std()

    # The target is the price one period ahead (tomorrow's price)
    # This introduces the shift/lag needed for prediction.
    df['target_price'] = df['price'].shift(-1)

    # Drop any remaining NaNs after feature creation and shift
    df.dropna(inplace=True)

    # Define features and target based on the cleaned data
    features = ['pct_change', 'rolling_mean', 'rolling_std', 'volatility', 'atr', 'ema_diff', 'ema_10', 'ema_30']
    X = df[features].values
    y = df['target_price'].values

    # We must ensure X and y are perfectly aligned
    # X includes the features up to time T
    # y includes the price at time T+1 (the target)
    return df, X, y, features

# Train and Optimize Model using Grid Search
def train_model(X_train, y_train, model_name='SVR'):
    """Performs Grid Search for hyperparameter optimization."""
    if model_name == 'SVR':
        model = SVR()
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.1, 1],
            'kernel': ['rbf'] # RBF is usually sufficient for SVR time series
        }
    elif model_name == 'RF':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    logging.info(f"Starting GridSearchCV for {model_name}...")
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    logging.info(f"Best {model_name} Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# --- CRITICAL IMPROVEMENT: ROLLING PREDICTION LOGIC ---
def predict_next_steps(model, df_last_row, scaler, features, steps=3):
    """
    Performs multi-step-ahead prediction using the model's output as the input for the next step.
    This requires re-calculating the features at each step.
    """
    predictions = []
    
    # Store history for feature re-calculation (using only 'price' and 'timestamp')
    prediction_history = df_last_row[['price']].copy()
    
    # Calculate initial features on the historical data
    current_df = calculate_features(df_last_row.copy())
    current_df['pct_change'] = current_df['price'].pct_change()
    current_df['rolling_mean'] = current_df['price'].rolling(window=5).mean()
    current_df['rolling_std'] = current_df['price'].rolling(window=5).std()
    current_df.dropna(inplace=True)
    
    last_price = current_df['price'].iloc[-1]
    
    for i in range(steps):
        # 1. Calculate features for the last known/predicted price point
        current_features = current_df.tail(1)[features].values
        
        # 2. Scale the features
        X_pred = scaler.transform(current_features)
        
        # 3. Predict the next price
        predicted_price = model.predict(X_pred)[0]
        
        predictions.append(predicted_price)
        
        # 4. Update the history DataFrame with the new prediction
        # Get the timestamp of the next day/period
        next_timestamp = current_df.index[-1] + pd.Timedelta(days=1) # Assuming daily data
        
        # Create a new row for the predicted price
        new_row = pd.Series({'price': predicted_price}, name=next_timestamp)
        prediction_history = pd.concat([prediction_history, new_row.to_frame().T])
        
        # 5. Re-calculate ALL rolling/lagged features on the updated history
        # This is the key difference from the original code!
        temp_df = calculate_features(prediction_history.copy())
        
        # Calculate the additional features from prepare_data
        temp_df['pct_change'] = temp_df['price'].pct_change()
        temp_df['rolling_mean'] = temp_df['price'].rolling(window=5).mean()
        temp_df['rolling_std'] = temp_df['price'].rolling(window=5).std()
        temp_df.dropna(inplace=True)
        
        # The new current_df must contain the *newly calculated features* for the latest predicted price
        current_df = temp_df.tail(1).copy()
        
    return predictions, prediction_history.iloc[-steps:]


# Main Execution
if __name__ == "__main__":
    try:
        # Step 1: Load and preprocess data
        df = load_data(FILE_PATH)
        df_processed = calculate_features(df.copy()) # Use a copy to avoid side effects

        # Step 2: Prepare data
        df_final, X, y, features = prepare_data(df_processed.copy())
        
        # Step 3: Time Series Split (maintains order)
        n_samples = len(X)
        split_point = int(n_samples * (1 - TEST_SIZE))
        
        X_train = X[:split_point]
        X_test = X[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]
        
        # Keep the unscaled training data for feature calculations during prediction
        df_train = df_final.iloc[:split_point]
        df_test = df_final.iloc[split_point:]

        logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Step 4: Scale Data (fit only on training data)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Step 5: Train Model (using SVR as in original)
        model = train_model(X_train_scaled, y_train, model_name='SVR')

        # Step 6: Evaluate Model on Test Set (using un-shuffled data)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"Model Test MSE: {mse:.4f}")
        logging.info(f"Model Test RMSE: {np.sqrt(mse):.4f}")

        # Step 7: Predict Next Steps (3 days)
        # We need the last N rows of the FULL processed DataFrame to correctly calculate
        # the rolling features for the first prediction step. N must be >= max window size (e.g., 30 for EMA)
        last_N_rows = df_final.tail(35)[['price']].copy() # Get enough data for re-calculation

        predictions, prediction_history = predict_next_steps(
            model=model, 
            df_last_row=last_N_rows, 
            scaler=scaler, 
            features=features, 
            steps=PREDICT_STEPS
        )

        # Step 8: Display Results
        print("\n" + "="*50)
        print("MODEL SUMMARY")
        print("="*50)
        print(f"Features Used: {', '.join(features)}")
        print(f"Model: SVR with best params: {model.get_params()}")
        print(f"Test Set RMSE: {np.sqrt(mse):.4f} (Lower is better)")
        print("\n--- Predicted Prices for Next 3 Days ---")
        for date, price in prediction_history['price'].items():
            print(f"{date.strftime('%Y-%m-%d')}: {price:.2f}")
        print("\n--- Last 7 Days of Data (Actual) ---")
        print(df_final[['price']].tail(7))
        print("="*50)

    except Exception as e:
        logging.error(f"An error occurred in the main execution block: {e}")
