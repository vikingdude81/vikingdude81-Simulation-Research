import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load Data from CSV
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

# Calculate Features: Enhanced Indicators
def calculate_features(df):
    df['returns'] = df['price'].pct_change()
    df['volatility'] = df['returns'].rolling(window=7).std()

    # ATR calculation
    df['high'] = df['price']  # Assume price as high
    df['low'] = df['price']  # Assume price as low
    df['prev_close'] = df['price'].shift(1)
    df['tr'] = df[['high', 'low', 'prev_close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['prev_close']), abs(x['low'] - x['prev_close'])),
        axis=1
    )
    df['atr'] = df['tr'].rolling(window=14).mean()

    # Trend-based features
    df['ema_10'] = df['price'].ewm(span=10, adjust=False).mean()  # 10-period EMA
    df['ema_30'] = df['price'].ewm(span=30, adjust=False).mean()  # 30-period EMA
    df['ema_diff'] = df['ema_10'] - df['ema_30']  # Difference between short-term and long-term EMA

    # Drop NaN values
    df.dropna(inplace=True)
    return df

# Prepare Dataset for Model
def prepare_data(df):
    df['pct_change'] = df['price'].pct_change()
    df['rolling_mean'] = df['price'].rolling(window=5).mean()
    df['rolling_std'] = df['price'].rolling(window=5).std()

    # Define features and target
    features = ['pct_change', 'rolling_mean', 'rolling_std', 'volatility', 'atr', 'ema_diff']
    df.dropna(inplace=True)
    X = df[features].values
    y = df['price'].shift(-1).dropna().values
    X = X[:-1]  # Align with y
    return df, X, y

# Train and Optimize SVR
def train_model(X_train, y_train):
    # Perform Grid Search for SVR Hyperparameters
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    svr = SVR()
    grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0)
    grid_search.fit(X_train, y_train)

    print("Best SVR Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# Main Execution
if __name__ == "__main__":
    try:
        # Step 1: Load and preprocess data
        file_path = './DATA/BTC_90day_data.csv'
        df = load_data(file_path)
        df = calculate_features(df)

        # Step 2: Prepare data
        df, X, y = prepare_data(df)

        # Step 3: Train-Test Split
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Step 4: Train Model
        model = train_model(X_train, y_train)

        # Step 5: Predict Next 3 Days
        last_known_data = X_scaled[-1].reshape(1, -1)  # Start with the last known data point
        predictions = []
        for _ in range(3):  # Predict 3 days ahead
            next_price = model.predict(last_known_data)[0]
            predictions.append(next_price)

            # Update features for the next prediction
            new_features = np.append(last_known_data[0, 1:], next_price).reshape(1, -1)
            last_known_data = scaler.transform(new_features)

        # Step 6: Display Results
        print("\nLast 7 Days of Data:")
        print(df[['price', 'volume']].tail(7))

        print("\nPredicted Prices for Next 3 Days:")
        for i, price in enumerate(predictions, start=1):
            print(f"Day {i}: {price:.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")
