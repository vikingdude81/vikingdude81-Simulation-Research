
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

def fetch_and_save_data():
    """Fetch BTC-USD data from Yahoo Finance for multiple timeframes."""
    
    # Define timeframes and their parameters
    timeframes = {
        '1h': {'period': '730d', 'interval': '1h', 'file': SCRIPT_DIR / 'DATA' / 'yf_btc_1h.csv'},  # 2 years of hourly
        '4h': {'period': 'max', 'interval': '1h', 'file': SCRIPT_DIR / 'DATA' / 'yf_btc_4h.csv'},   # Will resample
        '12h': {'period': 'max', 'interval': '1h', 'file': SCRIPT_DIR / 'DATA' / 'yf_btc_12h.csv'}, # Will resample
        '1d': {'period': 'max', 'interval': '1d', 'file': SCRIPT_DIR / 'DATA' / 'yf_btc_1d.csv'},
        '1w': {'period': 'max', 'interval': '1wk', 'file': SCRIPT_DIR / 'DATA' / 'yf_btc_1w.csv'},
    }
    
    ticker = yf.Ticker("BTC-USD")
    
    for tf_name, params in timeframes.items():
        try:
            logging.info(f"Fetching {tf_name} data...")
            
            # Download data
            df = ticker.history(period=params['period'], interval=params['interval'])
            
            if df.empty:
                logging.warning(f"No data retrieved for {tf_name}")
                continue
            
            # Resample if needed for 4h and 12h (before renaming columns)
            if tf_name == '4h':
                df = df.resample('4h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            elif tf_name == '12h':
                df = df.resample('12h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            
            # Reset index to make timestamp a column
            df.reset_index(inplace=True)
            
            # Rename columns (yfinance uses 'Datetime' as index name)
            df.rename(columns={
                'Datetime': 'time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # Ensure time column is timezone-aware and convert to unix timestamp
            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize('UTC')
            df['timestamp'] = df['time'].astype(int) // 10**9
            
            # Save to CSV
            df.to_csv(params['file'], index=False)
            
            logging.info(f"Saved {tf_name} data: {len(df)} rows from {df['time'].min()} to {df['time'].max()}")
            logging.info(f"File saved to: {params['file']}")
            
        except Exception as e:
            logging.error(f"Error fetching {tf_name} data: {e}")
    
    logging.info("Data fetching complete!")

if __name__ == "__main__":
    fetch_and_save_data()
