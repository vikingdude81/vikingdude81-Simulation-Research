"""
Multi-Asset Data Fetcher
========================

Fetches historical price data for SOL and ETH to match BTC data structure.
Downloads multiple timeframes: 1h, 4h, 1d, 1w, 1M

Author: AI Trading System
Date: October 25, 2025
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Create DATA directory if it doesn't exist
os.makedirs('DATA', exist_ok=True)

def fetch_crypto_data(symbol, period='2y', interval='1h'):
    """
    Fetch crypto data from Yahoo Finance.
    
    Args:
        symbol: Ticker symbol (e.g., 'SOL-USD', 'ETH-USD')
        period: Data period ('1y', '2y', 'max')
        interval: Data interval ('1h', '4h', '1d', '1wk', '1mo')
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching {symbol} {interval} data...")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print(f"  ❌ No data returned for {symbol} {interval}")
            return None
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Add timestamp column
        df['timestamp'] = df.index
        df['time'] = df.index
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Reorder columns
        cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits', 'timestamp']
        df = df[[col for col in cols if col in df.columns]]
        
        print(f"  ✅ Fetched {len(df)} rows ({df.iloc[0]['time']} to {df.iloc[-1]['time']})")
        
        return df
        
    except Exception as e:
        print(f"  ❌ Error fetching {symbol} {interval}: {e}")
        return None


def save_data(df, filename):
    """Save DataFrame to CSV file."""
    if df is not None and not df.empty:
        filepath = os.path.join('DATA', filename)
        df.to_csv(filepath, index=False)
        print(f"  💾 Saved to {filepath}")
        return True
    return False


def fetch_all_timeframes(symbol, asset_name):
    """
    Fetch all timeframes for a given asset.
    
    Args:
        symbol: Yahoo Finance symbol (e.g., 'SOL-USD')
        asset_name: Short name for file naming (e.g., 'sol', 'eth')
    """
    print("\n" + "="*80)
    print(f"📊 Fetching {asset_name.upper()} Data")
    print("="*80)
    
    timeframes = {
        '1h': ('2y', '1h'),
        '4h': ('2y', '1h'),  # Will resample from 1h
        '12h': ('2y', '1h'), # Will resample from 1h
        '1d': ('2y', '1d'),
        '1w': ('2y', '1wk'),
        '1M': ('2y', '1mo')
    }
    
    results = {}
    
    # Fetch hourly data first (can resample for 4h and 12h)
    df_1h = fetch_crypto_data(symbol, period='2y', interval='1h')
    if df_1h is not None:
        save_data(df_1h, f'yf_{asset_name}_1h.csv')
        results['1h'] = df_1h
        
        # Create 4h from 1h data
        print(f"Creating 4h data from 1h...")
        df_4h = resample_ohlcv(df_1h, '4H')
        if df_4h is not None:
            save_data(df_4h, f'yf_{asset_name}_4h.csv')
            results['4h'] = df_4h
        
        # Create 12h from 1h data
        print(f"Creating 12h data from 1h...")
        df_12h = resample_ohlcv(df_1h, '12H')
        if df_12h is not None:
            save_data(df_12h, f'yf_{asset_name}_12h.csv')
            results['12h'] = df_12h
    
    # Fetch daily data
    df_1d = fetch_crypto_data(symbol, period='2y', interval='1d')
    if df_1d is not None:
        save_data(df_1d, f'yf_{asset_name}_1d.csv')
        results['1d'] = df_1d
    
    # Fetch weekly data
    df_1w = fetch_crypto_data(symbol, period='2y', interval='1wk')
    if df_1w is not None:
        save_data(df_1w, f'yf_{asset_name}_1w.csv')
        results['1w'] = df_1w
    
    # Fetch monthly data
    df_1M = fetch_crypto_data(symbol, period='2y', interval='1mo')
    if df_1M is not None:
        save_data(df_1M, f'yf_{asset_name}_1M.csv')
        results['1M'] = df_1M
    
    return results


def resample_ohlcv(df, freq):
    """
    Resample OHLCV data to different frequency.
    
    Args:
        df: DataFrame with OHLCV data
        freq: Target frequency (e.g., '4H', '12H', '1D')
    
    Returns:
        Resampled DataFrame
    """
    try:
        df_copy = df.copy()
        df_copy['time'] = pd.to_datetime(df_copy['time'])
        df_copy = df_copy.set_index('time')
        
        # Resample OHLCV
        resampled = df_copy.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Reset index
        resampled = resampled.reset_index()
        resampled['timestamp'] = resampled['time']
        
        # Add missing columns if present
        if 'dividends' in df.columns:
            resampled['dividends'] = 0
        if 'stock splits' in df.columns:
            resampled['stock splits'] = 0
        
        print(f"  ✅ Resampled to {freq}: {len(resampled)} rows")
        
        return resampled
        
    except Exception as e:
        print(f"  ❌ Error resampling to {freq}: {e}")
        return None


def check_data_quality(df, asset_name, timeframe):
    """Check data quality and print statistics."""
    if df is None or df.empty:
        return
    
    print(f"\n📊 {asset_name.upper()} {timeframe} Data Quality:")
    print(f"  • Total Rows: {len(df)}")
    print(f"  • Date Range: {df['time'].min()} to {df['time'].max()}")
    print(f"  • Missing Values: {df.isnull().sum().sum()}")
    print(f"  • Price Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"  • Current Price: ${df['close'].iloc[-1]:.2f}")
    print(f"  • Avg Volume: {df['volume'].mean():,.0f}")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("🚀 MULTI-ASSET DATA FETCHER")
    print("="*80)
    print("\nFetching historical data for SOL and ETH...")
    print("This will match the BTC data structure for multi-asset training.\n")
    
    # Fetch SOL data
    sol_data = fetch_all_timeframes('SOL-USD', 'sol')
    
    # Fetch ETH data
    eth_data = fetch_all_timeframes('ETH-USD', 'eth')
    
    # Summary
    print("\n" + "="*80)
    print("📊 DATA FETCH SUMMARY")
    print("="*80)
    
    print("\n✅ SOL Data:")
    for tf, df in sol_data.items():
        if df is not None:
            print(f"  • {tf}: {len(df)} rows")
            check_data_quality(df, 'sol', tf)
    
    print("\n✅ ETH Data:")
    for tf, df in eth_data.items():
        if df is not None:
            print(f"  • {tf}: {len(df)} rows")
            check_data_quality(df, 'eth', tf)
    
    # Check existing BTC data for comparison
    print("\n📊 Existing BTC Data (for comparison):")
    btc_files = ['yf_btc_1h.csv', 'yf_btc_4h.csv', 'yf_btc_12h.csv', 'yf_btc_1d.csv']
    for filename in btc_files:
        filepath = os.path.join('DATA', filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"  • {filename}: {len(df)} rows")
    
    print("\n" + "="*80)
    print("✅ DATA FETCH COMPLETE!")
    print("="*80)
    print("\n📁 Files saved in DATA/ folder:")
    print("  SOL: yf_sol_1h.csv, yf_sol_4h.csv, yf_sol_12h.csv, yf_sol_1d.csv, yf_sol_1w.csv, yf_sol_1M.csv")
    print("  ETH: yf_eth_1h.csv, yf_eth_4h.csv, yf_eth_12h.csv, yf_eth_1d.csv, yf_eth_1w.csv, yf_eth_1M.csv")
    print("\n🚀 Ready for multi-asset model training!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
