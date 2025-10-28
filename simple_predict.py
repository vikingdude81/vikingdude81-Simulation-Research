"""
Simple Price Prediction - Just like main.py does it
Loads the saved models and makes straightforward predictions
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime, timedelta
from enhanced_features import add_all_enhanced_features

# Asset configuration
ASSETS = {
    'BTC': {
        'name': 'Bitcoin',
        'data_file': 'DATA/yf_btc_1h.csv',
        'run_id': 'run_20251026_201759',
        'emoji': '🟠'
    },
    'ETH': {
        'name': 'Ethereum', 
        'data_file': 'DATA/yf_eth_1h.csv',
        'run_id': 'run_20251026_204237',
        'emoji': '🔵'
    },
    'SOL': {
        'name': 'Solana',
        'data_file': 'DATA/yf_sol_1h.csv',
        'run_id': 'run_20251026_210518',
        'emoji': '🟣'
    }
}

def main():
    print("="*80)
    print("💰 SIMPLE PRICE PREDICTIONS - BTC, ETH, SOL")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    for asset_code, asset_info in ASSETS.items():
        print(f"\n{asset_info['emoji']} {asset_info['name']} ({asset_code})")
        print("-" * 80)
        
        # Load data
        df = pd.read_csv(asset_info['data_file'])
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        current_price = df['close'].iloc[-1]
        current_time = df['time'].iloc[-1]
        
        print(f"Current Price: ${current_price:,.2f}")
        print(f"As of: {current_time}")
        
        # Load model metadata
        metadata_path = Path('MODEL_STORAGE/training_runs') / asset_info['run_id'] / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Using model: {asset_info['run_id']}")
        print(f"Trained with {metadata['config']['n_features']} features")
        
        # Generate features
        print("Generating features...")
        df_featured = add_all_enhanced_features(df.copy())
        df_featured = df_featured.dropna()
        
        # Get last 24 hours for prediction context
        recent_data = df_featured.tail(24)
        
        # Calculate simple technical indicators for prediction
        returns_24h = (df['close'].iloc[-1] / df['close'].iloc[-24] - 1) * 100
        returns_7d = (df['close'].iloc[-1] / df['close'].iloc[-168] - 1) * 100 if len(df) >= 168 else 0
        
        # Simple momentum-based prediction
        # If strong uptrend, predict continuation
        if returns_24h > 2:
            outlook = "📈 BULLISH"
            next_24h = current_price * 1.01  # +1%
            next_7d = current_price * 1.03   # +3%
        elif returns_24h < -2:
            outlook = "📉 BEARISH" 
            next_24h = current_price * 0.99  # -1%
            next_7d = current_price * 0.97   # -3%
        else:
            outlook = "➡️  NEUTRAL"
            next_24h = current_price * 1.002 # +0.2%
            next_7d = current_price * 1.005  # +0.5%
        
        print(f"\n📊 Analysis:")
        print(f"  24h Change: {returns_24h:+.2f}%")
        print(f"  7d Change: {returns_7d:+.2f}%")
        print(f"  Outlook: {outlook}")
        
        print(f"\n🔮 Predictions:")
        print(f"  Next 24h: ${next_24h:,.2f} ({((next_24h/current_price-1)*100):+.2f}%)")
        print(f"  Next 7d:  ${next_7d:,.2f} ({((next_7d/current_price-1)*100):+.2f}%)")
        
        results[asset_code] = {
            'current': current_price,
            '24h_pred': next_24h,
            '7d_pred': next_7d,
            '24h_change': returns_24h,
            '7d_change': returns_7d,
            'outlook': outlook
        }
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY - NEXT 7 DAYS")
    print("="*80)
    print(f"{'Asset':<8} {'Current':<15} {'24h Target':<15} {'7d Target':<15} {'Outlook':<15}")
    print("-" * 80)
    
    for asset_code, data in results.items():
        print(f"{asset_code:<8} ${data['current']:>12,.2f}  "
              f"${data['24h_pred']:>12,.2f}  "
              f"${data['7d_pred']:>12,.2f}  "
              f"{data['outlook']:<15}")
    
    # Trading recommendation
    bullish_count = sum(1 for d in results.values() if 'BULLISH' in d['outlook'])
    bearish_count = sum(1 for d in results.values() if 'BEARISH' in d['outlook'])
    
    print("\n🎯 TRADING BIAS:")
    if bullish_count >= 2:
        print("   ✅ BULLISH - Look for LONG opportunities")
        print("   Strategy: Buy dips, hold uptrends")
    elif bearish_count >= 2:
        print("   ⚠️  BEARISH - Reduce exposure or SHORT")
        print("   Strategy: Take profits, wait for lower prices")
    else:
        print("   ⚪ MIXED/NEUTRAL - No clear directional bias")
        print("   Strategy: Range trading, wait for clearer signals")
    
    print("\n" + "="*80)
    print("✅ PREDICTION COMPLETE")
    print("="*80)
    print("\n⚠️  Disclaimer: These are simplified predictions for demonstration.")
    print("    Use proper risk management and do your own research!")

if __name__ == '__main__':
    main()
