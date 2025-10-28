"""
Generate Real-Time Market Predictions
Pull latest data, train models, generate signals for tomorrow + next week
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("\n" + "="*80)
print("🔮 REAL-TIME MARKET PREDICTIONS - TOMORROW & NEXT WEEK")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# STEP 1: Fetch Latest Data
# ============================================================================
print("\n📡 STEP 1: Fetching Latest Market Data...")

import yfinance as yf

# Fetch fresh data for all assets
assets_tickers = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD', 
    'SOL': 'SOL-USD'
}

for asset, ticker_symbol in assets_tickers.items():
    print(f"\n   Downloading {asset}...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period='730d', interval='1h')
        
        if not df.empty:
            # Prepare data
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={'date': 'time', 'datetime': 'time'})
            
            # Save
            output_file = f'DATA/yf_{asset.lower()}_1h.csv'
            df.to_csv(output_file, index=False)
            print(f"   ✅ {asset}: {len(df)} bars saved to {output_file}")
        else:
            print(f"   ⚠️ {asset}: No data retrieved")
    except Exception as e:
        print(f"   ❌ {asset}: Error - {e}")

print("\n✅ Latest data download complete")

# ============================================================================
# STEP 2: Train Models on Latest Data
# ============================================================================
print("\n" + "="*80)
print("🧠 STEP 2: Training Models on Latest Data...")
print("="*80)

import subprocess
import sys

# Train BTC model
print("\n🟠 Training BTC Model...")
result_btc = subprocess.run([sys.executable, 'main.py'], 
                            capture_output=True, text=True)
if result_btc.returncode == 0:
    print("   ✅ BTC model trained successfully")
else:
    print(f"   ⚠️ BTC training had issues (continuing anyway)")

# Load the latest BTC data (needed for predictions)
df_btc = pd.read_csv('DATA/yf_btc_1h.csv')
df_btc['time'] = pd.to_datetime(df_btc['time'])
df_btc = df_btc.sort_values('time').reset_index(drop=True)

print(f"\n📊 Latest BTC data loaded: {len(df_btc)} bars")
print(f"   Last update: {df_btc['time'].iloc[-1]}")

# ============================================================================
# STEP 3: Generate Predictions
# ============================================================================
print("\n" + "="*80)
print("🔮 STEP 3: Generating Price Predictions...")
print("="*80)

# Use existing data to calculate basic trend predictions
df_btc_pred = df_btc.tail(168).copy()  # Last week
current_price_btc = df_btc['close'].iloc[-1]
week_ago_price = df_btc['close'].iloc[-168] if len(df_btc) >= 168 else df_btc['close'].iloc[0]

# Calculate momentum-based projection
week_change_historical = ((current_price_btc - week_ago_price) / week_ago_price) * 100
week_high = df_btc['high'].tail(168).max()
week_low = df_btc['low'].tail(168).min()

print(f"\n📊 RECENT MARKET PERFORMANCE (Last 7 Days)")
print("="*80)
print(f"Current BTC Price:  ${current_price_btc:,.2f}")
print(f"Price 7 days ago:   ${week_ago_price:,.2f}")
print(f"7-Day Change:       {week_change_historical:+.2f}%")
print(f"7-Day High:         ${week_high:,.2f}")
print(f"7-Day Low:          ${week_low:,.2f}")
print(f"7-Day Range:        ${week_high - week_low:,.2f} ({((week_high - week_low) / week_low * 100):.2f}%)")

# Simple momentum projection
if week_change_historical > 5:
    outlook = "🚀 STRONG BULLISH MOMENTUM"
elif week_change_historical > 2:
    outlook = "📈 BULLISH MOMENTUM"
elif week_change_historical > -2:
    outlook = "➡️ CONSOLIDATION"
elif week_change_historical > -5:
    outlook = "📉 BEARISH PRESSURE"
else:
    outlook = "🔻 STRONG SELLING"

print(f"\nMarket Outlook:     {outlook}")

# ============================================================================
# STEP 4: Generate Trading Signals (GMA + ML)
# ============================================================================
print("\n" + "="*80)
print("🎯 STEP 4: Generating Trading Signals...")
print("="*80)

print(f"\n📊 Using BTC data: {len(df_btc)} bars")
print(f"   Current price: ${df_btc['close'].iloc[-1]:,.2f}")

# Calculate Geometric MA signals
from geometric_ma_crossover import GeometricMACrossover
from geometric_ma_settings import GEOMETRIC_MA_SETTINGS

# Use BTC optimal settings
gma_settings = GEOMETRIC_MA_SETTINGS['BTC']
gma = GeometricMACrossover(
    len_fast=gma_settings['len_fast'],
    len_slow=gma_settings['len_slow'],
    atr_length=14,
    stop_atr_mult=gma_settings['stop_atr_mult'],
    tp_rr=gma_settings['tp_rr']
)

# Prepare data for GMA
df_gma = df_btc[['close', 'high', 'low']].copy()
df_gma = gma.calculate(df_gma)

current_signal = gma.get_current_signal(df_gma)

print("\n" + "="*80)
print("🚀 GEOMETRIC MA CROSSOVER SIGNAL (Champion Indicator)")
print("="*80)
print(f"Signal:          {current_signal['signal_text']}")
print(f"Trend:           {current_signal['trend']}")
print(f"Current Price:   ${current_signal['price']:,.2f}")
print(f"GMA Fast (25):   ${current_signal['gma_fast']:,.2f}")
print(f"GMA Slow (75):   ${current_signal['gma_slow']:,.2f}")
print(f"Spread:          {current_signal['gma_spread_pct']:.2f}%")

if current_signal['signal'] != 0:
    print(f"\n🎯 Trade Setup:")
    print(f"   Entry:        ${current_signal['price']:,.2f}")
    print(f"   Stop Loss:    ${current_signal['stop_price']:,.2f}")
    print(f"   Take Profit:  ${current_signal['target_price']:,.2f}")
    print(f"   Risk:         ${abs(current_signal['price'] - current_signal['stop_price']):,.2f}")
    print(f"   Reward:       ${abs(current_signal['target_price'] - current_signal['price']):,.2f}")
    print(f"   R/R Ratio:    1:{gma_settings['tp_rr']}")

# ============================================================================
# STEP 5: Multi-Asset Analysis
# ============================================================================
print("\n" + "="*80)
print("💎 STEP 5: Multi-Asset Analysis (BTC, ETH, SOL)")
print("="*80)

multi_asset_signals = {}

for asset in ['BTC', 'ETH', 'SOL']:
    print(f"\n🔍 {asset} Analysis:")
    print("-" * 80)
    
    try:
        # Load data
        df = pd.read_csv(f'DATA/yf_{asset.lower()}_1h.csv')
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').tail(1000).reset_index(drop=True)
        
        current_price = df['close'].iloc[-1]
        print(f"   Current Price: ${current_price:,.2f}")
        
        # Calculate GMA signal
        asset_settings = GEOMETRIC_MA_SETTINGS.get(asset, GEOMETRIC_MA_SETTINGS['BTC'])
        gma_asset = GeometricMACrossover(
            len_fast=asset_settings['len_fast'],
            len_slow=asset_settings['len_slow'],
            atr_length=14,
            stop_atr_mult=asset_settings['stop_atr_mult'],
            tp_rr=asset_settings['tp_rr']
        )
        
        df_gma_asset = df[['close', 'high', 'low']].copy()
        df_gma_asset = gma_asset.calculate(df_gma_asset)
        signal_asset = gma_asset.get_current_signal(df_gma_asset)
        
        print(f"   GMA Signal: {signal_asset['signal_text']}")
        print(f"   Trend: {signal_asset['trend']}")
        print(f"   Spread: {signal_asset['gma_spread_pct']:.2f}%")
        
        # 24h change
        price_24h_ago = df['close'].iloc[-24] if len(df) >= 24 else df['close'].iloc[0]
        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
        print(f"   24h Change: {change_24h:+.2f}%")
        
        # Store signal
        multi_asset_signals[asset] = {
            'price': current_price,
            'signal': signal_asset['signal_text'],
            'trend': signal_asset['trend'],
            'spread_pct': signal_asset['gma_spread_pct'],
            'change_24h': change_24h
        }
        
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
        multi_asset_signals[asset] = {'error': str(e)}

# ============================================================================
# STEP 6: Final Recommendations
# ============================================================================
print("\n" + "="*80)
print("📋 FINAL TRADING RECOMMENDATIONS")
print("="*80)

print("\n🎯 IMMEDIATE SIGNALS (Next 24 Hours):")
print("-" * 80)

for asset, signal_data in multi_asset_signals.items():
    if 'error' not in signal_data:
        signal = signal_data['signal']
        trend = signal_data['trend']
        
        if signal == 'LONG':
            emoji = "🟢"
            action = "BUY"
        elif signal == 'SHORT':
            emoji = "🔴"
            action = "SELL"
        else:
            emoji = "⚪"
            action = "HOLD"
        
        print(f"{emoji} {asset}: {action:6s} | Trend: {trend:8s} | 24h: {signal_data['change_24h']:+6.2f}%")

print("\n📊 WEEKLY STRATEGY:")
print("-" * 80)

if week_change_historical > 0:
    print("✅ BULLISH BIAS continuing")
    print("   Strategy: Hold existing longs, look for dip entries")
    print("   Targets: Recent highs, breakout levels")
    print("   Risk: Use GMA levels as support")
else:
    print("⚠️ BEARISH PRESSURE detected")
    print("   Strategy: Caution on longs, watch for reversal signals")
    print("   Targets: Support levels, oversold bounces")
    print("   Risk: Use GMA levels as resistance")

print("\n💡 POSITION SIZING RECOMMENDATIONS:")
print("-" * 80)

# Count strong signals
strong_signals = sum(1 for s in multi_asset_signals.values() 
                    if 'signal' in s and s['signal'] in ['LONG', 'SHORT'])

if strong_signals >= 2:
    print("🔥 MULTIPLE CONFIRMATIONS - Medium-High conviction")
    print("   Suggested allocation: 50-75% of normal position")
elif strong_signals == 1:
    print("📊 SINGLE SIGNAL - Medium conviction")
    print("   Suggested allocation: 25-50% of normal position")
else:
    print("⚪ NO CLEAR SIGNALS - Low conviction")
    print("   Suggested allocation: Watch and wait, or 10-25% position")

print("\n⚠️ RISK MANAGEMENT:")
print("-" * 80)
print("✓ Always use stop losses (ATR-based from GMA signals)")
print("✓ Don't risk more than 1-2% of capital per trade")
print("✓ Consider partial profit taking at 1:1 R/R")
print("✓ Let winners run to 2:1 or 3:1 R/R")
print("✓ Monitor GMA 200 for major trend changes")

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'btc_recent_performance': {
        'current_price': float(current_price_btc),
        'week_ago_price': float(week_ago_price),
        'week_change_pct': float(week_change_historical),
        'outlook': outlook
    },
    'signals': multi_asset_signals
}

output_file = f'market_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n💾 Results saved to: {output_file}")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print("\n🚀 Good luck with your trades!")
print("📊 Remember: This is analysis, not financial advice. Trade responsibly!")
print("="*80 + "\n")
