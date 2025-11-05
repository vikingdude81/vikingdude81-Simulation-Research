"""
Calibrate Regime Detector Thresholds for Crypto Markets

Analyzes historical BTC data to determine appropriate thresholds for:
- VIX levels (volatile vs crisis)
- ADX levels (trending vs ranging)
- ATR patterns

This will give us crypto-specific thresholds instead of stock market defaults.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=close.index).rolling(window=period).sum()
    minus_dm = pd.Series(minus_dm, index=close.index).rolling(window=period).sum()
    
    # Calculate Directional Indicators
    plus_di = 100 * (plus_dm / atr)
    minus_di = 100 * (minus_dm / atr)
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def estimate_vix(close, window=30):
    """Estimate VIX-like volatility index from price data"""
    returns = close.pct_change()
    realized_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
    return realized_vol

def analyze_market_events(df):
    """Identify known market events for validation"""
    events = []
    
    # COVID Crash (Feb-Mar 2020)
    covid_crash = df[(df.index >= '2020-02-15') & (df.index <= '2020-03-31')]
    if len(covid_crash) > 0:
        events.append({
            'name': 'COVID Crash',
            'period': 'Feb-Mar 2020',
            'expected_regime': 'crisis',
            'data': covid_crash
        })
    
    # Bull Run (Nov 2020 - Apr 2021)
    bull_run = df[(df.index >= '2020-11-01') & (df.index <= '2021-04-30')]
    if len(bull_run) > 0:
        events.append({
            'name': 'Bull Run',
            'period': 'Nov 2020 - Apr 2021',
            'expected_regime': 'trending',
            'data': bull_run
        })
    
    # Bear Market (May 2022 - Nov 2022)
    bear_market = df[(df.index >= '2022-05-01') & (df.index <= '2022-11-30')]
    if len(bear_market) > 0:
        events.append({
            'name': 'Bear Market',
            'period': 'May-Nov 2022',
            'expected_regime': 'volatile/ranging',
            'data': bear_market
        })
    
    # Recovery (2023)
    recovery = df[(df.index >= '2023-01-01') & (df.index <= '2023-12-31')]
    if len(recovery) > 0:
        events.append({
            'name': 'Recovery',
            'period': '2023',
            'expected_regime': 'trending',
            'data': recovery
        })
    
    # Recent Period (2024-2025)
    recent = df[df.index >= '2024-01-01']
    if len(recent) > 0:
        events.append({
            'name': 'Recent Period',
            'period': '2024-2025',
            'expected_regime': 'mixed',
            'data': recent
        })
    
    return events

def main():
    print("🔬 CALIBRATING CRYPTO REGIME THRESHOLDS\n")
    print("=" * 70)
    
    # Load BTC data
    data_path = Path("DATA/yf_btc_1d.csv")
    print(f"\n📊 Loading BTC data from {data_path}...")
    
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    print(f"✅ Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    
    # Calculate indicators across full dataset
    print("\n📈 Calculating technical indicators...")
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    df['adx'] = calculate_adx(df['high'], df['low'], df['close'])
    df['vix_estimate'] = estimate_vix(df['close'])
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Drop NaN values from calculations
    df = df.dropna()
    print(f"✅ Calculated indicators for {len(df)} rows")
    
    # Analyze VIX distribution
    print("\n" + "=" * 70)
    print("📊 VIX ANALYSIS (Estimated from Realized Volatility)")
    print("=" * 70)
    
    vix_stats = {
        'min': df['vix_estimate'].min(),
        'max': df['vix_estimate'].max(),
        'mean': df['vix_estimate'].mean(),
        'median': df['vix_estimate'].median(),
        'std': df['vix_estimate'].std(),
        'p25': df['vix_estimate'].quantile(0.25),
        'p50': df['vix_estimate'].quantile(0.50),
        'p75': df['vix_estimate'].quantile(0.75),
        'p90': df['vix_estimate'].quantile(0.90),
        'p95': df['vix_estimate'].quantile(0.95),
        'p99': df['vix_estimate'].quantile(0.99),
    }
    
    print(f"\nVIX Range: {vix_stats['min']:.1f} - {vix_stats['max']:.1f}")
    print(f"Mean: {vix_stats['mean']:.1f} | Median: {vix_stats['median']:.1f} | Std: {vix_stats['std']:.1f}")
    print(f"\nPercentiles:")
    print(f"  25th: {vix_stats['p25']:.1f}")
    print(f"  50th: {vix_stats['p50']:.1f}")
    print(f"  75th: {vix_stats['p75']:.1f}")
    print(f"  90th: {vix_stats['p90']:.1f}")
    print(f"  95th: {vix_stats['p95']:.1f}")
    print(f"  99th: {vix_stats['p99']:.1f}")
    
    # Analyze ADX distribution
    print("\n" + "=" * 70)
    print("📊 ADX ANALYSIS (Trend Strength)")
    print("=" * 70)
    
    adx_stats = {
        'min': df['adx'].min(),
        'max': df['adx'].max(),
        'mean': df['adx'].mean(),
        'median': df['adx'].median(),
        'std': df['adx'].std(),
        'p25': df['adx'].quantile(0.25),
        'p50': df['adx'].quantile(0.50),
        'p75': df['adx'].quantile(0.75),
        'p90': df['adx'].quantile(0.90),
    }
    
    print(f"\nADX Range: {adx_stats['min']:.1f} - {adx_stats['max']:.1f}")
    print(f"Mean: {adx_stats['mean']:.1f} | Median: {adx_stats['median']:.1f}")
    print(f"\nPercentiles:")
    print(f"  25th: {adx_stats['p25']:.1f} (ranging)")
    print(f"  50th: {adx_stats['p50']:.1f}")
    print(f"  75th: {adx_stats['p75']:.1f} (trending)")
    print(f"  90th: {adx_stats['p90']:.1f} (strong trend)")
    
    # Analyze ATR distribution
    print("\n" + "=" * 70)
    print("📊 ATR ANALYSIS (Volatility)")
    print("=" * 70)
    
    atr_pct_stats = {
        'min': df['atr_pct'].min(),
        'max': df['atr_pct'].max(),
        'mean': df['atr_pct'].mean(),
        'median': df['atr_pct'].median(),
        'p75': df['atr_pct'].quantile(0.75),
        'p90': df['atr_pct'].quantile(0.90),
    }
    
    print(f"\nATR % of Price:")
    print(f"Mean: {atr_pct_stats['mean']:.2f}% | Median: {atr_pct_stats['median']:.2f}%")
    print(f"75th percentile: {atr_pct_stats['p75']:.2f}%")
    print(f"90th percentile: {atr_pct_stats['p90']:.2f}%")
    
    # Analyze known market events
    print("\n" + "=" * 70)
    print("🎯 KNOWN MARKET EVENTS ANALYSIS")
    print("=" * 70)
    
    events = analyze_market_events(df)
    event_stats = []
    
    for event in events:
        data = event['data']
        if len(data) == 0:
            continue
            
        stats = {
            'name': event['name'],
            'period': event['period'],
            'expected': event['expected_regime'],
            'vix_mean': data['vix_estimate'].mean(),
            'vix_max': data['vix_estimate'].max(),
            'adx_mean': data['adx'].mean(),
            'adx_max': data['adx'].max(),
            'price_range': f"${data['close'].min():.0f} - ${data['close'].max():.0f}",
            'price_change': ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
        }
        event_stats.append(stats)
        
        print(f"\n{event['name']} ({event['period']})")
        print(f"  Expected: {event['expected_regime']}")
        print(f"  VIX: {stats['vix_mean']:.1f} avg, {stats['vix_max']:.1f} max")
        print(f"  ADX: {stats['adx_mean']:.1f} avg, {stats['adx_max']:.1f} max")
        print(f"  Price: {stats['price_range']} ({stats['price_change']:+.1f}%)")
    
    # Recommend thresholds
    print("\n" + "=" * 70)
    print("💡 RECOMMENDED THRESHOLDS FOR CRYPTO")
    print("=" * 70)
    
    # VIX thresholds based on percentiles
    vix_low = vix_stats['p25']  # 25th percentile = ranging
    vix_medium = vix_stats['p75']  # 75th percentile = volatile
    vix_high = vix_stats['p95']  # 95th percentile = crisis
    
    # ADX thresholds
    adx_weak = adx_stats['p25']  # 25th percentile = ranging
    adx_strong = adx_stats['p75']  # 75th percentile = trending
    
    recommendations = {
        'vix_thresholds': {
            'ranging': f'< {vix_low:.1f}',
            'volatile': f'{vix_low:.1f} - {vix_high:.1f}',
            'crisis': f'> {vix_high:.1f}',
            'values': {
                'low': round(vix_low, 1),
                'high': round(vix_high, 1)
            }
        },
        'adx_thresholds': {
            'ranging': f'< {adx_weak:.1f}',
            'trending': f'> {adx_strong:.1f}',
            'values': {
                'weak': round(adx_weak, 1),
                'strong': round(adx_strong, 1)
            }
        },
        'comparison': {
            'stock_market': {
                'vix_volatile': 25,
                'vix_crisis': 35,
                'adx_trending': 25
            },
            'crypto_market': {
                'vix_volatile': round(vix_low, 1),
                'vix_crisis': round(vix_high, 1),
                'adx_trending': round(adx_strong, 1)
            }
        }
    }
    
    print(f"\n📈 VIX Thresholds:")
    print(f"  Ranging (calm):  VIX < {vix_low:.1f}")
    print(f"  Volatile:        VIX {vix_low:.1f} - {vix_high:.1f}")
    print(f"  Crisis:          VIX > {vix_high:.1f}")
    
    print(f"\n📊 ADX Thresholds:")
    print(f"  Ranging (weak):  ADX < {adx_weak:.1f}")
    print(f"  Trending:        ADX > {adx_strong:.1f}")
    
    print(f"\n🔄 Comparison with Stock Market:")
    print(f"  Stock VIX thresholds: 25 (volatile), 35 (crisis)")
    print(f"  Crypto VIX thresholds: {vix_low:.1f} (volatile), {vix_high:.1f} (crisis)")
    print(f"  → Crypto is {vix_low/25:.1f}x more volatile than stocks!")
    
    print(f"\n📝 Code Update:")
    print(f"```python")
    print(f"# In regime_detector.py __init__:")
    print(f"vix_threshold_high: float = {vix_low:.1f},      # 75th percentile")
    print(f"vix_threshold_extreme: float = {vix_high:.1f},  # 95th percentile")
    print(f"adx_threshold_low: float = {adx_weak:.1f},      # 25th percentile")
    print(f"adx_threshold_high: float = {adx_strong:.1f},   # 75th percentile")
    print(f"```")
    
    # Save results
    output = {
        'date': datetime.now().isoformat(),
        'dataset': {
            'source': 'DATA/yf_btc_1d.csv',
            'rows': len(df),
            'date_range': f"{df.index[0].date()} to {df.index[-1].date()}"
        },
        'vix_statistics': vix_stats,
        'adx_statistics': adx_stats,
        'atr_statistics': atr_pct_stats,
        'event_analysis': event_stats,
        'recommendations': recommendations
    }
    
    output_path = Path("outputs/regime_threshold_calibration.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("✅ CALIBRATION COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review the recommended thresholds above")
    print("2. Update regime_detector.py with new values")
    print("3. Re-run regime_detector.py to validate")
    print("4. Check if market events are classified correctly")
    
    return recommendations

if __name__ == "__main__":
    recommendations = main()
