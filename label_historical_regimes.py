"""
Label Historical Data by Market Regime

Processes historical crypto data and labels each period with its market regime
(volatile, trending, ranging, crisis) using the calibrated RegimeDetector.

This creates training datasets for regime-specific trading specialists.

Author: GA Conductor Research Team
Date: November 5, 2025
"""

import pandas as pd
import numpy as np
from regime_detector import RegimeDetector
import json
from datetime import datetime

def label_historical_data(symbol='BTC', data_path='DATA/yf_btc_1d.csv'):
    """
    Label each day in historical data with its market regime
    
    Args:
        symbol: Crypto symbol (BTC, ETH, SOL)
        data_path: Path to historical OHLCV data
    
    Returns:
        DataFrame with regime labels added
    """
    print(f"\n{'='*60}")
    print(f"📊 LABELING HISTORICAL DATA: {symbol}")
    print(f"{'='*60}\n")
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Initialize regime detector
    detector = RegimeDetector()
    print(f"Using calibrated thresholds:")
    print(f"  Volatile: VIX > {detector.vix_threshold_high:.1f}")
    print(f"  Crisis:   VIX > {detector.vix_threshold_extreme:.1f}")
    print(f"  Trending: ADX > {detector.adx_trending:.1f}")
    print(f"  Ranging:  ADX < {detector.adx_ranging:.1f}")
    
    # Calculate returns and volatility for detection
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(365)
    
    # Label each period
    print(f"\nLabeling {len(df)} days...")
    regimes = []
    
    for i in range(len(df)):
        if i < 100:
            # Need at least 100 days of history for accurate detection
            regimes.append('unknown')
            continue
        
        # Get window of data
        window_data = df.iloc[max(0, i-100):i+1].copy()
        
        # Detect regime
        regime = detector.detect_regime(window_data)
        regimes.append(regime)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(df)} days...")
    
    df['regime'] = regimes
    
    # Print distribution
    print(f"\n{'='*60}")
    print("REGIME DISTRIBUTION")
    print(f"{'='*60}")
    
    regime_counts = df['regime'].value_counts()
    total_labeled = len(df[df['regime'] != 'unknown'])
    
    for regime in ['volatile', 'trending', 'ranging', 'crisis', 'unknown']:
        if regime in regime_counts.index:
            count = regime_counts[regime]
            pct = (count / len(df)) * 100
            print(f"  {regime.upper():12s}: {count:5d} days ({pct:5.1f}%)")
    
    # Save labeled data
    output_path = data_path.replace('.csv', '_labeled.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved labeled data to: {output_path}")
    
    # Split by regime and save separate files
    print(f"\n{'='*60}")
    print("SPLITTING BY REGIME")
    print(f"{'='*60}\n")
    
    regime_data = {}
    
    for regime in ['volatile', 'trending', 'ranging', 'crisis']:
        regime_df = df[df['regime'] == regime].copy()
        
        if len(regime_df) > 0:
            regime_path = data_path.replace('.csv', f'_{regime}.csv')
            regime_df.to_csv(regime_path, index=False)
            
            regime_data[regime] = {
                'days': int(len(regime_df)),
                'path': regime_path,
                'date_range': {
                    'start': str(regime_df['date'].iloc[0]) if 'date' in regime_df.columns else str(regime_df.index[0]),
                    'end': str(regime_df['date'].iloc[-1]) if 'date' in regime_df.columns else str(regime_df.index[-1])
                },
                'price_stats': {
                    'mean': float(regime_df['close'].mean()),
                    'std': float(regime_df['close'].std()),
                    'min': float(regime_df['close'].min()),
                    'max': float(regime_df['close'].max())
                },
                'volatility_stats': {
                    'mean': float(regime_df['volatility'].mean()),
                    'std': float(regime_df['volatility'].std())
                }
            }
            
            print(f"✅ {regime.upper():12s}: {len(regime_df):4d} days → {regime_path}")
        else:
            print(f"⚠️  {regime.upper():12s}: No data found!")
            regime_data[regime] = None
    
    # Save regime summary
    summary = {
        'symbol': symbol,
        'total_days': int(len(df)),
        'labeled_days': int(total_labeled),
        'timestamp': datetime.now().isoformat(),
        'regimes': regime_data
    }
    
    summary_path = f'DATA/{symbol.lower()}_regime_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Saved regime summary to: {summary_path}")
    
    return df, regime_data


def analyze_regime_transitions(df):
    """
    Analyze how often regimes transition
    
    Args:
        df: DataFrame with regime labels
    """
    print(f"\n{'='*60}")
    print("REGIME TRANSITION ANALYSIS")
    print(f"{'='*60}\n")
    
    # Remove unknown periods
    df_labeled = df[df['regime'] != 'unknown'].copy()
    
    # Count transitions
    transitions = {}
    prev_regime = None
    
    for regime in df_labeled['regime']:
        if prev_regime is not None and regime != prev_regime:
            key = f"{prev_regime} → {regime}"
            transitions[key] = transitions.get(key, 0) + 1
        prev_regime = regime
    
    # Sort by frequency
    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    
    print("Most common transitions:")
    for transition, count in sorted_transitions[:10]:
        print(f"  {transition:30s}: {count:3d} times")
    
    # Calculate regime persistence (average days in each regime)
    print(f"\n{'='*60}")
    print("REGIME PERSISTENCE")
    print(f"{'='*60}\n")
    
    regime_runs = []
    current_regime = None
    run_length = 0
    
    for regime in df_labeled['regime']:
        if regime == current_regime:
            run_length += 1
        else:
            if current_regime is not None:
                regime_runs.append((current_regime, run_length))
            current_regime = regime
            run_length = 1
    
    # Add final run
    if current_regime is not None:
        regime_runs.append((current_regime, run_length))
    
    # Calculate statistics
    for regime in ['volatile', 'trending', 'ranging', 'crisis']:
        regime_lengths = [length for r, length in regime_runs if r == regime]
        if regime_lengths:
            print(f"{regime.upper():12s}: avg={np.mean(regime_lengths):.1f} days, "
                  f"median={np.median(regime_lengths):.0f}, "
                  f"max={np.max(regime_lengths):.0f}")


def visualize_regimes(df, symbol='BTC'):
    """
    Create visualization of regimes over time
    
    Args:
        df: DataFrame with regime labels
        symbol: Crypto symbol
    """
    try:
        import matplotlib.pyplot as plt
        
        print(f"\n{'='*60}")
        print("CREATING REGIME VISUALIZATION")
        print(f"{'='*60}\n")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot price
        df_plot = df[df['regime'] != 'unknown'].copy()
        
        # Color map for regimes
        regime_colors = {
            'volatile': 'orange',
            'trending': 'green',
            'ranging': 'blue',
            'crisis': 'red'
        }
        
        # Plot price with regime background colors
        for regime, color in regime_colors.items():
            mask = df_plot['regime'] == regime
            ax1.scatter(df_plot.index[mask], df_plot['close'][mask], 
                       c=color, alpha=0.5, s=1, label=regime)
        
        ax1.plot(df_plot.index, df_plot['close'], 'k-', alpha=0.2, linewidth=0.5)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(f'{symbol} Price with Regime Labels', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot regime as categorical
        regime_map = {'volatile': 3, 'trending': 2, 'ranging': 1, 'crisis': 4}
        df_plot['regime_num'] = df_plot['regime'].map(regime_map)
        
        ax2.scatter(df_plot.index, df_plot['regime_num'], 
                   c=[regime_colors[r] for r in df_plot['regime']], 
                   alpha=0.5, s=5)
        ax2.set_ylabel('Regime', fontsize=12)
        ax2.set_xlabel('Days', fontsize=12)
        ax2.set_yticks([1, 2, 3, 4])
        ax2.set_yticklabels(['Ranging', 'Trending', 'Volatile', 'Crisis'])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = f'outputs/regime_labels_{symbol.lower()}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to: {output_path}")
        
        plt.close()
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping visualization")


if __name__ == '__main__':
    # Label BTC data
    df, regime_data = label_historical_data(
        symbol='BTC',
        data_path='DATA/yf_btc_1d.csv'
    )
    
    # Analyze transitions
    analyze_regime_transitions(df)
    
    # Create visualization
    visualize_regimes(df, symbol='BTC')
    
    print(f"\n{'='*60}")
    print("✅ REGIME LABELING COMPLETE!")
    print(f"{'='*60}\n")
    
    # Print summary for next steps
    print("📋 NEXT STEPS:")
    print("  1. Use regime-specific CSV files for training specialists")
    print("  2. Each specialist will train on its own regime data")
    print("  3. Validate cross-regime performance (trending specialist on volatile data)")
    print()
    
    # Print data availability warning
    for regime, info in regime_data.items():
        if info is not None:
            days = info['days']
            if days < 100:
                print(f"⚠️  WARNING: Only {days} days of {regime} data - may not be enough for training!")
            elif days < 500:
                print(f"⚠️  CAUTION: Only {days} days of {regime} data - use conservative training settings")
