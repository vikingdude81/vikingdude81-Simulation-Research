"""
RECOMMENDED INDICATOR SETTINGS
Based on ML-guided optimization results from October 26, 2025

These settings have been tested across 90 days of historical data
and optimized for each asset (BTC/ETH/SOL).
"""

# ============================================================================
# RUBBER-BAND REVERSION INDICATOR
# ============================================================================
# Best for: BTC and ETH mean-reversion trades
# Performance: 80% win rate, 0.52-0.88 Sharpe ratio
# Use case: Secondary confirmation for ML signals

RUBBERBAND_SETTINGS = {
    'BTC': {
        'z_lookback': 200,              # Lookback period for z-score calculation
        'upper_threshold': 2.5,         # Upper threshold for mean reversion
        'lower_threshold': -2.0,        # Lower threshold for mean reversion
        'smoothing': 3,                 # EMA smoothing periods (fast response)
        'cooldown_periods': 5,          # Minimum bars between signals
        'use_dynamic_thresholds': True, # Enable ATR-based threshold adjustment
        'use_keltner_filter': True,     # Enable Keltner Channel filter
        'use_adaptive_weights': False,  # Disable adaptive MA weighting
        'ma_type': 'ema',              # Moving average type
        'use_20': True,                # Use 20-period MA
        'use_50': True,                # Use 50-period MA
        'use_100': True,               # Use 100-period MA
        'use_200': True,               # Use 200-period MA
        # Performance:
        # - Win Rate: 80.0%
        # - Sharpe: 0.883
        # - Total Return: +2.34%
        # - Trades: 5 (very selective)
    },
    
    'ETH': {
        'z_lookback': 200,              # Same as BTC
        'upper_threshold': 2.5,         # Same as BTC
        'lower_threshold': -2.0,        # Same as BTC
        'smoothing': 7,                 # Longer smoothing (ETH more volatile)
        'cooldown_periods': 5,          # Same as BTC
        'use_dynamic_thresholds': True,
        'use_keltner_filter': True,
        'use_adaptive_weights': False,
        'ma_type': 'ema',
        'use_20': True,
        'use_50': True,
        'use_100': True,
        'use_200': True,
        # Performance:
        # - Win Rate: 80.0%
        # - Sharpe: 0.519
        # - Total Return: +3.15%
        # - Trades: 5 (very selective)
    },
    
    'SOL': {
        'z_lookback': 150,              # Shorter lookback (SOL faster moving)
        'upper_threshold': 2.5,         # Same as BTC/ETH
        'lower_threshold': -1.5,        # Less aggressive lower threshold
        'smoothing': 7,                 # Longer smoothing like ETH
        'cooldown_periods': 5,
        'use_dynamic_thresholds': True,
        'use_keltner_filter': True,
        'use_adaptive_weights': False,
        'ma_type': 'ema',
        'use_20': True,
        'use_50': True,
        'use_100': True,
        'use_200': True,
        # Performance:
        # - Win Rate: 60.0%
        # - Sharpe: 0.608
        # - Total Return: +11.32%
        # - Trades: 5
        # NOTE: Volatility Hole performs better for SOL (1.40 Sharpe)
    }
}


# ============================================================================
# VOLATILITY HOLE DETECTOR
# ============================================================================
# Best for: SOL breakout trades (BEATS ML baseline!)
# Performance: 80% win rate, 1.395 Sharpe ratio
# Use case: Primary filter for SOL, secondary for BTC

VOLATILITY_HOLE_SETTINGS = {
    'BTC': {
        'compression_threshold': 75,        # Compression score threshold (0-100)
        'bb_pct_threshold': 15,            # Max BB width % during compression
        'expansion_min_bb_roc': 3.0,       # Min BB width ROC% for expansion signal
        'lookback_periods': 8,             # Lookback for hole detection
        'adx_quiet_threshold': 18,         # Max ADX during quiet period (stricter)
        'require_adx': True,               # Enable ADX filter
        'use_vol_quiet': True,             # Enable volume quiet filter
        'osc_smooth': 3,                   # Oscillator smoothing
        'bb_length': 20,                   # Bollinger Band period
        'bb_std': 2.0,                     # Bollinger Band std dev
        'atr_length': 14,                  # ATR calculation period
        'adx_length': 14,                  # ADX calculation period
        # Performance:
        # - Win Rate: 68.8%
        # - Sharpe: 0.555
        # - Total Return: +38.77%
        # - Trades: 32 (good frequency)
    },
    
    'ETH': {
        'compression_threshold': 75,        # Same as BTC
        'bb_pct_threshold': 15,            # Same as BTC
        'expansion_min_bb_roc': 3.0,       # Same as BTC
        'lookback_periods': 8,             # Same as BTC
        'adx_quiet_threshold': 22,         # More relaxed ADX (ETH more active)
        'require_adx': True,
        'use_vol_quiet': True,
        'osc_smooth': 3,
        'bb_length': 20,
        'bb_std': 2.0,
        'atr_length': 14,
        'adx_length': 14,
        # Performance:
        # - Win Rate: 43.5%
        # - Sharpe: 0.085
        # - Total Return: +7.99%
        # - Trades: 46
        # NOTE: Rubber-Band performs better for ETH (0.52 Sharpe)
    },
    
    'SOL': {
        'compression_threshold': 75,        # Same as BTC/ETH
        'bb_pct_threshold': 10,            # TIGHTER (SOL compresses more)
        'expansion_min_bb_roc': 5.0,       # HIGHER threshold (stronger breakouts)
        'lookback_periods': 3,             # FASTER detection (SOL moves quick)
        'adx_quiet_threshold': 22,         # Relaxed like ETH
        'require_adx': True,
        'use_vol_quiet': True,
        'osc_smooth': 3,
        'bb_length': 20,
        'bb_std': 2.0,
        'atr_length': 14,
        'adx_length': 14,
        # Performance:
        # - Win Rate: 80.0%
        # - Sharpe: 1.395 ⭐⭐⭐ BEST OVERALL
        # - Total Return: +5.50%
        # - Trades: 5 (very selective, very accurate)
        # *** USE THIS AS PRIMARY FILTER FOR SOL ***
    }
}


# ============================================================================
# INTEGRATION STRATEGY
# ============================================================================

INTEGRATION_RECOMMENDATIONS = {
    'BTC': {
        'primary': 'ML Model',
        'filter': 'Rubber-Band',
        'strategy': 'Only take ML LONG signals when Rubber-Band confirms oversold',
        'expected_improvement': 'Higher win rate, fewer trades',
        'settings': RUBBERBAND_SETTINGS['BTC']
    },
    
    'ETH': {
        'primary': 'ML Model',
        'filter': 'Rubber-Band',
        'strategy': 'Only take ML signals when Rubber-Band confirms mean reversion',
        'expected_improvement': 'Reduced false signals, better entries',
        'settings': RUBBERBAND_SETTINGS['ETH']
    },
    
    'SOL': {
        'primary': 'Volatility Hole',  # NOTE: This BEATS ML!
        'secondary': 'ML Model',
        'strategy': 'Use Volatility Hole as primary for SOL, ML as confirmation',
        'expected_improvement': 'Significantly better Sharpe (1.40 vs 1.23)',
        'settings': VOLATILITY_HOLE_SETTINGS['SOL']
    }
}


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_rubberband_btc():
    """Example: Using Rubber-Band for BTC"""
    from rubberband_indicator import RubberBandOscillator
    import pandas as pd
    
    # Load BTC data
    df = pd.read_csv('DATA/yf_btc_1h.csv')
    df['time'] = pd.to_datetime(df['time'])
    
    # Initialize with optimized settings
    rb = RubberBandOscillator(**RUBBERBAND_SETTINGS['BTC'])
    
    # Calculate signals
    df_result = rb.calculate(df)
    
    # Get current signal
    current = rb.get_current_signal(df_result)
    print(f"BTC Rubber-Band Signal: {current['signal']}")
    print(f"Z-Score: {current['composite_z']:.2f}")
    
    return df_result


def example_volatility_hole_sol():
    """Example: Using Volatility Hole for SOL"""
    from volatility_hole_detector import VolatilityHoleDetector
    import pandas as pd
    
    # Load SOL data
    df = pd.read_csv('DATA/yf_sol_1h.csv')
    df['time'] = pd.to_datetime(df['time'])
    
    # Initialize with optimized settings
    vh = VolatilityHoleDetector(**VOLATILITY_HOLE_SETTINGS['SOL'])
    
    # Calculate signals
    df_result = vh.calculate(df)
    
    # Get current state
    current = vh.get_current_state(df_result)
    print(f"SOL Volatility Hole State:")
    print(f"  In Hole: {current['in_hole']}")
    print(f"  Expansion Signal: {current['expansion_signal']}")
    print(f"  Compression Score: {current['compression_score']:.1f}")
    
    return df_result


def example_combined_strategy():
    """Example: Combined ML + Indicator strategy"""
    from rubberband_indicator import RubberBandOscillator
    from volatility_hole_detector import VolatilityHoleDetector
    # Assume you have ML predictions in ml_signal
    
    # For BTC: ML + Rubber-Band filter
    rb_btc = RubberBandOscillator(**RUBBERBAND_SETTINGS['BTC'])
    # df_btc_rb = rb_btc.calculate(df_btc)
    # Take ML signal only if Rubber-Band confirms:
    # combined_signal = ml_signal if (rb_signal != 0 and rb_signal == ml_signal) else 0
    
    # For SOL: Volatility Hole primary, ML confirmation
    vh_sol = VolatilityHoleDetector(**VOLATILITY_HOLE_SETTINGS['SOL'])
    # df_sol_vh = vh_sol.calculate(df_sol)
    # Take Volatility Hole signal if ML agrees:
    # combined_signal = vh_signal if (ml_signal == vh_signal) else 0
    
    print("Combined strategy initialized!")


# ============================================================================
# QUICK REFERENCE
# ============================================================================

QUICK_REFERENCE = """
INDICATOR QUICK REFERENCE
========================

BTC: Use Rubber-Band
  - z_lookback: 200, thresholds: [2.5, -2.0], smooth: 3
  - 80% win, 0.88 Sharpe
  - Very selective (5 trades/90 days)

ETH: Use Rubber-Band
  - z_lookback: 200, thresholds: [2.5, -2.0], smooth: 7
  - 80% win, 0.52 Sharpe
  - Very selective (5 trades/90 days)

SOL: Use Volatility Hole ⭐
  - comp_thresh: 75, bb_pct≤10, exp_roc≥5.0%, lookback: 3
  - 80% win, 1.40 Sharpe (BEATS ML!)
  - Very selective (5 trades/90 days)

Integration:
  BTC/ETH: ML primary + Rubber-Band filter
  SOL:     Volatility Hole primary + ML confirmation
"""

if __name__ == "__main__":
    print(QUICK_REFERENCE)
    print("\nSettings loaded! Import this file to use optimized parameters.")
    print("\nExample usage:")
    print("  from indicator_settings import RUBBERBAND_SETTINGS, VOLATILITY_HOLE_SETTINGS")
    print("  rb = RubberBandOscillator(**RUBBERBAND_SETTINGS['BTC'])")
    print("  vh = VolatilityHoleDetector(**VOLATILITY_HOLE_SETTINGS['SOL'])")
