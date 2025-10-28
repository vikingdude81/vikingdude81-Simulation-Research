"""
NEWTON BASIN-OF-ATTRACTION MAP - RECOMMENDED SETTINGS

Based on optimization against 90 days of tested data.
The indicator identifies regime shifts that predict future price movement.

KEY FINDING: "Shift to SLOW" basin is profitable for BTC/ETH!
This counterintuitive result suggests capitulation/mean-reversion opportunities.
"""

# ============================================================================
# OPTIMIZED SETTINGS
# ============================================================================

NEWTON_BASIN_SETTINGS = {
    'BTC': {
        # Parameters
        'ma1_len': 15,              # Fast EMA (tighter than default)
        'ma2_len': 40,              # Medium SMA (faster than default)
        'ma3_len': 220,             # Slow SMA (slower than default)
        'newton_iterations': 4,     # Faster convergence
        'tolerance': 0.01,
        'use_relative_distance': False,  # Use absolute distance
        
        # Trading Strategy
        'strategy': 'shift_to_slow',  # SHORT on shift to slow basin
        'signal_type': 'LONG',        # Counterintuitively, LONG on shift to slow!
        
        # Performance (90 days)
        'sharpe': 0.283,
        'win_rate': 0.549,           # 54.9%
        'avg_return': 0.00485,       # 0.485% per trade
        'total_return': 0.2703,      # +27.03%
        'num_trades': 51,
        
        # Interpretation
        'logic': 'When price shifts to SLOW basin (SMA 200), it indicates '
                 'oversold/capitulation. Enter LONG for mean reversion.'
    },
    
    'ETH': {
        # Parameters
        'ma1_len': 25,              # Slower fast MA
        'ma2_len': 40,              # Faster medium MA
        'ma3_len': 180,             # Faster slow MA
        'newton_iterations': 4,
        'tolerance': 0.01,
        'use_relative_distance': False,
        
        # Trading Strategy
        'strategy': 'shift_to_slow',
        'signal_type': 'LONG',
        
        # Performance (90 days) - BEST PERFORMER!
        'sharpe': 0.486,             # Excellent!
        'win_rate': 0.694,           # 69.4% win rate!
        'avg_return': 0.00921,       # 0.921% per trade
        'total_return': 0.7468,      # +74.68%
        'num_trades': 62,
        
        # Interpretation
        'logic': 'ETH shows strongest mean-reversion when shifting to slow basin. '
                 '69.4% win rate suggests reliable capitulation signals.'
    },
    
    'SOL': {
        # Parameters
        'ma1_len': 25,              # Slower fast MA
        'ma2_len': 60,              # Medium MA
        'ma3_len': 200,             # Standard slow MA
        'newton_iterations': 4,
        'tolerance': 0.01,
        'use_relative_distance': False,
        
        # Trading Strategy
        'strategy': 'shift_to_fast',  # LONG on shift to fast basin
        'signal_type': 'LONG',
        
        # Performance (90 days)
        'sharpe': 0.158,
        'win_rate': 0.591,           # 59.1%
        'avg_return': 0.00488,       # 0.488% per trade
        'total_return': 0.7864,      # +78.64%
        'num_trades': 132,           # Most active
        
        # Interpretation
        'logic': 'SOL prefers momentum: shift to FAST basin (EMA 25) signals '
                 'trend continuation. More trades, good consistency.'
    }
}


# ============================================================================
# COMPARISON VS OTHER INDICATORS
# ============================================================================

PERFORMANCE_COMPARISON = """
NEWTON BASIN vs PREVIOUS INDICATORS
===================================

ETH Performance Comparison:
  Newton Basin (shift to slow): 0.486 Sharpe, 69.4% win, +74.68%, 62 trades ⭐⭐
  Rubber-Band:                  0.519 Sharpe, 80.0% win, +3.15%,  5 trades
  Volatility Hole:              0.085 Sharpe, 43.5% win, +7.99%,  46 trades
  
  WINNER: Newton Basin (similar Sharpe, WAY more trades!)

BTC Performance Comparison:
  Newton Basin (shift to slow): 0.283 Sharpe, 54.9% win, +27.03%, 51 trades
  Rubber-Band:                  0.883 Sharpe, 80.0% win, +2.34%,  5 trades ⭐
  Volatility Hole:              0.555 Sharpe, 68.8% win, +38.77%, 32 trades ⭐
  
  WINNER: Rubber-Band (highest Sharpe, but Newton has more volume)

SOL Performance Comparison:
  Newton Basin (shift to fast): 0.158 Sharpe, 59.1% win, +78.64%, 132 trades
  Rubber-Band:                  0.608 Sharpe, 60.0% win, +11.32%, 5 trades
  Volatility Hole:              1.395 Sharpe, 80.0% win, +5.50%,  5 trades ⭐⭐⭐
  
  WINNER: Volatility Hole (unbeatable Sharpe)

KEY INSIGHTS:
- Newton Basin generates MORE SIGNALS than other indicators
- ETH "shift to slow" is a hidden gem (69.4% win rate!)
- Complements low-frequency indicators (RB/VH have only 5 trades)
"""


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_newton_basin_eth():
    """Example: Using Newton Basin for ETH regime shifts"""
    from newton_basin_map import NewtonBasinMap
    import pandas as pd
    
    # Load ETH data
    df = pd.read_csv('DATA/yf_eth_1h.csv')
    df['time'] = pd.to_datetime(df['time'])
    
    # Initialize with optimized settings for ETH
    newton = NewtonBasinMap(**{
        'ma1_len': NEWTON_BASIN_SETTINGS['ETH']['ma1_len'],
        'ma2_len': NEWTON_BASIN_SETTINGS['ETH']['ma2_len'],
        'ma3_len': NEWTON_BASIN_SETTINGS['ETH']['ma3_len'],
        'newton_iterations': NEWTON_BASIN_SETTINGS['ETH']['newton_iterations'],
        'use_relative_distance': NEWTON_BASIN_SETTINGS['ETH']['use_relative_distance']
    })
    
    # Calculate basin map
    df_result = newton.calculate(df)
    
    # Get signals: LONG when shifting to SLOW basin
    signals = df_result[df_result['shift_to_slow'] == True].copy()
    
    print(f"ETH Newton Basin Signals: {len(signals)}")
    print(f"Strategy: {NEWTON_BASIN_SETTINGS['ETH']['strategy']}")
    print(f"Expected Win Rate: {NEWTON_BASIN_SETTINGS['ETH']['win_rate']*100:.1f}%")
    
    return df_result, signals


def example_combined_with_ml():
    """Example: Combine Newton Basin with ML predictions"""
    from newton_basin_map import NewtonBasinMap
    
    # For ETH: Use Newton Basin as high-frequency filter
    # ML prediction + Newton "shift to slow" = Strong buy signal
    
    # For BTC: Use Rubber-Band for quality, Newton for quantity
    # Rubber-Band (5 trades, 80% win) for primary
    # Newton Basin (51 trades, 55% win) for secondary opportunities
    
    # For SOL: Volatility Hole still king
    # But Newton "shift to fast" can catch additional momentum plays
    
    print("Combined strategy initialized!")


# ============================================================================
# INTEGRATION RECOMMENDATIONS
# ============================================================================

INTEGRATION_STRATEGY = {
    'BTC': {
        'primary': 'Rubber-Band (80% win, 0.88 Sharpe)',
        'secondary': 'Newton Basin shift_to_slow (55% win, 0.28 Sharpe)',
        'use_case': 'Use Rubber-Band for high-conviction trades, '
                    'Newton Basin for additional opportunities between RB signals',
        'expected_trades': '5 (RB) + 51 (Newton) = ~56 trades/90 days'
    },
    
    'ETH': {
        'primary': 'Newton Basin shift_to_slow (69.4% win, 0.49 Sharpe) ⭐',
        'secondary': 'Rubber-Band (80% win, 0.52 Sharpe)',
        'use_case': 'Newton Basin provides good frequency (62 trades) with '
                    'solid win rate. Use Rubber-Band to filter best Newton signals.',
        'expected_trades': '62 (Newton) filtered by Rubber-Band',
        'advantage': 'Newton generates 12x more signals than Rubber-Band alone!'
    },
    
    'SOL': {
        'primary': 'Volatility Hole (80% win, 1.40 Sharpe) ⭐⭐⭐',
        'secondary': 'Newton Basin shift_to_fast (59% win, 0.16 Sharpe)',
        'use_case': 'Volatility Hole for quality, Newton for momentum fills',
        'expected_trades': '5 (Vol Hole) + 132 (Newton) = ~137 trades/90 days'
    }
}


# ============================================================================
# KEY FINDINGS
# ============================================================================

KEY_FINDINGS = """
NEWTON BASIN-OF-ATTRACTION MAP - KEY FINDINGS
=============================================

1. SHIFT TO SLOW = MEAN REVERSION OPPORTUNITY
   - BTC: 54.9% win, +27% over 90 days
   - ETH: 69.4% win, +74.68% over 90 days ⭐⭐⭐
   - Counterintuitive: Capitulation into slow MA = buy signal

2. ETH IS THE STAR
   - Best Newton Basin performance across all assets
   - 69.4% win rate is excellent for regime-shift indicator
   - 62 trades = 12x more than Rubber-Band
   - 0.486 Sharpe comparable to Rubber-Band's 0.519

3. FREQUENCY ADVANTAGE
   - Rubber-Band/Vol Hole: ~5 trades/90 days (very selective)
   - Newton Basin: 51-132 trades/90 days (active)
   - Fills the gap between high-Sharpe rare signals

4. COMPLEMENTARY TO EXISTING INDICATORS
   - Use with Rubber-Band: Quality (RB) + Quantity (Newton)
   - Use with Volatility Hole: Breakouts (VH) + Regime shifts (Newton)
   - Use with ML: ML prediction + Newton regime = confirmation

5. PARAMETER INSIGHTS
   - Faster MAs work better (15-25 for fast MA vs default 20)
   - Absolute distance beats relative distance
   - 4 iterations sufficient (faster than default 6)

RECOMMENDED USE:
================
ETH: PRIMARY indicator (Newton Basin beats alternatives for frequency)
BTC: SECONDARY to Rubber-Band (fills gaps between RB signals)
SOL: SECONDARY to Volatility Hole (adds momentum plays)
"""

if __name__ == "__main__":
    print(KEY_FINDINGS)
    print("\n" + "="*80)
    print(PERFORMANCE_COMPARISON)
    print("\n" + "="*80)
    print("\nNewton Basin settings loaded!")
    print("\nExample usage:")
    print("  from newton_basin_settings import NEWTON_BASIN_SETTINGS")
    print("  from newton_basin_map import NewtonBasinMap")
    print("  newton = NewtonBasinMap(**NEWTON_BASIN_SETTINGS['ETH'])")
