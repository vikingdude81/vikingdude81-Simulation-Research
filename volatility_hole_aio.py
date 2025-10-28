"""
Volatility Hole All-In-One (AIO) - Python Implementation

This is an ENHANCED version combining:
1. Compression Score (4 volatility metrics)
2. Volume quietness filter
3. ADX trend strength filter
4. Expansion detection (BB width ROC)
5. Directional breakout (Bollinger or Donchian)
6. Optional trend filter (EMA 50 vs 200)

Designed to work both as:
- Overlay on price chart (shows bands, labels, background shading)
- Oscillator in separate pane (compression score with diagnostics)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class VolatilityHoleAIO:
    """
    Enhanced Volatility Hole detector with multiple filters
    
    Core concept: Detect extreme compression across multiple volatility measures,
    then trigger on expansion + directional breakout.
    """
    
    def __init__(
        self,
        # Volatility measures
        bb_length: int = 20,
        bb_mult: float = 2.0,
        atr_length: int = 14,
        rv_length: int = 20,
        dc_length: int = 20,
        rank_length: int = 120,
        vol_length: int = 30,
        
        # Compression thresholds
        compression_threshold: int = 80,
        bb_pct_threshold: int = 10,
        atr_pct_threshold: int = 15,
        rv_pct_threshold: int = 15,
        dc_pct_threshold: int = 15,
        
        # Volume filter
        use_volume_quiet: bool = True,
        rvol_pct_threshold: int = 35,
        
        # ADX filter
        require_adx: bool = True,
        adx_length: int = 14,
        adx_quiet: int = 18,
        
        # Expansion/trigger
        exp_min_bb_roc: float = 5.0,
        use_donchian_break: bool = False,
        trend_filter: bool = False,
        lookback_hole_bars: int = 5,
        
        # Oscillator smoothing
        osc_smooth: int = 3
    ):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.atr_length = atr_length
        self.rv_length = rv_length
        self.dc_length = dc_length
        self.rank_length = rank_length
        self.vol_length = vol_length
        
        self.compression_threshold = compression_threshold
        self.bb_pct_threshold = bb_pct_threshold
        self.atr_pct_threshold = atr_pct_threshold
        self.rv_pct_threshold = rv_pct_threshold
        self.dc_pct_threshold = dc_pct_threshold
        
        self.use_volume_quiet = use_volume_quiet
        self.rvol_pct_threshold = rvol_pct_threshold
        
        self.require_adx = require_adx
        self.adx_length = adx_length
        self.adx_quiet = adx_quiet
        
        self.exp_min_bb_roc = exp_min_bb_roc
        self.use_donchian_break = use_donchian_break
        self.trend_filter = trend_filter
        self.lookback_hole_bars = lookback_hole_bars
        
        self.osc_smooth = osc_smooth
    
    def _calculate_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df['bb_basis'] = df['close'].rolling(window=self.bb_length).mean()
        df['bb_dev'] = self.bb_mult * df['close'].rolling(window=self.bb_length).std()
        df['bb_upper'] = df['bb_basis'] + df['bb_dev']
        df['bb_lower'] = df['bb_basis'] - df['bb_dev']
        # Avoid division by zero
        denominator = df['bb_basis'].where(df['bb_basis'] != 0, df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / denominator
        return df
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.atr_length).mean()
        df['atr_pct'] = df['atr'] / df['close']
        return df
    
    def _calculate_realized_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate realized volatility (std of log returns)"""
        log_returns = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol'] = log_returns.rolling(window=self.rv_length).std()
        return df
    
    def _calculate_donchian(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Donchian Channel"""
        df['dc_high'] = df['high'].rolling(window=self.dc_length).max()
        df['dc_low'] = df['low'].rolling(window=self.dc_length).min()
        df['dc_width'] = (df['dc_high'] - df['dc_low']) / df['close']
        return df
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Directional movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothed values (RMA = Wilder's smoothing)
        alpha = 1.0 / self.adx_length
        sm_tr = tr.ewm(alpha=alpha, adjust=False).mean()
        sm_plus_dm = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
        sm_minus_dm = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
        
        # Directional Indicators (avoid division by zero)
        plus_di = 100 * sm_plus_dm / sm_tr.where(sm_tr != 0, 1)
        minus_di = 100 * sm_minus_dm / sm_tr.where(sm_tr != 0, 1)
        
        # DX and ADX
        di_sum = (plus_di + minus_di).where((plus_di + minus_di) != 0, 1)
        dx = 100 * abs(plus_di - minus_di) / di_sum
        df['adx'] = dx.ewm(alpha=alpha, adjust=False).mean()
        
        return df
    
    def _percentrank(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate percentile rank over rolling window"""
        def rank_pct(x):
            if len(x) == 0:
                return np.nan
            return (x < x.iloc[-1]).sum() / len(x) * 100
        
        return series.rolling(window=window).apply(rank_pct, raw=False)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all Volatility Hole AIO indicators"""
        df = df.copy()
        
        # Core volatility measures
        df = self._calculate_bollinger(df)
        df = self._calculate_atr(df)
        df = self._calculate_realized_vol(df)
        df = self._calculate_donchian(df)
        
        # Volume percentile
        df['vol_pct'] = self._percentrank(df['volume'], self.rank_length)
        
        # ADX
        df = self._calculate_adx(df)
        
        # Percent ranks for compression score
        df['pr_bb'] = self._percentrank(df['bb_width'], self.rank_length)
        df['pr_atr'] = self._percentrank(df['atr_pct'], self.rank_length)
        df['pr_rv'] = self._percentrank(df['realized_vol'], self.rank_length)
        df['pr_dc'] = self._percentrank(df['dc_width'], self.rank_length)
        
        # Compression score (raw)
        df['comp_score_raw'] = 100.0 - (df['pr_bb'] + df['pr_atr'] + df['pr_rv'] + df['pr_dc']) / 4.0
        
        # Smoothed compression score (for oscillator)
        df['comp_score'] = df['comp_score_raw'].ewm(span=self.osc_smooth, adjust=False).mean()
        
        # Hole detection gates
        df['hole_core'] = (
            (df['comp_score_raw'] >= self.compression_threshold) &
            (df['pr_bb'] <= self.bb_pct_threshold) &
            (df['pr_atr'] <= self.atr_pct_threshold) &
            (df['pr_rv'] <= self.rv_pct_threshold) &
            (df['pr_dc'] <= self.dc_pct_threshold)
        )
        
        df['hole_vol'] = ~self.use_volume_quiet | (df['vol_pct'] <= self.rvol_pct_threshold)
        df['hole_adx'] = ~self.require_adx | (df['adx'] <= self.adx_quiet)
        
        df['in_hole'] = df['hole_core'] & df['hole_vol'] & df['hole_adx']
        
        # Expansion detection
        df['bb_roc_pct'] = df['bb_width'].pct_change(1) * 100
        df['expanding'] = df['bb_roc_pct'] >= self.exp_min_bb_roc
        
        # Breakout detection
        df['long_break_bb'] = df['close'] > df['bb_upper']
        df['short_break_bb'] = df['close'] < df['bb_lower']
        df['long_break_dc'] = df['close'] > df['dc_high'].shift(1)
        df['short_break_dc'] = df['close'] < df['dc_low'].shift(1)
        
        if self.use_donchian_break:
            df['long_break'] = df['long_break_dc']
            df['short_break'] = df['short_break_dc']
        else:
            df['long_break'] = df['long_break_bb']
            df['short_break'] = df['short_break_bb']
        
        # Trend filter (EMA 50 vs 200)
        if self.trend_filter:
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
            df['long_tilt'] = df['ema_50'] > df['ema_200']
            df['short_tilt'] = df['ema_50'] < df['ema_200']
        else:
            df['long_tilt'] = True
            df['short_tilt'] = True
        
        # Recent hole detection
        df['recent_hole'] = df['in_hole'].astype(int).rolling(window=self.lookback_hole_bars).max() > 0
        
        # Final signals
        df['signal'] = 0
        df.loc[
            df['recent_hole'] & df['expanding'] & df['long_break'] & df['long_tilt'],
            'signal'
        ] = 1  # LONG
        
        df.loc[
            df['recent_hole'] & df['expanding'] & df['short_break'] & df['short_tilt'],
            'signal'
        ] = -1  # SHORT
        
        return df
    
    def get_current_state(self, df: pd.DataFrame) -> Dict:
        """Get current state with diagnostics"""
        if len(df) == 0:
            return {}
        
        last = df.iloc[-1]
        
        signal_text = "LONG" if last['signal'] == 1 else "SHORT" if last['signal'] == -1 else "NEUTRAL"
        
        result = {
            'signal': int(last['signal']),
            'signal_text': signal_text,
            'in_hole': bool(last['in_hole']),
            'comp_score': last['comp_score'],
            'comp_score_raw': last['comp_score_raw'],
            
            # Diagnostics
            'bb_width': last['bb_width'],
            'pr_bb': last['pr_bb'],
            'pr_atr': last['pr_atr'],
            'pr_rv': last['pr_rv'],
            'pr_dc': last['pr_dc'],
            
            'vol_pct': last['vol_pct'],
            'adx': last['adx'],
            
            'expanding': bool(last['expanding']),
            'bb_roc_pct': last['bb_roc_pct'],
            
            'long_break': bool(last['long_break']),
            'short_break': bool(last['short_break']),
            
            # Bollinger Bands for overlay
            'bb_upper': last['bb_upper'],
            'bb_basis': last['bb_basis'],
            'bb_lower': last['bb_lower'],
            'price': last['close']
        }
        
        # Pass/fail checks
        result['checks'] = {
            'comp_score': last['comp_score_raw'] >= self.compression_threshold,
            'bb_pct': last['pr_bb'] <= self.bb_pct_threshold,
            'atr_pct': last['pr_atr'] <= self.atr_pct_threshold,
            'rv_pct': last['pr_rv'] <= self.rv_pct_threshold,
            'dc_pct': last['pr_dc'] <= self.dc_pct_threshold,
            'vol_quiet': not self.use_volume_quiet or last['vol_pct'] <= self.rvol_pct_threshold,
            'adx_quiet': not self.require_adx or last['adx'] <= self.adx_quiet
        }
        
        return result


if __name__ == "__main__":
    print("="*80)
    print("VOLATILITY HOLE AIO - Python Test")
    print("="*80)
    
    # Load BTC data
    print("\nLoading BTC data...")
    df = pd.read_csv('DATA/yf_btc_1h.csv')
    df['time'] = pd.to_datetime(df['time'])
    
    if 'Close' in df.columns:
        df = df.rename(columns={
            'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
    
    df = df.tail(2160).reset_index(drop=True)  # Last 90 days
    print(f"   Loaded {len(df)} bars")
    
    # Initialize with optimized SOL settings (best performer)
    print("\nCalculating Volatility Hole AIO indicators...")
    vh = VolatilityHoleAIO(
        compression_threshold=75,
        bb_pct_threshold=10,
        exp_min_bb_roc=5.0,
        lookback_hole_bars=3,
        require_adx=True,
        adx_quiet=22
    )
    
    df_result = vh.calculate(df)
    
    # Current state with diagnostics
    print("\n" + "="*80)
    print("CURRENT STATE (with Diagnostics)")
    print("="*80)
    
    state = vh.get_current_state(df_result)
    
    print(f"\nSignal: {state['signal_text']}")
    print(f"In Hole: {'YES' if state['in_hole'] else 'NO'}")
    print(f"\nCompression Score: {state['comp_score']:.1f} (raw: {state['comp_score_raw']:.1f})")
    
    print(f"\nBollinger Bands:")
    print(f"  Upper: ${state['bb_upper']:,.2f}")
    print(f"  Basis: ${state['bb_basis']:,.2f}")
    print(f"  Lower: ${state['bb_lower']:,.2f}")
    print(f"  Price: ${state['price']:,.2f}")
    
    print(f"\nDiagnostics:")
    checks = state['checks']
    print(f"  ✓ Comp Score ≥ 75:  {'PASS' if checks['comp_score'] else 'FAIL'} ({state['comp_score_raw']:.1f})")
    print(f"  ✓ BB pct ≤ 10%:     {'PASS' if checks['bb_pct'] else 'FAIL'} ({state['pr_bb']:.1f}%)")
    print(f"  ✓ ATR pct ≤ 15%:    {'PASS' if checks['atr_pct'] else 'FAIL'} ({state['pr_atr']:.1f}%)")
    print(f"  ✓ RV pct ≤ 15%:     {'PASS' if checks['rv_pct'] else 'FAIL'} ({state['pr_rv']:.1f}%)")
    print(f"  ✓ DC pct ≤ 15%:     {'PASS' if checks['dc_pct'] else 'FAIL'} ({state['pr_dc']:.1f}%)")
    print(f"  ✓ Vol quiet ≤ 35%:  {'PASS' if checks['vol_quiet'] else 'FAIL'} ({state['vol_pct']:.1f}%)")
    print(f"  ✓ ADX ≤ 22:         {'PASS' if checks['adx_quiet'] else 'FAIL'} ({state['adx']:.1f})")
    
    print(f"\nExpansion:")
    print(f"  Expanding: {'YES' if state['expanding'] else 'NO'} (BB ROC: {state['bb_roc_pct']:.2f}%)")
    print(f"  Long Break: {'YES' if state['long_break'] else 'NO'}")
    print(f"  Short Break: {'YES' if state['short_break'] else 'NO'}")
    
    # Historical signals
    print("\n" + "="*80)
    print("SIGNAL SUMMARY (90 days)")
    print("="*80)
    
    signals = df_result[df_result['signal'] != 0].copy()
    long_signals = signals[signals['signal'] == 1]
    short_signals = signals[signals['signal'] == -1]
    
    print(f"\nTotal signals: {len(signals)}")
    print(f"  Long (expansion up):   {len(long_signals)}")
    print(f"  Short (expansion down): {len(short_signals)}")
    
    # Hole periods
    holes = df_result[df_result['in_hole']].copy()
    print(f"\nHole periods: {len(holes)} bars")
    if len(holes) > 0:
        avg_comp = holes['comp_score_raw'].mean()
        print(f"  Avg compression score: {avg_comp:.1f}")
    
    # Last 10 signals
    if len(signals) > 0:
        print("\n" + "="*80)
        print("LAST 10 SIGNALS")
        print("="*80)
        
        last_signals = signals.tail(10)
        for idx, row in last_signals.iterrows():
            signal_type = "LONG ↑" if row['signal'] == 1 else "SHORT ↓"
            print(f"{row['time'].strftime('%Y-%m-%d %H:%M')} {signal_type:8s} @ ${row['close']:>9,.2f}  "
                  f"(comp: {row['comp_score_raw']:.0f}, BB ROC: {row['bb_roc_pct']:+.1f}%)")
    
    print("\n" + "="*80)
    print("Test complete!")
