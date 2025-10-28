"""
Volatility Hole Detector - Python Implementation
Based on TradingView Pine Script "Volatility Hole AIO"

Detects low-volatility consolidation periods (holes) followed by expansions.
Uses compression score from BB width, ATR%, Realized Vol, and Donchian width.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class VolatilityHoleDetector:
    """
    Volatility Hole Detection System
    
    Identifies low-volatility periods (compression/squeezes) that often
    precede significant price moves. Generates signals when expansion begins.
    """
    
    def __init__(
        self,
        bb_len: int = 20,
        bb_mult: float = 2.0,
        atr_len: int = 14,
        rv_len: int = 20,
        dc_len: int = 20,
        rank_len: int = 120,
        vol_len: int = 30,
        comp_thresh: int = 80,
        bb_pct_thresh: int = 10,
        atr_pct_thresh: int = 15,
        rv_pct_thresh: int = 15,
        dc_pct_thresh: int = 15,
        use_vol_quiet: bool = True,
        rvol_pct_thresh: int = 35,
        require_adx: bool = True,
        adx_len: int = 14,
        adx_quiet: int = 18,
        exp_min_bb_roc: float = 5.0,
        use_donchian_break: bool = False,
        trend_filter: bool = False,
        lookback_hole_bars: int = 5,
        osc_smooth: int = 3
    ):
        """Initialize Volatility Hole Detector with parameters"""
        self.bb_len = bb_len
        self.bb_mult = bb_mult
        self.atr_len = atr_len
        self.rv_len = rv_len
        self.dc_len = dc_len
        self.rank_len = rank_len
        self.vol_len = vol_len
        self.comp_thresh = comp_thresh
        self.bb_pct_thresh = bb_pct_thresh
        self.atr_pct_thresh = atr_pct_thresh
        self.rv_pct_thresh = rv_pct_thresh
        self.dc_pct_thresh = dc_pct_thresh
        self.use_vol_quiet = use_vol_quiet
        self.rvol_pct_thresh = rvol_pct_thresh
        self.require_adx = require_adx
        self.adx_len = adx_len
        self.adx_quiet = adx_quiet
        self.exp_min_bb_roc = exp_min_bb_roc
        self.use_donchian_break = use_donchian_break
        self.trend_filter = trend_filter
        self.lookback_hole_bars = lookback_hole_bars
        self.osc_smooth = osc_smooth
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all volatility hole indicators
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Bollinger Bands
        df['bb_basis'] = df['close'].rolling(window=self.bb_len).mean()
        df['bb_dev'] = self.bb_mult * df['close'].rolling(window=self.bb_len).std()
        df['bb_upper'] = df['bb_basis'] + df['bb_dev']
        df['bb_lower'] = df['bb_basis'] - df['bb_dev']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_basis'].where(df['bb_basis'] != 0, df['close'])
        
        # ATR percentage
        df['atr'] = self._calculate_atr(df, self.atr_len)
        df['atr_pct'] = df['atr'] / df['close']
        
        # Realized Volatility (std of log returns)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['rv'] = df['log_return'].rolling(window=self.rv_len).std()
        
        # Donchian Channel
        df['don_high'] = df['high'].rolling(window=self.dc_len).max()
        df['don_low'] = df['low'].rolling(window=self.dc_len).min()
        df['dc_width'] = (df['don_high'] - df['don_low']) / df['close']
        
        # Volume percentile
        df['vol_pct'] = df['volume'].rolling(window=self.rank_len).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        
        # ADX calculation
        if self.require_adx:
            df = self._calculate_adx(df, self.adx_len)
        
        # Percent ranks for compression score
        df['pr_bb'] = df['bb_width'].rolling(window=self.rank_len).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        df['pr_atr'] = df['atr_pct'].rolling(window=self.rank_len).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        df['pr_rv'] = df['rv'].rolling(window=self.rank_len).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        df['pr_dc'] = df['dc_width'].rolling(window=self.rank_len).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        
        # Compression Score
        df['comp_score_raw'] = 100.0 - (df['pr_bb'] + df['pr_atr'] + df['pr_rv'] + df['pr_dc']) / 4.0
        df['comp_score'] = df['comp_score_raw'].ewm(span=self.osc_smooth, adjust=False).mean()
        
        # Hole detection gates
        df['hole_core'] = (
            (df['comp_score_raw'] >= self.comp_thresh) &
            (df['pr_bb'] <= self.bb_pct_thresh) &
            (df['pr_atr'] <= self.atr_pct_thresh) &
            (df['pr_rv'] <= self.rv_pct_thresh) &
            (df['pr_dc'] <= self.dc_pct_thresh)
        )
        
        df['hole_vol'] = (not self.use_vol_quiet) | (df['vol_pct'] <= self.rvol_pct_thresh)
        df['hole_adx'] = (not self.require_adx) | (df['adx'] <= self.adx_quiet) if 'adx' in df.columns else True
        df['in_hole'] = df['hole_core'] & df['hole_vol'] & df['hole_adx']
        
        # Expansion detection
        df['bb_roc_pct'] = df['bb_width'].pct_change(1) * 100
        
        # Breakout detection
        if self.use_donchian_break:
            df['long_break'] = df['close'] > df['don_high'].shift(1)
            df['short_break'] = df['close'] < df['don_low'].shift(1)
        else:
            df['long_break'] = df['close'] > df['bb_upper']
            df['short_break'] = df['close'] < df['bb_lower']
        
        # Trend filter (optional)
        if self.trend_filter:
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
            df['long_tilt'] = df['ema50'] > df['ema200']
            df['short_tilt'] = df['ema50'] < df['ema200']
        else:
            df['long_tilt'] = True
            df['short_tilt'] = True
        
        # Recent hole check
        df['recent_hole'] = df['in_hole'].rolling(window=self.lookback_hole_bars).max() > 0
        
        # Expansion flag
        df['expanding'] = df['bb_roc_pct'] >= self.exp_min_bb_roc
        
        # Signals
        df['long_signal'] = df['recent_hole'] & df['expanding'] & df['long_break'] & df['long_tilt']
        df['short_signal'] = df['recent_hole'] & df['expanding'] & df['short_break'] & df['short_tilt']
        
        # Combined signal (-1 = SHORT, 0 = NEUTRAL, 1 = LONG)
        df['signal'] = 0
        df.loc[df['long_signal'], 'signal'] = 1
        df.loc[df['short_signal'], 'signal'] = -1
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index)"""
        # Directional movement
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # True range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Smoothed values (Wilder's smoothing = EMA with alpha = 1/period)
        alpha = 1.0 / period
        sm_tr = tr.ewm(alpha=alpha, adjust=False).mean()
        sm_plus_dm = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
        sm_minus_dm = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
        
        # Directional indicators
        plus_di = 100 * sm_plus_dm / sm_tr.replace(0, np.nan)
        minus_di = 100 * sm_minus_dm / sm_tr.replace(0, np.nan)
        
        # DX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        
        # ADX (smoothed DX)
        df['adx'] = dx.ewm(alpha=alpha, adjust=False).mean()
        
        return df
    
    def get_current_state(self, df: pd.DataFrame) -> Dict[str, any]:
        """Get the most recent state and signal"""
        if len(df) == 0:
            return None
        
        latest = df.iloc[-1]
        
        return {
            'comp_score': latest.get('comp_score', np.nan),
            'in_hole': latest.get('in_hole', False),
            'expanding': latest.get('expanding', False),
            'signal': int(latest.get('signal', 0)),
            'signal_text': {1: 'LONG', -1: 'SHORT', 0: 'NEUTRAL'}.get(int(latest.get('signal', 0)), 'NEUTRAL'),
            'bb_width': latest.get('bb_width', np.nan),
            'atr_pct': latest.get('atr_pct', np.nan),
            'rv': latest.get('rv', np.nan),
            'dc_width': latest.get('dc_width', np.nan),
            'pr_bb': latest.get('pr_bb', np.nan),
            'pr_atr': latest.get('pr_atr', np.nan),
            'pr_rv': latest.get('pr_rv', np.nan),
            'pr_dc': latest.get('pr_dc', np.nan),
            'adx': latest.get('adx', np.nan) if 'adx' in df.columns else None
        }


if __name__ == "__main__":
    """Test the indicator on BTC data"""
    import os
    
    print("="*80)
    print("🎯 VOLATILITY HOLE DETECTOR - Python Implementation")
    print("="*80)
    
    # Load BTC data
    data_path = "DATA/yf_btc_1h.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        exit(1)
    
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # Standardize column names
    if 'Close' in df.columns:
        df = df.rename(columns={
            'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
    
    print(f"\n📊 Loaded {len(df)} bars of BTC data")
    print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Initialize detector with default settings
    vhd = VolatilityHoleDetector(
        comp_thresh=80,
        require_adx=True,
        use_vol_quiet=True
    )
    
    print("\n🔧 Calculating volatility hole indicators...")
    df_with_signals = vhd.calculate(df)
    
    # Get current state
    current = vhd.get_current_state(df_with_signals)
    
    print("\n📈 Current State:")
    print(f"   Compression Score: {current['comp_score']:.1f}")
    print(f"   In Hole: {'YES ✅' if current['in_hole'] else 'NO'}")
    print(f"   Expanding: {'YES ⚡' if current['expanding'] else 'NO'}")
    print(f"   Signal: {current['signal_text']}")
    print(f"\n   Component Percentiles:")
    print(f"      BB Width:  {current['pr_bb']:.1f}%")
    print(f"      ATR%:      {current['pr_atr']:.1f}%")
    print(f"      Real Vol:  {current['pr_rv']:.1f}%")
    print(f"      Donchian:  {current['pr_dc']:.1f}%")
    if current['adx'] is not None:
        print(f"   ADX: {current['adx']:.1f}")
    
    # Count holes and signals
    holes = (df_with_signals['in_hole'] == True).sum()
    long_signals = (df_with_signals['signal'] == 1).sum()
    short_signals = (df_with_signals['signal'] == -1).sum()
    
    print(f"\n📊 Historical Summary:")
    print(f"   Volatility Holes detected: {holes}")
    print(f"   Long expansion signals:    {long_signals}")
    print(f"   Short expansion signals:   {short_signals}")
    print(f"   Total signals:             {long_signals + short_signals}")
    
    # Show last 5 signals
    signals_df = df_with_signals[df_with_signals['signal'] != 0][['time', 'close', 'comp_score', 'signal']].tail(5)
    if len(signals_df) > 0:
        print(f"\n📋 Last 5 Expansion Signals:")
        for _, row in signals_df.iterrows():
            signal_text = "🟢 LONG" if row['signal'] == 1 else "🔴 SHORT"
            print(f"   {row['time']}: {signal_text} @ ${row['close']:.2f} (comp={row['comp_score']:.1f})")
    
    print("\n✅ Test complete!")
