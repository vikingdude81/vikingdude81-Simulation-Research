"""
Rubber-Band Reversion Oscillator - Python Implementation
Based on TradingView Pine Script v6.3

This indicator measures mean-reversion opportunities by calculating z-scores
of price distance from multiple moving averages (20/50/100/200).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class RubberBandOscillator:
    """
    Rubber-Band Reversion Oscillator
    
    Measures how far price has stretched from multiple MAs and identifies
    extreme deviations (rubber-band stretched) that tend to revert.
    """
    
    def __init__(
        self,
        ma_type: str = "SMA",
        use_20: bool = True,
        use_50: bool = True,
        use_100: bool = True,
        use_200: bool = True,
        z_lookback: int = 200,
        w_20: float = 1.0,
        w_50: float = 1.5,
        w_100: float = 2.0,
        w_200: float = 3.0,
        up_threshold: float = 2.0,
        down_threshold: float = -2.0,
        trend_filter: str = "200MA Slope",
        slope_len: int = 50,
        smooth_len: int = 5,
        use_adaptive_weights: bool = False,
        corr_len: int = 100,
        fwd_horizon: int = 10,
        use_dynamic_thresholds: bool = True,
        atr_len_thr: int = 20,
        thr_scale: float = 0.50,
        use_keltner_filter: bool = True,
        kc_len: int = 20,
        kc_mult: float = 2.0,
        min_bars_between: int = 8
    ):
        """
        Initialize Rubber-Band Oscillator with parameters
        
        Args:
            ma_type: "SMA" or "EMA"
            use_20/50/100/200: Which MAs to include
            z_lookback: Lookback period for z-score calculation
            w_20/50/100/200: Base weights for each MA
            up_threshold: Overbought z-score (SHORT signal)
            down_threshold: Oversold z-score (LONG signal)
            trend_filter: "None" or "200MA Slope"
            slope_len: Bars to lookback for MA slope
            smooth_len: EMA smoothing for final oscillator
            use_adaptive_weights: Adjust weights by correlation to future returns
            corr_len: Correlation window for adaptive weights
            fwd_horizon: Forward return horizon for correlation
            use_dynamic_thresholds: Adjust thresholds based on ATR
            atr_len_thr: ATR length for threshold scaling
            thr_scale: Threshold scale vs ATR%
            use_keltner_filter: Skip signals outside Keltner Channel
            kc_len: Keltner length
            kc_mult: Keltner multiplier
            min_bars_between: Minimum bars between signals
        """
        self.ma_type = ma_type
        self.use_20 = use_20
        self.use_50 = use_50
        self.use_100 = use_100
        self.use_200 = use_200
        self.z_lookback = z_lookback
        self.w_20 = w_20
        self.w_50 = w_50
        self.w_100 = w_100
        self.w_200 = w_200
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.trend_filter = trend_filter
        self.slope_len = slope_len
        self.smooth_len = smooth_len
        self.use_adaptive_weights = use_adaptive_weights
        self.corr_len = corr_len
        self.fwd_horizon = fwd_horizon
        self.use_dynamic_thresholds = use_dynamic_thresholds
        self.atr_len_thr = atr_len_thr
        self.thr_scale = thr_scale
        self.use_keltner_filter = use_keltner_filter
        self.kc_len = kc_len
        self.kc_mult = kc_mult
        self.min_bars_between = min_bars_between
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicator values
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Calculate MAs
        if self.use_20:
            df['ma20'] = self._ma(df['close'], 20)
        if self.use_50:
            df['ma50'] = self._ma(df['close'], 50)
        if self.use_100:
            df['ma100'] = self._ma(df['close'], 100)
        if self.use_200:
            df['ma200'] = self._ma(df['close'], 200)
        
        # Calculate % distance from each MA
        if self.use_20:
            df['d20'] = self._pct_dist(df['close'], df['ma20'])
        if self.use_50:
            df['d50'] = self._pct_dist(df['close'], df['ma50'])
        if self.use_100:
            df['d100'] = self._pct_dist(df['close'], df['ma100'])
        if self.use_200:
            df['d200'] = self._pct_dist(df['close'], df['ma200'])
        
        # Calculate z-scores of distances
        if self.use_20:
            df['z20'] = self._zscore(df['d20'], self.z_lookback)
        if self.use_50:
            df['z50'] = self._zscore(df['d50'], self.z_lookback)
        if self.use_100:
            df['z100'] = self._zscore(df['d100'], self.z_lookback)
        if self.use_200:
            df['z200'] = self._zscore(df['d200'], self.z_lookback)
        
        # Calculate adaptive weights if enabled
        if self.use_adaptive_weights:
            df = self._calculate_adaptive_weights(df)
            # Use adaptive weights
            df['eff_w20'] = df['w20_adaptive'].fillna(self.w_20)
            df['eff_w50'] = df['w50_adaptive'].fillna(self.w_50)
            df['eff_w100'] = df['w100_adaptive'].fillna(self.w_100)
            df['eff_w200'] = df['w200_adaptive'].fillna(self.w_200)
        else:
            # Use base weights
            df['eff_w20'] = self.w_20 if self.use_20 else 0.0
            df['eff_w50'] = self.w_50 if self.use_50 else 0.0
            df['eff_w100'] = self.w_100 if self.use_100 else 0.0
            df['eff_w200'] = self.w_200 if self.use_200 else 0.0
        
        # Calculate composite z-score
        df['composite_z_raw'] = self._composite_z(df)
        df['composite_z'] = df['composite_z_raw'].ewm(span=self.smooth_len, adjust=False).mean()
        
        # Calculate dynamic thresholds
        if self.use_dynamic_thresholds:
            df = self._calculate_dynamic_thresholds(df)
            df['up_thr_eff'] = df['up_thr_dynamic']
            df['down_thr_eff'] = df['down_thr_dynamic']
        else:
            df['up_thr_eff'] = self.up_threshold
            df['down_thr_eff'] = self.down_threshold
        
        # Trend filter
        if self.trend_filter == "200MA Slope" and self.use_200:
            df['ma200_slope'] = df['ma200'] - df['ma200'].shift(self.slope_len)
            df['allow_long'] = df['ma200_slope'] >= 0
            df['allow_short'] = df['ma200_slope'] <= 0
        else:
            df['allow_long'] = True
            df['allow_short'] = True
        
        # Keltner Channel filter
        if self.use_keltner_filter:
            df = self._calculate_keltner(df)
            df['outside_kc'] = (df['close'] > df['kc_upper']) | (df['close'] < df['kc_lower'])
        else:
            df['outside_kc'] = False
        
        # Generate signals
        df = self._generate_signals(df)
        
        return df
    
    def _ma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate moving average (SMA or EMA)"""
        if self.ma_type == "EMA":
            return series.ewm(span=period, adjust=False).mean()
        else:
            return series.rolling(window=period).mean()
    
    def _pct_dist(self, price: pd.Series, ma: pd.Series) -> pd.Series:
        """Calculate percentage distance from MA"""
        return 100.0 * (price - ma) / ma
    
    def _zscore(self, series: pd.Series, lookback: int) -> pd.Series:
        """Calculate rolling z-score"""
        mean = series.rolling(window=lookback).mean()
        std = series.rolling(window=lookback).std()
        return (series - mean) / std
    
    def _composite_z(self, df: pd.DataFrame) -> pd.Series:
        """Calculate weighted composite z-score"""
        numerator = 0.0
        denominator = 0.0
        
        if self.use_20 and 'z20' in df.columns:
            numerator += df['z20'].fillna(0) * df['eff_w20']
            denominator += df['eff_w20']
        if self.use_50 and 'z50' in df.columns:
            numerator += df['z50'].fillna(0) * df['eff_w50']
            denominator += df['eff_w50']
        if self.use_100 and 'z100' in df.columns:
            numerator += df['z100'].fillna(0) * df['eff_w100']
            denominator += df['eff_w100']
        if self.use_200 and 'z200' in df.columns:
            numerator += df['z200'].fillna(0) * df['eff_w200']
            denominator += df['eff_w200']
        
        return numerator / denominator.replace(0, np.nan)
    
    def _calculate_adaptive_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate adaptive weights based on correlation to forward returns"""
        # Calculate forward returns (log returns)
        df['fwd_return'] = np.log(df['close']) - np.log(df['close'].shift(-self.fwd_horizon))
        
        # Calculate rolling correlation for each z-score
        if self.use_20 and 'z20' in df.columns:
            df['w20_adaptive'] = df['z20'].rolling(self.corr_len).corr(df['fwd_return']).clip(lower=0)
        if self.use_50 and 'z50' in df.columns:
            df['w50_adaptive'] = df['z50'].rolling(self.corr_len).corr(df['fwd_return']).clip(lower=0)
        if self.use_100 and 'z100' in df.columns:
            df['w100_adaptive'] = df['w100'].rolling(self.corr_len).corr(df['fwd_return']).clip(lower=0)
        if self.use_200 and 'z200' in df.columns:
            df['w200_adaptive'] = df['z200'].rolling(self.corr_len).corr(df['fwd_return']).clip(lower=0)
        
        return df
    
    def _calculate_dynamic_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR-based dynamic thresholds"""
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(self.atr_len_thr).mean()
        
        # ATR as % of price
        atr_pct = atr / df['close']
        atr_pct_ma = atr_pct.rolling(self.atr_len_thr).mean()
        
        # ATR adjustment (how much current ATR% deviates from average)
        atr_adj = (atr_pct / atr_pct_ma - 1.0).fillna(0)
        
        # Adjust thresholds
        df['up_thr_dynamic'] = self.up_threshold * (1 + self.thr_scale * atr_adj)
        df['down_thr_dynamic'] = self.down_threshold * (1 + self.thr_scale * atr_adj)
        
        return df
    
    def _calculate_keltner(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channel"""
        df['kc_basis'] = df['close'].ewm(span=self.kc_len, adjust=False).mean()
        
        # Calculate ATR for Keltner
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        kc_range = true_range.rolling(self.kc_len).mean()
        
        df['kc_upper'] = df['kc_basis'] + self.kc_mult * kc_range
        df['kc_lower'] = df['kc_basis'] - self.kc_mult * kc_range
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with cooldown"""
        # Detect crossovers
        df['cross_under'] = (df['composite_z'].shift(1) >= df['up_thr_eff'].shift(1)) & (df['composite_z'] < df['up_thr_eff'])
        df['cross_over'] = (df['composite_z'].shift(1) <= df['down_thr_eff'].shift(1)) & (df['composite_z'] > df['down_thr_eff'])
        
        # Candidate signals (before cooldown)
        df['cand_short'] = df['cross_under'] & df['allow_short'] & ~df['outside_kc']
        df['cand_long'] = df['cross_over'] & df['allow_long'] & ~df['outside_kc']
        
        # Apply cooldown
        df['signal'] = 0
        last_signal_bar = -999999
        
        for i in range(len(df)):
            bars_since = i - last_signal_bar
            
            if df['cand_long'].iloc[i] and bars_since >= self.min_bars_between:
                df.loc[df.index[i], 'signal'] = 1  # LONG
                last_signal_bar = i
            elif df['cand_short'].iloc[i] and bars_since >= self.min_bars_between:
                df.loc[df.index[i], 'signal'] = -1  # SHORT
                last_signal_bar = i
        
        return df
    
    def get_current_signal(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get the most recent signal and oscillator state
        
        Returns:
            Dictionary with current state
        """
        if len(df) == 0:
            return None
        
        latest = df.iloc[-1]
        
        return {
            'composite_z': latest.get('composite_z', np.nan),
            'up_threshold': latest.get('up_thr_eff', self.up_threshold),
            'down_threshold': latest.get('down_thr_eff', self.down_threshold),
            'signal': int(latest.get('signal', 0)),
            'signal_text': {1: 'LONG', -1: 'SHORT', 0: 'NEUTRAL'}.get(int(latest.get('signal', 0)), 'NEUTRAL'),
            'z20': latest.get('z20', np.nan),
            'z50': latest.get('z50', np.nan),
            'z100': latest.get('z100', np.nan),
            'z200': latest.get('z200', np.nan),
            'eff_w20': latest.get('eff_w20', 0),
            'eff_w50': latest.get('eff_w50', 0),
            'eff_w100': latest.get('eff_w100', 0),
            'eff_w200': latest.get('eff_w200', 0),
            'outside_kc': latest.get('outside_kc', False),
            'ma200_slope': latest.get('ma200_slope', 0)
        }


if __name__ == "__main__":
    """Test the indicator on BTC data"""
    import os
    
    print("="*80)
    print("RUBBER-BAND REVERSION OSCILLATOR - Python Implementation")
    print("="*80)
    
    # Load BTC data
    data_path = "DATA/yf_btc_1h.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        exit(1)
    
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # Ensure OHLCV columns exist
    if 'close' not in df.columns and 'Close' in df.columns:
        df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    
    print(f"\n📊 Loaded {len(df)} bars of BTC data")
    print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Initialize indicator with default settings
    rbo = RubberBandOscillator(
        use_adaptive_weights=False,  # Faster without adaptive weights
        use_dynamic_thresholds=True,
        use_keltner_filter=True
    )
    
    print("\n🔧 Calculating indicator...")
    df_with_signals = rbo.calculate(df)
    
    # Get current state
    current = rbo.get_current_signal(df_with_signals)
    
    print("\n📈 Current State:")
    print(f"   Composite Z-Score: {current['composite_z']:.3f}")
    print(f"   Thresholds: {current['down_threshold']:.2f} (oversold) to {current['up_threshold']:.2f} (overbought)")
    print(f"   Signal: {current['signal_text']}")
    print(f"   Component Z-Scores:")
    print(f"      Z20:  {current['z20']:.3f} (weight: {current['eff_w20']:.2f})")
    print(f"      Z50:  {current['z50']:.3f} (weight: {current['eff_w50']:.2f})")
    print(f"      Z100: {current['z100']:.3f} (weight: {current['eff_w100']:.2f})")
    print(f"      Z200: {current['z200']:.3f} (weight: {current['eff_w200']:.2f})")
    print(f"   200MA Slope: {current['ma200_slope']:.2f}")
    print(f"   Outside Keltner: {current['outside_kc']}")
    
    # Count signals
    long_signals = (df_with_signals['signal'] == 1).sum()
    short_signals = (df_with_signals['signal'] == -1).sum()
    
    print(f"\n📊 Signal Summary (last {len(df)} bars):")
    print(f"   Long signals:  {long_signals}")
    print(f"   Short signals: {short_signals}")
    print(f"   Total signals: {long_signals + short_signals}")
    
    # Show last 5 signals
    signals_df = df_with_signals[df_with_signals['signal'] != 0][['time', 'close', 'composite_z', 'signal']].tail(5)
    if len(signals_df) > 0:
        print(f"\n📋 Last 5 Signals:")
        for _, row in signals_df.iterrows():
            signal_text = "🟢 LONG" if row['signal'] == 1 else "🔴 SHORT"
            print(f"   {row['time']}: {signal_text} @ ${row['close']:.2f} (z={row['composite_z']:.2f})")
    
    print("\n✅ Test complete!")
