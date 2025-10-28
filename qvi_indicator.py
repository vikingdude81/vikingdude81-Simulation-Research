"""
QVI - Quant Volume Intelligence - Python Implementation
Based on TradingView Pine Script v1.2

Volume-based oscillator combining:
1. Relative Volume (log z-score)
2. Chaikin Money Flow (CMF z-score)  
3. Up/Down Volume Delta (directional volume z-score)

Includes VSA-style climax/churn detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class QuantVolumeIntelligence:
    """
    QVI - Quant Volume Intelligence
    
    Advanced volume analysis combining multiple volume metrics:
    - Relative volume strength
    - Money flow (CMF)
    - Directional volume delta
    """
    
    def __init__(
        self,
        len_rvol: int = 50,
        len_cmf: int = 20,
        len_delta: int = 34,
        len_smooth: int = 5,
        len_band: int = 100,
        band_k: float = 1.6,
        vwma_len: int = 20,
        osc_gain: float = 1.5,
        w1: float = 0.5,
        w2: float = 0.3,
        w3: float = 0.2,
        eps: float = 1e-10
    ):
        """Initialize QVI with parameters"""
        self.len_rvol = len_rvol
        self.len_cmf = len_cmf
        self.len_delta = len_delta
        self.len_smooth = len_smooth
        self.len_band = len_band
        self.band_k = band_k
        self.vwma_len = vwma_len
        self.osc_gain = osc_gain
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.eps = eps
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate QVI oscillator and signals
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with QVI indicators
        """
        df = df.copy()
        
        # 1. Relative Volume (log z-score)
        df['log_vol'] = np.log(df['volume'] + 1.0)
        df['rvol_z'] = self._zscore(df['log_vol'], self.len_rvol)
        df['rvol_z'] = df['rvol_z'].ewm(span=3, adjust=False).mean()
        
        # 2. Chaikin Money Flow (CMF)
        df['cmf'] = self._calculate_cmf(df, self.len_cmf)
        df['cmf_z'] = self._zscore(df['cmf'], self.len_cmf)
        
        # 3. Up/Down Volume Delta
        df['return'] = df['close'] - df['close'].shift(1)
        df['udv_sign'] = np.where(df['return'] > 0, 1.0, np.where(df['return'] < 0, -1.0, 0.0))
        df['udv'] = df['udv_sign'] * df['volume']
        df['udv_ema'] = df['udv'].ewm(span=self.len_delta, adjust=False).mean()
        df['delta_z'] = self._zscore(df['udv_ema'], self.len_delta)
        
        # Clip extreme values
        df['rvol_z_clip'] = df['rvol_z'].clip(-4, 4)
        df['cmf_z_clip'] = df['cmf_z'].clip(-4, 4)
        df['delta_z_clip'] = df['delta_z'].clip(-4, 4)
        
        # Blend components
        df['qvi_raw'] = (
            self.w1 * df['rvol_z_clip'] +
            self.w2 * df['cmf_z_clip'] +
            self.w3 * df['delta_z_clip']
        )
        
        # Smooth
        df['qvi'] = df['qvi_raw'].ewm(span=self.len_smooth, adjust=False).mean()
        
        # Adaptive bands
        df['qvi_std'] = df['qvi'].rolling(window=self.len_band).std()
        df['band'] = self.band_k * df['qvi_std']
        df['upper_band'] = df['band']
        df['lower_band'] = -df['band']
        
        # VWMA trend filter
        df['vwma'] = self._vwma(df, self.vwma_len)
        
        # Context filters (simplified - no HTF in this version)
        df['long_ok'] = (df['qvi'] > 0) & (df['close'] > df['vwma'])
        df['short_ok'] = (df['qvi'] < 0) & (df['close'] < df['vwma'])
        
        # Signals (crossovers with bands)
        df['cross_lower'] = (df['qvi'].shift(1) <= df['lower_band'].shift(1)) & (df['qvi'] > df['lower_band'])
        df['cross_upper'] = (df['qvi'].shift(1) >= df['upper_band'].shift(1)) & (df['qvi'] < df['upper_band'])
        
        df['long_signal'] = df['cross_lower'] & df['long_ok']
        df['short_signal'] = df['cross_upper'] & df['short_ok']
        
        # VSA-style Climax/Churn detection
        df['range'] = df['high'] - df['low']
        df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + self.eps)
        df['range_z'] = self._zscore(df['range'], 20)
        
        df['climax_up'] = (df['close'] > df['open']) & (df['vol_ratio'] > 1.5) & (df['range_z'] > 1.0)
        df['climax_down'] = (df['close'] < df['open']) & (df['vol_ratio'] > 1.5) & (df['range_z'] > 1.0)
        df['churn'] = (df['vol_ratio'] > 1.5) & (df['range_z'] < 0.3)
        
        # VSA-based signals (more practical than band crossovers)
        # Climax signals indicate exhaustion/reversal opportunities
        df['vsa_long'] = df['climax_down'] & (df['qvi'] < -df['lower_band'])  # Selling exhaustion
        df['vsa_short'] = df['climax_up'] & (df['qvi'] > df['upper_band'])   # Buying exhaustion
        
        # Combined signal - use VSA by default (more signals than band crossovers)
        df['signal'] = 0
        df.loc[df['vsa_long'], 'signal'] = 1    # LONG on downside exhaustion
        df.loc[df['vsa_short'], 'signal'] = -1   # SHORT on upside exhaustion
        
        # Alternative: Use band crossover signals (very conservative)
        # df.loc[df['long_signal'], 'signal'] = 1
        # df.loc[df['short_signal'], 'signal'] = -1
        
        return df

    
    def _zscore(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score"""
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return (series - mean) / (std + self.eps)
    
    def _calculate_cmf(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        # Money Flow Multiplier
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + self.eps)
        
        # Money Flow Volume
        mfv = mfm * df['volume']
        
        # CMF = Sum(MFV) / Sum(Volume)
        cmf = mfv.rolling(window=period).sum() / (df['volume'].rolling(window=period).sum() + self.eps)
        
        return cmf
    
    def _vwma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Weighted Moving Average"""
        return (df['close'] * df['volume']).rolling(window=period).sum() / (df['volume'].rolling(window=period).sum() + self.eps)
    
    def get_current_state(self, df: pd.DataFrame) -> Dict[str, any]:
        """Get current QVI state"""
        if len(df) == 0:
            return None
        
        latest = df.iloc[-1]
        
        return {
            'qvi': latest.get('qvi', np.nan),
            'upper_band': latest.get('upper_band', np.nan),
            'lower_band': latest.get('lower_band', np.nan),
            'signal': int(latest.get('signal', 0)),
            'signal_text': {1: 'LONG', -1: 'SHORT', 0: 'NEUTRAL'}.get(int(latest.get('signal', 0)), 'NEUTRAL'),
            'rvol_z': latest.get('rvol_z', np.nan),
            'cmf_z': latest.get('cmf_z', np.nan),
            'delta_z': latest.get('delta_z', np.nan),
            'climax_up': latest.get('climax_up', False),
            'climax_down': latest.get('climax_down', False),
            'churn': latest.get('churn', False),
            'vwma': latest.get('vwma', np.nan),
            'close': latest.get('close', np.nan)
        }


if __name__ == "__main__":
    """Test QVI on BTC data"""
    import os
    
    print("="*80)
    print("📊 QVI - QUANT VOLUME INTELLIGENCE - Python Implementation")
    print("="*80)
    
    # Load BTC data
    data_path = "DATA/yf_btc_1h.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        exit(1)
    
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # Standardize columns
    if 'Close' in df.columns:
        df = df.rename(columns={
            'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
    
    print(f"\n📊 Loaded {len(df)} bars of BTC data")
    print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Initialize QVI with default (Swing) settings
    qvi = QuantVolumeIntelligence(
        len_rvol=60,
        len_cmf=20,
        len_delta=34,
        len_smooth=5,
        len_band=120,
        band_k=1.6,
        vwma_len=20
    )
    
    print("\n🔧 Calculating QVI indicators...")
    df_with_signals = qvi.calculate(df)
    
    # Get current state
    current = qvi.get_current_state(df_with_signals)
    
    print("\n📈 Current State:")
    print(f"   QVI Oscillator: {current['qvi']:.3f}")
    print(f"   Bands: {current['lower_band']:.3f} to {current['upper_band']:.3f}")
    print(f"   Signal: {current['signal_text']}")
    print(f"\n   Components (z-scores):")
    print(f"      Rel Volume:  {current['rvol_z']:.3f}")
    print(f"      CMF:         {current['cmf_z']:.3f}")
    print(f"      Delta Vol:   {current['delta_z']:.3f}")
    print(f"\n   VSA Events:")
    print(f"      Climax Up:   {'YES ⬆️' if current['climax_up'] else 'NO'}")
    print(f"      Climax Down: {'YES ⬇️' if current['climax_down'] else 'NO'}")
    print(f"      Churn:       {'YES 🔄' if current['churn'] else 'NO'}")
    print(f"\n   Trend Context:")
    print(f"      Price: ${current['close']:.2f}")
    print(f"      VWMA:  ${current['vwma']:.2f}")
    print(f"      Above VWMA: {'YES ✅' if current['close'] > current['vwma'] else 'NO ❌'}")
    
    # Count signals
    long_signals = (df_with_signals['signal'] == 1).sum()
    short_signals = (df_with_signals['signal'] == -1).sum()
    climax_up = df_with_signals['climax_up'].sum()
    climax_down = df_with_signals['climax_down'].sum()
    churn = df_with_signals['churn'].sum()
    
    print(f"\n📊 Historical Summary:")
    print(f"   Long signals:      {long_signals}")
    print(f"   Short signals:     {short_signals}")
    print(f"   Total signals:     {long_signals + short_signals}")
    print(f"   Climax Up events:  {climax_up}")
    print(f"   Climax Down:       {climax_down}")
    print(f"   Churn events:      {churn}")
    
    # Show last 5 signals
    signals_df = df_with_signals[df_with_signals['signal'] != 0][['time', 'close', 'qvi', 'signal']].tail(5)
    if len(signals_df) > 0:
        print(f"\n📋 Last 5 Signals:")
        for _, row in signals_df.iterrows():
            signal_text = "🟢 LONG" if row['signal'] == 1 else "🔴 SHORT"
            print(f"   {row['time']}: {signal_text} @ ${row['close']:.2f} (QVI={row['qvi']:.3f})")
    
    print("\n✅ Test complete!")
