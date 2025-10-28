"""
Geometric MA Crossover (GMA x GMA) - Python Implementation

Geometric Moving Average uses log-space averaging for better handling
of exponential price movements (common in crypto).

Formula: GMA = exp(SMA(log(price)))

This is superior to arithmetic MA for trending assets because it:
- Weighs percentage changes equally (not absolute changes)
- Better fits exponential growth/decay
- Less lag on strong trends
"""

import pandas as pd
import numpy as np
from typing import Dict


class GeometricMACrossover:
    """
    Geometric Moving Average crossover strategy
    
    Fast GMA x Slow GMA with optional:
    - ATR-based stops and targets
    - Higher timeframe (HTF) trend confirmation
    """
    
    def __init__(
        self,
        len_fast: int = 20,
        len_slow: int = 50,
        use_atr_exit: bool = True,
        atr_length: int = 14,
        stop_atr_mult: float = 2.0,
        tp_rr: float = 2.0,  # Take profit = stop_distance * tp_rr
        min_price: float = 1e-10  # Prevent log(0)
    ):
        self.len_fast = len_fast
        self.len_slow = len_slow
        self.use_atr_exit = use_atr_exit
        self.atr_length = atr_length
        self.stop_atr_mult = stop_atr_mult
        self.tp_rr = tp_rr
        self.min_price = min_price
    
    def _gma(self, series: pd.Series, length: int) -> pd.Series:
        """
        Calculate Geometric Moving Average
        
        GMA = exp(SMA(log(price)))
        """
        # Protect against log(0) or log(negative)
        safe_series = series.clip(lower=self.min_price)
        
        # Log space
        log_series = np.log(safe_series)
        
        # SMA in log space
        log_sma = log_series.rolling(window=length).mean()
        
        # Back to normal space
        gma = np.exp(log_sma)
        
        return gma
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_length).mean()
        
        return atr
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate GMA crossover signals"""
        df = df.copy()
        
        # Calculate GMAs
        df['gma_fast'] = self._gma(df['close'], self.len_fast)
        df['gma_slow'] = self._gma(df['close'], self.len_slow)
        
        # Crossover detection
        df['gma_fast_above'] = (df['gma_fast'] > df['gma_slow']).astype(bool)
        df['gma_fast_above_prev'] = df['gma_fast_above'].shift(1).fillna(False)
        
        df['cross_up'] = (~df['gma_fast_above_prev']) & df['gma_fast_above']
        df['cross_down'] = df['gma_fast_above_prev'] & (~df['gma_fast_above'])
        
        # Base signals
        df['signal'] = 0
        df.loc[df['cross_up'], 'signal'] = 1    # LONG
        df.loc[df['cross_down'], 'signal'] = -1  # SHORT
        
        # ATR for risk management
        if self.use_atr_exit:
            df['atr'] = self._calculate_atr(df)
            df['stop_distance'] = self.stop_atr_mult * df['atr']
            df['target_distance'] = self.tp_rr * df['stop_distance']
        
        # Trend strength (distance between GMAs)
        df['gma_spread'] = (df['gma_fast'] - df['gma_slow']) / df['gma_slow']
        df['gma_spread_pct'] = df['gma_spread'] * 100
        
        return df
    
    def get_current_signal(self, df: pd.DataFrame) -> Dict:
        """Get current GMA state"""
        if len(df) == 0:
            return {}
        
        last = df.iloc[-1]
        
        signal_text = "LONG" if last['signal'] == 1 else "SHORT" if last['signal'] == -1 else "NEUTRAL"
        
        trend = "BULLISH" if last['gma_fast'] > last['gma_slow'] else "BEARISH"
        
        result = {
            'signal': int(last['signal']),
            'signal_text': signal_text,
            'trend': trend,
            'gma_fast': last['gma_fast'],
            'gma_slow': last['gma_slow'],
            'gma_spread_pct': last['gma_spread_pct'],
            'price': last['close']
        }
        
        if self.use_atr_exit:
            result['atr'] = last['atr']
            result['stop_distance'] = last['stop_distance']
            result['target_distance'] = last['target_distance']
            
            if last['signal'] == 1:  # LONG
                result['stop_price'] = last['close'] - last['stop_distance']
                result['target_price'] = last['close'] + last['target_distance']
            elif last['signal'] == -1:  # SHORT
                result['stop_price'] = last['close'] + last['stop_distance']
                result['target_price'] = last['close'] - last['target_distance']
        
        return result


if __name__ == "__main__":
    print("="*80)
    print("GEOMETRIC MA CROSSOVER - Python Test")
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
    
    # Initialize GMA crossover
    print("\nCalculating Geometric MA crossover...")
    gma = GeometricMACrossover(
        len_fast=20,
        len_slow=50,
        use_atr_exit=True,
        atr_length=14,
        stop_atr_mult=2.0,
        tp_rr=2.0
    )
    
    df_result = gma.calculate(df)
    
    # Current signal
    print("\n" + "="*80)
    print("CURRENT STATE")
    print("="*80)
    
    current = gma.get_current_signal(df_result)
    print(f"\nSignal: {current['signal_text']}")
    print(f"Trend: {current['trend']}")
    print(f"\nPrice: ${current['price']:,.2f}")
    print(f"  GMA Fast (20): ${current['gma_fast']:,.2f}")
    print(f"  GMA Slow (50): ${current['gma_slow']:,.2f}")
    print(f"  Spread: {current['gma_spread_pct']:.2f}%")
    
    if 'stop_price' in current:
        print(f"\nRisk Management (ATR-based):")
        print(f"  ATR: ${current['atr']:,.2f}")
        print(f"  Stop Distance: ${current['stop_distance']:,.2f}")
        print(f"  Target Distance: ${current['target_distance']:,.2f}")
        if current['signal'] != 0:
            print(f"  Stop Price: ${current.get('stop_price', 0):,.2f}")
            print(f"  Target Price: ${current.get('target_price', 0):,.2f}")
    
    # Historical signals
    print("\n" + "="*80)
    print("SIGNAL SUMMARY (90 days)")
    print("="*80)
    
    signals = df_result[df_result['signal'] != 0].copy()
    long_signals = signals[signals['signal'] == 1]
    short_signals = signals[signals['signal'] == -1]
    
    print(f"\nTotal signals: {len(signals)}")
    print(f"  Long (cross up):   {len(long_signals)}")
    print(f"  Short (cross down): {len(short_signals)}")
    
    # Last 10 signals
    print("\n" + "="*80)
    print("LAST 10 SIGNALS")
    print("="*80)
    
    last_signals = signals.tail(10)
    for idx, row in last_signals.iterrows():
        signal_type = "LONG ↑" if row['signal'] == 1 else "SHORT ↓"
        print(f"{row['time'].strftime('%Y-%m-%d %H:%M')} {signal_type:8s} @ ${row['close']:>9,.2f}  "
              f"(spread: {row['gma_spread_pct']:+.2f}%)")
    
    # Compare GMA vs SMA
    print("\n" + "="*80)
    print("GMA vs SMA COMPARISON (Current)")
    print("="*80)
    
    current_close = df_result.iloc[-1]['close']
    sma_20 = df_result['close'].rolling(20).mean().iloc[-1]
    sma_50 = df_result['close'].rolling(50).mean().iloc[-1]
    gma_20 = df_result.iloc[-1]['gma_fast']
    gma_50 = df_result.iloc[-1]['gma_slow']
    
    print(f"\nFast MA (20):")
    print(f"  SMA: ${sma_20:,.2f}  (diff from price: {((current_close - sma_20)/sma_20)*100:+.2f}%)")
    print(f"  GMA: ${gma_20:,.2f}  (diff from price: {((current_close - gma_20)/gma_20)*100:+.2f}%)")
    print(f"\nSlow MA (50):")
    print(f"  SMA: ${sma_50:,.2f}  (diff from price: {((current_close - sma_50)/sma_50)*100:+.2f}%)")
    print(f"  GMA: ${gma_50:,.2f}  (diff from price: {((current_close - gma_50)/gma_50)*100:+.2f}%)")
    
    print("\n" + "="*80)
    print("Test complete!")
