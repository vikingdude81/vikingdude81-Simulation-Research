"""
Support/Resistance Level Calculator
Phase D: Advanced technical analysis for better entry/exit timing

Features:
- Pivot Points (Standard, Fibonacci, Camarilla)
- Volume Profile (HVN/LVN detection)
- Historical Swing Levels (Local max/min with touch counting)
- Signal Enhancement (proximity bonuses, stop-loss, take-profit)
- Dynamic Risk/Reward Calculation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json


class SupportResistanceAnalyzer:
    """
    Calculates support and resistance levels using multiple methods.
    Enhances trading signals with proximity-based adjustments.
    """
    
    def __init__(self, data_path: str = 'DATA'):
        """
        Initialize the S/R analyzer.
        
        Args:
            data_path: Path to historical price data
        """
        self.data_path = data_path
        self.proximity_threshold = 0.02  # 2% proximity for bonus
        self.swing_window = 20  # Window for swing high/low detection
        self.volume_bins = 50  # Bins for volume profile
        
    def load_data(self, asset: str, timeframe: str = '1h', lookback_days: int = 30) -> pd.DataFrame:
        """
        Load historical price data for analysis.
        
        Args:
            asset: Asset symbol (BTC, ETH, SOL)
            timeframe: Timeframe (1h, 4h, 12h, 1d)
            lookback_days: Days of historical data to load
            
        Returns:
            DataFrame with OHLCV data
        """
        filename = f"{self.data_path}/yf_{asset.lower()}_{timeframe}.csv"
        
        try:
            df = pd.read_csv(filename)
            df['time'] = pd.to_datetime(df['time'])
            
            # Get recent data (make cutoff timezone-aware if needed)
            cutoff = datetime.now() - timedelta(days=lookback_days)
            if df['time'].dt.tz is not None:
                # If data is timezone-aware, make cutoff timezone-aware too
                cutoff = pd.Timestamp(cutoff).tz_localize('UTC')
            df = df[df['time'] >= cutoff].copy()
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                raise ValueError(f"Missing required columns in {filename}")
            
            return df.sort_values('time').reset_index(drop=True)
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return pd.DataFrame()
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate pivot points using three methods.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary with Standard, Fibonacci, and Camarilla pivots
        """
        # Use most recent day's data
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        # Standard Pivots
        pivot = (high + low + close) / 3
        standard = {
            'pivot': pivot,
            'r1': 2 * pivot - low,
            'r2': pivot + (high - low),
            'r3': high + 2 * (pivot - low),
            's1': 2 * pivot - high,
            's2': pivot - (high - low),
            's3': low - 2 * (high - pivot)
        }
        
        # Fibonacci Pivots
        fib_range = high - low
        fibonacci = {
            'pivot': pivot,
            'r1': pivot + 0.382 * fib_range,
            'r2': pivot + 0.618 * fib_range,
            'r3': pivot + fib_range,
            's1': pivot - 0.382 * fib_range,
            's2': pivot - 0.618 * fib_range,
            's3': pivot - fib_range
        }
        
        # Camarilla Pivots
        camarilla_range = high - low
        camarilla = {
            'pivot': close,
            'r1': close + camarilla_range * 1.1 / 12,
            'r2': close + camarilla_range * 1.1 / 6,
            'r3': close + camarilla_range * 1.1 / 4,
            'r4': close + camarilla_range * 1.1 / 2,
            's1': close - camarilla_range * 1.1 / 12,
            's2': close - camarilla_range * 1.1 / 6,
            's3': close - camarilla_range * 1.1 / 4,
            's4': close - camarilla_range * 1.1 / 2
        }
        
        return {
            'standard': standard,
            'fibonacci': fibonacci,
            'camarilla': camarilla
        }
    
    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Calculate volume profile to identify HVN/LVN levels.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary with high volume nodes and low volume nodes
        """
        if df.empty:
            return {'hvn': [], 'lvn': []}
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        bins = np.linspace(price_min, price_max, self.volume_bins)
        
        # Aggregate volume by price level
        volume_by_price = np.zeros(len(bins) - 1)
        
        for _, row in df.iterrows():
            # Find which bins this candle's range covers
            low_idx = np.searchsorted(bins, row['low'], side='right') - 1
            high_idx = np.searchsorted(bins, row['high'], side='left')
            
            # Distribute volume across bins
            if low_idx < 0:
                low_idx = 0
            if high_idx >= len(volume_by_price):
                high_idx = len(volume_by_price) - 1
            
            bins_covered = max(1, high_idx - low_idx + 1)
            volume_per_bin = row['volume'] / bins_covered
            
            for i in range(low_idx, min(high_idx + 1, len(volume_by_price))):
                volume_by_price[i] += volume_per_bin
        
        # Find high volume nodes (top 20%)
        threshold_high = np.percentile(volume_by_price, 80)
        hvn_indices = np.where(volume_by_price >= threshold_high)[0]
        hvn_levels = [(bins[i] + bins[i+1]) / 2 for i in hvn_indices]
        
        # Find low volume nodes (bottom 20%)
        threshold_low = np.percentile(volume_by_price, 20)
        lvn_indices = np.where(volume_by_price <= threshold_low)[0]
        lvn_levels = [(bins[i] + bins[i+1]) / 2 for i in lvn_indices]
        
        return {
            'hvn': hvn_levels[:10],  # Top 10 HVN levels
            'lvn': lvn_levels[:10]   # Top 10 LVN levels
        }
    
    def find_swing_levels(self, df: pd.DataFrame) -> Dict[str, List[Tuple[float, int]]]:
        """
        Find swing highs and lows with touch counting.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary with resistance (swing highs) and support (swing lows)
            Each level is (price, touch_count)
        """
        if len(df) < self.swing_window * 2:
            return {'resistance': [], 'support': []}
        
        resistance_levels = []
        support_levels = []
        
        # Find swing highs (local maxima)
        for i in range(self.swing_window, len(df) - self.swing_window):
            if df['high'].iloc[i] == df['high'].iloc[i - self.swing_window:i + self.swing_window + 1].max():
                resistance_levels.append(df['high'].iloc[i])
        
        # Find swing lows (local minima)
        for i in range(self.swing_window, len(df) - self.swing_window):
            if df['low'].iloc[i] == df['low'].iloc[i - self.swing_window:i + self.swing_window + 1].min():
                support_levels.append(df['low'].iloc[i])
        
        # Cluster similar levels and count touches
        def cluster_levels(levels: List[float], tolerance: float = 0.01) -> List[Tuple[float, int]]:
            if not levels:
                return []
            
            clustered = []
            sorted_levels = sorted(levels)
            
            current_cluster = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if abs(level - current_cluster[0]) / current_cluster[0] < tolerance:
                    current_cluster.append(level)
                else:
                    # Save cluster
                    avg_price = np.mean(current_cluster)
                    touch_count = len(current_cluster)
                    clustered.append((avg_price, touch_count))
                    
                    # Start new cluster
                    current_cluster = [level]
            
            # Don't forget last cluster
            if current_cluster:
                avg_price = np.mean(current_cluster)
                touch_count = len(current_cluster)
                clustered.append((avg_price, touch_count))
            
            # Sort by touch count (stronger levels first)
            return sorted(clustered, key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'resistance': cluster_levels(resistance_levels),
            'support': cluster_levels(support_levels)
        }
    
    def get_all_levels(self, asset: str, timeframe: str = '1h') -> Dict:
        """
        Calculate all support/resistance levels for an asset.
        
        Args:
            asset: Asset symbol (BTC, ETH, SOL)
            timeframe: Timeframe for analysis
            
        Returns:
            Comprehensive dictionary with all S/R levels
        """
        df = self.load_data(asset, timeframe)
        
        if df.empty:
            return {
                'asset': asset,
                'current_price': 0,
                'pivots': {},
                'volume_profile': {'hvn': [], 'lvn': []},
                'swing_levels': {'resistance': [], 'support': []}
            }
        
        current_price = df['close'].iloc[-1]
        
        return {
            'asset': asset,
            'current_price': current_price,
            'pivots': self.calculate_pivot_points(df),
            'volume_profile': self.calculate_volume_profile(df),
            'swing_levels': self.find_swing_levels(df),
            'timestamp': datetime.now().isoformat()
        }
    
    def find_nearest_levels(self, current_price: float, levels_dict: Dict) -> Dict[str, Optional[float]]:
        """
        Find nearest support and resistance levels.
        
        Args:
            current_price: Current asset price
            levels_dict: Dictionary from get_all_levels()
            
        Returns:
            Dictionary with nearest_support and nearest_resistance
        """
        all_resistance = []
        all_support = []
        
        # Collect from pivots
        for method in ['standard', 'fibonacci', 'camarilla']:
            if method in levels_dict.get('pivots', {}):
                pivots = levels_dict['pivots'][method]
                for key, value in pivots.items():
                    if 'r' in key.lower() and value > current_price:
                        all_resistance.append(value)
                    elif 's' in key.lower() and value < current_price:
                        all_support.append(value)
        
        # Collect from HVN
        for hvn in levels_dict.get('volume_profile', {}).get('hvn', []):
            if hvn > current_price:
                all_resistance.append(hvn)
            elif hvn < current_price:
                all_support.append(hvn)
        
        # Collect from swing levels
        for price, _ in levels_dict.get('swing_levels', {}).get('resistance', []):
            if price > current_price:
                all_resistance.append(price)
        
        for price, _ in levels_dict.get('swing_levels', {}).get('support', []):
            if price < current_price:
                all_support.append(price)
        
        # Find nearest
        nearest_resistance = min(all_resistance) if all_resistance else None
        nearest_support = max(all_support) if all_support else None
        
        return {
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support
        }
    
    def calculate_proximity_bonus(self, current_price: float, nearest_support: Optional[float], 
                                  nearest_resistance: Optional[float]) -> Dict[str, float]:
        """
        Calculate signal strength bonuses based on proximity to S/R.
        
        Args:
            current_price: Current asset price
            nearest_support: Nearest support level
            nearest_resistance: Nearest resistance level
            
        Returns:
            Dictionary with buy_bonus and sell_bonus
        """
        buy_bonus = 0.0
        sell_bonus = 0.0
        
        # Buy bonus near support
        if nearest_support:
            distance_to_support = (current_price - nearest_support) / current_price
            if distance_to_support <= self.proximity_threshold:
                # Closer = stronger bonus (up to +10%)
                buy_bonus = 0.10 * (1 - distance_to_support / self.proximity_threshold)
        
        # Sell bonus near resistance
        if nearest_resistance:
            distance_to_resistance = (nearest_resistance - current_price) / current_price
            if distance_to_resistance <= self.proximity_threshold:
                # Closer = stronger bonus (up to +10%)
                sell_bonus = 0.10 * (1 - distance_to_resistance / self.proximity_threshold)
        
        return {
            'buy_bonus': buy_bonus,
            'sell_bonus': sell_bonus
        }
    
    def calculate_stop_loss_take_profit(self, current_price: float, signal: str,
                                        nearest_support: Optional[float],
                                        nearest_resistance: Optional[float]) -> Dict[str, Optional[float]]:
        """
        Calculate stop-loss and take-profit levels.
        
        Args:
            current_price: Current asset price
            signal: Trading signal (BUY or SELL)
            nearest_support: Nearest support level
            nearest_resistance: Nearest resistance level
            
        Returns:
            Dictionary with stop_loss and take_profit levels
        """
        if signal == 'BUY':
            # Stop-loss just below support (or 2% below if no support)
            if nearest_support:
                stop_loss = nearest_support * 0.995  # 0.5% buffer below support
            else:
                stop_loss = current_price * 0.98  # 2% stop-loss
            
            # Take-profit at resistance (but only if above current price)
            if nearest_resistance and nearest_resistance > current_price * 1.005:
                # Resistance is meaningfully above current price (>0.5%)
                take_profit = nearest_resistance * 0.999  # Take profit just before resistance (0.1% buffer)
            else:
                # No valid resistance above or too close, use percentage target
                take_profit = current_price * 1.04  # 4% take-profit
        
        elif signal == 'SELL':
            # Stop-loss just above resistance (or 2% above if no resistance)
            if nearest_resistance:
                stop_loss = nearest_resistance * 1.005  # 0.5% buffer above resistance
            else:
                stop_loss = current_price * 1.02  # 2% stop-loss
            
            # Take-profit at support (but only if below current price)
            if nearest_support and nearest_support < current_price * 0.995:
                # Support is meaningfully below current price (>0.5%)
                take_profit = nearest_support * 1.001  # Take profit just before support (0.1% buffer)
            else:
                # No valid support below or too close, use percentage target
                take_profit = current_price * 0.96  # 4% take-profit
        
        else:
            stop_loss = None
            take_profit = None
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def calculate_risk_reward(self, current_price: float, stop_loss: Optional[float],
                             take_profit: Optional[float]) -> Optional[float]:
        """
        Calculate risk/reward ratio.
        
        Args:
            current_price: Current asset price
            stop_loss: Stop-loss level
            take_profit: Take-profit level
            
        Returns:
            Risk/reward ratio (or None if invalid)
        """
        if not stop_loss or not take_profit:
            return None
        
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        
        if risk == 0:
            return None
        
        return reward / risk
    
    def enhance_signal(self, asset: str, signal: str, current_price: float,
                      expected_return: float, timeframe: str = '1h') -> Dict:
        """
        Enhance trading signal with S/R analysis.
        
        Args:
            asset: Asset symbol
            signal: Original signal (BUY/SELL/HOLD)
            current_price: Current price
            expected_return: Expected return from prediction
            timeframe: Timeframe for S/R calculation
            
        Returns:
            Enhanced signal with S/R data
        """
        # Get all S/R levels
        levels = self.get_all_levels(asset, timeframe)
        
        # Find nearest levels
        nearest = self.find_nearest_levels(current_price, levels)
        
        # Calculate proximity bonuses
        bonuses = self.calculate_proximity_bonus(
            current_price, 
            nearest['nearest_support'],
            nearest['nearest_resistance']
        )
        
        # Calculate stop-loss and take-profit
        sl_tp = self.calculate_stop_loss_take_profit(
            current_price,
            signal,
            nearest['nearest_support'],
            nearest['nearest_resistance']
        )
        
        # Calculate risk/reward
        risk_reward = self.calculate_risk_reward(
            current_price,
            sl_tp['stop_loss'],
            sl_tp['take_profit']
        )
        
        # Apply bonuses to expected return
        enhanced_return = expected_return
        if signal == 'BUY':
            enhanced_return += bonuses['buy_bonus']
        elif signal == 'SELL':
            enhanced_return += bonuses['sell_bonus']
        
        return {
            'asset': asset,
            'original_signal': signal,
            'enhanced_signal': signal,  # Can upgrade HOLD to BUY/SELL if near S/R
            'current_price': current_price,
            'expected_return': expected_return,
            'enhanced_return': enhanced_return,
            'proximity_bonus': bonuses['buy_bonus'] if signal == 'BUY' else bonuses['sell_bonus'],
            'nearest_support': nearest['nearest_support'],
            'nearest_resistance': nearest['nearest_resistance'],
            'stop_loss': sl_tp['stop_loss'],
            'take_profit': sl_tp['take_profit'],
            'risk_reward_ratio': risk_reward,
            'levels_detail': levels
        }
    
    def print_sr_analysis(self, enhanced_signal: Dict):
        """
        Pretty print S/R analysis.
        
        Args:
            enhanced_signal: Dictionary from enhance_signal()
        """
        print(f"\n{'='*60}")
        print(f"📊 SUPPORT/RESISTANCE ANALYSIS - {enhanced_signal['asset']}")
        print(f"{'='*60}")
        
        print(f"\n💰 Current Price: ${enhanced_signal['current_price']:,.2f}")
        
        if enhanced_signal['nearest_support']:
            distance_to_support = (enhanced_signal['current_price'] - enhanced_signal['nearest_support']) / enhanced_signal['current_price'] * 100
            print(f"🛡️  Nearest Support: ${enhanced_signal['nearest_support']:,.2f} ({distance_to_support:.2f}% below)")
        else:
            print(f"🛡️  Nearest Support: None found")
        
        if enhanced_signal['nearest_resistance']:
            distance_to_resistance = (enhanced_signal['nearest_resistance'] - enhanced_signal['current_price']) / enhanced_signal['current_price'] * 100
            print(f"🚧 Nearest Resistance: ${enhanced_signal['nearest_resistance']:,.2f} ({distance_to_resistance:.2f}% above)")
        else:
            print(f"🚧 Nearest Resistance: None found")
        
        print(f"\n📈 Signal Analysis:")
        print(f"   Original Signal: {enhanced_signal['original_signal']}")
        print(f"   Expected Return: {enhanced_signal['expected_return']:.2%}")
        if enhanced_signal['proximity_bonus'] > 0:
            print(f"   Proximity Bonus: +{enhanced_signal['proximity_bonus']:.2%}")
            print(f"   Enhanced Return: {enhanced_signal['enhanced_return']:.2%} ⭐")
        
        if enhanced_signal['stop_loss']:
            sl_distance = abs(enhanced_signal['current_price'] - enhanced_signal['stop_loss']) / enhanced_signal['current_price'] * 100
            print(f"\n🛑 Stop-Loss: ${enhanced_signal['stop_loss']:,.2f} ({sl_distance:.2f}% risk)")
        
        if enhanced_signal['take_profit']:
            tp_distance = abs(enhanced_signal['take_profit'] - enhanced_signal['current_price']) / enhanced_signal['current_price'] * 100
            print(f"🎯 Take-Profit: ${enhanced_signal['take_profit']:,.2f} ({tp_distance:.2f}% gain)")
        
        if enhanced_signal['risk_reward_ratio']:
            print(f"⚖️  Risk/Reward: 1:{enhanced_signal['risk_reward_ratio']:.2f}", end="")
            if enhanced_signal['risk_reward_ratio'] >= 2.0:
                print(" ✅ (Excellent)")
            elif enhanced_signal['risk_reward_ratio'] >= 1.5:
                print(" ✅ (Good)")
            else:
                print(" ⚠️ (Marginal)")
        
        print(f"\n{'='*60}\n")


def test_sr_analyzer():
    """Test the S/R analyzer with BTC data."""
    print("\n" + "="*60)
    print("🧪 TESTING SUPPORT/RESISTANCE ANALYZER")
    print("="*60)
    
    analyzer = SupportResistanceAnalyzer()
    
    # Test with BTC
    print("\n📊 Analyzing BTC Support/Resistance Levels...")
    levels = analyzer.get_all_levels('btc', '1h')
    
    print(f"\n💰 Current BTC Price: ${levels['current_price']:,.2f}")
    
    # Print pivot points
    print(f"\n📍 PIVOT POINTS:")
    for method, pivots in levels['pivots'].items():
        print(f"\n   {method.upper()}:")
        for key, value in sorted(pivots.items()):
            if 'r' in key.lower():
                print(f"      {key.upper()}: ${value:,.2f} 🚧")
            elif 's' in key.lower():
                print(f"      {key.upper()}: ${value:,.2f} 🛡️")
            else:
                print(f"      {key.upper()}: ${value:,.2f} ⭐")
    
    # Print volume profile
    print(f"\n📊 VOLUME PROFILE:")
    print(f"   High Volume Nodes (HVN): {len(levels['volume_profile']['hvn'])} levels")
    for i, hvn in enumerate(levels['volume_profile']['hvn'][:5]):
        print(f"      HVN{i+1}: ${hvn:,.2f}")
    
    # Print swing levels
    print(f"\n🎢 SWING LEVELS:")
    print(f"   Resistance (swing highs): {len(levels['swing_levels']['resistance'])} levels")
    for price, touches in levels['swing_levels']['resistance'][:5]:
        print(f"      ${price:,.2f} (tested {touches}x) 🚧")
    
    print(f"   Support (swing lows): {len(levels['swing_levels']['support'])} levels")
    for price, touches in levels['swing_levels']['support'][:5]:
        print(f"      ${price:,.2f} (tested {touches}x) 🛡️")
    
    # Test signal enhancement
    print(f"\n🎯 TESTING SIGNAL ENHANCEMENT:")
    enhanced = analyzer.enhance_signal(
        asset='btc',
        signal='BUY',
        current_price=levels['current_price'],
        expected_return=0.0186,  # Example: +1.86%
        timeframe='1h'
    )
    
    analyzer.print_sr_analysis(enhanced)
    
    print("✅ S/R Analyzer test complete!")


if __name__ == '__main__':
    test_sr_analyzer()
