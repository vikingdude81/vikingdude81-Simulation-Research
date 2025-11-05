"""
Market Regime Detection System
Identifies current market regime to select appropriate trading specialist

Based on validated prisoner's dilemma patterns:
- Volatile → EarlyGame specialist
- Trending → MidGame specialist  
- Ranging → LateGame specialist
- Crisis → Crisis_Manager specialist
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime, timedelta

# Simple implementations of technical indicators (no talib dependency)
def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # Simplified ADX calculation
    tr = pd.DataFrame()
    tr['h-l'] = high - low
    tr['h-pc'] = abs(high - close.shift())
    tr['l-pc'] = abs(low - close.shift())
    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    atr = tr['tr'].rolling(period).mean()
    
    # Directional movement
    up = high - high.shift()
    down = low.shift() - low
    
    pos_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=high.index)
    neg_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=high.index)
    
    pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
    neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
    
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
    adx = dx.rolling(period).mean()
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr = pd.DataFrame()
    tr['h-l'] = high - low
    tr['h-pc'] = abs(high - close.shift())
    tr['l-pc'] = abs(low - close.shift())
    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    atr = tr['tr'].rolling(period).mean()
    return atr

class RegimeDetector:
    """
    Detects market regime using multiple indicators:
    - VIX (volatility)
    - ADX (trend strength)
    - ATR (average true range)
    - Volume profile
    - Correlation patterns
    """
    
    def __init__(self, 
                 vix_threshold_high: float = 62.2,      # 75th percentile for crypto
                 vix_threshold_extreme: float = 99.2,   # 95th percentile for crypto
                 adx_trending: float = 51.1,            # 75th percentile for crypto
                 adx_ranging: float = 27.0,             # 25th percentile for crypto
                 atr_multiplier: float = 1.5,
                 lookback_period: int = 20):
        """
        Initialize regime detector with CRYPTO-SPECIFIC thresholds
        (Calibrated from BTC historical data 2014-2025)
        
        Args:
            vix_threshold_high: VIX level for volatile market (62.2 - crypto 75th percentile)
            vix_threshold_extreme: VIX level for crisis (99.2 - crypto 95th percentile)
            adx_trending: ADX threshold for trending market (51.1 - crypto 75th percentile)
            adx_ranging: ADX threshold for ranging market (27.0 - crypto 25th percentile)
            atr_multiplier: Multiplier for ATR volatility check (1.5)
            lookback_period: Period for rolling calculations (20)
        """
        self.vix_threshold_high = vix_threshold_high
        self.vix_threshold_extreme = vix_threshold_extreme
        self.adx_trending = adx_trending
        self.adx_ranging = adx_ranging
        self.atr_multiplier = atr_multiplier
        self.lookback_period = lookback_period
        
        # Regime history for persistence checking
        self.regime_history: List[str] = []
        self.regime_confidence: Dict[str, float] = {}
    
    def detect_regime(self, df: pd.DataFrame, include_vix: bool = False) -> str:
        """
        Detect current market regime
        
        Args:
            df: DataFrame with OHLCV data (last row is current)
            include_vix: Whether VIX data is available
            
        Returns:
            regime: 'volatile', 'trending', 'ranging', or 'crisis'
        """
        # Calculate indicators
        indicators = self.calculate_indicators(df)
        
        # Get VIX level (or estimate if not available)
        if include_vix and 'vix' in df.columns:
            vix = df['vix'].iloc[-1]
        else:
            vix = self.estimate_vix(df)
        
        adx = indicators['adx']
        atr_ratio = indicators['atr_ratio']
        volume_spike = indicators['volume_spike']
        price_range = indicators['price_range']
        
        # Calculate regime scores
        scores = self.calculate_regime_scores(
            vix, adx, atr_ratio, volume_spike, price_range
        )
        
        # Determine regime
        regime = self.select_regime(scores, vix, adx)
        
        # Add to history
        self.regime_history.append(regime)
        if len(self.regime_history) > 20:
            self.regime_history.pop(0)
        
        # Store confidence
        self.regime_confidence = scores
        
        return regime
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators for regime detection"""
        
        # ADX (trend strength)
        adx_series = calculate_adx(df['high'], df['low'], df['close'], period=14)
        adx = adx_series.iloc[-1] if not adx_series.empty else 20.0
        
        # ATR (volatility)
        atr_series = calculate_atr(df['high'], df['low'], df['close'], period=14)
        atr = atr_series.iloc[-1] if not atr_series.empty else 1.0
        atr_avg = atr_series.rolling(self.lookback_period).mean().iloc[-1]
        atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
        
        # Volume spike
        volume_avg = df['volume'].rolling(self.lookback_period).mean().iloc[-1]
        volume_current = df['volume'].iloc[-1]
        volume_spike = volume_current / volume_avg if volume_avg > 0 else 1.0
        
        # Price range (for ranging market detection)
        high_20 = df['high'].rolling(self.lookback_period).max().iloc[-1]
        low_20 = df['low'].rolling(self.lookback_period).min().iloc[-1]
        current_close = df['close'].iloc[-1]
        price_range = (current_close - low_20) / (high_20 - low_20) if (high_20 - low_20) > 0 else 0.5
        
        return {
            'adx': adx,
            'atr_ratio': atr_ratio,
            'volume_spike': volume_spike,
            'price_range': price_range
        }
    
    def estimate_vix(self, df: pd.DataFrame) -> float:
        """
        Estimate VIX-like volatility from price data
        Uses realized volatility scaled to VIX-like levels
        """
        returns = df['close'].pct_change().dropna()
        
        # Calculate realized volatility (annualized)
        realized_vol = returns.std() * np.sqrt(252) * 100
        
        # Scale to VIX-like levels (typical range 10-80)
        # Realized vol typically 15-50%, VIX typically 10-50
        vix_estimate = realized_vol
        
        return vix_estimate
    
    def calculate_regime_scores(self, vix: float, adx: float, 
                                 atr_ratio: float, volume_spike: float,
                                 price_range: float) -> Dict[str, float]:
        """
        Calculate confidence scores for each regime
        Returns dict with score for each regime (0-100)
        """
        scores = {
            'crisis': 0,
            'volatile': 0,
            'trending': 0,
            'ranging': 0
        }
        
        # Crisis scoring (VIX > 35, extreme volatility)
        if vix > self.vix_threshold_extreme:
            scores['crisis'] += 50
        if vix > 40:
            scores['crisis'] += 30
        if atr_ratio > 2.0:
            scores['crisis'] += 20
        
        # Volatile scoring (VIX 25-35, high volatility, weak trend)
        if self.vix_threshold_high < vix < self.vix_threshold_extreme:
            scores['volatile'] += 40
        if atr_ratio > self.atr_multiplier:
            scores['volatile'] += 30
        if adx < self.adx_ranging:
            scores['volatile'] += 20
        if volume_spike > 1.5:
            scores['volatile'] += 10
        
        # Trending scoring (ADX > 25, moderate volatility)
        if adx > self.adx_trending:
            scores['trending'] += 50
        if adx > 30:
            scores['trending'] += 20
        if self.atr_multiplier * 0.8 < atr_ratio < self.atr_multiplier * 1.2:
            scores['trending'] += 20
        if vix < self.vix_threshold_high:
            scores['trending'] += 10
        
        # Ranging scoring (ADX < 20, low volatility, mean reversion)
        if adx < self.adx_ranging:
            scores['ranging'] += 40
        if vix < 20:
            scores['ranging'] += 30
        if 0.3 < price_range < 0.7:  # Price in middle of range
            scores['ranging'] += 20
        if atr_ratio < 1.0:  # Below average volatility
            scores['ranging'] += 10
        
        return scores
    
    def select_regime(self, scores: Dict[str, float], vix: float, adx: float) -> str:
        """
        Select regime based on scores with hierarchical logic
        
        Crisis > Volatile > Trending/Ranging
        """
        # Crisis takes priority (capital preservation)
        if scores['crisis'] >= 50 or vix > self.vix_threshold_extreme:
            return 'crisis'
        
        # Find highest score among other regimes
        regime_scores = {k: v for k, v in scores.items() if k != 'crisis'}
        max_regime = max(regime_scores, key=regime_scores.get)
        max_score = regime_scores[max_regime]
        
        # Require minimum confidence (40 points)
        if max_score < 40:
            # Default logic based on simple rules
            if vix > self.vix_threshold_high:
                return 'volatile'
            elif adx > self.adx_trending:
                return 'trending'
            else:
                return 'ranging'
        
        return max_regime
    
    def get_regime_confidence(self) -> Dict[str, float]:
        """
        Get confidence level for current regime
        Returns normalized confidence scores (0-1)
        """
        if not self.regime_confidence:
            return {'crisis': 0, 'volatile': 0, 'trending': 0, 'ranging': 0}
        
        # Normalize scores to 0-1
        total = sum(self.regime_confidence.values())
        if total == 0:
            return self.regime_confidence
        
        return {k: v/total for k, v in self.regime_confidence.items()}
    
    def check_regime_persistence(self, min_consistency: int = 3) -> Tuple[bool, str]:
        """
        Check if regime has persisted for minimum periods
        Helps avoid thrashing between specialists
        
        Args:
            min_consistency: Minimum periods regime must persist (3)
            
        Returns:
            (is_persistent, regime): Whether regime is stable and which regime
        """
        if len(self.regime_history) < min_consistency:
            return False, 'unknown'
        
        # Check last N periods
        recent_regimes = self.regime_history[-min_consistency:]
        
        # Count most common regime
        from collections import Counter
        regime_counts = Counter(recent_regimes)
        most_common_regime, count = regime_counts.most_common(1)[0]
        
        # Is it persistent enough?
        is_persistent = (count >= min_consistency)
        
        return is_persistent, most_common_regime if is_persistent else 'mixed'
    
    def get_regime_statistics(self) -> Dict:
        """Get statistics about regime history"""
        if not self.regime_history:
            return {}
        
        from collections import Counter
        counts = Counter(self.regime_history)
        total = len(self.regime_history)
        
        return {
            'total_periods': total,
            'regime_distribution': {k: v/total for k, v in counts.items()},
            'current_regime': self.regime_history[-1],
            'current_streak': self._calculate_current_streak()
        }
    
    def _calculate_current_streak(self) -> int:
        """Calculate how many consecutive periods current regime has lasted"""
        if not self.regime_history:
            return 0
        
        current = self.regime_history[-1]
        streak = 0
        
        for regime in reversed(self.regime_history):
            if regime == current:
                streak += 1
            else:
                break
        
        return streak


def test_regime_detector():
    """Test regime detector with sample data from existing project data"""
    from pathlib import Path
    
    print("\n" + "="*70)
    print("🔍 TESTING REGIME DETECTOR")
    print("="*70)
    
    # Load existing BTC data from the project
    print("\n📊 Loading existing BTC data...")
    data_dir = Path(__file__).parent / 'DATA'
    
    # Try to load daily data
    btc_file = data_dir / 'yf_btc_1d.csv'
    
    if not btc_file.exists():
        print(f"❌ Data file not found: {btc_file}")
        print("Run fetch_data.py first to download data!")
        return []
    
    # Load data - already formatted correctly!
    btc = pd.read_csv(btc_file)
    
    # Convert time to datetime
    if 'time' in btc.columns:
        btc['time'] = pd.to_datetime(btc['time'])
        btc.set_index('time', inplace=True)
    
    print(f"✅ Loaded {len(btc)} rows from {btc.index[0]} to {btc.index[-1]}")
    
    # Initialize detector
    detector = RegimeDetector()
    
    # Test on rolling windows (last year of data)
    print("\n📈 Detecting regimes over time...\n")
    
    # Use last 365 days
    start_idx = max(0, len(btc) - 365)
    
    results = []
    for i in range(start_idx + 50, len(btc), 10):  # Every 10 days
        window = btc.iloc[:i]
        regime = detector.detect_regime(window)
        confidence = detector.get_regime_confidence()
        date = btc.index[i-1]
        
        results.append({
            'date': date,
            'regime': regime,
            'confidence': confidence[regime],
            'price': btc['close'].iloc[i-1]
        })
        
        print(f"{date.strftime('%Y-%m-%d')}: {regime:>10s} (confidence: {confidence[regime]:.2f}, price: ${btc['close'].iloc[i-1]:,.0f})")
    
    # Summary statistics
    print("\n" + "="*70)
    print("📊 REGIME STATISTICS")
    print("="*70)
    
    stats = detector.get_regime_statistics()
    print(f"\nTotal periods analyzed: {stats['total_periods']}")
    print(f"\nRegime distribution:")
    for regime, pct in stats['regime_distribution'].items():
        print(f"  {regime:>10s}: {pct:.1%}")
    
    print(f"\nCurrent regime: {stats['current_regime']}")
    print(f"Current streak: {stats['current_streak']} periods")
    
    # Regime persistence check
    is_persistent, persistent_regime = detector.check_regime_persistence(min_consistency=3)
    if is_persistent:
        print(f"\n✅ Regime is PERSISTENT: {persistent_regime} (last 3+ periods)")
    else:
        print(f"\n⚠️  Regime is MIXED: No clear persistence")
    
    print("\n✅ Regime detector test complete!")
    print(f"\n💡 Tip: This detector is ready to use with your trading system!")
    print(f"   Just call: regime = detector.detect_regime(market_data)")
    
    return results


if __name__ == "__main__":
    # Run test
    results = test_regime_detector()
