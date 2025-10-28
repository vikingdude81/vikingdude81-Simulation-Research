"""
Dominance Analyzer for Market Regime Detection
Analyzes USDT.D and BTC.D to determine market conditions and optimal asset allocation
"""

from typing import Dict, Tuple
from datetime import datetime
import json
from pathlib import Path
import numpy as np


class DominanceAnalyzer:
    """
    Analyze crypto dominance metrics to detect market regimes and optimize allocation
    
    Key Metrics:
    - USDT.D (Tether Dominance): Fear/Greed indicator
    - BTC.D (Bitcoin Dominance): BTC vs Alt season indicator
    """
    
    def __init__(self):
        """Initialize dominance analyzer with thresholds"""
        
        # USDT.D thresholds (fear gauge)
        self.usdt_high_fear = 5.0      # >5% = High fear, defensive
        self.usdt_low_greed = 4.0      # <4% = Greed, aggressive
        
        # BTC.D thresholds (market regime)
        self.btc_rally_threshold = 60.0    # >60% = BTC rally
        self.btc_alt_season = 50.0         # <50% = Alt season
        
        # Allocation modifiers
        self.fear_reduction = 0.4    # Reduce to 40% in high fear
        self.neutral_reduction = 0.7  # 70% in neutral
        self.greed_boost = 1.0       # 100% in greed
    
    def get_current_dominance(self) -> Dict[str, float]:
        """
        Get current dominance values from latest external data
        
        Returns:
            Dict with USDT.D and BTC.D values
        """
        try:
            # Find most recent external data file
            external_files = list(Path('EXTERNAL_DATA_CACHE').glob('external_*.json'))
            
            if not external_files:
                # Fallback to MODEL_STORAGE
                external_files = list(Path('MODEL_STORAGE/external_data').glob('*.json'))
            
            if not external_files:
                print("⚠️  No external data found, using defaults")
                return {'USDT.D': 4.5, 'BTC.D': 55.0}  # Neutral defaults
            
            # Get most recent file
            latest_file = max(external_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            usdt_d = data.get('usdt_dominance', 4.5)
            btc_d = data.get('btc_dominance', 55.0)
            
            return {
                'USDT.D': usdt_d,
                'BTC.D': btc_d
            }
            
        except Exception as e:
            print(f"⚠️  Error loading dominance data: {e}")
            return {'USDT.D': 4.5, 'BTC.D': 55.0}  # Neutral defaults
    
    def analyze_market_regime(self, usdt_d: float, btc_d: float) -> Dict:
        """
        Determine market regime based on dominance metrics
        
        Args:
            usdt_d: Tether dominance percentage
            btc_d: Bitcoin dominance percentage
            
        Returns:
            Dict with regime analysis and recommendations
        """
        
        # Analyze fear/greed (USDT.D)
        if usdt_d > self.usdt_high_fear:
            fear_state = 'HIGH_FEAR'
            fear_description = 'Market in fear - capital fleeing to stablecoins'
            position_modifier = self.fear_reduction
            fear_emoji = '😨'
        elif usdt_d < self.usdt_low_greed:
            fear_state = 'GREED'
            fear_description = 'Market in greed - capital flowing into crypto'
            position_modifier = self.greed_boost
            fear_emoji = '🤑'
        else:
            fear_state = 'NEUTRAL'
            fear_description = 'Market neutral - balanced sentiment'
            position_modifier = self.neutral_reduction
            fear_emoji = '😐'
        
        # Analyze BTC vs Alt (BTC.D)
        if btc_d > self.btc_rally_threshold:
            btc_regime = 'BTC_RALLY'
            btc_description = 'Bitcoin rallying - favor BTC over alts'
            asset_preference = {
                'BTC': 0.60,
                'ETH': 0.25,
                'SOL': 0.15
            }
            regime_emoji = '₿'
        elif btc_d < self.btc_alt_season:
            btc_regime = 'ALT_SEASON'
            btc_description = 'Alt season - favor ETH/SOL over BTC'
            asset_preference = {
                'BTC': 0.25,
                'ETH': 0.35,
                'SOL': 0.40
            }
            regime_emoji = '🚀'
        else:
            btc_regime = 'BALANCED'
            btc_description = 'Balanced market - diversified allocation'
            asset_preference = {
                'BTC': 0.45,
                'ETH': 0.30,
                'SOL': 0.25
            }
            regime_emoji = '⚖️'
        
        # Determine overall market phase
        if fear_state == 'HIGH_FEAR':
            market_phase = 'DEFENSIVE'
            phase_description = 'Reduce positions and preserve capital'
            action = 'REDUCE_EXPOSURE'
        elif fear_state == 'GREED' and btc_regime == 'ALT_SEASON':
            market_phase = 'AGGRESSIVE_ALT'
            phase_description = 'Maximum exposure to altcoins'
            action = 'FULL_ALT_ALLOCATION'
        elif fear_state == 'GREED' and btc_regime == 'BTC_RALLY':
            market_phase = 'AGGRESSIVE_BTC'
            phase_description = 'Maximum exposure to Bitcoin'
            action = 'FULL_BTC_ALLOCATION'
        else:
            market_phase = 'MODERATE'
            phase_description = 'Moderate allocation with diversification'
            action = 'BALANCED_ALLOCATION'
        
        return {
            # Fear/Greed Analysis
            'fear_state': fear_state,
            'fear_description': fear_description,
            'fear_emoji': fear_emoji,
            'position_modifier': position_modifier,
            
            # BTC/Alt Analysis
            'btc_regime': btc_regime,
            'btc_description': btc_description,
            'regime_emoji': regime_emoji,
            'asset_preference': asset_preference,
            
            # Overall Market Phase
            'market_phase': market_phase,
            'phase_description': phase_description,
            'action': action,
            
            # Current Values
            'usdt_d': usdt_d,
            'btc_d': btc_d
        }
    
    def adjust_allocation(self, 
                         base_allocation: Dict[str, float], 
                         regime: Dict) -> Dict[str, float]:
        """
        Adjust portfolio allocation based on market regime
        
        Args:
            base_allocation: Original allocation from signals
            regime: Market regime analysis
            
        Returns:
            Adjusted allocation dict
        """
        
        # Start with base allocation
        adjusted = base_allocation.copy()
        
        # Get position modifier (overall market exposure)
        position_mod = regime['position_modifier']
        
        # Get asset preferences (how to distribute within crypto)
        preferences = regime['asset_preference']
        
        # Calculate total crypto allocation (everything except CASH)
        total_crypto = sum(v for k, v in adjusted.items() if k != 'CASH')
        
        if total_crypto > 0:
            # Apply position modifier to overall exposure
            adjusted_crypto_total = total_crypto * position_mod
            
            # Redistribute within crypto based on regime preferences
            for asset in ['BTC', 'ETH', 'SOL']:
                if asset in adjusted and adjusted[asset] > 0:
                    # Asset gets its share of the adjusted crypto allocation
                    adjusted[asset] = adjusted_crypto_total * preferences[asset]
            
            # Remaining goes to cash
            adjusted['CASH'] = 1.0 - sum(v for k, v in adjusted.items() if k != 'CASH')
        
        return adjusted
    
    def get_regime_summary(self) -> Dict:
        """
        Get complete market regime analysis with current dominance values
        
        Returns:
            Complete regime analysis
        """
        
        # Get current dominance
        dominance = self.get_current_dominance()
        
        # Analyze regime
        regime = self.analyze_market_regime(
            dominance['USDT.D'],
            dominance['BTC.D']
        )
        
        return regime
    
    def detect_extreme_market(self, asset: str = 'BTC') -> Dict:
        """
        Detect if we're in an extreme market regime
        
        Based on Moskowitz, Ooi, Pedersen (2012) finding:
        "Momentum strategies perform BEST during extreme markets"
        
        Args:
            asset: Asset to check (BTC, ETH, SOL)
        
        Returns:
            Dict with extreme market analysis
        """
        try:
            import pandas as pd
            
            # Load recent data to calculate volatility
            data_file = f'DATA/yf_{asset.lower()}_12h.csv'
            df = pd.read_csv(data_file)
            
            # Calculate recent volatility
            df['returns'] = df['close'].pct_change()
            
            # 30-day rolling volatility (720 hours)
            vol_30d = df['returns'].rolling(720).std() * np.sqrt(24 * 365)
            
            # Current volatility percentile
            current_vol = vol_30d.iloc[-1]
            vol_percentile = (vol_30d <= current_vol).sum() / len(vol_30d.dropna())
            
            # Extreme market if in top 10% volatility
            is_extreme = vol_percentile > 0.9
            
            # Volume surge check (if available)
            if 'volume' in df.columns:
                vol_ma = df['volume'].rolling(720).mean()
                volume_surge = df['volume'].iloc[-1] > (vol_ma.iloc[-1] * 1.5)
            else:
                volume_surge = False
            
            # Determine boost level
            if is_extreme and volume_surge:
                boost_level = 1.2  # 20% boost
                confidence = 'HIGH'
            elif is_extreme:
                boost_level = 1.15  # 15% boost
                confidence = 'MEDIUM'
            elif vol_percentile > 0.8:
                boost_level = 1.1  # 10% boost
                confidence = 'LOW'
            else:
                boost_level = 1.0  # No boost
                confidence = 'NORMAL'
            
            return {
                'is_extreme': is_extreme,
                'vol_percentile': vol_percentile,
                'current_vol': current_vol,
                'volume_surge': volume_surge,
                'boost_level': boost_level,
                'confidence': confidence,
                'description': (
                    f"Extreme volatility (top {(1-vol_percentile)*100:.1f}%)" 
                    if is_extreme else 
                    f"Normal volatility ({vol_percentile*100:.0f}th percentile)"
                )
            }
        
        except Exception as e:
            # Fallback if data not available
            return {
                'is_extreme': False,
                'vol_percentile': 0.5,
                'current_vol': 0.0,
                'volume_surge': False,
                'boost_level': 1.0,
                'confidence': 'UNKNOWN',
                'description': f'Could not detect (error: {str(e)})'
            }
    
    def adjust_for_extreme_market(self, signal: Dict, asset: str = 'BTC') -> Dict:
        """
        Adjust trading signal based on extreme market regime
        
        Boosts momentum-based signals during extreme volatility
        (Moskowitz et al. 2012 key finding)
        
        Args:
            signal: Trading signal dict with expected_return, position_size, etc.
            asset: Asset name
        
        Returns:
            Adjusted signal
        """
        extreme_analysis = self.detect_extreme_market(asset)
        
        if extreme_analysis['is_extreme']:
            # Boost position size (momentum works best in extreme markets)
            signal['position_size'] *= extreme_analysis['boost_level']
            
            # Boost expected return estimate
            signal['expected_return'] *= (1 + (extreme_analysis['boost_level'] - 1) * 0.5)
            
            # Add extreme market note
            signal['extreme_market'] = True
            signal['extreme_boost'] = extreme_analysis['boost_level']
            signal['extreme_confidence'] = extreme_analysis['confidence']
            signal['extreme_note'] = (
                f"EXTREME MARKET BOOST: {(extreme_analysis['boost_level']-1)*100:.0f}% "
                f"(Moskowitz et al. 2012 - momentum best in extremes)"
            )
        else:
            signal['extreme_market'] = False
            signal['extreme_boost'] = 1.0
        
        return signal
    
    def print_regime_analysis(self, regime: Dict = None) -> None:
        """Print formatted regime analysis"""
        
        if regime is None:
            regime = self.get_regime_summary()
        
        print()
        print("="*80)
        print("📊 MARKET REGIME ANALYSIS (DOMINANCE-BASED)")
        print("="*80)
        
        print()
        print(f"📈 Current Dominance Metrics:")
        print(f"   • USDT.D: {regime['usdt_d']:.2f}%")
        print(f"   • BTC.D:  {regime['btc_d']:.2f}%")
        
        print()
        print(f"{regime['fear_emoji']} Fear/Greed State: {regime['fear_state']}")
        print(f"   {regime['fear_description']}")
        print(f"   Position Modifier: {regime['position_modifier']*100:.0f}%")
        
        print()
        print(f"{regime['regime_emoji']} BTC/Alt Regime: {regime['btc_regime']}")
        print(f"   {regime['btc_description']}")
        print(f"   Preferred Allocation:")
        for asset, pct in regime['asset_preference'].items():
            print(f"   • {asset}: {pct*100:.0f}%")
        
        print()
        print(f"🎯 Market Phase: {regime['market_phase']}")
        print(f"   {regime['phase_description']}")
        print(f"   Recommended Action: {regime['action']}")
        
        print("="*80)


def main():
    """Test dominance analyzer"""
    
    analyzer = DominanceAnalyzer()
    
    # Get current regime
    regime = analyzer.get_regime_summary()
    
    # Print analysis
    analyzer.print_regime_analysis(regime)
    
    # Example: Adjust a sample allocation
    print()
    print("="*80)
    print("💡 ALLOCATION ADJUSTMENT EXAMPLE")
    print("="*80)
    
    # Sample base allocation (from signals)
    base_allocation = {
        'BTC': 0.45,
        'ETH': 0.30,
        'SOL': 0.25,
        'CASH': 0.0
    }
    
    print()
    print("Original Allocation (from signals):")
    for asset, pct in base_allocation.items():
        if pct > 0:
            print(f"   {asset}: {pct*100:.1f}%")
    
    # Adjust based on regime
    adjusted = analyzer.adjust_allocation(base_allocation, regime)
    
    print()
    print("Adjusted Allocation (with dominance intelligence):")
    for asset, pct in adjusted.items():
        if pct > 0:
            emoji = "📈" if asset != 'CASH' else "💰"
            print(f"   {emoji} {asset}: {pct*100:.1f}%")
    
    print()
    print("="*80)
    print("✅ Dominance Analyzer Test Complete!")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
