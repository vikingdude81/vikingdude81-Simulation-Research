"""
Multi-Asset Trading Signal Generator
Combines BTC, ETH, and SOL predictions to generate portfolio allocation signals
Enhanced with UCB (Upper Confidence Bound) reinforcement learning for adaptive allocation
Enhanced with support/resistance technical analysis (Phase D)
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import UCB asset selector (replaces fixed dominance rules)
from ucb_asset_selector import UCBAssetSelector
# Import dominance analyzer (kept for market context)
from dominance_analyzer import DominanceAnalyzer
# Import S/R analyzer (Phase D)
from support_resistance import SupportResistanceAnalyzer


class MultiAssetSignalGenerator:
    """
    Generate trading signals for a multi-asset portfolio (BTC, ETH, SOL)
    
    Features:
    - Individual asset signal generation
    - Portfolio allocation based on confidence and volatility
    - Risk-adjusted position sizing
    - Diversification benefits
    """
    
    def __init__(self, 
                 base_threshold: float = 0.003,
                 confidence_threshold: float = 0.002,
                 max_position_per_asset: float = 0.45,
                 min_position_per_asset: float = 0.10,
                 use_ucb: bool = True,
                 use_dominance: bool = True,
                 use_support_resistance: bool = True):
        """
        Initialize multi-asset signal generator
        
        Args:
            base_threshold: Base percentage move to generate signal (0.3%)
            confidence_threshold: Confidence band for signal strength (0.2%)
            max_position_per_asset: Maximum allocation to single asset (45%)
            min_position_per_asset: Minimum allocation to active asset (10%)
            use_ucb: Whether to use UCB reinforcement learning for allocation
            use_dominance: Whether to use dominance for market context
            use_support_resistance: Whether to use S/R technical analysis (Phase D)
        """
        self.base_threshold = base_threshold
        self.confidence_threshold = confidence_threshold
        self.max_position = max_position_per_asset
        self.min_position = min_position_per_asset
        self.use_ucb = use_ucb
        self.use_dominance = use_dominance
        self.use_sr = use_support_resistance
        
        # Initialize UCB asset selector (replaces fixed dominance allocation)
        self.ucb_selector = UCBAssetSelector(
            assets=['BTC', 'ETH', 'SOL'],
            exploration_param=1.5,  # Balanced exploration/exploitation
            persistence_file='ucb_state.json'
        ) if use_ucb else None
        
        # Initialize dominance analyzer (for market context, not allocation)
        self.dominance_analyzer = DominanceAnalyzer() if use_dominance else None
        
        # Initialize S/R analyzer (Phase D)
        self.sr_analyzer = SupportResistanceAnalyzer() if use_support_resistance else None
        
        # Asset characteristics (from training results)
        self.asset_info = {
            'BTC': {
                'name': 'Bitcoin',
                'rmse': 0.0045,  # 0.45%
                'volatility': 'LOW',
                'volatility_score': 1.0,
                'risk_weight': 1.0
            },
            'ETH': {
                'name': 'Ethereum',
                'rmse': 0.0091,  # 0.91%
                'volatility': 'MEDIUM',
                'volatility_score': 2.0,
                'risk_weight': 0.85
            },
            'SOL': {
                'name': 'Solana',
                'rmse': 0.0101,  # 1.01%
                'volatility': 'HIGH',
                'volatility_score': 2.5,
                'risk_weight': 0.70
            }
        }
    
    def load_predictions(self, asset: str) -> Optional[pd.DataFrame]:
        """Load latest predictions for an asset"""
        try:
            # For BTC, use the standard predictions_forecast.csv
            if asset == 'BTC' and Path('predictions_forecast.csv').exists():
                df = pd.read_csv('predictions_forecast.csv')
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                # Add current actual price from data file
                data_file = f'DATA/yf_{asset.lower()}_1h.csv'
                if Path(data_file).exists():
                    data_df = pd.read_csv(data_file)
                    df['current_actual_price'] = data_df['close'].iloc[-1]
                return df
            
            # For ETH and SOL, find prediction files from their training runs
            # Look for the most recent file
            prediction_files = list(Path('MODEL_STORAGE/predictions').glob(f'*predictions*.csv'))
            
            if not prediction_files:
                print(f"⚠️  No prediction files found")
                return None
            
            # Sort by modification time
            prediction_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # For ETH and SOL, we need to match based on recency and check the price range
            # ETH: ~$3900-4000, SOL: ~$190-200
            for pred_file in prediction_files[:5]:  # Check last 5 files
                df = pd.read_csv(pred_file)
                
                if 'price' in df.columns:
                    avg_price = df['price'].mean()
                    
                    # Match based on price range
                    if asset == 'ETH' and 2000 < avg_price < 6000:
                        if 'time' in df.columns:
                            df['time'] = pd.to_datetime(df['time'])
                        # Add current actual price
                        data_file = f'DATA/yf_{asset.lower()}_1h.csv'
                        if Path(data_file).exists():
                            data_df = pd.read_csv(data_file)
                            df['current_actual_price'] = data_df['close'].iloc[-1]
                        return df
                    
                    elif asset == 'SOL' and 50 < avg_price < 500:
                        if 'time' in df.columns:
                            df['time'] = pd.to_datetime(df['time'])
                        # Add current actual price
                        data_file = f'DATA/yf_{asset.lower()}_1h.csv'
                        if Path(data_file).exists():
                            data_df = pd.read_csv(data_file)
                            df['current_actual_price'] = data_df['close'].iloc[-1]
                        return df
            
            print(f"⚠️  Could not find predictions for {asset}")
            return None
            
        except Exception as e:
            print(f"⚠️  Error loading predictions for {asset}: {e}")
            return None
    
    def calculate_asset_signal(self, df: pd.DataFrame, asset: str) -> Dict:
        """
        Calculate trading signal for a single asset
        
        Returns:
            Dictionary with signal, confidence, and metrics
        """
        if df is None or len(df) == 0:
            return {
                'signal': 'HOLD',
                'confidence': 'NONE',
                'position_size': 0,
                'expected_return': 0,
                'reason': 'No data available'
            }
        
        # Get current actual price if available, otherwise use first prediction
        if 'current_actual_price' in df.columns:
            current_price = df['current_actual_price'].iloc[0]
        elif 'Most_Likely_Price' in df.columns:  # BTC forecast format
            current_price = df['Most_Likely_Price'].iloc[0]
        elif 'price' in df.columns:
            current_price = df['price'].iloc[0]
        elif 'predicted' in df.columns:
            current_price = df['predicted'].iloc[0]
        else:
            return {
                'signal': 'HOLD',
                'confidence': 'NONE',
                'position_size': 0,
                'expected_return': 0,
                'reason': 'No price column found'
            }
        
        # Get 12-hour prediction (or closest available)
        if 'Most_Likely_Price' in df.columns:  # BTC forecast format
            price_col = 'Most_Likely_Price'
        elif 'price' in df.columns:
            price_col = 'price'
        else:
            price_col = 'predicted'
            
        if len(df) >= 12:
            predicted_price = df[price_col].iloc[11]
        else:
            predicted_price = df[price_col].iloc[-1]
        
        # Calculate expected return
        expected_return = (predicted_price - current_price) / current_price
        
        # Get confidence band (use std of predictions as proxy)
        if 'lower_bound' in df.columns and 'upper_bound' in df.columns:
            idx = min(11, len(df)-1)
            confidence_band = (df['upper_bound'].iloc[idx] - df['lower_bound'].iloc[idx]) / current_price
        elif 'Worst_Case_Price' in df.columns and 'Best_Case_Price' in df.columns:
            idx = min(11, len(df)-1)
            confidence_band = (df['Best_Case_Price'].iloc[idx] - df['Worst_Case_Price'].iloc[idx]) / current_price
        else:
            # Estimate based on RMSE
            confidence_band = self.asset_info[asset]['rmse'] * 2
        
        # Adjust threshold based on asset characteristics
        asset_threshold = self.base_threshold * self.asset_info[asset]['volatility_score']
        
        # Determine signal
        if expected_return > asset_threshold:
            signal = 'BUY'
        elif expected_return < -asset_threshold:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        # Determine confidence level
        confidence_ratio = abs(expected_return) / confidence_band if confidence_band > 0 else 0
        
        if confidence_ratio > 2.0:
            confidence = 'HIGH'
            confidence_score = 1.0
        elif confidence_ratio > 1.0:
            confidence = 'MEDIUM'
            confidence_score = 0.6
        else:
            confidence = 'LOW'
            confidence_score = 0.3
        
        # Calculate position size (0-100%)
        if signal == 'HOLD':
            position_size = 0
        else:
            # Base position on confidence and expected return
            base_size = min(abs(expected_return) / self.base_threshold * 0.3, 1.0)
            position_size = base_size * confidence_score * self.asset_info[asset]['risk_weight']
        
        # ✅ PHASE D: Support/Resistance Enhancement
        sr_data = None
        if self.use_sr and self.sr_analyzer and signal != 'HOLD':
            try:
                # Get S/R analysis for this asset
                sr_enhanced = self.sr_analyzer.enhance_signal(
                    asset=asset,
                    signal=signal,
                    current_price=current_price,
                    expected_return=expected_return,
                    timeframe='1h'
                )
                
                # Apply proximity bonus to expected return
                if sr_enhanced.get('proximity_bonus', 0) > 0:
                    enhanced_return = sr_enhanced['enhanced_return']
                    proximity_bonus = sr_enhanced['proximity_bonus']
                    
                    # Also boost position size slightly
                    position_size = min(position_size * 1.1, 1.0)
                    
                    # Store S/R data
                    sr_data = {
                        'proximity_bonus': proximity_bonus,
                        'enhanced_return': enhanced_return,
                        'nearest_support': sr_enhanced.get('nearest_support'),
                        'nearest_resistance': sr_enhanced.get('nearest_resistance'),
                        'stop_loss': sr_enhanced.get('stop_loss'),
                        'take_profit': sr_enhanced.get('take_profit'),
                        'risk_reward_ratio': sr_enhanced.get('risk_reward_ratio')
                    }
                    
                    # Use enhanced return for display
                    expected_return = enhanced_return
            except Exception as e:
                print(f"⚠️  S/R analysis failed for {asset}: {e}")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'confidence_score': confidence_score,
            'position_size': position_size,
            'expected_return': expected_return,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': predicted_price - current_price,
            'confidence_band': confidence_band,
            'reason': f"{expected_return*100:.2f}% expected move (threshold: {asset_threshold*100:.2f}%)",
            'sr_analysis': sr_data  # Include S/R data in output
        }
    
    def calculate_portfolio_allocation(self, signals: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate optimal portfolio allocation based on individual signals
        
        Strategy:
        - Allocate based on signal strength (confidence + expected return)
        - Apply volatility adjustments (lower vol = higher allocation)
        - Ensure diversification (max 45% per asset)
        - Reserve cash for HOLD/SELL signals
        """
        allocation = {'BTC': 0.0, 'ETH': 0.0, 'SOL': 0.0, 'CASH': 1.0}
        
        # Calculate signal strength scores
        scores = {}
        for asset, signal_data in signals.items():
            if signal_data['signal'] == 'BUY':
                # Score = expected_return * confidence * risk_weight
                score = (
                    abs(signal_data['expected_return']) * 
                    signal_data['confidence_score'] * 
                    self.asset_info[asset]['risk_weight']
                )
                scores[asset] = max(score, 0)
            else:
                scores[asset] = 0
        
        total_score = sum(scores.values())
        
        if total_score == 0:
            # No buy signals - stay in cash
            return allocation
        
        # Normalize scores to allocations
        total_allocated = 0
        for asset, score in scores.items():
            if score > 0:
                # Allocate proportionally
                raw_allocation = (score / total_score)
                
                # Apply constraints
                allocation[asset] = np.clip(
                    raw_allocation,
                    self.min_position if raw_allocation > 0 else 0,
                    self.max_position
                )
                total_allocated += allocation[asset]
        
        # Normalize to ensure total <= 1.0
        if total_allocated > 1.0:
            scale_factor = 1.0 / total_allocated
            for asset in ['BTC', 'ETH', 'SOL']:
                allocation[asset] *= scale_factor
            total_allocated = 1.0
        
        # Remaining goes to cash
        allocation['CASH'] = 1.0 - total_allocated
        
        return allocation
    
    def generate_portfolio_signal(self) -> Dict:
        """
        Generate complete portfolio signal with allocation recommendations
        
        Returns:
            Dictionary containing:
            - Individual asset signals
            - Portfolio allocation
            - Expected portfolio return
            - Risk metrics
            - Trading recommendations
        """
        print("\n" + "="*80)
        print("🎯 MULTI-ASSET PORTFOLIO SIGNAL GENERATOR")
        print("="*80)
        print(f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Load predictions for all assets
        print("📊 Loading predictions for all assets...")
        predictions = {}
        for asset in ['BTC', 'ETH', 'SOL']:
            print(f"   Loading {asset}...", end=" ")
            df = self.load_predictions(asset)
            if df is not None:
                predictions[asset] = df
                print("✅")
            else:
                print("❌")
        
        if len(predictions) == 0:
            print("\n❌ ERROR: No predictions available for any asset")
            return {}
        
        print(f"\n✅ Loaded predictions for {len(predictions)} assets")
        print()
        
        # Generate individual signals
        print("🔍 Analyzing individual asset signals...")
        print()
        signals = {}
        for asset, df in predictions.items():
            signal_data = self.calculate_asset_signal(df, asset)
            signals[asset] = signal_data
            
            # Print asset signal
            signal_emoji = "🟢" if signal_data['signal'] == 'BUY' else "🔴" if signal_data['signal'] == 'SELL' else "🟡"
            print(f"{signal_emoji} {asset} ({self.asset_info[asset]['name']})")
            print(f"   Signal: {signal_data['signal']} ({signal_data['confidence']} confidence)")
            print(f"   Current: ${signal_data['current_price']:.2f}")
            print(f"   Predicted (12h): ${signal_data['predicted_price']:.2f}")
            print(f"   Expected Return: {signal_data['expected_return']*100:+.2f}%")
            print(f"   Position Size: {signal_data['position_size']*100:.1f}%")
            print(f"   Reason: {signal_data['reason']}")
            
            # ✅ PHASE D: Display S/R analysis if available
            if signal_data.get('sr_analysis'):
                sr = signal_data['sr_analysis']
                if sr.get('proximity_bonus', 0) > 0:
                    print(f"   🎯 S/R Bonus: +{sr['proximity_bonus']*100:.2f}% (near support/resistance)")
                if sr.get('nearest_support'):
                    print(f"   🛡️  Support: ${sr['nearest_support']:.2f}")
                if sr.get('nearest_resistance'):
                    print(f"   🚧 Resistance: ${sr['nearest_resistance']:.2f}")
                if sr.get('stop_loss'):
                    risk_pct = abs(signal_data['current_price'] - sr['stop_loss']) / signal_data['current_price'] * 100
                    print(f"   🛑 Stop-Loss: ${sr['stop_loss']:.2f} ({risk_pct:.2f}% risk)")
                if sr.get('take_profit'):
                    reward_pct = abs(sr['take_profit'] - signal_data['current_price']) / signal_data['current_price'] * 100
                    print(f"   🎯 Take-Profit: ${sr['take_profit']:.2f} ({reward_pct:.2f}% target)")
                if sr.get('risk_reward_ratio'):
                    rr_emoji = "✅" if sr['risk_reward_ratio'] >= 2.0 else "⚠️"
                    print(f"   ⚖️  Risk/Reward: 1:{sr['risk_reward_ratio']:.2f} {rr_emoji}")
            
            print()
        
        # Calculate portfolio allocation
        print("💼 Calculating optimal portfolio allocation...")
        base_allocation = self.calculate_portfolio_allocation(signals)
        
        # Apply UCB reinforcement learning or dominance-based adjustments
        regime = None
        
        if self.use_ucb and self.ucb_selector:
            # UCB REINFORCEMENT LEARNING (adaptive allocation)
            print()
            print("🎰 UCB Reinforcement Learning - Adaptive Allocation...")
            
            # Get UCB-based allocation
            allocation = self.ucb_selector.get_allocation_weights(top_n=3)
            
            # Optional: Get dominance context for logging
            if self.dominance_analyzer:
                regime = self.dominance_analyzer.get_regime_summary()
                print(f"   Market Context: {regime['btc_regime']}, {regime['fear_state']}")
            
            print(f"   UCB Selected Allocation:")
            for asset, weight in allocation.items():
                if weight > 0:
                    print(f"      {asset}: {weight*100:.1f}%")
        
        elif self.use_dominance and self.dominance_analyzer:
            # FALLBACK: Dominance-based allocation (original method)
            print()
            print("🔍 Analyzing market regime (dominance-based)...")
            regime = self.dominance_analyzer.get_regime_summary()
            
            # Print regime analysis
            self.dominance_analyzer.print_regime_analysis(regime)
            
            print()
            print("⚙️  Adjusting allocation based on market regime...")
            allocation = self.dominance_analyzer.adjust_allocation(base_allocation, regime)
            
            print()
            print(f"   Position Modifier: {regime['position_modifier']*100:.0f}% (from {regime['fear_state']})")
            print(f"   Asset Preference: {regime['btc_regime']}")
        else:
            # NO ADJUSTMENTS: Use base allocation
            allocation = base_allocation
        
        print()
        print("="*80)
        print("📊 RECOMMENDED PORTFOLIO ALLOCATION")
        print("="*80)
        for asset, pct in allocation.items():
            if pct > 0:
                emoji = "💰" if asset == 'CASH' else "📈"
                print(f"{emoji} {asset:4s}: {pct*100:5.1f}%")
        print("="*80)
        
        # Calculate expected portfolio return
        expected_portfolio_return = sum(
            signals[asset]['expected_return'] * allocation[asset]
            for asset in ['BTC', 'ETH', 'SOL']
        )
        
        print()
        print(f"📈 Expected Portfolio Return (12h): {expected_portfolio_return*100:+.2f}%")
        
        # Annualized (rough estimate)
        periods_per_year = 365 * 2  # 12h periods
        annualized = (1 + expected_portfolio_return) ** periods_per_year - 1
        print(f"📊 Annualized (extrapolated): {annualized*100:+.1f}%")
        
        # Trading recommendations
        print()
        print("="*80)
        print("💡 TRADING RECOMMENDATIONS")
        print("="*80)
        
        active_signals = [a for a in ['BTC', 'ETH', 'SOL'] if signals[a]['signal'] != 'HOLD']
        
        if len(active_signals) == 0:
            print("🟡 HOLD - No strong signals at this time")
            print("   Stay in cash and wait for better opportunities")
        else:
            for asset in active_signals:
                sig = signals[asset]
                if sig['signal'] == 'BUY':
                    print(f"🟢 BUY {asset}")
                    print(f"   Allocate {allocation[asset]*100:.1f}% of portfolio")
                    print(f"   Target: ${sig['predicted_price']:.2f} (+{sig['expected_return']*100:.2f}%)")
                    print(f"   Confidence: {sig['confidence']}")
                elif sig['signal'] == 'SELL':
                    print(f"🔴 SELL/SHORT {asset}")
                    print(f"   Expected decline: {sig['expected_return']*100:.2f}%")
                    print(f"   Consider reducing exposure or shorting")
                print()
        
        print("="*80)
        
        # Build complete result
        result = {
            'timestamp': datetime.now().isoformat(),
            'individual_signals': signals,
            'portfolio_allocation': allocation,
            'base_allocation': base_allocation if self.use_dominance else allocation,
            'expected_return_12h': expected_portfolio_return,
            'annualized_return': annualized,
            'active_positions': len(active_signals),
            'diversification': len([a for a in allocation.values() if a > 0.05]),
            'market_regime': regime if regime else None
        }
        
        # Save to JSON
        output_file = 'multi_asset_signal.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n💾 Signal saved to: {output_file}")
        print()
        
        return result
    
    def compare_with_single_asset(self, portfolio_result: Dict) -> None:
        """
        Compare multi-asset portfolio with single-asset (BTC-only) strategy
        """
        print()
        print("="*80)
        print("📊 MULTI-ASSET vs SINGLE-ASSET COMPARISON")
        print("="*80)
        
        btc_only_return = portfolio_result['individual_signals']['BTC']['expected_return']
        portfolio_return = portfolio_result['expected_return_12h']
        
        print(f"\n🏦 BTC-Only Strategy:")
        print(f"   Expected Return: {btc_only_return*100:+.2f}%")
        print(f"   Assets: 1")
        print(f"   Diversification: None")
        
        print(f"\n💼 Multi-Asset Portfolio:")
        print(f"   Expected Return: {portfolio_return*100:+.2f}%")
        print(f"   Active Assets: {portfolio_result['active_positions']}")
        print(f"   Diversification: {portfolio_result['diversification']} positions")
        
        improvement = ((portfolio_return / btc_only_return) - 1) * 100 if btc_only_return != 0 else 0
        
        print()
        if improvement > 5:
            print(f"✅ Multi-asset strategy is {improvement:+.1f}% better!")
        elif improvement < -5:
            print(f"⚠️  Single-asset strategy is {-improvement:+.1f}% better")
        else:
            print(f"➡️  Strategies similar ({improvement:+.1f}% difference)")
        
        print()
        print("💡 Benefits of Multi-Asset:")
        print("   • Diversification reduces risk")
        print("   • Capture opportunities across assets")
        print("   • Lower correlation = smoother returns")
        print("   • Better risk-adjusted returns")
        print("="*80)


def main():
    """Main execution function"""
    
    # Initialize generator with dominance intelligence and S/R analysis
    generator = MultiAssetSignalGenerator(
        base_threshold=0.003,      # 0.3% - optimal from backtesting
        confidence_threshold=0.002, # 0.2%
        max_position_per_asset=0.45,  # 45% max per asset
        min_position_per_asset=0.10,   # 10% min if active
        use_dominance=True,  # ✅ PHASE C: Dominance-based regime detection
        use_support_resistance=True  # ✅ PHASE D: Support/Resistance analysis
    )
    
    # Generate portfolio signal
    result = generator.generate_portfolio_signal()
    
    if result:
        # Compare with single-asset
        generator.compare_with_single_asset(result)
        
        print()
        print("="*80)
        print("✅ MULTI-ASSET SIGNAL GENERATION COMPLETE!")
        print("="*80)
        print()
        print("📁 Output files:")
        print("   • multi_asset_signal.json - Full signal data")
        print()
        print("🚀 Next steps:")
        print("   1. Review allocation recommendations")
        print("   2. Execute trades based on signals")
        print("   3. Run multi-asset backtest to validate")
        print("   4. Add dominance indicators (Phase C)")
        print("   5. Add support/resistance (Phase D)")
        print()
        
        return result
    else:
        print("\n❌ Failed to generate multi-asset signals")
        return None


if __name__ == '__main__':
    main()
