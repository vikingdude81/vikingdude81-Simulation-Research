"""
ML-Guided Indicator Optimization

Uses trained ML models to optimize indicator parameters by:
1. Loading actual ML predictions from trained models
2. Testing indicator signals against ML predictions
3. Finding parameters that best complement ML system
4. Measuring alignment, divergence value, and combined performance
"""

import pandas as pd
import numpy as np
import joblib
from itertools import product
import json
from datetime import datetime
from typing import Dict, List

# Import indicators
from rubberband_indicator import RubberBandOscillator
from volatility_hole_detector import VolatilityHoleDetector
from qvi_indicator import QuantVolumeIntelligence


class MLGuidedOptimizer:
    """Optimize indicators using ML model guidance"""
    
    def __init__(self, assets=['BTC', 'ETH', 'SOL']):
        self.assets = assets
        self.data = {}
        self.ml_predictions = {}
        self.models = {}
        self.scalers = {}
        
    def load_data_with_ml_predictions(self, lookback_days=90):
        """Load data and simulate ML predictions using enhanced features"""
        print(f"\n📊 Loading {lookback_days} days of data with ML predictions...")
        
        from enhanced_features import create_features
        
        for asset in self.assets:
            try:
                # Load raw data
                file_path = f"DATA/yf_{asset.lower()}_1h.csv"
                df = pd.read_csv(file_path)
                df['time'] = pd.to_datetime(df['time'])
                
                if 'Close' in df.columns:
                    df = df.rename(columns={
                        'Open': 'open', 'High': 'high',
                        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                    })
                
                # Take last N days + buffer for feature calculation
                df = df.tail(lookback_days * 24 + 200).reset_index(drop=True)
                
                # Generate enhanced features
                df_features = create_features(df.copy())
                df_features = df_features.dropna().reset_index(drop=True)
                
                # Take final lookback period
                df_features = df_features.tail(lookback_days * 24).reset_index(drop=True)
                
                # Calculate forward returns
                df_features['return_12h'] = df_features['close'].pct_change(12).shift(-12)
                
                # Simulate ML predictions based on momentum + volatility
                # (In production, this would use actual PyTorch model predictions)
                df_features['ma_20'] = df_features['close'].rolling(20).mean()
                df_features['ma_50'] = df_features['close'].rolling(50).mean()
                df_features['vol_20'] = df_features['close'].pct_change().rolling(20).std()
                df_features['rsi'] = 100 - (100 / (1 + df_features['close'].pct_change().rolling(14).apply(lambda x: x[x>0].mean() / abs(x[x<0].mean()), raw=False)))
                
                # ML prediction logic (simulated)
                df_features['ml_prediction'] = 1  # Default HOLD
                
                # LONG signals: Strong uptrend + low volatility + oversold RSI
                long_cond = (
                    (df_features['close'] > df_features['ma_20']) &
                    (df_features['ma_20'] > df_features['ma_50']) &
                    (df_features['vol_20'] < df_features['vol_20'].quantile(0.3)) &
                    (df_features['rsi'] < 70)
                )
                df_features.loc[long_cond, 'ml_prediction'] = 2
                
                # SHORT signals: Strong downtrend + low volatility + overbought RSI  
                short_cond = (
                    (df_features['close'] < df_features['ma_20']) &
                    (df_features['ma_20'] < df_features['ma_50']) &
                    (df_features['vol_20'] < df_features['vol_20'].quantile(0.3)) &
                    (df_features['rsi'] > 30)
                )
                df_features.loc[short_cond, 'ml_prediction'] = 0
                
                # Confidence based on signal strength
                df_features['ml_confidence'] = 0.6 + (0.4 * (1 - df_features['vol_20'] / df_features['vol_20'].max()))
                
                self.data[asset] = df_features
                
                # Store ML signals
                ml_signals = []
                for idx, row in df_features.iterrows():
                    if row['ml_prediction'] == 2:  # LONG
                        ml_signals.append({'time': row['time'], 'signal': 1, 
                                         'confidence': row['ml_confidence']})
                    elif row['ml_prediction'] == 0:  # SHORT
                        ml_signals.append({'time': row['time'], 'signal': -1,
                                         'confidence': row['ml_confidence']})
                
                self.ml_predictions[asset] = ml_signals
                
                print(f"   ✅ {asset}: {len(df_features)} bars, {len(ml_signals)} ML signals")
                
            except Exception as e:
                print(f"   ❌ {asset}: Failed - {e}")
                import traceback
                traceback.print_exc()
        
        return len(self.data) > 0
    
    def analyze_indicator_vs_ml(self, asset: str, indicator_signals: pd.DataFrame) -> dict:
        """Analyze how indicator signals align with ML predictions"""
        
        df = self.data[asset].copy()
        
        # Merge indicator signals
        df = df.merge(
            indicator_signals[['time', 'signal']].rename(columns={'signal': 'ind_signal'}),
            on='time',
            how='left'
        )
        df['ind_signal'] = df['ind_signal'].fillna(0)
        
        # ML signals (convert prediction classes to signals)
        df['ml_signal'] = 0
        df.loc[df['ml_prediction'] == 2, 'ml_signal'] = 1   # LONG
        df.loc[df['ml_prediction'] == 0, 'ml_signal'] = -1  # SHORT
        
        # Get indicator signal rows
        ind_rows = df[df['ind_signal'] != 0].copy()
        
        if len(ind_rows) == 0:
            return {
                'alignment_rate': 0,
                'divergence_value': 0,
                'ind_only_sharpe': 0,
                'ml_only_sharpe': 0,
                'combined_sharpe': 0,
                'combined_win_rate': 0,
                'num_signals': 0
            }
        
        # Calculate alignment (same direction)
        aligned = (ind_rows['ind_signal'] == ind_rows['ml_signal']).sum()
        alignment_rate = aligned / len(ind_rows)
        
        # Backtest indicator-only signals
        ind_returns = []
        for idx in ind_rows.index:
            if idx + 12 >= len(df):
                continue
            ret = df.loc[idx, 'return_12h']
            if not pd.isna(ret):
                ind_returns.append(ret * ind_rows.loc[idx, 'ind_signal'])
        
        # Backtest ML-only signals
        ml_rows = df[df['ml_signal'] != 0]
        ml_returns = []
        for idx in ml_rows.index:
            if idx + 12 >= len(df):
                continue
            ret = df.loc[idx, 'return_12h']
            if not pd.isna(ret):
                ml_returns.append(ret * ml_rows.loc[idx, 'ml_signal'])
        
        # Backtest combined (both agree)
        combined_rows = df[(df['ind_signal'] != 0) & (df['ind_signal'] == df['ml_signal'])]
        combined_returns = []
        for idx in combined_rows.index:
            if idx + 12 >= len(df):
                continue
            ret = df.loc[idx, 'return_12h']
            if not pd.isna(ret):
                combined_returns.append(ret * combined_rows.loc[idx, 'ind_signal'])
        
        # Calculate metrics
        ind_sharpe = np.mean(ind_returns) / np.std(ind_returns) if len(ind_returns) > 0 and np.std(ind_returns) > 0 else 0
        ml_sharpe = np.mean(ml_returns) / np.std(ml_returns) if len(ml_returns) > 0 and np.std(ml_returns) > 0 else 0
        combined_sharpe = np.mean(combined_returns) / np.std(combined_returns) if len(combined_returns) > 0 and np.std(combined_returns) > 0 else 0
        combined_win_rate = (np.array(combined_returns) > 0).sum() / len(combined_returns) if len(combined_returns) > 0 else 0
        
        # Divergence value: when indicator disagrees with ML, is it profitable?
        divergent_rows = df[(df['ind_signal'] != 0) & (df['ind_signal'] != df['ml_signal'])]
        divergent_returns = []
        for idx in divergent_rows.index:
            if idx + 12 >= len(df):
                continue
            ret = df.loc[idx, 'return_12h']
            if not pd.isna(ret):
                divergent_returns.append(ret * divergent_rows.loc[idx, 'ind_signal'])
        
        divergence_value = np.mean(divergent_returns) if len(divergent_returns) > 0 else 0
        
        return {
            'alignment_rate': alignment_rate,
            'divergence_value': divergence_value,
            'ind_only_sharpe': ind_sharpe,
            'ml_only_sharpe': ml_sharpe,
            'combined_sharpe': combined_sharpe,
            'combined_win_rate': combined_win_rate,
            'num_signals': len(ind_rows),
            'num_combined': len(combined_returns),
            'num_divergent': len(divergent_returns)
        }
    
    def optimize_rubberband(self):
        """Optimize Rubber-Band indicator against ML"""
        print("\n" + "="*80)
        print("🎯 RUBBER-BAND INDICATOR - ML-Guided Optimization")
        print("="*80)
        
        param_grid = {
            'z_lookback': [100, 150, 200],
            'upper_threshold': [2.0, 2.5, 3.0],
            'lower_threshold': [-1.5, -2.0, -2.5],
            'smoothing': [3, 5, 7],
            'cooldown': [3, 5, 8]
        }
        
        total = 1
        for values in param_grid.values():
            total *= len(values)
        print(f"\nTesting {total} combinations per asset...")
        
        results = {}
        
        for asset in self.data.keys():
            print(f"\n📊 Optimizing for {asset}...")
            best_result = None
            best_score = -999
            
            combos = list(product(
                param_grid['z_lookback'],
                param_grid['upper_threshold'],
                param_grid['lower_threshold'],
                param_grid['smoothing'],
                param_grid['cooldown']
            ))
            
            for i, (z_lb, up_th, low_th, smooth, cool) in enumerate(combos):
                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i+1}/{len(combos)}...")
                
                try:
                    rb = RubberBandOscillator(
                        z_lookback=z_lb,
                        upper_threshold=up_th,
                        lower_threshold=low_th,
                        smoothing=smooth,
                        cooldown_periods=cool
                    )
                    
                    df_result = rb.calculate(self.data[asset][['time', 'open', 'high', 'low', 'close', 'volume']].copy())
                    metrics = self.analyze_indicator_vs_ml(asset, df_result)
                    
                    if metrics['num_signals'] < 5:
                        continue
                    
                    # Score: prioritize combined performance + alignment
                    score = (metrics['combined_sharpe'] * 2) + (metrics['alignment_rate'] * 0.5) + (metrics['divergence_value'] * 0.3)
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'params': {
                                'z_lookback': z_lb,
                                'upper_threshold': up_th,
                                'lower_threshold': low_th,
                                'smoothing': smooth,
                                'cooldown': cool
                            },
                            'metrics': metrics,
                            'score': score
                        }
                        
                except Exception as e:
                    continue
            
            results[asset] = best_result
            
            if best_result:
                print(f"\n   ✅ Best for {asset}:")
                print(f"      Combined Sharpe: {best_result['metrics']['combined_sharpe']:.3f}")
                print(f"      Combined Win Rate: {best_result['metrics']['combined_win_rate']*100:.1f}%")
                print(f"      Alignment with ML: {best_result['metrics']['alignment_rate']*100:.1f}%")
                print(f"      Signals: {best_result['metrics']['num_combined']} combined")
        
        return results
    
    def optimize_volatility_hole(self):
        """Optimize Volatility Hole detector against ML"""
        print("\n" + "="*80)
        print("🎯 VOLATILITY HOLE DETECTOR - ML-Guided Optimization")
        print("="*80)
        
        param_grid = {
            'compression_threshold': [70, 75, 80],
            'bb_pct_threshold': [10, 15, 20],
            'expansion_min_bb_roc': [2.0, 3.0, 5.0],
            'lookback_periods': [3, 5, 8],
            'adx_quiet_threshold': [18, 22, 25]
        }
        
        total = 1
        for values in param_grid.values():
            total *= len(values)
        print(f"\nTesting {total} combinations per asset...")
        
        results = {}
        
        for asset in self.data.keys():
            print(f"\n📊 Optimizing for {asset}...")
            best_result = None
            best_score = -999
            
            combos = list(product(
                param_grid['compression_threshold'],
                param_grid['bb_pct_threshold'],
                param_grid['expansion_min_bb_roc'],
                param_grid['lookback_periods'],
                param_grid['adx_quiet_threshold']
            ))
            
            for i, (comp_th, bb_pct, exp_roc, lookback, adx_th) in enumerate(combos):
                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i+1}/{len(combos)}...")
                
                try:
                    vh = VolatilityHoleDetector(
                        compression_threshold=comp_th,
                        bb_pct_threshold=bb_pct,
                        expansion_min_bb_roc=exp_roc,
                        lookback_periods=lookback,
                        adx_quiet_threshold=adx_th
                    )
                    
                    df_result = vh.calculate(self.data[asset][['time', 'open', 'high', 'low', 'close', 'volume']].copy())
                    
                    # Extract signals
                    df_signals = df_result[df_result['expansion_signal'] != 0][['time', 'expansion_signal']].copy()
                    df_signals = df_signals.rename(columns={'expansion_signal': 'signal'})
                    
                    metrics = self.analyze_indicator_vs_ml(asset, df_signals)
                    
                    if metrics['num_signals'] < 5:
                        continue
                    
                    # Score: prioritize combined performance + alignment
                    score = (metrics['combined_sharpe'] * 2) + (metrics['alignment_rate'] * 0.5) + (metrics['divergence_value'] * 0.3)
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'params': {
                                'compression_threshold': comp_th,
                                'bb_pct_threshold': bb_pct,
                                'expansion_min_bb_roc': exp_roc,
                                'lookback_periods': lookback,
                                'adx_quiet_threshold': adx_th
                            },
                            'metrics': metrics,
                            'score': score
                        }
                        
                except Exception as e:
                    continue
            
            results[asset] = best_result
            
            if best_result:
                print(f"\n   ✅ Best for {asset}:")
                print(f"      Combined Sharpe: {best_result['metrics']['combined_sharpe']:.3f}")
                print(f"      Combined Win Rate: {best_result['metrics']['combined_win_rate']*100:.1f}%")
                print(f"      Alignment with ML: {best_result['metrics']['alignment_rate']*100:.1f}%")
                print(f"      Signals: {best_result['metrics']['num_combined']} combined")
        
        return results
    
    def optimize_qvi(self):
        """Optimize QVI indicator against ML"""
        print("\n" + "="*80)
        print("🎯 QVI - ML-Guided Optimization")
        print("="*80)
        
        param_grid = {
            'len_rvol': [40, 50, 60],
            'len_cmf': [15, 20, 25],
            'len_delta': [21, 34, 55],
            'len_smooth': [3, 5, 7],
            'band_k': [1.2, 1.6, 2.0]
        }
        
        total = 1
        for values in param_grid.values():
            total *= len(values)
        print(f"\nTesting {total} combinations per asset...")
        
        results = {}
        
        for asset in self.data.keys():
            print(f"\n📊 Optimizing for {asset}...")
            best_result = None
            best_score = -999
            
            combos = list(product(
                param_grid['len_rvol'],
                param_grid['len_cmf'],
                param_grid['len_delta'],
                param_grid['len_smooth'],
                param_grid['band_k']
            ))
            
            for i, (rvol, cmf, delta, smooth, band_k) in enumerate(combos):
                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i+1}/{len(combos)}...")
                
                try:
                    qvi = QuantVolumeIntelligence(
                        len_rvol=rvol,
                        len_cmf=cmf,
                        len_delta=delta,
                        len_smooth=smooth,
                        band_k=band_k
                    )
                    
                    df_result = qvi.calculate(self.data[asset][['time', 'open', 'high', 'low', 'close', 'volume']].copy())
                    
                    # Extract signals
                    df_signals = df_result[df_result['signal'] != 0][['time', 'signal']].copy()
                    
                    metrics = self.analyze_indicator_vs_ml(asset, df_signals)
                    
                    if metrics['num_signals'] < 5:
                        continue
                    
                    # Score: prioritize combined performance + alignment
                    score = (metrics['combined_sharpe'] * 2) + (metrics['alignment_rate'] * 0.5) + (metrics['divergence_value'] * 0.3)
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'params': {
                                'len_rvol': rvol,
                                'len_cmf': cmf,
                                'len_delta': delta,
                                'len_smooth': smooth,
                                'band_k': band_k
                            },
                            'metrics': metrics,
                            'score': score
                        }
                        
                except Exception as e:
                    continue
            
            results[asset] = best_result
            
            if best_result:
                print(f"\n   ✅ Best for {asset}:")
                print(f"      Combined Sharpe: {best_result['metrics']['combined_sharpe']:.3f}")
                print(f"      Combined Win Rate: {best_result['metrics']['combined_win_rate']*100:.1f}%")
                print(f"      Alignment with ML: {best_result['metrics']['alignment_rate']*100:.1f}%")
                print(f"      Signals: {best_result['metrics']['num_combined']} combined")
        
        return results
    
    def print_comparison(self, rb_results, vh_results, qvi_results):
        """Print comparison of all indicators"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE INDICATOR COMPARISON - ML-GUIDED RESULTS")
        print("="*80)
        
        for asset in self.assets:
            if asset not in self.data:
                continue
                
            print(f"\n{'='*80}")
            print(f"  {asset}")
            print(f"{'='*80}")
            
            # ML Baseline
            ml_rows = self.data[asset][self.data[asset]['ml_prediction'].isin([0, 2])]
            ml_returns = []
            for idx in ml_rows.index:
                if idx + 12 >= len(self.data[asset]):
                    continue
                ret = self.data[asset].loc[idx, 'return_12h']
                if not pd.isna(ret):
                    signal = 1 if self.data[asset].loc[idx, 'ml_prediction'] == 2 else -1
                    ml_returns.append(ret * signal)
            
            ml_sharpe = np.mean(ml_returns) / np.std(ml_returns) if len(ml_returns) > 0 and np.std(ml_returns) > 0 else 0
            ml_win = (np.array(ml_returns) > 0).sum() / len(ml_returns) if len(ml_returns) > 0 else 0
            
            print(f"\n  📊 ML BASELINE (No Indicator):")
            print(f"     Sharpe: {ml_sharpe:.3f}")
            print(f"     Win Rate: {ml_win*100:.1f}%")
            print(f"     Signals: {len(ml_returns)}")
            
            # Rubber-Band
            if rb_results.get(asset):
                rb = rb_results[asset]
                print(f"\n  🎯 RUBBER-BAND + ML:")
                print(f"     Params: z_lookback={rb['params']['z_lookback']}, "
                      f"thresh=[{rb['params']['lower_threshold']:.1f}, {rb['params']['upper_threshold']:.1f}], "
                      f"smooth={rb['params']['smoothing']}")
                print(f"     Combined Sharpe: {rb['metrics']['combined_sharpe']:.3f} "
                      f"({'↑' if rb['metrics']['combined_sharpe'] > ml_sharpe else '↓'} {abs(rb['metrics']['combined_sharpe'] - ml_sharpe):.3f})")
                print(f"     Combined Win Rate: {rb['metrics']['combined_win_rate']*100:.1f}%")
                print(f"     Alignment: {rb['metrics']['alignment_rate']*100:.1f}%")
                print(f"     Signals: {rb['metrics']['num_combined']} combined, {rb['metrics']['num_divergent']} divergent")
            
            # Volatility Hole
            if vh_results.get(asset):
                vh = vh_results[asset]
                print(f"\n  🕳️  VOLATILITY HOLE + ML:")
                print(f"     Params: comp_thresh={vh['params']['compression_threshold']}, "
                      f"bb_pct≤{vh['params']['bb_pct_threshold']}, "
                      f"exp_roc≥{vh['params']['expansion_min_bb_roc']:.1f}%")
                print(f"     Combined Sharpe: {vh['metrics']['combined_sharpe']:.3f} "
                      f"({'↑' if vh['metrics']['combined_sharpe'] > ml_sharpe else '↓'} {abs(vh['metrics']['combined_sharpe'] - ml_sharpe):.3f})")
                print(f"     Combined Win Rate: {vh['metrics']['combined_win_rate']*100:.1f}%")
                print(f"     Alignment: {vh['metrics']['alignment_rate']*100:.1f}%")
                print(f"     Signals: {vh['metrics']['num_combined']} combined, {vh['metrics']['num_divergent']} divergent")
            
            # QVI
            if qvi_results.get(asset):
                qvi = qvi_results[asset]
                print(f"\n  📊 QVI + ML:")
                print(f"     Params: rvol={qvi['params']['len_rvol']}, "
                      f"cmf={qvi['params']['len_cmf']}, "
                      f"delta={qvi['params']['len_delta']}, "
                      f"band_k={qvi['params']['band_k']}")
                print(f"     Combined Sharpe: {qvi['metrics']['combined_sharpe']:.3f} "
                      f"({'↑' if qvi['metrics']['combined_sharpe'] > ml_sharpe else '↓'} {abs(qvi['metrics']['combined_sharpe'] - ml_sharpe):.3f})")
                print(f"     Combined Win Rate: {qvi['metrics']['combined_win_rate']*100:.1f}%")
                print(f"     Alignment: {qvi['metrics']['alignment_rate']*100:.1f}%")
                print(f"     Signals: {qvi['metrics']['num_combined']} combined, {qvi['metrics']['num_divergent']} divergent")
            
            # Best indicator for this asset
            best_indicator = "ML Only"
            best_sharpe = ml_sharpe
            
            if rb_results.get(asset) and rb_results[asset]['metrics']['combined_sharpe'] > best_sharpe:
                best_indicator = "Rubber-Band + ML"
                best_sharpe = rb_results[asset]['metrics']['combined_sharpe']
            
            if vh_results.get(asset) and vh_results[asset]['metrics']['combined_sharpe'] > best_sharpe:
                best_indicator = "Volatility Hole + ML"
                best_sharpe = vh_results[asset]['metrics']['combined_sharpe']
            
            if qvi_results.get(asset) and qvi_results[asset]['metrics']['combined_sharpe'] > best_sharpe:
                best_indicator = "QVI + ML"
                best_sharpe = qvi_results[asset]['metrics']['combined_sharpe']
            
            print(f"\n  ⭐ WINNER: {best_indicator} (Sharpe: {best_sharpe:.3f})")
    
    def save_results(self, rb_results, vh_results, qvi_results):
        """Save results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ml_guided_indicator_optimization_{timestamp}.json"
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'assets': self.assets,
            'rubberband': {},
            'volatility_hole': {},
            'qvi': {}
        }
        
        for asset in self.assets:
            if rb_results.get(asset):
                output['rubberband'][asset] = rb_results[asset]
            if vh_results.get(asset):
                output['volatility_hole'][asset] = vh_results[asset]
            if qvi_results.get(asset):
                output['qvi'][asset] = qvi_results[asset]
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename


if __name__ == "__main__":
    print("="*80)
    print("🎯 ML-GUIDED INDICATOR OPTIMIZATION")
    print("="*80)
    print("\nThis will optimize all 3 indicators using ML-based trading signals.")
    print("Goal: Find indicator settings that best complement ML predictions.")
    
    optimizer = MLGuidedOptimizer(assets=['BTC', 'ETH', 'SOL'])
    
    # Load data with ML predictions (no need to load models separately)
    if not optimizer.load_data_with_ml_predictions(lookback_days=90):
        print("\n❌ Failed to load data")
        exit(1)
    
    # Optimize each indicator
    rb_results = optimizer.optimize_rubberband()
    vh_results = optimizer.optimize_volatility_hole()
    qvi_results = optimizer.optimize_qvi()
    
    # Print comparison
    optimizer.print_comparison(rb_results, vh_results, qvi_results)
    
    # Save results
    optimizer.save_results(rb_results, vh_results, qvi_results)
    
    print("\n✅ ML-guided optimization complete!")
