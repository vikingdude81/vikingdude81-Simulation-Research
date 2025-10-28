"""
Parameter Optimizer for QVI (Quant Volume Intelligence)

Finds optimal settings for volume-based signals across BTC/ETH/SOL.
"""

import pandas as pd
import numpy as np
from qvi_indicator import QuantVolumeIntelligence
from itertools import product
import json
from datetime import datetime


class QVIOptimizer:
    """Optimize QVI parameters"""
    
    def __init__(self, assets=['BTC', 'ETH', 'SOL']):
        self.assets = assets
        self.data = {}
        self.results = []
    
    def load_data(self, lookback_days=90):
        """Load historical data"""
        print(f"\n📊 Loading {lookback_days} days of data...")
        
        for asset in self.assets:
            file_path = f"DATA/yf_{asset.lower()}_1h.csv"
            try:
                df = pd.read_csv(file_path)
                df['time'] = pd.to_datetime(df['time'])
                
                if 'Close' in df.columns:
                    df = df.rename(columns={
                        'Open': 'open', 'High': 'high',
                        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                    })
                
                df = df.tail(lookback_days * 24).reset_index(drop=True)
                df['return_12h'] = df['close'].pct_change(12).shift(-12)
                
                self.data[asset] = df
                print(f"   ✅ {asset}: {len(df)} bars")
            except Exception as e:
                print(f"   ❌ {asset}: Failed - {e}")
        
        return len(self.data) > 0
    
    def backtest_signals(self, df: pd.DataFrame) -> dict:
        """Backtest QVI signals"""
        df_signals = df[df['signal'] != 0].copy()
        
        if len(df_signals) == 0:
            return {
                'win_rate': 0, 'avg_return': 0, 'sharpe': 0,
                'total_return': 0, 'num_trades': 0
            }
        
        returns = []
        for idx in df_signals.index:
            if idx + 12 >= len(df):
                continue
            
            entry_price = df.loc[idx, 'close']
            exit_price = df.loc[idx + 12, 'close']
            signal = df.loc[idx, 'signal']
            
            if signal == 1:  # LONG
                ret = (exit_price - entry_price) / entry_price
            else:  # SHORT
                ret = (entry_price - exit_price) / entry_price
            
            returns.append(ret)
        
        if len(returns) == 0:
            return {
                'win_rate': 0, 'avg_return': 0, 'sharpe': 0,
                'total_return': 0, 'num_trades': 0
            }
        
        returns = np.array(returns)
        return {
            'win_rate': (returns > 0).sum() / len(returns),
            'avg_return': returns.mean(),
            'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
            'total_return': (1 + returns).prod() - 1,
            'num_trades': len(returns)
        }
    
    def optimize_grid_search(self, param_grid: dict, target_metric: str = 'sharpe', min_trades: int = 5):
        """Grid search optimization"""
        print(f"\n🔍 Starting Grid Search...")
        print(f"   Target: {target_metric}, Min trades: {min_trades}")
        
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        combinations = list(product(*param_values))
        
        print(f"   Testing {len(combinations)} combinations...")
        
        all_results = []
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{len(combinations)}...")
            
            asset_results = {}
            
            for asset in self.data.keys():
                try:
                    qvi = QuantVolumeIntelligence(**params)
                    df_result = qvi.calculate(self.data[asset])
                    metrics = self.backtest_signals(df_result)
                    asset_results[asset] = metrics
                except Exception as e:
                    asset_results[asset] = {
                        'win_rate': 0, 'avg_return': 0, 'sharpe': 0,
                        'total_return': 0, 'num_trades': 0
                    }
            
            valid_assets = [a for a in asset_results if asset_results[a]['num_trades'] >= min_trades]
            
            if len(valid_assets) > 0:
                universal_metric = np.mean([asset_results[a][target_metric] for a in valid_assets])
            else:
                universal_metric = -999
            
            all_results.append({
                'params': params,
                'asset_results': asset_results,
                'universal_metric': universal_metric,
                'valid_assets': len(valid_assets)
            })
        
        self.results = all_results
        
        # Best per asset
        best_per_asset = {}
        for asset in self.data.keys():
            valid = [r for r in all_results if r['asset_results'][asset]['num_trades'] >= min_trades]
            
            if len(valid) > 0:
                best = max(valid, key=lambda x: x['asset_results'][asset][target_metric])
                best_per_asset[asset] = {
                    'params': best['params'],
                    'metrics': best['asset_results'][asset]
                }
            else:
                best_per_asset[asset] = None
        
        # Best universal
        valid_universal = [r for r in all_results if r['valid_assets'] >= len(self.data)]
        if len(valid_universal) > 0:
            best_universal = max(valid_universal, key=lambda x: x['universal_metric'])
        else:
            best_universal = None
        
        return {
            'best_per_asset': best_per_asset,
            'best_universal': best_universal,
            'all_results': all_results
        }
    
    def print_results(self, results: dict):
        """Print results"""
        print("\n" + "="*80)
        print("📊 QVI - OPTIMIZATION RESULTS")
        print("="*80)
        
        print("\n🎯 BEST PARAMETERS PER ASSET:\n")
        for asset, result in results['best_per_asset'].items():
            if result is None:
                print(f"❌ {asset}: No valid results")
                continue
            
            print(f"{'='*60}")
            print(f"  {asset}")
            print(f"{'='*60}")
            print(f"  Parameters:")
            for param, value in result['params'].items():
                print(f"    {param}: {value}")
            
            print(f"\n  Performance:")
            m = result['metrics']
            print(f"    Win Rate:     {m['win_rate']*100:.1f}%")
            print(f"    Avg Return:   {m['avg_return']*100:.3f}%")
            print(f"    Sharpe Ratio: {m['sharpe']:.3f}")
            print(f"    Total Return: {m['total_return']*100:.2f}%")
            print(f"    Num Trades:   {m['num_trades']}")
            print()
        
        if results['best_universal']:
            print(f"\n{'='*80}")
            print("🌍 BEST UNIVERSAL PARAMETERS:")
            print(f"{'='*80}")
            
            best = results['best_universal']
            print(f"\nParameters:")
            for param, value in best['params'].items():
                print(f"  {param}: {value}")
            
            print(f"\nPerformance by Asset:")
            for asset, metrics in best['asset_results'].items():
                print(f"\n  {asset}:")
                print(f"    Win Rate:     {metrics['win_rate']*100:.1f}%")
                print(f"    Avg Return:   {metrics['avg_return']*100:.3f}%")
                print(f"    Sharpe Ratio: {metrics['sharpe']:.3f}")
                print(f"    Total Return: {metrics['total_return']*100:.2f}%")
                print(f"    Num Trades:   {metrics['num_trades']}")
            
            print(f"\n  Universal Metric: {best['universal_metric']:.3f}")
    
    def save_results(self, results: dict):
        """Save to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qvi_optimization_{timestamp}.json"
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'assets': self.assets,
            'best_per_asset': {},
            'best_universal': None
        }
        
        for asset, result in results['best_per_asset'].items():
            if result:
                output['best_per_asset'][asset] = {
                    'params': result['params'],
                    'metrics': result['metrics']
                }
        
        if results['best_universal']:
            best = results['best_universal']
            output['best_universal'] = {
                'params': best['params'],
                'asset_results': best['asset_results'],
                'universal_metric': best['universal_metric']
            }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename


if __name__ == "__main__":
    print("="*80)
    print("🎯 QVI - PARAMETER OPTIMIZATION")
    print("="*80)
    
    optimizer = QVIOptimizer(assets=['BTC', 'ETH', 'SOL'])
    
    if not optimizer.load_data(lookback_days=90):
        print("❌ Failed to load data")
        exit(1)
    
    # Parameter grid - test various settings
    param_grid = {
        'len_rvol': [40, 50, 60],          # Relative volume window
        'len_cmf': [15, 20, 25],           # CMF window
        'len_delta': [21, 34, 55],         # Delta volume window
        'len_smooth': [3, 5, 7],           # Smoothing
        'len_band': [80, 100, 120],        # Band lookback
        'band_k': [1.2, 1.6, 2.0],         # Band multiplier
        'vwma_len': [10, 20, 30],          # VWMA filter
        'w1': [0.5],                       # Weight for rvol (keep fixed)
        'w2': [0.3],                       # Weight for cmf (keep fixed)
        'w3': [0.2]                        # Weight for delta (keep fixed)
    }
    
    print(f"\n📋 Parameter Grid:")
    for param, values in param_grid.items():
        print(f"   {param}: {values}")
    
    total = 1
    for values in param_grid.values():
        total *= len(values)
    print(f"\n   Total combinations: {total}")
    
    results = optimizer.optimize_grid_search(
        param_grid=param_grid,
        target_metric='sharpe',
        min_trades=5
    )
    
    optimizer.print_results(results)
    optimizer.save_results(results)
    
    print("\n✅ Optimization complete!")
