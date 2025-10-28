"""
Newton Basin Map Optimizer

Find optimal settings and trading strategies based on basin regime shifts.
Tests various configurations against historical data to find profitable patterns.
"""

import pandas as pd
import numpy as np
from newton_basin_map import NewtonBasinMap
from itertools import product
import json
from datetime import datetime


class NewtonBasinOptimizer:
    """Optimize Newton Basin Map for trading applications"""
    
    def __init__(self, assets=['BTC', 'ETH', 'SOL']):
        self.assets = assets
        self.data = {}
        
    def load_data(self, lookback_days=90):
        """Load historical data"""
        print(f"\nLoading {lookback_days} days of data...")
        
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
                print(f"   {asset}: {len(df)} bars")
            except Exception as e:
                print(f"   {asset}: Failed - {e}")
        
        return len(self.data) > 0
    
    def test_shift_strategy(self, df: pd.DataFrame, strategy: str) -> dict:
        """
        Test different shift-based strategies:
        - 'shift_to_fast': Buy when shifting to fast basin
        - 'shift_to_slow': Short when shifting to slow basin
        - 'shift_any': Trade on any basin shift
        - 'fast_entry_slow_exit': Enter in fast, exit in slow
        - 'persistence': Trade based on basin persistence
        """
        
        if strategy == 'shift_to_fast':
            # Buy on shift to fast basin
            signals = df[df['shift_to_fast'] == True].copy()
            direction = 1  # LONG
            
        elif strategy == 'shift_to_slow':
            # Short on shift to slow basin
            signals = df[df['shift_to_slow'] == True].copy()
            direction = -1  # SHORT
            
        elif strategy == 'shift_any':
            # Trade on any shift
            signals = df[df['basin_shift'] != 0].copy()
            direction = 1  # LONG (simplified)
            
        elif strategy == 'fast_entry_slow_exit':
            # Enter LONG in fast basin, exit when shift to slow
            signals = df[df['shift_to_fast'] == True].copy()
            direction = 1
            
        elif strategy == 'persistence':
            # Trade when persistence > threshold
            signals = df[df['basin_persistence'] > 10].copy()
            direction = 1
            
        else:
            return {'num_trades': 0, 'win_rate': 0, 'sharpe': 0, 'avg_return': 0, 'total_return': 0}
        
        if len(signals) == 0:
            return {'num_trades': 0, 'win_rate': 0, 'sharpe': 0, 'avg_return': 0, 'total_return': 0}
        
        # Calculate returns
        returns = []
        for idx in signals.index:
            if idx + 12 >= len(df):
                continue
            ret = df.loc[idx, 'return_12h']
            if not pd.isna(ret):
                returns.append(ret * direction)
        
        if len(returns) == 0:
            return {'num_trades': 0, 'win_rate': 0, 'sharpe': 0, 'avg_return': 0, 'total_return': 0}
        
        returns = np.array(returns)
        
        return {
            'num_trades': len(returns),
            'win_rate': (returns > 0).sum() / len(returns),
            'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
            'avg_return': returns.mean(),
            'total_return': (1 + returns).prod() - 1
        }
    
    def optimize_parameters(self):
        """Find optimal Newton basin parameters"""
        print("\n" + "="*80)
        print("NEWTON BASIN MAP - PARAMETER OPTIMIZATION")
        print("="*80)
        
        param_grid = {
            'ma1_len': [15, 20, 25],
            'ma2_len': [40, 50, 60],
            'ma3_len': [180, 200, 220],
            'newton_iterations': [4, 6, 8],
            'use_relative_distance': [True, False]
        }
        
        strategies = [
            'shift_to_fast',
            'shift_to_slow',
            'fast_entry_slow_exit'
        ]
        
        results_by_asset = {}
        
        for asset in self.data.keys():
            print(f"\nOptimizing for {asset}...")
            best_result = None
            best_score = -999
            
            combos = list(product(
                param_grid['ma1_len'],
                param_grid['ma2_len'],
                param_grid['ma3_len'],
                param_grid['newton_iterations'],
                param_grid['use_relative_distance']
            ))
            
            for i, (ma1, ma2, ma3, iterations, use_rel) in enumerate(combos):
                if (i + 1) % 20 == 0:
                    print(f"   Testing combo {i+1}/{len(combos)}...")
                
                try:
                    newton = NewtonBasinMap(
                        ma1_len=ma1,
                        ma2_len=ma2,
                        ma3_len=ma3,
                        newton_iterations=iterations,
                        use_relative_distance=use_rel
                    )
                    
                    df_result = newton.calculate(self.data[asset].copy())
                    
                    # Test each strategy
                    for strategy in strategies:
                        metrics = self.test_shift_strategy(df_result, strategy)
                        
                        if metrics['num_trades'] < 5:
                            continue
                        
                        # Score: Sharpe + win rate
                        score = metrics['sharpe'] + (metrics['win_rate'] * 0.5)
                        
                        if score > best_score:
                            best_score = score
                            best_result = {
                                'params': {
                                    'ma1_len': ma1,
                                    'ma2_len': ma2,
                                    'ma3_len': ma3,
                                    'newton_iterations': iterations,
                                    'use_relative_distance': use_rel
                                },
                                'strategy': strategy,
                                'metrics': metrics,
                                'score': score
                            }
                
                except Exception as e:
                    continue
            
            results_by_asset[asset] = best_result
            
            if best_result:
                print(f"\n   Best for {asset}:")
                print(f"      Strategy: {best_result['strategy']}")
                print(f"      Sharpe: {best_result['metrics']['sharpe']:.3f}")
                print(f"      Win Rate: {best_result['metrics']['win_rate']*100:.1f}%")
                print(f"      Avg Return: {best_result['metrics']['avg_return']*100:.3f}%")
                print(f"      Trades: {best_result['metrics']['num_trades']}")
        
        return results_by_asset
    
    def analyze_basin_characteristics(self):
        """Analyze basin characteristics without optimization"""
        print("\n" + "="*80)
        print("BASIN CHARACTERISTICS ANALYSIS")
        print("="*80)
        
        for asset in self.data.keys():
            print(f"\n{asset}:")
            
            newton = NewtonBasinMap()
            df_result = newton.calculate(self.data[asset].copy())
            analysis = newton.analyze_regime_shifts(df_result)
            
            print(f"  Basin Occupancy:")
            print(f"    Fast:   {analysis['pct_time_fast']:.1f}%")
            print(f"    Medium: {analysis['pct_time_medium']:.1f}%")
            print(f"    Slow:   {analysis['pct_time_slow']:.1f}%")
            
            print(f"  Shift Performance:")
            print(f"    Shift to FAST: {analysis['shift_to_fast_win_rate']*100:.1f}% win, "
                  f"{analysis['shift_to_fast_avg_return']*100:.3f}% avg return")
            print(f"    Shift to SLOW: {analysis['shift_to_slow_win_rate']*100:.1f}% win, "
                  f"{analysis['shift_to_slow_avg_return']*100:.3f}% avg return")
            
            print(f"  Persistence:")
            print(f"    Fast:   {analysis['fast_persistence_avg']:.1f} bars")
            print(f"    Medium: {analysis['medium_persistence_avg']:.1f} bars")
            print(f"    Slow:   {analysis['slow_persistence_avg']:.1f} bars")
    
    def save_results(self, results: dict):
        """Save optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"newton_basin_optimization_{timestamp}.json"
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'assets': self.assets,
            'results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n\nResults saved to: {filename}")
        return filename


if __name__ == "__main__":
    print("="*80)
    print("NEWTON BASIN-OF-ATTRACTION MAP - OPTIMIZER")
    print("="*80)
    
    optimizer = NewtonBasinOptimizer(assets=['BTC', 'ETH', 'SOL'])
    
    if not optimizer.load_data(lookback_days=90):
        print("\nFailed to load data")
        exit(1)
    
    # First, analyze baseline characteristics
    optimizer.analyze_basin_characteristics()
    
    # Then optimize parameters
    print("\n\nStarting parameter optimization...")
    print("(This may take a few minutes...)\n")
    
    results = optimizer.optimize_parameters()
    
    # Print summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    
    for asset, result in results.items():
        if result:
            print(f"\n{asset}:")
            print(f"  Best Strategy: {result['strategy']}")
            print(f"  Parameters: MA1={result['params']['ma1_len']}, "
                  f"MA2={result['params']['ma2_len']}, MA3={result['params']['ma3_len']}")
            print(f"  Performance:")
            print(f"    Sharpe:      {result['metrics']['sharpe']:.3f}")
            print(f"    Win Rate:    {result['metrics']['win_rate']*100:.1f}%")
            print(f"    Avg Return:  {result['metrics']['avg_return']*100:.3f}%")
            print(f"    Total Return: {result['metrics']['total_return']*100:.2f}%")
            print(f"    Trades:      {result['metrics']['num_trades']}")
        else:
            print(f"\n{asset}: No valid results")
    
    # Save
    optimizer.save_results(results)
    
    print("\n\nOptimization complete!")
