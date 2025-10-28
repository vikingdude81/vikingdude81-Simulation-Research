"""
Optimize Geometric MA Crossover parameters across BTC/ETH/SOL

Tests different GMA length combinations to find optimal settings.
"""

import pandas as pd
import numpy as np
from geometric_ma_crossover import GeometricMACrossover
from typing import Dict, List, Tuple
import json
from datetime import datetime


def load_data(symbol: str, days: int = 90) -> pd.DataFrame:
    """Load historical data"""
    file_map = {
        'BTC': 'DATA/yf_btc_1h.csv',
        'ETH': 'DATA/yf_eth_1h.csv',
        'SOL': 'DATA/yf_sol_1h.csv'
    }
    
    df = pd.read_csv(file_map[symbol])
    df['time'] = pd.to_datetime(df['time'])
    
    if 'Close' in df.columns:
        df = df.rename(columns={
            'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
    
    # Get last N days
    total_bars = days * 24
    df = df.tail(total_bars).reset_index(drop=True)
    
    return df


def backtest_strategy(
    df: pd.DataFrame,
    len_fast: int,
    len_slow: int,
    use_atr_exit: bool = True,
    atr_length: int = 14,
    stop_atr_mult: float = 2.0,
    tp_rr: float = 2.0
) -> Dict:
    """
    Backtest GMA crossover strategy
    
    Returns performance metrics
    """
    gma = GeometricMACrossover(
        len_fast=len_fast,
        len_slow=len_slow,
        use_atr_exit=use_atr_exit,
        atr_length=atr_length,
        stop_atr_mult=stop_atr_mult,
        tp_rr=tp_rr
    )
    
    df_calc = gma.calculate(df)
    
    # Get signals
    signals = df_calc[df_calc['signal'] != 0].copy()
    
    if len(signals) == 0:
        return {
            'sharpe': 0.0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'total_return': 0.0,
            'num_trades': 0,
            'score': 0.0
        }
    
    # Calculate forward returns
    returns = []
    wins = 0
    
    for idx in signals.index:
        signal = signals.loc[idx, 'signal']
        entry_price = df_calc.loc[idx, 'close']
        
        if use_atr_exit:
            # Use ATR-based stop/target
            stop_dist = signals.loc[idx, 'stop_distance']
            target_dist = signals.loc[idx, 'target_distance']
            
            if signal == 1:  # LONG
                stop_price = entry_price - stop_dist
                target_price = entry_price + target_dist
            else:  # SHORT
                stop_price = entry_price + stop_dist
                target_price = entry_price - target_dist
            
            # Look forward for stop/target hit
            exit_price = None
            exit_idx = None
            
            for future_idx in range(idx + 1, min(idx + 240, len(df_calc))):  # 10 days max
                future_high = df_calc.loc[future_idx, 'high']
                future_low = df_calc.loc[future_idx, 'low']
                
                if signal == 1:  # LONG
                    if future_low <= stop_price:
                        exit_price = stop_price
                        exit_idx = future_idx
                        break
                    elif future_high >= target_price:
                        exit_price = target_price
                        exit_idx = future_idx
                        break
                else:  # SHORT
                    if future_high >= stop_price:
                        exit_price = stop_price
                        exit_idx = future_idx
                        break
                    elif future_low <= target_price:
                        exit_price = target_price
                        exit_idx = future_idx
                        break
            
            # If no stop/target hit, use close at end of window
            if exit_price is None:
                exit_idx = min(idx + 240, len(df_calc) - 1)
                exit_price = df_calc.loc[exit_idx, 'close']
        
        else:
            # Simple exit on opposite signal or 10 days later
            exit_idx = min(idx + 240, len(df_calc) - 1)
            exit_price = df_calc.loc[exit_idx, 'close']
        
        # Calculate return
        if signal == 1:  # LONG
            ret = (exit_price - entry_price) / entry_price
        else:  # SHORT
            ret = (entry_price - exit_price) / entry_price
        
        returns.append(ret)
        if ret > 0:
            wins += 1
    
    # Calculate metrics
    returns = np.array(returns)
    
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0.0
    win_rate = (wins / len(returns)) * 100 if len(returns) > 0 else 0.0
    avg_return = np.mean(returns) * 100 if len(returns) > 0 else 0.0
    total_return = (np.prod(1 + returns) - 1) * 100 if len(returns) > 0 else 0.0
    num_trades = len(returns)
    
    # Scoring function (balance Sharpe and win rate)
    score = (sharpe * 2.0) + (win_rate / 100.0 * 0.5)
    
    return {
        'sharpe': float(sharpe),
        'win_rate': float(win_rate),
        'avg_return': float(avg_return),
        'total_return': float(total_return),
        'num_trades': int(num_trades),
        'score': float(score)
    }


def optimize_parameters(symbol: str) -> Dict:
    """Optimize GMA parameters for a symbol"""
    print(f"\n{'='*80}")
    print(f"Optimizing {symbol}")
    print(f"{'='*80}")
    
    # Load data
    df = load_data(symbol, days=90)
    print(f"Loaded {len(df)} bars")
    
    # Parameter grid
    fast_lengths = [10, 15, 20, 25, 30]
    slow_lengths = [40, 50, 60, 75, 100]
    atr_stops = [1.5, 2.0, 2.5]
    tp_ratios = [1.5, 2.0, 2.5, 3.0]
    
    results = []
    total_combos = len(fast_lengths) * len(slow_lengths) * len(atr_stops) * len(tp_ratios)
    
    print(f"Testing {total_combos} parameter combinations...")
    
    count = 0
    for fast_len in fast_lengths:
        for slow_len in slow_lengths:
            if fast_len >= slow_len:
                continue
            
            for stop_mult in atr_stops:
                for tp_rr in tp_ratios:
                    count += 1
                    
                    if count % 50 == 0:
                        print(f"  Progress: {count}/{total_combos}")
                    
                    perf = backtest_strategy(
                        df,
                        len_fast=fast_len,
                        len_slow=slow_len,
                        use_atr_exit=True,
                        atr_length=14,
                        stop_atr_mult=stop_mult,
                        tp_rr=tp_rr
                    )
                    
                    result = {
                        'symbol': symbol,
                        'len_fast': fast_len,
                        'len_slow': slow_len,
                        'atr_length': 14,
                        'stop_atr_mult': stop_mult,
                        'tp_rr': tp_rr,
                        **perf
                    }
                    
                    results.append(result)
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Top 5
    print(f"\nTop 5 Results for {symbol}:")
    print(f"{'Rank':<6} {'Fast':<6} {'Slow':<6} {'Stop':<7} {'TP_RR':<7} {'Sharpe':<8} {'Win%':<7} {'Ret%':<9} {'Trades':<7} {'Score':<7}")
    print("-" * 80)
    
    for i, result in enumerate(results[:5], 1):
        print(f"{i:<6} {result['len_fast']:<6} {result['len_slow']:<6} "
              f"{result['stop_atr_mult']:<7.1f} {result['tp_rr']:<7.1f} "
              f"{result['sharpe']:<8.3f} {result['win_rate']:<7.1f} "
              f"{result['total_return']:<9.2f} {result['num_trades']:<7} "
              f"{result['score']:<7.3f}")
    
    return {
        'symbol': symbol,
        'all_results': results,
        'best': results[0] if results else None
    }


if __name__ == "__main__":
    print("="*80)
    print("GEOMETRIC MA CROSSOVER - Parameter Optimization")
    print("="*80)
    print("\nTesting 90 days of hourly data for BTC/ETH/SOL")
    print("ATR-based stops and targets enabled")
    
    all_results = {}
    
    # Optimize each symbol
    for symbol in ['BTC', 'ETH', 'SOL']:
        result = optimize_parameters(symbol)
        all_results[symbol] = result
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"geometric_ma_optimization_{timestamp}.json"
    
    # Convert to serializable format
    save_data = {}
    for symbol, data in all_results.items():
        save_data[symbol] = {
            'best': data['best'],
            'top_10': data['all_results'][:10]
        }
    
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {filename}")
    
    # Summary
    print("\n" + "="*80)
    print("BEST PARAMETERS SUMMARY")
    print("="*80)
    
    for symbol in ['BTC', 'ETH', 'SOL']:
        best = all_results[symbol]['best']
        if best:
            print(f"\n{symbol}:")
            print(f"  Fast GMA: {best['len_fast']}")
            print(f"  Slow GMA: {best['len_slow']}")
            print(f"  ATR Stop: {best['stop_atr_mult']}x")
            print(f"  TP Ratio: {best['tp_rr']}x")
            print(f"  Performance:")
            print(f"    Sharpe: {best['sharpe']:.3f}")
            print(f"    Win Rate: {best['win_rate']:.1f}%")
            print(f"    Total Return: {best['total_return']:.2f}%")
            print(f"    Trades: {best['num_trades']}")
