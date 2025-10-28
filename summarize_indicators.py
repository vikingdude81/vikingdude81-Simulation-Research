"""
Indicator Comparison Summary - Best Settings from All Tests

Compiles the best settings for each indicator from optimization results.
"""

import json
from datetime import datetime


def load_and_summarize():
    """Load all optimization results and create summary"""
    
    print("="*80)
    print("INDICATOR OPTIMIZATION SUMMARY - BEST SETTINGS")
    print("="*80)
    print("\nCompiling best parameters from all optimizations...")
    
    results_summary = {
        'rubberband': {},
        'volatility_hole': {},
        'qvi': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Load Rubber-Band results
    try:
        with open('rubberband_optimization_20251026_142510.json', 'r') as f:
            rb_data = json.load(f)
            results_summary['rubberband'] = rb_data.get('best_per_asset', {})
        print("\n[OK] Loaded Rubber-Band optimization results")
    except Exception as e:
        print(f"\n[SKIP] Rubber-Band: {e}")
    
    # Load Volatility Hole results
    try:
        with open('volatility_hole_optimization_20251026_145019.json', 'r') as f:
            vh_data = json.load(f)
            results_summary['volatility_hole'] = vh_data.get('best_per_asset', {})
        print("[OK] Loaded Volatility Hole optimization results")
    except Exception as e:
        print(f"[SKIP] Volatility Hole: {e}")
    
    # Load QVI results (latest)
    try:
        with open('qvi_optimization_20251026_153919.json', 'r') as f:
            qvi_data = json.load(f)
            results_summary['qvi'] = qvi_data.get('best_per_asset', {})
        print("[OK] Loaded QVI optimization results")
    except Exception as e:
        print(f"[SKIP] QVI: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("BEST SETTINGS FOR EACH INDICATOR (By Asset)")
    print("="*80)
    
    for asset in ['BTC', 'ETH', 'SOL']:
        print(f"\n{'='*80}")
        print(f"  {asset}")
        print(f"{'='*80}")
        
        # Rubber-Band
        if asset in results_summary['rubberband']:
            rb = results_summary['rubberband'][asset]
            print(f"\n  RUBBER-BAND REVERSION:")
            print(f"    Parameters:")
            for param, value in rb['params'].items():
                print(f"      {param}: {value}")
            print(f"    Performance:")
            print(f"      Win Rate:     {rb['metrics']['win_rate']*100:.1f}%")
            print(f"      Sharpe Ratio: {rb['metrics']['sharpe']:.3f}")
            print(f"      Total Return: {rb['metrics']['total_return']*100:.2f}%")
            print(f"      Num Trades:   {rb['metrics']['num_trades']}")
        else:
            print(f"\n  RUBBER-BAND: No valid results")
        
        # Volatility Hole
        if asset in results_summary['volatility_hole']:
            vh = results_summary['volatility_hole'][asset]
            print(f"\n  VOLATILITY HOLE DETECTOR:")
            print(f"    Parameters:")
            for param, value in vh['params'].items():
                print(f"      {param}: {value}")
            print(f"    Performance:")
            print(f"      Win Rate:     {vh['metrics']['win_rate']*100:.1f}%")
            print(f"      Sharpe Ratio: {vh['metrics']['sharpe']:.3f}")
            print(f"      Total Return: {vh['metrics']['total_return']*100:.2f}%")
            print(f"      Num Trades:   {vh['metrics']['num_trades']}")
        else:
            print(f"\n  VOLATILITY HOLE: No valid results")
        
        # QVI
        if asset in results_summary['qvi']:
            qvi = results_summary['qvi'][asset]
            print(f"\n  QVI - VOLUME INTELLIGENCE:")
            print(f"    Parameters:")
            for param, value in qvi['params'].items():
                print(f"      {param}: {value}")
            print(f"    Performance:")
            print(f"      Win Rate:     {qvi['metrics']['win_rate']*100:.1f}%")
            print(f"      Sharpe Ratio: {qvi['metrics']['sharpe']:.3f}")
            print(f"      Total Return: {qvi['metrics']['total_return']*100:.2f}%")
            print(f"      Num Trades:   {qvi['metrics']['num_trades']}")
        else:
            print(f"\n  QVI: No valid results")
        
        # Recommendation
        best_indicator = None
        best_sharpe = -999
        
        if asset in results_summary['rubberband']:
            sharpe = results_summary['rubberband'][asset]['metrics']['sharpe']
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_indicator = "Rubber-Band"
        
        if asset in results_summary['volatility_hole']:
            sharpe = results_summary['volatility_hole'][asset]['metrics']['sharpe']
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_indicator = "Volatility Hole"
        
        if asset in results_summary['qvi']:
            sharpe = results_summary['qvi'][asset]['metrics']['sharpe']
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_indicator = "QVI"
        
        if best_indicator:
            print(f"\n  >>> WINNER: {best_indicator} (Sharpe: {best_sharpe:.3f})")
        else:
            print(f"\n  >>> No valid indicator results for {asset}")
    
    # Save summary
    filename = f"indicator_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n\nSummary saved to: {filename}")
    
    # Print overall recommendations
    print("\n" + "="*80)
    print("INTEGRATION RECOMMENDATIONS")
    print("="*80)
    
    print("\nBased on standalone indicator performance:")
    print("\n1. VOLATILITY HOLE - Best for SOL")
    print("   - 80% win rate, 1.40 Sharpe")
    print("   - Detects compression/expansion breakouts effectively")
    print("   - RECOMMEND: Use as filter for SOL trades")
    
    print("\n2. RUBBER-BAND - Good for BTC/ETH")
    print("   - 80% win rate on BTC/ETH, 0.52-0.88 Sharpe")
    print("   - Mean-reversion from multiple MAs")
    print("   - RECOMMEND: Use as secondary confirmation")
    
    print("\n3. QVI - Underperforms")
    print("   - 49-60% win rate, max 0.070 Sharpe")
    print("   - Volume climax detection works but not profitable standalone")
    print("   - RECOMMEND: Skip or use VSA events as alerts only")
    
    print("\n4. ML BASELINE - Still King")
    print("   - From previous tests: 97.6% win rate, 1.23 Sharpe")
    print("   - RECOMMEND: Keep ML as primary, use Vol Hole as filter")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    load_and_summarize()
