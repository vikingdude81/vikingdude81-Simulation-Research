"""
Conductor-Enhanced Multi-Regime Ensemble System

Uses RegimeDetector to identify market regime, then deploys the appropriate
conductor-enhanced specialist for optimal performance.

Combines 3 conductor-enhanced specialists:
- Volatile specialist (fitness 71.92)
- Trending specialist (expected ~64)
- Ranging specialist (expected ~1.5-2.0)
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Tuple
from regime_detector import RegimeDetector
from trading_specialist import TradingSpecialist


class ConductorEnsemble:
    """Ensemble system using conductor-enhanced specialists"""
    
    def __init__(self):
        """Initialize ensemble with regime detector and specialists"""
        self.regime_detector = RegimeDetector()
        self.specialists = {}
        self.specialist_results = {}
        
    def load_specialists(self, results_dir: str = 'outputs'):
        """Load all conductor-enhanced specialists from results files"""
        regimes = ['volatile', 'trending', 'ranging']
        
        print("Loading conductor-enhanced specialists...")
        for regime in regimes:
            # Find most recent result file for this regime
            import glob
            pattern = f"{results_dir}/conductor_enhanced_{regime}_*.json"
            files = glob.glob(pattern)
            
            if not files:
                print(f"  ⚠️  No results found for {regime} regime")
                continue
                
            # Use most recent file
            result_file = max(files, key=lambda x: x.split('_')[-1].replace('.json', ''))
            
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Create specialist from best genome
            genome = np.array(data['best_agent']['genome'])
            specialist = TradingSpecialist(genome=genome, regime_type=regime)
            
            self.specialists[regime] = specialist
            self.specialist_results[regime] = data
            
            fitness = data['best_fitness']
            print(f"  ✓ Loaded {regime}: fitness {fitness:.2f}")
        
        print(f"\n✓ Ensemble ready with {len(self.specialists)} specialists\n")
        
    def detect_regime(self, df: pd.DataFrame, window: int = 60) -> str:
        """Detect current market regime using RegimeDetector"""
        # Use last window days for regime detection
        recent_data = df.tail(window)
        regime = self.regime_detector.detect_regime(recent_data)
        return regime
    
    def generate_signal(self, df: pd.DataFrame, regime: str = None) -> Tuple[int, float, str]:
        """
        Generate trading signal using appropriate specialist
        
        Args:
            df: Market data DataFrame
            regime: Optional regime override (otherwise auto-detect)
            
        Returns:
            (signal, position_size) tuple
        """
        # Auto-detect regime if not provided
        if regime is None:
            regime = self.detect_regime(df)
        
        # Map crisis to volatile (fallback)
        if regime == 'crisis':
            regime = 'volatile'
        
        # Get specialist for this regime
        if regime not in self.specialists:
            print(f"  ⚠️  No specialist for {regime}, using volatile")
            regime = 'volatile'
        
        specialist = self.specialists[regime]
        
        # Generate signal - specialist needs DataFrame and predictions array
        predictions = df['predictions'].values if 'predictions' in df.columns else np.zeros(len(df))
        
        signal, size = specialist.generate_signal(df, predictions)
        
        return signal, size, regime
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
        """
        Backtest ensemble on full dataset with regime switching
        
        Args:
            df: Full market data with all regimes
            initial_capital: Starting capital
            
        Returns:
            Dictionary with performance metrics
        """
        print("="*80)
        print("CONDUCTOR ENSEMBLE BACKTEST")
        print("="*80)
        
        capital = initial_capital
        position = 0
        position_price = 0
        trades = []
        equity_curve = [initial_capital]
        regime_usage = {'volatile': 0, 'trending': 0, 'ranging': 0, 'crisis': 0}
        
        # Add regime labels
        df = df.copy()
        df['regime'] = df.apply(lambda row: self.regime_detector.detect_regime(
            df.loc[:row.name].tail(60) if len(df.loc[:row.name]) > 60 else df.loc[:row.name]
        ), axis=1)
        
        # Generate predictions (simple momentum for now)
        df['returns'] = df['close'].pct_change()
        df['predictions'] = df['returns'].rolling(5).mean().fillna(0)
        
        print(f"Testing on {len(df)} days...")
        print(f"Regimes: Volatile={sum(df['regime']=='volatile')}, "
              f"Trending={sum(df['regime']=='trending')}, "
              f"Ranging={sum(df['regime']=='ranging')}, "
              f"Crisis={sum(df['regime']=='crisis')}")
        print()
        
        for i in range(60, len(df)):  # Start after regime detection window
            current_data = df.iloc[:i+1]
            regime = current_data.iloc[-1]['regime']
            regime_usage[regime] += 1
            
            # Get signal from appropriate specialist
            signal, size, used_regime = self.generate_signal(current_data, regime)
            
            current_price = current_data.iloc[-1]['close']
            
            # Execute trades
            if signal == 1 and position == 0:  # Buy
                position = (capital * size) / current_price
                position_price = current_price
                capital -= position * current_price
                trades.append({
                    'type': 'BUY',
                    'price': current_price,
                    'regime': used_regime,
                    'date': current_data.iloc[-1].name if hasattr(current_data.iloc[-1], 'name') else i
                })
                
            elif signal == -1 and position > 0:  # Sell
                pnl = position * (current_price - position_price)
                capital += position * current_price
                position = 0
                trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'pnl': pnl,
                    'regime': used_regime,
                    'date': current_data.iloc[-1].name if hasattr(current_data.iloc[-1], 'name') else i
                })
            
            # Update equity
            equity = capital + (position * current_price if position > 0 else 0)
            equity_curve.append(equity)
        
        # Close any open position
        if position > 0:
            final_price = df.iloc[-1]['close']
            pnl = position * (final_price - position_price)
            capital += position * final_price
            trades.append({
                'type': 'SELL',
                'price': final_price,
                'pnl': pnl,
                'regime': 'final',
                'date': df.iloc[-1].name if hasattr(df.iloc[-1], 'name') else len(df)-1
            })
        
        # Calculate metrics
        final_capital = capital
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        
        # Calculate Sharpe ratio
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        winning_trades = [t for t in trades if 'pnl' in t and t['pnl'] > 0]
        win_rate = (len(winning_trades) / len([t for t in trades if 'pnl' in t])) * 100 if len([t for t in trades if 'pnl' in t]) > 0 else 0
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len([t for t in trades if t['type'] == 'BUY']),
            'win_rate': win_rate,
            'regime_usage': regime_usage,
            'trades': trades,
            'equity_curve': equity_curve
        }
        
        # Print results
        print("\n" + "="*80)
        print("ENSEMBLE PERFORMANCE")
        print("="*80)
        print(f"Total Return:       {total_return:+.2f}%")
        print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
        print(f"Max Drawdown:       {max_drawdown:.2f}%")
        print(f"Number of Trades:   {results['num_trades']}")
        print(f"Win Rate:           {win_rate:.1f}%")
        print(f"\nRegime Usage:")
        for regime, count in regime_usage.items():
            pct = (count / sum(regime_usage.values())) * 100
            print(f"  {regime:>10}: {count:4d} days ({pct:5.1f}%)")
        print("="*80)
        
        return results


def main():
    """Test ensemble on full BTC dataset"""
    
    # Load full BTC dataset
    print("\nLoading full BTC dataset...")
    df = pd.read_csv('DATA/yf_btc_1d.csv', parse_dates=['time'], index_col='time')
    print(f"✓ Loaded {len(df)} days of data\n")
    
    # Create and load ensemble
    ensemble = ConductorEnsemble()
    ensemble.load_specialists()
    
    # Backtest ensemble
    results = ensemble.backtest(df)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"outputs/ensemble_conductor_{timestamp}.json"
    
    # Convert to JSON-serializable format
    results_json = {
        'initial_capital': results['initial_capital'],
        'final_capital': results['final_capital'],
        'total_return': results['total_return'],
        'sharpe_ratio': results['sharpe_ratio'],
        'max_drawdown': results['max_drawdown'],
        'num_trades': results['num_trades'],
        'win_rate': results['win_rate'],
        'regime_usage': results['regime_usage'],
        'num_trades_detail': len(results['trades']),
        'timestamp': timestamp
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
