"""
Combined Backtest: Rubber-Band Indicator + ML Predictions

Tests three strategies:
1. ML predictions only (your current system)
2. Rubber-Band indicator only (mean-reversion)
3. Combined (ML predictions filtered by Rubber-Band signals)
"""

import pandas as pd
import numpy as np
from rubberband_indicator import RubberBandOscillator
import json
from datetime import datetime
import matplotlib.pyplot as plt


class CombinedBacktest:
    """
    Backtest ML predictions with and without Rubber-Band indicator filter
    """
    
    def __init__(self):
        self.data = {}
        self.rb_params = {}  # Optimized parameters per asset
        self.results = {}
    
    def load_optimized_params(self, filename='rubberband_optimization_20251026_142510.json'):
        """Load optimized Rubber-Band parameters"""
        print(f"\n📊 Loading optimized parameters from {filename}...")
        
        try:
            with open(filename, 'r') as f:
                opt_results = json.load(f)
            
            # Extract best parameters per asset
            for asset in ['BTC', 'ETH', 'SOL']:
                if asset in opt_results['best_per_asset']:
                    self.rb_params[asset] = opt_results['best_per_asset'][asset]['params']
                    print(f"   ✅ {asset}: Loaded optimized params")
                else:
                    # Use universal params as fallback
                    self.rb_params[asset] = opt_results['best_universal']['params']
                    print(f"   ⚠️  {asset}: Using universal params (no asset-specific)")
            
            return True
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            return False
    
    def load_data(self, lookback_days=90):
        """Load historical price data"""
        print(f"\n📊 Loading {lookback_days} days of data...")
        
        for asset in ['BTC', 'ETH', 'SOL']:
            file_path = f"DATA/yf_{asset.lower()}_1h.csv"
            try:
                df = pd.read_csv(file_path)
                df['time'] = pd.to_datetime(df['time'])
                
                # Standardize column names
                if 'Close' in df.columns:
                    df = df.rename(columns={
                        'Open': 'open', 'High': 'high',
                        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                    })
                
                # Take last N days
                df = df.tail(lookback_days * 24).reset_index(drop=True)
                
                # Calculate actual 12h forward returns
                df['return_12h'] = df['close'].pct_change(12).shift(-12)
                
                self.data[asset] = df
                print(f"   ✅ {asset}: {len(df)} bars ({df['time'].min().date()} to {df['time'].max().date()})")
            except Exception as e:
                print(f"   ❌ {asset}: Failed - {e}")
        
        return len(self.data) > 0
    
    def simulate_ml_predictions(self):
        """
        Simulate ML predictions (in real use, these come from your trained models)
        For this backtest, we'll use actual returns + noise to simulate predictions
        """
        print("\n🤖 Simulating ML predictions...")
        
        for asset in self.data.keys():
            df = self.data[asset]
            
            # Simulate predictions: actual return + small noise
            np.random.seed(42)  # For reproducibility
            df['ml_predicted_return'] = df['return_12h'] + np.random.normal(0, 0.005, len(df))
            
            # ML signal: BUY if predicted return > 0.5%, SELL if < -0.5%, else HOLD
            df['ml_signal'] = 0
            df.loc[df['ml_predicted_return'] > 0.005, 'ml_signal'] = 1   # BUY
            df.loc[df['ml_predicted_return'] < -0.005, 'ml_signal'] = -1  # SELL
            
            print(f"   {asset}: {(df['ml_signal'] == 1).sum()} BUY, {(df['ml_signal'] == -1).sum()} SELL signals")
    
    def calculate_rubberband_signals(self):
        """Calculate Rubber-Band indicator signals"""
        print("\n🎯 Calculating Rubber-Band signals...")
        
        for asset in self.data.keys():
            # Get optimized parameters for this asset
            params = self.rb_params.get(asset, {})
            
            # Create indicator
            rbo = RubberBandOscillator(**params)
            
            # Calculate
            df_result = rbo.calculate(self.data[asset])
            
            # Update data with RB signals
            self.data[asset] = df_result
            
            long_signals = (df_result['signal'] == 1).sum()
            short_signals = (df_result['signal'] == -1).sum()
            print(f"   {asset}: {long_signals} LONG, {short_signals} SHORT signals")
    
    def backtest_strategies(self):
        """
        Backtest three strategies:
        1. ML only
        2. Rubber-Band only
        3. Combined (ML + RB confirmation)
        """
        print("\n📈 Running backtests...")
        
        results = {}
        
        for asset in self.data.keys():
            df = self.data[asset].copy()
            
            # Strategy 1: ML Only
            ml_trades = []
            for idx in range(len(df) - 12):
                if df['ml_signal'].iloc[idx] != 0:
                    signal = df['ml_signal'].iloc[idx]
                    actual_return = df['return_12h'].iloc[idx]
                    
                    if not np.isnan(actual_return):
                        # Return based on signal direction
                        trade_return = actual_return if signal == 1 else -actual_return
                        ml_trades.append(trade_return)
            
            # Strategy 2: Rubber-Band Only
            rb_trades = []
            for idx in range(len(df) - 12):
                if df['signal'].iloc[idx] != 0:
                    signal = df['signal'].iloc[idx]
                    actual_return = df['return_12h'].iloc[idx]
                    
                    if not np.isnan(actual_return):
                        trade_return = actual_return if signal == 1 else -actual_return
                        rb_trades.append(trade_return)
            
            # Strategy 3: Combined (both must agree)
            combined_trades = []
            for idx in range(len(df) - 12):
                ml_sig = df['ml_signal'].iloc[idx]
                rb_sig = df['signal'].iloc[idx]
                
                # Both must agree on direction
                if ml_sig != 0 and rb_sig != 0 and ml_sig == rb_sig:
                    actual_return = df['return_12h'].iloc[idx]
                    
                    if not np.isnan(actual_return):
                        trade_return = actual_return if ml_sig == 1 else -actual_return
                        combined_trades.append(trade_return)
            
            # Calculate metrics for each strategy
            results[asset] = {
                'ml_only': self._calculate_metrics(ml_trades),
                'rb_only': self._calculate_metrics(rb_trades),
                'combined': self._calculate_metrics(combined_trades)
            }
            
            print(f"\n   {asset} Results:")
            print(f"      ML Only:     {results[asset]['ml_only']['num_trades']} trades, {results[asset]['ml_only']['sharpe']:.3f} Sharpe")
            print(f"      RB Only:     {results[asset]['rb_only']['num_trades']} trades, {results[asset]['rb_only']['sharpe']:.3f} Sharpe")
            print(f"      Combined:    {results[asset]['combined']['num_trades']} trades, {results[asset]['combined']['sharpe']:.3f} Sharpe")
        
        self.results = results
        return results
    
    def _calculate_metrics(self, returns):
        """Calculate performance metrics"""
        if len(returns) == 0:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe': 0,
                'total_return': 0,
                'max_drawdown': 0
            }
        
        returns = np.array(returns)
        wins = returns > 0
        
        # Calculate metrics
        win_rate = wins.sum() / len(returns)
        avg_return = returns.mean()
        sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
        total_return = (1 + returns).prod() - 1
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'num_trades': len(returns),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown
        }
    
    def print_summary(self):
        """Print comprehensive results summary"""
        print("\n" + "="*80)
        print("📊 COMBINED BACKTEST RESULTS - ML + RUBBER-BAND")
        print("="*80)
        
        for asset in self.results.keys():
            print(f"\n{'='*80}")
            print(f"  {asset}")
            print(f"{'='*80}")
            
            strategies = ['ml_only', 'rb_only', 'combined']
            strategy_names = ['ML Only', 'Rubber-Band Only', 'Combined (ML + RB)']
            
            for strat, name in zip(strategies, strategy_names):
                metrics = self.results[asset][strat]
                print(f"\n  {name}:")
                print(f"    Trades:       {metrics['num_trades']}")
                print(f"    Win Rate:     {metrics['win_rate']*100:.1f}%")
                print(f"    Avg Return:   {metrics['avg_return']*100:.3f}%")
                print(f"    Sharpe Ratio: {metrics['sharpe']:.3f}")
                print(f"    Total Return: {metrics['total_return']*100:.2f}%")
                print(f"    Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        
        # Overall summary
        print(f"\n{'='*80}")
        print("📈 STRATEGY COMPARISON (Average across all assets):")
        print(f"{'='*80}")
        
        strategies = ['ml_only', 'rb_only', 'combined']
        strategy_names = ['ML Only', 'Rubber-Band Only', 'Combined']
        
        for strat, name in zip(strategies, strategy_names):
            avg_sharpe = np.mean([self.results[a][strat]['sharpe'] for a in self.results.keys()])
            avg_win_rate = np.mean([self.results[a][strat]['win_rate'] for a in self.results.keys()])
            avg_return = np.mean([self.results[a][strat]['total_return'] for a in self.results.keys()])
            total_trades = sum([self.results[a][strat]['num_trades'] for a in self.results.keys()])
            
            print(f"\n  {name}:")
            print(f"    Avg Sharpe:      {avg_sharpe:.3f}")
            print(f"    Avg Win Rate:    {avg_win_rate*100:.1f}%")
            print(f"    Avg Total Return: {avg_return*100:.2f}%")
            print(f"    Total Trades:    {total_trades}")
        
        # Determine best strategy
        sharpes = {
            'ML Only': np.mean([self.results[a]['ml_only']['sharpe'] for a in self.results.keys()]),
            'RB Only': np.mean([self.results[a]['rb_only']['sharpe'] for a in self.results.keys()]),
            'Combined': np.mean([self.results[a]['combined']['sharpe'] for a in self.results.keys()])
        }
        
        best_strategy = max(sharpes.items(), key=lambda x: x[1])
        
        print(f"\n{'='*80}")
        print(f"🏆 WINNER: {best_strategy[0]} (Sharpe: {best_strategy[1]:.3f})")
        print(f"{'='*80}")
    
    def save_results(self):
        """Save results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_rubberband_ml_{timestamp}.json"
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': {}
        }
        
        for asset in self.results.keys():
            output['results'][asset] = self.results[asset]
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename
    
    def plot_comparison(self):
        """Create comparison charts"""
        fig, axes = plt.subplots(len(self.results), 2, figsize=(14, 4 * len(self.results)))
        
        if len(self.results) == 1:
            axes = axes.reshape(1, -1)
        
        for i, asset in enumerate(self.results.keys()):
            # Plot 1: Sharpe Ratio comparison
            strategies = ['ML Only', 'RB Only', 'Combined']
            sharpes = [
                self.results[asset]['ml_only']['sharpe'],
                self.results[asset]['rb_only']['sharpe'],
                self.results[asset]['combined']['sharpe']
            ]
            
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            axes[i, 0].bar(strategies, sharpes, color=colors, alpha=0.7)
            axes[i, 0].set_ylabel('Sharpe Ratio')
            axes[i, 0].set_title(f'{asset} - Sharpe Ratio Comparison')
            axes[i, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot 2: Total Return comparison
            returns = [
                self.results[asset]['ml_only']['total_return'] * 100,
                self.results[asset]['rb_only']['total_return'] * 100,
                self.results[asset]['combined']['total_return'] * 100
            ]
            
            axes[i, 1].bar(strategies, returns, color=colors, alpha=0.7)
            axes[i, 1].set_ylabel('Total Return (%)')
            axes[i, 1].set_title(f'{asset} - Total Return Comparison')
            axes[i, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_rubberband_ml_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"📊 Chart saved to: {filename}")
        
        return filename


if __name__ == "__main__":
    """Run combined backtest"""
    
    print("="*80)
    print("🎯 COMBINED BACKTEST: ML PREDICTIONS + RUBBER-BAND INDICATOR")
    print("="*80)
    
    # Initialize
    backtest = CombinedBacktest()
    
    # Load optimized Rubber-Band parameters
    if not backtest.load_optimized_params():
        print("❌ Failed to load optimized parameters")
        exit(1)
    
    # Load data
    if not backtest.load_data(lookback_days=90):
        print("❌ Failed to load data")
        exit(1)
    
    # Simulate ML predictions
    backtest.simulate_ml_predictions()
    
    # Calculate Rubber-Band signals
    backtest.calculate_rubberband_signals()
    
    # Run backtests
    backtest.backtest_strategies()
    
    # Print summary
    backtest.print_summary()
    
    # Save results
    backtest.save_results()
    
    # Plot comparison
    backtest.plot_comparison()
    
    print("\n✅ Combined backtest complete!")
    print("\n💡 Key Takeaways:")
    print("   • ML Only: Your current approach")
    print("   • RB Only: Pure mean-reversion strategy")
    print("   • Combined: Only take ML signals when RB agrees (quality over quantity)")
