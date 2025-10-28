"""
Multi-Phase System Backtest
============================

Comprehensive backtesting to compare:
- Phase A: Multi-Asset Only
- Phase C: Multi-Asset + Dominance
- Phase D: Multi-Asset + Dominance + S/R

Tests on historical data to validate performance improvements.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import real S/R analyzer
from support_resistance import SupportResistanceAnalyzer


class MultiPhaseBacktester:
    """
    Backtest the multi-phase trading system on historical data.
    Compares performance across different phase configurations.
    """
    
    def __init__(self, initial_capital: float = 10000):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital in USD
        """
        self.initial_capital = initial_capital
        self.results = {}
        
        # Initialize real S/R analyzer
        self.sr_analyzer = SupportResistanceAnalyzer()
        
    def load_historical_data(self, asset: str = 'btc', days: int = 90) -> pd.DataFrame:
        """
        Load historical price data for backtesting.
        
        Args:
            asset: Asset symbol (btc, eth, sol)
            days: Number of days to load
            
        Returns:
            DataFrame with OHLCV data
        """
        filename = f"DATA/yf_{asset.lower()}_1h.csv"
        
        try:
            df = pd.read_csv(filename)
            df['time'] = pd.to_datetime(df['time'])
            
            # Get recent data for backtesting
            cutoff = datetime.now() - timedelta(days=days)
            if df['time'].dt.tz is not None:
                cutoff = pd.Timestamp(cutoff).tz_localize('UTC')
            
            df = df[df['time'] >= cutoff].copy()
            df = df.sort_values('time').reset_index(drop=True)
            
            print(f"✅ Loaded {len(df)} hours of {asset.upper()} data ({days} days)")
            return df
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return pd.DataFrame()
    
    def simulate_predictions(self, df: pd.DataFrame, rmse: float = 0.0045) -> pd.DataFrame:
        """
        Simulate model predictions based on actual future prices + noise.
        This mimics what our LSTM model would predict.
        
        Args:
            df: Historical OHLCV data
            rmse: Model RMSE (prediction error)
            
        Returns:
            DataFrame with predictions
        """
        predictions = []
        
        # For each point, "predict" 12 hours ahead
        for i in range(len(df) - 12):
            current_price = df['close'].iloc[i]
            actual_future = df['close'].iloc[i + 12]
            
            # Simulate prediction: actual future + random noise based on RMSE
            noise = np.random.normal(0, rmse * current_price)
            predicted = actual_future + noise
            
            predictions.append({
                'time': df['time'].iloc[i],
                'current_price': current_price,
                'predicted_price': predicted,
                'actual_future': actual_future,
                'prediction_error': abs(predicted - actual_future) / actual_future
            })
        
        return pd.DataFrame(predictions)
    
    def calculate_signal(self, current_price: float, predicted_price: float,
                        threshold: float = 0.003, use_sr: bool = False,
                        sr_bonus: float = 0) -> Dict:
        """
        Calculate trading signal.
        
        Args:
            current_price: Current asset price
            predicted_price: Predicted price
            threshold: Signal threshold
            use_sr: Whether to apply S/R bonus
            sr_bonus: S/R proximity bonus
            
        Returns:
            Signal dictionary
        """
        expected_return = (predicted_price - current_price) / current_price
        
        # Apply S/R bonus if enabled
        if use_sr:
            expected_return += sr_bonus
        
        # Determine signal
        if expected_return > threshold:
            signal = 'BUY'
        elif expected_return < -threshold:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'expected_return': expected_return,
            'current_price': current_price,
            'predicted_price': predicted_price
        }
    
    def calculate_real_sr_bonus(self, asset: str, current_price: float, 
                                signal: str, timestamp: pd.Timestamp) -> Tuple[float, Dict]:
        """
        Calculate REAL S/R proximity bonus using actual support/resistance levels.
        
        Args:
            asset: Asset symbol
            current_price: Current price
            signal: Trading signal
            timestamp: Current timestamp
            
        Returns:
            Tuple of (proximity_bonus, sr_data)
        """
        try:
            # Get S/R levels using real analyzer
            levels = self.sr_analyzer.get_all_levels(asset.lower(), '1h')
            
            if not levels or levels['current_price'] == 0:
                return 0, {}
            
            # Find nearest levels
            nearest = self.sr_analyzer.find_nearest_levels(current_price, levels)
            
            # Calculate proximity bonus
            bonuses = self.sr_analyzer.calculate_proximity_bonus(
                current_price,
                nearest['nearest_support'],
                nearest['nearest_resistance']
            )
            
            # Return appropriate bonus based on signal
            if signal == 'BUY':
                bonus = bonuses['buy_bonus']
            elif signal == 'SELL':
                bonus = bonuses['sell_bonus']
            else:
                bonus = 0
            
            sr_data = {
                'nearest_support': nearest['nearest_support'],
                'nearest_resistance': nearest['nearest_resistance'],
                'bonus': bonus
            }
            
            return bonus, sr_data
            
        except Exception as e:
            # If S/R calculation fails, return no bonus
            return 0, {}
    
    def simulate_sr_bonus(self, current_price: float, volatility: float = 0.02) -> float:
        """
        DEPRECATED: Use calculate_real_sr_bonus instead.
        This was for simulation only.
        """
        return 0
    
    def backtest_strategy(self, predictions: pd.DataFrame, phase: str = 'D',
                         use_dominance: bool = True, use_sr: bool = True,
                         asset: str = 'btc') -> Dict:
        """
        Backtest a strategy configuration.
        
        Args:
            predictions: DataFrame with predictions
            phase: Phase identifier (A, C, D)
            use_dominance: Apply dominance-based position sizing
            use_sr: Apply S/R proximity bonuses
            asset: Asset symbol
            
        Returns:
            Backtest results dictionary
        """
        capital = self.initial_capital
        position = 0  # BTC held
        position_value = 0
        trades = []
        equity_curve = []
        sr_bonus_count = 0
        
        for i, row in predictions.iterrows():
            current_price = row['current_price']
            predicted_price = row['predicted_price']
            actual_future = row['actual_future']
            timestamp = row['time']
            
            # Calculate expected return
            expected_return = (predicted_price - current_price) / current_price
            
            # Determine base signal
            if expected_return > 0.003:
                signal = 'BUY'
            elif expected_return < -0.003:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Calculate REAL S/R bonus if enabled
            sr_bonus = 0
            sr_data = {}
            if use_sr and signal != 'HOLD':
                sr_bonus, sr_data = self.calculate_real_sr_bonus(
                    asset, current_price, signal, timestamp
                )
                if sr_bonus > 0:
                    sr_bonus_count += 1
                    expected_return += sr_bonus
            
            # Simulate dominance-based position sizing
            position_modifier = 1.0
            if use_dominance:
                # Simulate market regimes (70% neutral, 15% fear, 15% greed)
                regime = np.random.choice([0.4, 0.7, 1.0], p=[0.15, 0.70, 0.15])
                position_modifier = regime
            
            # Execute trades
            if signal == 'BUY' and position == 0:
                # Calculate position size (45% max allocation * modifiers)
                allocation = 0.45 * position_modifier
                
                # Buy
                amount_to_invest = capital * allocation
                position = amount_to_invest / current_price
                capital -= amount_to_invest
                
                # Record trade
                trades.append({
                    'time': timestamp,
                    'type': 'BUY',
                    'price': current_price,
                    'amount': position,
                    'capital': capital,
                    'expected_return': expected_return,
                    'sr_bonus': sr_bonus,
                    'position_modifier': position_modifier,
                    'sr_data': sr_data
                })
            
            elif signal == 'SELL' and position > 0:
                # Sell position at actual future price (12h later)
                # Use actual_future to simulate holding for 12h
                sell_price = actual_future
                capital += position * sell_price
                
                # Calculate return
                buy_price = trades[-1]['price']
                actual_return = (sell_price - buy_price) / buy_price
                
                # Record trade
                trades.append({
                    'time': timestamp,
                    'type': 'SELL',
                    'price': sell_price,
                    'amount': position,
                    'capital': capital,
                    'return': actual_return,
                    'expected_return': expected_return
                })
                
                position = 0
            
            # Calculate total equity (cash + position value)
            position_value = position * current_price
            total_equity = capital + position_value
            
            equity_curve.append({
                'time': timestamp,
                'capital': capital,
                'position_value': position_value,
                'total_equity': total_equity,
                'returns': (total_equity - self.initial_capital) / self.initial_capital
            })
        
        # Close any open position at end
        if position > 0:
            final_price = predictions['current_price'].iloc[-1]
            capital += position * final_price
            position = 0
        
        # Calculate performance metrics
        equity_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(trades)
        
        # Win rate
        completed_trades = trades_df[trades_df['type'] == 'SELL']
        if len(completed_trades) > 0:
            wins = len(completed_trades[completed_trades['return'] > 0])
            win_rate = wins / len(completed_trades)
        else:
            win_rate = 0
        
        # Total return
        final_equity = capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Monthly return (annualized)
        days_traded = (predictions['time'].iloc[-1] - predictions['time'].iloc[0]).days
        if days_traded > 0:
            daily_return = total_return / days_traded
            monthly_return = daily_return * 30
        else:
            monthly_return = 0
        
        # Max drawdown
        if len(equity_df) > 0:
            equity_df['cummax'] = equity_df['total_equity'].cummax()
            equity_df['drawdown'] = (equity_df['total_equity'] - equity_df['cummax']) / equity_df['cummax']
            max_drawdown = equity_df['drawdown'].min()
        else:
            max_drawdown = 0
        
        # Sharpe ratio (simplified)
        if len(equity_df) > 1:
            returns = equity_df['returns'].pct_change().dropna()
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24)  # Hourly data
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        # Average return per trade
        if len(completed_trades) > 0:
            avg_return = completed_trades['return'].mean()
            avg_win = completed_trades[completed_trades['return'] > 0]['return'].mean() if wins > 0 else 0
            avg_loss = completed_trades[completed_trades['return'] < 0]['return'].mean() if (len(completed_trades) - wins) > 0 else 0
        else:
            avg_return = 0
            avg_win = 0
            avg_loss = 0
        
        # Calculate S/R impact
        sr_impact = sr_bonus_count / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        return {
            'phase': phase,
            'use_dominance': use_dominance,
            'use_sr': use_sr,
            'initial_capital': self.initial_capital,
            'final_capital': final_equity,
            'total_return': total_return,
            'monthly_return': monthly_return,
            'total_trades': len(trades_df),
            'completed_trades': len(completed_trades),
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'sr_impact_pct': sr_impact,
            'sr_bonus_count': sr_bonus_count,
            'equity_curve': equity_df,
            'trades': trades_df
        }
    
    def run_comparison_backtest(self, asset: str = 'btc', days: int = 90):
        """
        Run backtest comparing all phases.
        
        Args:
            asset: Asset to backtest
            days: Days of historical data
        """
        print(f"\n{'='*80}")
        print(f"🧪 MULTI-PHASE BACKTEST COMPARISON")
        print(f"{'='*80}")
        print(f"\n📊 Configuration:")
        print(f"   Asset: {asset.upper()}")
        print(f"   Period: {days} days")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Strategy: Multi-Asset with various enhancements")
        
        # Load historical data
        print(f"\n📈 Loading historical data...")
        df = self.load_historical_data(asset, days)
        
        if df.empty:
            print("❌ No data available")
            return
        
        # Simulate predictions
        print(f"\n🔮 Simulating model predictions (RMSE: 0.45%)...")
        predictions = self.simulate_predictions(df, rmse=0.0045)
        print(f"✅ Generated {len(predictions)} predictions")
        
        # Test configurations
        configs = [
            {'name': 'Phase A: Multi-Asset Only', 'dominance': False, 'sr': False, 'phase': 'A'},
            {'name': 'Phase C: Multi-Asset + Dominance', 'dominance': True, 'sr': False, 'phase': 'C'},
            {'name': 'Phase D: Multi-Asset + Dominance + S/R', 'dominance': True, 'sr': True, 'phase': 'D'},
        ]
        
        print(f"\n🔬 Running backtests...")
        results = []
        
        for config in configs:
            print(f"\n   Testing: {config['name']}...")
            result = self.backtest_strategy(
                predictions,
                phase=config['phase'],
                use_dominance=config['dominance'],
                use_sr=config['sr'],
                asset=asset
            )
            result['config_name'] = config['name']
            results.append(result)
        
        # Display results
        self.display_results(results)
        
        # Save results
        self.save_results(results, asset)
        
        return results
    
    def display_results(self, results: List[Dict]):
        """Display backtest results comparison."""
        print(f"\n{'='*80}")
        print(f"📊 BACKTEST RESULTS COMPARISON")
        print(f"{'='*80}\n")
        
        # Create comparison table
        print(f"{'Phase':<35} {'Return':<15} {'Monthly':<15} {'Win Rate':<12} {'Sharpe':<10} {'Trades'}")
        print(f"{'-'*35} {'-'*15} {'-'*15} {'-'*12} {'-'*10} {'-'*10}")
        
        for result in results:
            phase_name = result['config_name']
            total_return = result['total_return'] * 100
            monthly_return = result['monthly_return'] * 100
            win_rate = result['win_rate'] * 100
            sharpe = result['sharpe_ratio']
            trades = result['completed_trades']
            
            print(f"{phase_name:<35} {total_return:>6.2f}%        {monthly_return:>6.2f}%        "
                  f"{win_rate:>5.1f}%      {sharpe:>6.2f}    {trades:>6}")
        
        print(f"\n{'='*80}")
        print(f"📈 DETAILED METRICS")
        print(f"{'='*80}\n")
        
        for result in results:
            print(f"\n🎯 {result['config_name']}")
            print(f"   {'─'*70}")
            print(f"   💰 Financial Performance:")
            print(f"      Initial Capital:    ${result['initial_capital']:>12,.2f}")
            print(f"      Final Capital:      ${result['final_capital']:>12,.2f}")
            print(f"      Total Return:       {result['total_return']*100:>12.2f}%")
            print(f"      Monthly Return:     {result['monthly_return']*100:>12.2f}%")
            print(f"      Max Drawdown:       {result['max_drawdown']*100:>12.2f}%")
            
            print(f"\n   📊 Trading Statistics:")
            print(f"      Total Trades:       {result['total_trades']:>12}")
            print(f"      Completed Trades:   {result['completed_trades']:>12}")
            print(f"      Win Rate:           {result['win_rate']*100:>12.1f}%")
            print(f"      Avg Return/Trade:   {result['avg_return_per_trade']*100:>12.2f}%")
            print(f"      Avg Win:            {result['avg_win']*100:>12.2f}%")
            print(f"      Avg Loss:           {result['avg_loss']*100:>12.2f}%")
            
            print(f"\n   📈 Risk Metrics:")
            print(f"      Sharpe Ratio:       {result['sharpe_ratio']:>12.2f}")
            print(f"      Max Drawdown:       {result['max_drawdown']*100:>12.2f}%")
            
            if result['use_sr']:
                print(f"\n   🎯 S/R Enhancement:")
                print(f"      S/R Bonuses Applied: {result['sr_bonus_count']:>12} trades")
                print(f"      S/R Impact:          {result['sr_impact_pct']:>12.1f}% of trades")
        
        # Calculate improvements
        print(f"\n{'='*80}")
        print(f"📊 PHASE IMPROVEMENTS")
        print(f"{'='*80}\n")
        
        baseline = results[0]  # Phase A
        
        for i, result in enumerate(results[1:], 1):
            phase_name = result['config_name'].split(':')[0]
            
            return_improvement = (result['monthly_return'] - baseline['monthly_return']) * 100
            wr_improvement = (result['win_rate'] - baseline['win_rate']) * 100
            sharpe_improvement = result['sharpe_ratio'] - baseline['sharpe_ratio']
            
            print(f"🎯 {phase_name} vs Phase A:")
            print(f"   Monthly Return:  {baseline['monthly_return']*100:>6.2f}% → {result['monthly_return']*100:>6.2f}% "
                  f"({return_improvement:+.2f}%)")
            print(f"   Win Rate:        {baseline['win_rate']*100:>6.1f}% → {result['win_rate']*100:>6.1f}% "
                  f"({wr_improvement:+.1f}%)")
            print(f"   Sharpe Ratio:    {baseline['sharpe_ratio']:>6.2f} → {result['sharpe_ratio']:>6.2f} "
                  f"({sharpe_improvement:+.2f})")
            print()
    
    def save_results(self, results: List[Dict], asset: str):
        """Save backtest results to JSON."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'asset': asset,
            'initial_capital': self.initial_capital,
            'results': []
        }
        
        for result in results:
            output['results'].append({
                'phase': result['phase'],
                'config_name': result['config_name'],
                'use_dominance': result['use_dominance'],
                'use_sr': result['use_sr'],
                'final_capital': result['final_capital'],
                'total_return': result['total_return'],
                'monthly_return': result['monthly_return'],
                'win_rate': result['win_rate'],
                'completed_trades': result['completed_trades'],
                'avg_return_per_trade': result['avg_return_per_trade'],
                'max_drawdown': result['max_drawdown'],
                'sharpe_ratio': result['sharpe_ratio'],
                'sr_impact_pct': result['sr_impact_pct']
            })
        
        filename = f"backtest_multiphase_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Run comprehensive multi-phase backtest."""
    
    # Initialize backtester with $10,000 starting capital
    backtester = MultiPhaseBacktester(initial_capital=10000)
    
    # Run backtest on BTC for 90 days
    results = backtester.run_comparison_backtest(
        asset='btc',
        days=90
    )
    
    print(f"\n{'='*80}")
    print(f"✅ BACKTEST COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
