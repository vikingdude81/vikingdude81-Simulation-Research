"""
UCB Reinforcement Learning Backtest

Simulates how UCB learns optimal asset allocation over time by:
1. Loading historical predictions for BTC, ETH, SOL
2. Running UCB allocation at each time step
3. Updating UCB with actual returns
4. Tracking performance vs fixed strategies

Expected outcome: UCB learns to allocate more to consistently performing assets
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from ucb_asset_selector import UCBAssetSelector


class UCBBacktest:
    """Backtest UCB reinforcement learning for asset allocation"""
    
    def __init__(self, lookback_days=90):
        """
        Initialize UCB backtest
        
        Args:
            lookback_days: Number of days of historical data to use
        """
        self.lookback_days = lookback_days
        self.ucb = UCBAssetSelector(
            assets=['BTC', 'ETH', 'SOL'],
            exploration_param=1.5,
            persistence_file='ucb_backtest_state.json'
        )
        
        # Performance tracking
        self.history = []
        self.ucb_returns = []
        self.equal_weight_returns = []
        self.btc_only_returns = []
        
    def load_historical_data(self):
        """Load historical price data for all assets"""
        print("📊 Loading historical data...")
        
        data = {}
        for asset in ['BTC', 'ETH', 'SOL']:
            file_path = f'DATA/yf_{asset.lower()}_1h.csv'
            
            if not Path(file_path).exists():
                print(f"   ⚠️  {asset} data not found: {file_path}")
                continue
            
            df = pd.read_csv(file_path)
            df['time'] = pd.to_datetime(df['time'], utc=True)  # Ensure UTC timezone
            df = df.sort_values('time')
            
            # Rename 'close' to 'price' for consistency
            if 'close' in df.columns:
                df['price'] = df['close']
            
            # Keep last N days
            cutoff_date = df['time'].max() - timedelta(days=self.lookback_days)
            df = df[df['time'] >= cutoff_date].copy()
            
            # Calculate hourly returns
            df['returns'] = df['price'].pct_change()
            
            data[asset] = df
            print(f"   ✅ {asset}: {len(df)} hours ({df['time'].min().date()} to {df['time'].max().date()})")
        
        return data
    
    def simulate_predictions(self, data):
        """
        Simulate predictions based on historical data
        
        For backtesting, we'll use actual future returns as "predictions"
        with some noise to simulate model uncertainty
        """
        print("\n🔮 Simulating prediction signals...")
        
        # Find overlapping date range across all assets
        start_dates = [data[asset]['time'].min() for asset in ['BTC', 'ETH', 'SOL']]
        end_dates = [data[asset]['time'].max() for asset in ['BTC', 'ETH', 'SOL']]
        
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        print(f"   Common date range: {common_start.date()} to {common_end.date()}")
        
        predictions = {}
        
        for asset in ['BTC', 'ETH', 'SOL']:
            df = data[asset]
            
            # Filter to common date range
            mask = (df['time'] >= common_start) & (df['time'] <= common_end)
            df_filtered = df[mask].copy().reset_index(drop=True)
            
            pred_list = []
            
            # Generate predictions for each timepoint (except last 12 hours)
            for i in range(len(df_filtered) - 12):
                current_price = df_filtered.iloc[i]['price']
                future_price = df_filtered.iloc[i + 12]['price']
                actual_return = (future_price - current_price) / current_price
                
                # Add noise to simulate prediction uncertainty (±0.5% std)
                predicted_return = actual_return + np.random.normal(0, 0.005)
                
                pred_list.append({
                    'time': df_filtered.iloc[i]['time'],
                    'current_price': current_price,
                    'predicted_return': predicted_return,
                    'actual_return': actual_return
                })
            
            if len(pred_list) > 0:
                predictions[asset] = pd.DataFrame(pred_list)
                print(f"   ✅ {asset}: {len(pred_list)} predictions generated")
            else:
                print(f"   ⚠️  {asset}: No predictions generated")
        
        return predictions if len(predictions) == 3 else None
        
        return result_predictions
    
    def run_backtest(self, predictions):
        """
        Run UCB backtest simulation
        
        At each timestep:
        1. Get UCB allocation
        2. Calculate portfolio return based on actual returns
        3. Update UCB with observed returns
        """
        print("\n🎰 Running UCB backtest simulation...")
        print("="*80)
        
        # Reset UCB state
        self.ucb.reset()
        
        # Merge predictions on time for all assets
        df_merged = None
        for asset in ['BTC', 'ETH', 'SOL']:
            if asset not in predictions or len(predictions[asset]) == 0:
                print(f"❌ No predictions for {asset}")
                return
            
            df = predictions[asset][['time', 'actual_return']].copy()
            df = df.rename(columns={'actual_return': f'{asset}_return'})
            
            if df_merged is None:
                df_merged = df
            else:
                df_merged = df_merged.merge(df, on='time', how='inner')
        
        if df_merged is None or len(df_merged) == 0:
            print("❌ No common timepoints found across assets")
            return
        
        print(f"Backtesting {len(df_merged)} time periods...")
        print(f"Date range: {df_merged['time'].min().date() if hasattr(df_merged['time'].min(), 'date') else df_merged['time'].min()} to {df_merged['time'].max().date() if hasattr(df_merged['time'].max(), 'date') else df_merged['time'].max()}")
        print("="*80)
        
        for i, row in df_merged.iterrows():
            # Get UCB allocation
            allocation = self.ucb.get_allocation_weights(top_n=3)
            
            # Calculate returns for each strategy
            btc_return = row['BTC_return']
            eth_return = row['ETH_return']
            sol_return = row['SOL_return']
            
            ucb_return = (allocation['BTC'] * btc_return + 
                         allocation['ETH'] * eth_return + 
                         allocation['SOL'] * sol_return)
            
            equal_return = (btc_return + eth_return + sol_return) / 3
            
            # Update UCB with actual returns (this is how it learns!)
            self.ucb.update('BTC', btc_return)
            self.ucb.update('ETH', eth_return)
            self.ucb.update('SOL', sol_return)
            
            # Track performance
            self.history.append({
                'time': row['time'],
                'allocation_BTC': allocation['BTC'],
                'allocation_ETH': allocation['ETH'],
                'allocation_SOL': allocation['SOL'],
                'ucb_return': ucb_return,
                'equal_weight_return': equal_return,
                'btc_only_return': btc_return,
                'ucb_cumulative': (1 + ucb_return) if i == 0 else self.history[-1]['ucb_cumulative'] * (1 + ucb_return),
                'equal_cumulative': (1 + equal_return) if i == 0 else self.history[-1]['equal_cumulative'] * (1 + equal_return),
                'btc_cumulative': (1 + btc_return) if i == 0 else self.history[-1]['btc_cumulative'] * (1 + btc_return)
            })
            
            # Print progress every 200 periods
            if (i + 1) % 200 == 0:
                ucb_total = (self.history[-1]['ucb_cumulative'] - 1) * 100
                equal_total = (self.history[-1]['equal_cumulative'] - 1) * 100
                btc_total = (self.history[-1]['btc_cumulative'] - 1) * 100
                
                print(f"Period {i+1}/{len(df_merged)}: UCB={ucb_total:+.2f}%, Equal={equal_total:+.2f}%, BTC={btc_total:+.2f}%")
                print(f"   Current allocation: BTC={allocation['BTC']*100:.1f}%, ETH={allocation['ETH']*100:.1f}%, SOL={allocation['SOL']*100:.1f}%")
        
        self.history = pd.DataFrame(self.history)
        print("\n✅ Backtest complete!")
    
    def analyze_results(self):
        """Analyze and display backtest results"""
        print("\n" + "="*80)
        print("📊 UCB BACKTEST RESULTS")
        print("="*80)
        
        if len(self.history) == 0:
            print("⚠️  No backtest data available")
            return
        
        # Calculate final returns
        ucb_total = (self.history['ucb_cumulative'].iloc[-1] - 1) * 100
        equal_total = (self.history['equal_cumulative'].iloc[-1] - 1) * 100
        btc_total = (self.history['btc_cumulative'].iloc[-1] - 1) * 100
        
        print(f"\n🎯 FINAL RETURNS ({len(self.history)} periods):")
        print(f"   UCB Strategy:          {ucb_total:+.2f}%")
        print(f"   Equal Weight (33/33/33): {equal_total:+.2f}%")
        print(f"   BTC Only:              {btc_total:+.2f}%")
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"   UCB vs Equal Weight: {ucb_total - equal_total:+.2f}% {'✅ Better' if ucb_total > equal_total else '❌ Worse'}")
        print(f"   UCB vs BTC Only:     {ucb_total - btc_total:+.2f}% {'✅ Better' if ucb_total > btc_total else '❌ Worse'}")
        
        # Risk metrics
        ucb_sharpe = self.history['ucb_return'].mean() / (self.history['ucb_return'].std() + 1e-8) * np.sqrt(365 * 2)
        equal_sharpe = self.history['equal_weight_return'].mean() / (self.history['equal_weight_return'].std() + 1e-8) * np.sqrt(365 * 2)
        btc_sharpe = self.history['btc_only_return'].mean() / (self.history['btc_only_return'].std() + 1e-8) * np.sqrt(365 * 2)
        
        print(f"\n⚖️  RISK-ADJUSTED RETURNS (Sharpe Ratio):")
        print(f"   UCB Strategy:          {ucb_sharpe:.2f}")
        print(f"   Equal Weight:          {equal_sharpe:.2f}")
        print(f"   BTC Only:              {btc_sharpe:.2f}")
        
        # Allocation evolution
        print(f"\n📊 FINAL UCB ALLOCATION:")
        final_alloc = {
            'BTC': self.history['allocation_BTC'].iloc[-1],
            'ETH': self.history['allocation_ETH'].iloc[-1],
            'SOL': self.history['allocation_SOL'].iloc[-1]
        }
        for asset, weight in final_alloc.items():
            print(f"   {asset}: {weight*100:.1f}%")
        
        # Average allocation over time
        print(f"\n📊 AVERAGE ALLOCATION (learned preferences):")
        avg_alloc = {
            'BTC': self.history['allocation_BTC'].mean(),
            'ETH': self.history['allocation_ETH'].mean(),
            'SOL': self.history['allocation_SOL'].mean()
        }
        for asset, weight in avg_alloc.items():
            print(f"   {asset}: {weight*100:.1f}%")
        
        # UCB statistics
        print(f"\n🎰 UCB LEARNING STATISTICS:")
        summary = self.ucb.get_performance_summary()
        for asset, stats in summary['assets'].items():
            print(f"   {asset}:")
            print(f"      Selections: {stats['selections']} ({stats['selection_rate']*100:.1f}%)")
            print(f"      Avg Reward: {stats['avg_reward']*100:+.4f}%")
        
        print("\n" + "="*80)
        
        # Save results
        output = {
            'backtest_summary': {
                'periods': len(self.history),
                'date_range': {
                    'start': self.history['time'].iloc[0].isoformat() if hasattr(self.history['time'].iloc[0], 'isoformat') else str(self.history['time'].iloc[0]),
                    'end': self.history['time'].iloc[-1].isoformat() if hasattr(self.history['time'].iloc[-1], 'isoformat') else str(self.history['time'].iloc[-1])
                },
                'final_returns': {
                    'ucb': float(ucb_total),
                    'equal_weight': float(equal_total),
                    'btc_only': float(btc_total)
                },
                'sharpe_ratios': {
                    'ucb': float(ucb_sharpe),
                    'equal_weight': float(equal_sharpe),
                    'btc_only': float(btc_sharpe)
                },
                'final_allocation': {k: float(v) for k, v in final_alloc.items()},
                'avg_allocation': {k: float(v) for k, v in avg_alloc.items()},
                'ucb_statistics': summary
            }
        }
        
        with open('ucb_backtest_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Results saved to: ucb_backtest_results.json")
        
        return output
    
    def plot_results(self):
        """Plot backtest results"""
        if len(self.history) == 0:
            print("⚠️  No data to plot")
            return
        
        print("\n📊 Generating performance charts...")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Cumulative returns
        ax1 = axes[0]
        self.history.plot(x='time', y='ucb_cumulative', ax=ax1, label='UCB Strategy', linewidth=2)
        self.history.plot(x='time', y='equal_cumulative', ax=ax1, label='Equal Weight (33/33/33)', linewidth=2, linestyle='--')
        self.history.plot(x='time', y='btc_cumulative', ax=ax1, label='BTC Only', linewidth=2, linestyle=':')
        ax1.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return (1 = 100%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Allocation evolution
        ax2 = axes[1]
        self.history.plot(x='time', y='allocation_BTC', ax=ax2, label='BTC', linewidth=2)
        self.history.plot(x='time', y='allocation_ETH', ax=ax2, label='ETH', linewidth=2)
        self.history.plot(x='time', y='allocation_SOL', ax=ax2, label='SOL', linewidth=2)
        ax2.set_title('UCB Allocation Evolution (Learning Over Time)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Allocation Weight')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Plot 3: Rolling Sharpe ratio
        ax3 = axes[2]
        window = 50  # 50-period rolling window
        ucb_rolling_sharpe = self.history['ucb_return'].rolling(window).mean() / self.history['ucb_return'].rolling(window).std() * np.sqrt(365 * 2)
        equal_rolling_sharpe = self.history['equal_weight_return'].rolling(window).mean() / self.history['equal_weight_return'].rolling(window).std() * np.sqrt(365 * 2)
        
        ax3.plot(self.history['time'], ucb_rolling_sharpe, label=f'UCB (rolling {window}-period)', linewidth=2)
        ax3.plot(self.history['time'], equal_rolling_sharpe, label=f'Equal Weight (rolling {window}-period)', linewidth=2, linestyle='--')
        ax3.set_title('Risk-Adjusted Performance (Rolling Sharpe Ratio)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_xlabel('Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='red', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('ucb_backtest_results.png', dpi=150, bbox_inches='tight')
        print(f"📊 Chart saved to: ucb_backtest_results.png")
        
        # Show plot
        try:
            plt.show()
        except:
            print("   (Display not available, chart saved to file)")


def main():
    """Run UCB backtest"""
    print("="*80)
    print("🎰 UCB REINFORCEMENT LEARNING BACKTEST")
    print("="*80)
    
    # Initialize backtest
    backtest = UCBBacktest(lookback_days=90)
    
    # Load data
    data = backtest.load_historical_data()
    
    if len(data) < 3:
        print("❌ Not enough data for backtest")
        return
    
    # Simulate predictions
    predictions = backtest.simulate_predictions(data)
    
    # Run backtest
    backtest.run_backtest(predictions)
    
    # Analyze results
    results = backtest.analyze_results()
    
    # Plot results
    backtest.plot_results()
    
    print("\n✅ UCB backtest complete!")
    print("\n💡 Key Takeaway:")
    print("   UCB learns which assets perform best and dynamically adjusts allocation.")
    print("   Over time, it concentrates on winners while still exploring alternatives.")


if __name__ == '__main__':
    main()
