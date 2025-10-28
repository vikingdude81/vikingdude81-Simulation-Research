"""
Backtest Trading Signals on Historical Data
===========================================

Tests the trading signals strategy on actual historical BTC data
to validate performance and optimize thresholds.

Author: AI Trading System
Version: 1.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt
from trading_signals import TradingSignalGenerator, SignalBacktester


def load_historical_data(data_file='DATA/yf_btc_1h.csv'):
    """
    Load historical BTC price data.
    
    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Loaded {len(df)} rows from {data_file}")
        
        # Ensure timestamp column exists
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            print("⚠️  No timestamp column found, using index")
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
        
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {data_file}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def create_synthetic_predictions(historical_df, prediction_error_pct=0.45):
    """
    Create synthetic predictions based on historical data.
    
    Since we don't have historical predictions stored, we simulate them
    by adding realistic error to actual future prices.
    
    Args:
        historical_df: DataFrame with actual prices
        prediction_error_pct: Model RMSE (0.45% for your model)
    
    Returns:
        DataFrame with prediction scenarios
    """
    predictions = []
    
    for i in range(len(historical_df) - 12):  # Need 12 hours ahead data
        current_row = historical_df.iloc[i]
        future_1h = historical_df.iloc[i + 1]
        
        current_price = current_row['close']
        actual_1h_price = future_1h['close']
        
        # Simulate prediction with model error
        # Add normally distributed error around actual price
        error = np.random.normal(0, prediction_error_pct / 100)
        predicted_price = actual_1h_price * (1 + error)
        
        # Create prediction range (worst/best case)
        uncertainty = abs(error) * 2  # Uncertainty proportional to error
        uncertainty_pct = uncertainty * 100
        
        worst_case = predicted_price * (1 - uncertainty)
        best_case = predicted_price * (1 + uncertainty)
        
        # Estimated volatility (based on recent price changes)
        if i >= 24:
            recent_returns = historical_df.iloc[i-24:i]['close'].pct_change().dropna()
            predicted_volatility = recent_returns.std() * 100 * np.sqrt(24)  # Daily vol
        else:
            predicted_volatility = 3.0  # Default BTC volatility
        
        predictions.append({
            'timestamp': current_row['timestamp'],
            'current_price': current_price,
            'predicted_price': predicted_price,
            'actual_price': actual_1h_price,
            'worst_case': worst_case,
            'best_case': best_case,
            'uncertainty_pct': uncertainty_pct,
            'predicted_volatility_pct': predicted_volatility
        })
    
    return pd.DataFrame(predictions)


def run_backtest_with_thresholds(predictions_df, thresholds=[0.3, 0.5, 0.7, 1.0], initial_capital=10000):
    """
    Run backtests with different threshold values to find optimal settings.
    
    Args:
        predictions_df: DataFrame with predictions
        thresholds: List of threshold percentages to test
        initial_capital: Starting capital in USD
    
    Returns:
        DataFrame with results for each threshold
    """
    results = []
    
    print("\n" + "="*80)
    print("🧪 TESTING MULTIPLE THRESHOLDS")
    print("="*80 + "\n")
    
    for threshold in thresholds:
        print(f"Testing threshold: {threshold}%...")
        
        # Create signal generator with this threshold
        generator = TradingSignalGenerator(
            buy_threshold_pct=threshold,
            sell_threshold_pct=threshold,
            high_conf_threshold=0.3,
            medium_conf_threshold=0.6
        )
        
        # Run backtest
        backtester = SignalBacktester(
            initial_capital=initial_capital,
            trading_fee_pct=0.1
        )
        
        backtest_results = backtester.backtest(predictions_df, generator)
        
        # Store results
        results.append({
            'threshold_pct': threshold,
            'total_return_pct': backtest_results['total_return_pct'],
            'buy_hold_return_pct': backtest_results['buy_hold_return_pct'],
            'outperformance_pct': backtest_results['outperformance_pct'],
            'num_trades': backtest_results['num_trades'],
            'win_rate_pct': backtest_results['win_rate_pct'],
            'sharpe_ratio': backtest_results['sharpe_ratio'],
            'max_drawdown_pct': backtest_results['max_drawdown_pct'],
            'final_capital': backtest_results['final_capital']
        })
    
    return pd.DataFrame(results)


def plot_backtest_results(results_df, predictions_df, best_threshold):
    """
    Create visualization of backtest results.
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trading Signals Backtest Results', fontsize=16, fontweight='bold')
        
        # 1. Returns by Threshold
        ax1 = axes[0, 0]
        x = results_df['threshold_pct']
        ax1.plot(x, results_df['total_return_pct'], 'o-', label='Strategy Return', linewidth=2, markersize=8)
        ax1.axhline(y=results_df['buy_hold_return_pct'].iloc[0], color='r', linestyle='--', label='Buy & Hold')
        ax1.set_xlabel('Signal Threshold (%)')
        ax1.set_ylabel('Return (%)')
        ax1.set_title('Returns vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sharpe Ratio vs Number of Trades
        ax2 = axes[0, 1]
        scatter = ax2.scatter(results_df['num_trades'], results_df['sharpe_ratio'], 
                             c=results_df['threshold_pct'], cmap='viridis', s=100)
        ax2.set_xlabel('Number of Trades')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Risk-Adjusted Returns vs Trading Frequency')
        plt.colorbar(scatter, ax=ax2, label='Threshold %')
        ax2.grid(True, alpha=0.3)
        
        # 3. Win Rate and Max Drawdown
        ax3 = axes[1, 0]
        x_pos = np.arange(len(results_df))
        width = 0.35
        ax3.bar(x_pos - width/2, results_df['win_rate_pct'], width, label='Win Rate %', alpha=0.7)
        ax3.bar(x_pos + width/2, -results_df['max_drawdown_pct'], width, label='Max Drawdown %', alpha=0.7)
        ax3.set_xlabel('Threshold (%)')
        ax3.set_ylabel('Percentage')
        ax3.set_title('Win Rate vs Max Drawdown')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(results_df['threshold_pct'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # 4. Summary Table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Find best threshold row
        best_row = results_df[results_df['threshold_pct'] == best_threshold].iloc[0]
        
        table_data = [
            ['Metric', 'Value'],
            ['Best Threshold', f"{best_threshold}%"],
            ['Total Return', f"{best_row['total_return_pct']:.2f}%"],
            ['Buy & Hold Return', f"{best_row['buy_hold_return_pct']:.2f}%"],
            ['Outperformance', f"{best_row['outperformance_pct']:.2f}%"],
            ['Number of Trades', f"{int(best_row['num_trades'])}"],
            ['Win Rate', f"{best_row['win_rate_pct']:.1f}%"],
            ['Sharpe Ratio', f"{best_row['sharpe_ratio']:.2f}"],
            ['Max Drawdown', f"{best_row['max_drawdown_pct']:.2f}%"],
            ['Final Capital', f"${best_row['final_capital']:,.2f}"]
        ]
        
        table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Best Strategy Performance', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = 'backtest_results.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Backtest visualization saved to {plot_file}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")


def main():
    """
    Main backtesting workflow.
    """
    print("\n" + "="*80)
    print("📊 TRADING SIGNALS BACKTESTING")
    print("="*80 + "\n")
    
    # Configuration
    INITIAL_CAPITAL = 10000
    THRESHOLDS = [0.3, 0.5, 0.7, 1.0]
    PREDICTION_ERROR = 0.45  # Your model's RMSE
    
    print("⚙️  Configuration:")
    print(f"   • Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"   • Thresholds to Test: {THRESHOLDS}")
    print(f"   • Model RMSE: {PREDICTION_ERROR}%")
    print(f"   • Trading Fee: 0.1%\n")
    
    # Load historical data
    print("📁 Loading historical data...")
    historical_df = load_historical_data()
    
    if historical_df is None:
        print("❌ Could not load historical data. Exiting.")
        return
    
    # Use last 30 days for backtesting (720 hours)
    backtest_period = min(720, len(historical_df) - 12)
    historical_df = historical_df.tail(backtest_period + 12).reset_index(drop=True)
    
    print(f"   • Using last {backtest_period} hours for backtesting")
    print(f"   • Date Range: {historical_df.iloc[0]['timestamp']} to {historical_df.iloc[-1]['timestamp']}")
    
    # Create synthetic predictions
    print("\n🔮 Creating synthetic predictions...")
    predictions_df = create_synthetic_predictions(historical_df, PREDICTION_ERROR)
    print(f"   • Generated {len(predictions_df)} prediction scenarios")
    print(f"   • Average prediction error: {PREDICTION_ERROR}%")
    
    # Show sample predictions
    print("\n📋 Sample Predictions (first 3):")
    print("-" * 80)
    for i in range(min(3, len(predictions_df))):
        row = predictions_df.iloc[i]
        actual_return = ((row['actual_price'] - row['current_price']) / row['current_price']) * 100
        predicted_return = ((row['predicted_price'] - row['current_price']) / row['current_price']) * 100
        print(f"Time: {row['timestamp']}")
        print(f"  Current: ${row['current_price']:,.2f}")
        print(f"  Predicted: ${row['predicted_price']:,.2f} ({predicted_return:+.2f}%)")
        print(f"  Actual: ${row['actual_price']:,.2f} ({actual_return:+.2f}%)")
        print(f"  Uncertainty: {row['uncertainty_pct']:.2f}%")
        print()
    
    # Run backtests with multiple thresholds
    results_df = run_backtest_with_thresholds(predictions_df, THRESHOLDS, INITIAL_CAPITAL)
    
    # Display results
    print("\n" + "="*80)
    print("📈 BACKTEST RESULTS COMPARISON")
    print("="*80 + "\n")
    
    print(results_df.to_string(index=False))
    
    # Find best threshold (highest Sharpe ratio)
    best_idx = results_df['sharpe_ratio'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold_pct']
    best_return = results_df.loc[best_idx, 'total_return_pct']
    best_sharpe = results_df.loc[best_idx, 'sharpe_ratio']
    
    print("\n" + "="*80)
    print("🏆 BEST STRATEGY")
    print("="*80)
    print(f"Threshold: {best_threshold}%")
    print(f"Total Return: {best_return:+.2f}%")
    print(f"Sharpe Ratio: {best_sharpe:.2f}")
    print(f"Win Rate: {results_df.loc[best_idx, 'win_rate_pct']:.1f}%")
    print(f"Max Drawdown: {results_df.loc[best_idx, 'max_drawdown_pct']:.2f}%")
    print(f"Number of Trades: {int(results_df.loc[best_idx, 'num_trades'])}")
    print("="*80 + "\n")
    
    # Detailed backtest with best threshold
    print(f"📊 Running detailed backtest with {best_threshold}% threshold...\n")
    
    best_generator = TradingSignalGenerator(
        buy_threshold_pct=best_threshold,
        sell_threshold_pct=best_threshold
    )
    
    backtester = SignalBacktester(initial_capital=INITIAL_CAPITAL, trading_fee_pct=0.1)
    detailed_results = backtester.backtest(predictions_df, best_generator)
    
    # Show trade history
    print("📝 Trade History (First 10 trades):")
    print("-" * 80)
    for i, trade in enumerate(detailed_results['trades'][:10]):
        if trade['action'] == 'BUY':
            print(f"{i+1}. BUY  @ ${trade['price']:8,.2f} | {trade['amount']:.6f} BTC | "
                  f"Cost: ${trade['cost']:,.2f} | Conf: {trade['confidence']}")
        else:
            print(f"{i+1}. SELL @ ${trade['price']:8,.2f} | {trade['amount']:.6f} BTC | "
                  f"Value: ${trade['value']:,.2f} | Conf: {trade['confidence']}")
    
    if len(detailed_results['trades']) > 10:
        print(f"... and {len(detailed_results['trades']) - 10} more trades")
    print("-" * 80 + "\n")
    
    # Calculate annualized return
    days_tested = backtest_period / 24
    annualized_return = (best_return / days_tested) * 365
    
    print("📊 Performance Metrics:")
    print(f"   • Test Period: {days_tested:.1f} days")
    print(f"   • Total Return: {best_return:+.2f}%")
    print(f"   • Annualized Return: {annualized_return:+.2f}%")
    print(f"   • Buy & Hold Return: {results_df.loc[best_idx, 'buy_hold_return_pct']:+.2f}%")
    print(f"   • Outperformance: {results_df.loc[best_idx, 'outperformance_pct']:+.2f}%")
    print(f"   • Sharpe Ratio: {best_sharpe:.2f}")
    print(f"   • Max Drawdown: {results_df.loc[best_idx, 'max_drawdown_pct']:.2f}%")
    print(f"   • Win Rate: {results_df.loc[best_idx, 'win_rate_pct']:.1f}%")
    print(f"   • Total Trades: {int(results_df.loc[best_idx, 'num_trades'])}")
    
    # Save results
    results_file = 'backtest_results.json'
    results_data = {
        'config': {
            'initial_capital': INITIAL_CAPITAL,
            'thresholds_tested': THRESHOLDS,
            'prediction_error_pct': PREDICTION_ERROR,
            'backtest_period_hours': backtest_period,
            'backtest_period_days': days_tested
        },
        'best_strategy': {
            'threshold_pct': best_threshold,
            'total_return_pct': float(best_return),
            'annualized_return_pct': float(annualized_return),
            'sharpe_ratio': float(best_sharpe),
            'win_rate_pct': float(results_df.loc[best_idx, 'win_rate_pct']),
            'max_drawdown_pct': float(results_df.loc[best_idx, 'max_drawdown_pct']),
            'num_trades': int(results_df.loc[best_idx, 'num_trades'])
        },
        'all_results': results_df.to_dict('records')
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    
    # Create visualization
    plot_backtest_results(results_df, predictions_df, best_threshold)
    
    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    if best_return > 0:
        print(f"✅ Strategy is profitable: {best_return:+.2f}% return")
        if best_return > results_df.loc[best_idx, 'buy_hold_return_pct']:
            print(f"✅ Outperforms buy & hold by {results_df.loc[best_idx, 'outperformance_pct']:.2f}%")
        else:
            print(f"⚠️  Underperforms buy & hold by {-results_df.loc[best_idx, 'outperformance_pct']:.2f}%")
    else:
        print(f"❌ Strategy is unprofitable: {best_return:.2f}% loss")
    
    if best_sharpe > 1.0:
        print(f"✅ Good risk-adjusted returns (Sharpe {best_sharpe:.2f})")
    else:
        print(f"⚠️  Low risk-adjusted returns (Sharpe {best_sharpe:.2f})")
    
    if results_df.loc[best_idx, 'max_drawdown_pct'] < -20:
        print(f"⚠️  High drawdown risk ({results_df.loc[best_idx, 'max_drawdown_pct']:.2f}%)")
    else:
        print(f"✅ Acceptable drawdown ({results_df.loc[best_idx, 'max_drawdown_pct']:.2f}%)")
    
    print(f"\n🎯 Recommended threshold for live trading: {best_threshold}%")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
