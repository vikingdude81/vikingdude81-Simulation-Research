"""
Trading Signals Generator for BTC Price Predictions
===================================================

Converts model predictions into actionable trading signals with:
- Buy/Sell/Hold recommendations
- Confidence levels (High/Medium/Low)
- Position sizing based on volatility and confidence
- Risk-adjusted signal generation

Author: AI Trading System
Version: 1.0 - Basic Implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json


class TradingSignalGenerator:
    """
    Generate trading signals from price predictions.
    
    Signals are based on:
    1. Predicted price change vs threshold
    2. Model confidence (from Multi-Task uncertainty)
    3. Predicted volatility (for position sizing)
    """
    
    def __init__(
        self,
        buy_threshold_pct=0.5,      # Buy if predicted gain > 0.5%
        sell_threshold_pct=0.5,     # Sell if predicted loss > 0.5%
        high_conf_threshold=0.3,    # High confidence if uncertainty < 0.3%
        medium_conf_threshold=0.6,  # Medium confidence if uncertainty < 0.6%
        max_position_size=1.0,      # 100% max position
        min_position_size=0.1       # 10% min position
    ):
        """Initialize signal generator with thresholds."""
        self.buy_threshold_pct = buy_threshold_pct
        self.sell_threshold_pct = sell_threshold_pct
        self.high_conf_threshold = high_conf_threshold
        self.medium_conf_threshold = medium_conf_threshold
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        
    def calculate_expected_return(self, current_price, predicted_price):
        """Calculate expected return percentage."""
        return ((predicted_price - current_price) / current_price) * 100
    
    def get_confidence_level(self, uncertainty_pct):
        """
        Determine confidence level based on uncertainty.
        
        Lower uncertainty = Higher confidence
        """
        if uncertainty_pct < self.high_conf_threshold:
            return "HIGH"
        elif uncertainty_pct < self.medium_conf_threshold:
            return "MEDIUM"
        else:
            return "LOW"
    
    def calculate_position_size(self, confidence_level, predicted_volatility_pct):
        """
        Calculate position size based on confidence and volatility.
        
        Higher confidence + Lower volatility = Larger position
        """
        # Base position by confidence
        if confidence_level == "HIGH":
            base_size = 0.8
        elif confidence_level == "MEDIUM":
            base_size = 0.5
        else:  # LOW
            base_size = 0.2
        
        # Adjust for volatility (reduce size in high volatility)
        # Normal BTC volatility is around 3-4% daily
        if predicted_volatility_pct > 5.0:
            volatility_adjustment = 0.6
        elif predicted_volatility_pct > 3.0:
            volatility_adjustment = 0.8
        else:
            volatility_adjustment = 1.0
        
        position_size = base_size * volatility_adjustment
        
        # Clamp to min/max
        return max(self.min_position_size, min(self.max_position_size, position_size))
    
    def generate_signal(
        self,
        current_price,
        predicted_price,
        uncertainty_pct,
        predicted_volatility_pct=None
    ):
        """
        Generate trading signal from predictions.
        
        Returns:
            dict with keys: action, confidence, position_size, expected_return,
                           reasoning, current_price, predicted_price
        """
        # Calculate expected return
        expected_return = self.calculate_expected_return(current_price, predicted_price)
        
        # Get confidence level
        confidence = self.get_confidence_level(uncertainty_pct)
        
        # Use default volatility if not provided (3% for BTC)
        if predicted_volatility_pct is None:
            predicted_volatility_pct = 3.0
        
        # Determine action
        if expected_return > self.buy_threshold_pct and confidence in ["HIGH", "MEDIUM"]:
            action = "BUY"
            reasoning = f"Predicted gain of {expected_return:.2f}% exceeds threshold ({self.buy_threshold_pct}%)"
        elif expected_return < -self.sell_threshold_pct and confidence in ["HIGH", "MEDIUM"]:
            action = "SELL"
            reasoning = f"Predicted loss of {expected_return:.2f}% exceeds threshold ({self.sell_threshold_pct}%)"
        else:
            action = "HOLD"
            if confidence == "LOW":
                reasoning = f"Low confidence (uncertainty {uncertainty_pct:.2f}%) - staying in cash"
            else:
                reasoning = f"Expected return {expected_return:.2f}% below threshold ({self.buy_threshold_pct}%)"
        
        # Calculate position size
        position_size = self.calculate_position_size(confidence, predicted_volatility_pct)
        
        # Don't take position on HOLD or LOW confidence
        if action == "HOLD" or confidence == "LOW":
            position_size = 0.0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "confidence": confidence,
            "position_size": round(position_size, 2),
            "expected_return_pct": round(expected_return, 2),
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "uncertainty_pct": round(uncertainty_pct, 3),
            "predicted_volatility_pct": round(predicted_volatility_pct, 2),
            "reasoning": reasoning
        }


class SignalBacktester:
    """
    Backtest trading signals on historical data.
    """
    
    def __init__(self, initial_capital=10000, trading_fee_pct=0.1):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital in USD
            trading_fee_pct: Trading fee percentage (default 0.1% for most exchanges)
        """
        self.initial_capital = initial_capital
        self.trading_fee_pct = trading_fee_pct
        
    def backtest(self, predictions_df, signal_generator):
        """
        Backtest signals on historical predictions.
        
        Args:
            predictions_df: DataFrame with columns:
                - timestamp
                - current_price (actual price at prediction time)
                - predicted_price
                - actual_price (actual price at target time)
                - uncertainty_pct (optional)
                - predicted_volatility_pct (optional)
            signal_generator: TradingSignalGenerator instance
        
        Returns:
            dict with backtest results
        """
        # Ensure required columns exist
        required_cols = ['timestamp', 'current_price', 'predicted_price', 'actual_price']
        missing_cols = [col for col in required_cols if col not in predictions_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add default uncertainty if not present
        if 'uncertainty_pct' not in predictions_df.columns:
            predictions_df['uncertainty_pct'] = 0.5  # Medium confidence
        
        # Initialize tracking
        capital = self.initial_capital
        position = 0.0  # BTC held
        trades = []
        capital_history = [capital]
        
        for idx, row in predictions_df.iterrows():
            # Generate signal
            signal = signal_generator.generate_signal(
                current_price=row['current_price'],
                predicted_price=row['predicted_price'],
                uncertainty_pct=row['uncertainty_pct'],
                predicted_volatility_pct=row.get('predicted_volatility_pct', None)
            )
            
            # Execute trade based on signal
            if signal['action'] == 'BUY' and position == 0:
                # Buy BTC
                trade_amount = capital * signal['position_size']
                fee = trade_amount * (self.trading_fee_pct / 100)
                btc_bought = (trade_amount - fee) / row['current_price']
                
                position = btc_bought
                capital -= trade_amount
                
                trades.append({
                    'timestamp': row['timestamp'],
                    'action': 'BUY',
                    'price': row['current_price'],
                    'amount': btc_bought,
                    'cost': trade_amount,
                    'fee': fee,
                    'confidence': signal['confidence']
                })
                
            elif signal['action'] == 'SELL' and position > 0:
                # Sell BTC
                sell_value = position * row['current_price']
                fee = sell_value * (self.trading_fee_pct / 100)
                
                capital += (sell_value - fee)
                
                trades.append({
                    'timestamp': row['timestamp'],
                    'action': 'SELL',
                    'price': row['current_price'],
                    'amount': position,
                    'value': sell_value,
                    'fee': fee,
                    'confidence': signal['confidence']
                })
                
                position = 0.0
            
            # Calculate current portfolio value
            portfolio_value = capital + (position * row['actual_price'])
            capital_history.append(portfolio_value)
        
        # Close any open position at final price
        if position > 0:
            final_price = predictions_df.iloc[-1]['actual_price']
            final_value = position * final_price
            fee = final_value * (self.trading_fee_pct / 100)
            capital += (final_value - fee)
            
            trades.append({
                'timestamp': predictions_df.iloc[-1]['timestamp'],
                'action': 'SELL',
                'price': final_price,
                'amount': position,
                'value': final_value,
                'fee': fee,
                'confidence': 'FORCED'
            })
        
        # Calculate metrics
        final_capital = capital
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate buy & hold return
        buy_hold_return = ((predictions_df.iloc[-1]['actual_price'] - predictions_df.iloc[0]['current_price']) / 
                          predictions_df.iloc[0]['current_price']) * 100
        
        # Win rate
        winning_trades = 0
        total_trades = len([t for t in trades if t['action'] == 'SELL' and t['confidence'] != 'FORCED'])
        
        for i, trade in enumerate(trades):
            if trade['action'] == 'SELL' and i > 0:
                buy_trade = trades[i-1]
                if trade['price'] > buy_trade['price']:
                    winning_trades += 1
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate Sharpe ratio (simplified - using daily returns)
        returns = pd.Series(capital_history).pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365) if len(returns) > 0 else 0
        
        # Max drawdown
        capital_series = pd.Series(capital_history)
        running_max = capital_series.expanding().max()
        drawdown = (capital_series - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': round(final_capital, 2),
            'total_return_pct': round(total_return, 2),
            'buy_hold_return_pct': round(buy_hold_return, 2),
            'outperformance_pct': round(total_return - buy_hold_return, 2),
            'num_trades': len(trades),
            'winning_trades': winning_trades,
            'win_rate_pct': round(win_rate, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'trades': trades,
            'capital_history': capital_history
        }


def load_latest_predictions(predictions_file='predictions_forecast.csv'):
    """Load the latest predictions from file."""
    try:
        df = pd.read_csv(predictions_file)
        print(f"✅ Loaded {len(df)} predictions from {predictions_file}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {predictions_file}")
        return None
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return None


def generate_live_signal():
    """
    Generate a live trading signal from the latest prediction.
    """
    print("\n" + "="*80)
    print("🎯 LIVE TRADING SIGNAL GENERATOR")
    print("="*80 + "\n")
    
    # Load predictions
    predictions_df = load_latest_predictions()
    if predictions_df is None:
        return
    
    # First row is current price (0% change)
    current = predictions_df.iloc[0]
    current_price = current['Most_Likely_Price']
    
    # Get 1-hour ahead prediction (next row)
    if len(predictions_df) > 1:
        next_hour = predictions_df.iloc[1]
        prediction_time = next_hour['Time']
        predicted_price_low = next_hour['Worst_Case_Price']
        predicted_price_mid = next_hour['Most_Likely_Price']
        predicted_price_high = next_hour['Best_Case_Price']
    else:
        print("❌ Not enough prediction data")
        return
    
    # Get 12-hour ahead prediction for longer-term view
    if len(predictions_df) >= 12:
        future = predictions_df.iloc[11]  # 12 hours ahead
        future_time = future['Time']
        future_price_mid = future['Most_Likely_Price']
        future_change = float(future['Percent_Change_Mid'].strip('%'))
    else:
        future_time = "N/A"
        future_price_mid = None
        future_change = None
    
    # Create signal generator
    generator = TradingSignalGenerator(
        buy_threshold_pct=0.5,   # Buy if predicted gain > 0.5%
        sell_threshold_pct=0.5   # Sell if predicted loss > 0.5%
    )
    
    # Calculate uncertainty from prediction range
    price_range = predicted_price_high - predicted_price_low
    uncertainty_pct = (price_range / predicted_price_mid) * 100
    
    # Estimate volatility from prediction spread
    volatility_pct = uncertainty_pct * 0.5  # Rough approximation
    
    print("📊 Current Market Data:")
    print(f"   • Current Time: {current['Time']}")
    print(f"   • Current Price: ${current_price:,.2f}")
    print(f"\n🔮 1-Hour Prediction:")
    print(f"   • Time: {prediction_time}")
    print(f"   • Worst Case: ${predicted_price_low:,.2f} ({next_hour['Percent_Change_Low']})")
    print(f"   • Most Likely: ${predicted_price_mid:,.2f} ({next_hour['Percent_Change_Mid']})")
    print(f"   • Best Case: ${predicted_price_high:,.2f} ({next_hour['Percent_Change_High']})")
    
    if future_price_mid:
        print(f"\n📅 12-Hour Outlook:")
        print(f"   • Time: {future_time}")
        print(f"   • Predicted Price: ${future_price_mid:,.2f}")
        print(f"   • Expected Change: {future_change:+.2f}%")
    
    # Generate signal based on most likely prediction
    signal = generator.generate_signal(
        current_price=current_price,
        predicted_price=predicted_price_mid,
        uncertainty_pct=uncertainty_pct,
        predicted_volatility_pct=volatility_pct
    )
    
    # Display signal
    print("\n" + "="*80)
    print("📡 TRADING SIGNAL")
    print("="*80)
    print(f"⏰ Timestamp: {signal['timestamp']}")
    print(f"💰 Current Price: ${signal['current_price']:,.2f}")
    print(f"🎯 Predicted Price (1h): ${signal['predicted_price']:,.2f}")
    print(f"📈 Expected Return (1h): {signal['expected_return_pct']:+.2f}%")
    if future_change:
        print(f"📈 Expected Return (12h): {future_change:+.2f}%")
    print(f"\n🚦 ACTION: {signal['action']}")
    print(f"🎖️  CONFIDENCE: {signal['confidence']}")
    print(f"📊 POSITION SIZE: {signal['position_size']:.0%}")
    print(f"📉 Uncertainty: {signal['uncertainty_pct']:.3f}%")
    print(f"🌊 Estimated Volatility: {signal['predicted_volatility_pct']:.2f}%")
    print(f"\n💡 Reasoning: {signal['reasoning']}")
    
    # Add interpretation
    print("\n💭 Interpretation:")
    if signal['action'] == 'BUY':
        print(f"   ✅ Model predicts price increase to ${signal['predicted_price']:,.2f}")
        print(f"   ✅ Confidence is {signal['confidence']} - take {signal['position_size']:.0%} position")
        if future_change and future_change > 1.0:
            print(f"   ✅ Strong 12h outlook: +{future_change:.2f}% expected")
    elif signal['action'] == 'SELL':
        print(f"   ⚠️  Model predicts price decrease to ${signal['predicted_price']:,.2f}")
        print(f"   ⚠️  Consider reducing exposure or taking short position")
    else:
        print(f"   ⏸️  No clear signal - price change below threshold or confidence too low")
        print(f"   ⏸️  Wait for better opportunity")
    
    print("="*80 + "\n")
    
    # Save signal to file
    signal_data = signal.copy()
    signal_data['prediction_1h'] = {
        'time': prediction_time,
        'worst_case': float(predicted_price_low),
        'most_likely': float(predicted_price_mid),
        'best_case': float(predicted_price_high)
    }
    if future_price_mid:
        signal_data['prediction_12h'] = {
            'time': future_time,
            'price': float(future_price_mid),
            'change_pct': float(future_change)
        }
    
    signal_file = 'latest_signal.json'
    with open(signal_file, 'w') as f:
        json.dump(signal_data, f, indent=2)
    print(f"✅ Signal saved to {signal_file}\n")
    
    return signal


def run_backtest_example():
    """
    Example: Create synthetic data and run backtest.
    
    In practice, you would load actual historical predictions.
    """
    print("\n" + "="*80)
    print("📊 BACKTESTING EXAMPLE (Synthetic Data)")
    print("="*80 + "\n")
    
    # Create synthetic predictions for demonstration
    np.random.seed(42)
    n_predictions = 100
    
    # Simulate a trending market with noise
    base_price = 80000
    trend = np.linspace(0, 0.15, n_predictions)  # 15% uptrend
    noise = np.random.normal(0, 0.02, n_predictions)  # 2% noise
    
    current_prices = base_price * (1 + trend + noise)
    
    # Model predictions (slightly better than random)
    prediction_error = np.random.normal(0, 0.005, n_predictions)  # 0.5% prediction error
    predicted_prices = current_prices * (1 + trend[::-1] * 0.1 + prediction_error)
    
    # Actual future prices
    actual_prices = current_prices * (1 + np.random.normal(0.001, 0.02, n_predictions))
    
    # Create DataFrame
    predictions_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_predictions, freq='D'),
        'current_price': current_prices,
        'predicted_price': predicted_prices,
        'actual_price': actual_prices,
        'uncertainty_pct': np.random.uniform(0.2, 0.8, n_predictions),
        'predicted_volatility_pct': np.random.uniform(2.5, 4.5, n_predictions)
    })
    
    # Create signal generator and backtester
    generator = TradingSignalGenerator(
        buy_threshold_pct=0.5,
        sell_threshold_pct=0.5
    )
    
    backtester = SignalBacktester(
        initial_capital=10000,
        trading_fee_pct=0.1
    )
    
    # Run backtest
    results = backtester.backtest(predictions_df, generator)
    
    # Display results
    print("="*80)
    print("📈 BACKTEST RESULTS")
    print("="*80)
    print(f"💵 Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"💰 Final Capital: ${results['final_capital']:,.2f}")
    print(f"📊 Total Return: {results['total_return_pct']:+.2f}%")
    print(f"🎯 Buy & Hold Return: {results['buy_hold_return_pct']:+.2f}%")
    print(f"⚡ Outperformance: {results['outperformance_pct']:+.2f}%")
    print(f"\n📈 Number of Trades: {results['num_trades']}")
    print(f"✅ Winning Trades: {results['winning_trades']}")
    print(f"🎯 Win Rate: {results['win_rate_pct']:.1f}%")
    print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"📉 Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print("="*80 + "\n")
    
    # Show sample trades
    if results['trades']:
        print("📝 Sample Trades (First 5):")
        print("-" * 80)
        for trade in results['trades'][:5]:
            print(f"{trade['timestamp']} | {trade['action']:4s} | "
                  f"${trade['price']:8,.2f} | "
                  f"{trade.get('amount', 0):.6f} BTC | "
                  f"Confidence: {trade['confidence']}")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    # Run live signal generation
    generate_live_signal()
    
    # Uncomment to run backtest example
    # run_backtest_example()
