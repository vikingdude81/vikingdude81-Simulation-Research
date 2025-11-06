"""
Trading Specialist - Genome-Based Trading Agent

A trading agent optimized for specific market regimes through genetic evolution.
Each specialist has an 8-gene genome that controls its trading behavior.

Genome Structure:
[
    stop_loss_pct,        # 0.01-0.05 (1-5% stop loss)
    take_profit_pct,      # 0.02-0.20 (2-20% take profit)
    position_size_pct,    # 0.01-0.10 (1-10% of capital per trade)
    entry_threshold,      # 0.0-1.0 (minimum signal strength to enter)
    exit_threshold,       # 0.0-1.0 (signal strength to exit)
    max_hold_time,        # 1-14 days (maximum position hold time)
    volatility_scaling,   # 0.5-2.0 (ATR multiplier for position sizing)
    momentum_weight,      # 0.0-1.0 (trend vs mean-reversion preference)
]

Author: GA Conductor Research Team
Date: November 5, 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from dataclasses import dataclass, field

@dataclass
class Trade:
    """Record of a single trade"""
    entry_date: str
    entry_price: float
    entry_signal: float
    exit_date: str = None
    exit_price: float = None
    exit_signal: float = None
    direction: int = 1  # 1 for long, -1 for short
    position_size: float = 0.1
    pnl: float = 0.0
    return_pct: float = 0.0
    hold_time: int = 0
    exit_reason: str = None

@dataclass
class PerformanceMetrics:
    """Trading performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    avg_trade_return: float = 0.0
    profit_factor: float = 0.0
    fitness: float = 0.0
    
    def to_dict(self):
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'max_drawdown': self.max_drawdown,
            'num_trades': self.num_trades,
            'avg_trade_return': self.avg_trade_return,
            'profit_factor': self.profit_factor,
            'fitness': self.fitness
        }


class TradingSpecialist:
    """
    Trading agent with genome-controlled behavior
    
    Designed to be trained via genetic algorithm on regime-specific data
    """
    
    def __init__(self, genome: np.ndarray, regime_type: str):
        """
        Initialize trading specialist
        
        Args:
            genome: 8-element array of trading parameters
            regime_type: 'volatile', 'trending', 'ranging', or 'crisis'
        """
        self.genome = np.array(genome, dtype=float)
        self.regime_type = regime_type
        
        # Unpack genome for easy access
        self.stop_loss = self.genome[0]
        self.take_profit = self.genome[1]
        self.position_size = self.genome[2]
        self.entry_threshold = self.genome[3]
        self.exit_threshold = self.genome[4]
        self.max_hold_time = int(self.genome[5])
        self.volatility_scaling = self.genome[6]
        self.momentum_weight = self.genome[7]
        
        # Trading state
        self.current_position = 0  # 0=no position, 1=long, -1=short
        self.entry_price = 0.0
        self.position_size_actual = 0.0
        self.current_hold_time = 0
        self.entry_date = None
        self.entry_signal_strength = 0.0
        
        # Performance tracking
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.current_equity = 10000.0  # Start with $10k
        self.peak_equity = 10000.0
        
    def generate_signal(self, 
                       market_data: pd.DataFrame, 
                       predictions: np.ndarray) -> Tuple[int, float]:
        """
        Generate trading signal based on genome and market state
        
        Args:
            market_data: DataFrame with OHLCV + indicators (last row is current)
            predictions: Array of ML predictions (price direction probabilities)
        
        Returns:
            signal: -1 (sell/short), 0 (hold), 1 (buy/long)
            position_size: Fraction of capital to use (0.0-1.0)
        """
        # Get current market state
        current_idx = len(market_data) - 1
        current_price = market_data['close'].iloc[-1]
        
        # Calculate ATR for volatility scaling
        if 'atr' in market_data.columns:
            atr = market_data['atr'].iloc[-1]
        else:
            # Simple ATR approximation
            high_low = market_data['high'] - market_data['low']
            atr = high_low.rolling(14).mean().iloc[-1]
        
        # Get prediction strength
        if len(predictions) > 0:
            prediction = predictions[-1]
            prediction_strength = abs(prediction)
        else:
            prediction = 0.0
            prediction_strength = 0.0
        
        # NO POSITION - Look for entry
        if self.current_position == 0:
            return self._evaluate_entry(
                current_price, 
                prediction, 
                prediction_strength,
                atr,
                market_data
            )
        
        # HAVE POSITION - Check exit conditions
        else:
            return self._evaluate_exit(
                current_price,
                prediction,
                prediction_strength,
                atr,
                market_data
            )
    
    def _evaluate_entry(self, 
                        price: float, 
                        prediction: float,
                        prediction_strength: float,
                        atr: float,
                        market_data: pd.DataFrame) -> Tuple[int, float]:
        """Evaluate entry signal"""
        
        # Check if signal is strong enough
        if prediction_strength < self.entry_threshold:
            return 0, 0.0
        
        # Determine direction
        signal = 1 if prediction > 0 else -1
        
        # Calculate position size based on volatility
        volatility_factor = atr / price if price > 0 else 0.02
        scaled_position = self.position_size / (self.volatility_scaling * volatility_factor)
        scaled_position = np.clip(scaled_position, 0.01, 0.10)
        
        # Adjust for momentum weight
        if 'returns' in market_data.columns:
            recent_momentum = market_data['returns'].iloc[-5:].mean()
            
            # High momentum_weight = follow trends
            # Low momentum_weight = mean reversion
            if self.momentum_weight > 0.5:
                # Trend following - scale up in strong trends
                if abs(recent_momentum) > 0.01:
                    scaled_position *= (1.0 + self.momentum_weight * 0.5)
            else:
                # Mean reversion - scale down in strong trends
                if abs(recent_momentum) > 0.02:
                    scaled_position *= (1.0 - (1 - self.momentum_weight) * 0.3)
        
        scaled_position = np.clip(scaled_position, 0.01, 0.10)
        
        return signal, scaled_position
    
    def _evaluate_exit(self,
                      price: float,
                      prediction: float,
                      prediction_strength: float,
                      atr: float,
                      market_data: pd.DataFrame) -> Tuple[int, float]:
        """Evaluate exit conditions"""
        
        # Calculate current P&L
        pnl_pct = (price - self.entry_price) / self.entry_price * self.current_position
        
        # Exit reason tracking
        exit_reason = None
        
        # 1. Stop loss hit
        if pnl_pct <= -self.stop_loss:
            exit_reason = 'stop_loss'
            return 0, 0.0
        
        # 2. Take profit hit
        if pnl_pct >= self.take_profit:
            exit_reason = 'take_profit'
            return 0, 0.0
        
        # 3. Signal weakened
        if prediction_strength < self.exit_threshold:
            exit_reason = 'weak_signal'
            return 0, 0.0
        
        # 4. Signal reversed
        if self.current_position == 1 and prediction < -0.3:
            exit_reason = 'signal_reversal'
            return 0, 0.0
        elif self.current_position == -1 and prediction > 0.3:
            exit_reason = 'signal_reversal'
            return 0, 0.0
        
        # 5. Maximum hold time
        if self.current_hold_time >= self.max_hold_time:
            exit_reason = 'max_hold_time'
            return 0, 0.0
        
        # Hold position
        return self.current_position, self.position_size_actual
    
    def execute_signal(self,
                      signal: int,
                      position_size: float,
                      current_price: float,
                      current_date: str,
                      signal_strength: float = 0.0):
        """
        Execute trading signal (update internal state)
        
        Args:
            signal: -1, 0, or 1
            position_size: Fraction of capital
            current_price: Current market price
            current_date: Current date string
            signal_strength: Strength of entry signal (for tracking)
        """
        # Close existing position
        if signal != self.current_position and self.current_position != 0:
            self._close_position(current_price, current_date)
        
        # Open new position
        if signal != 0 and self.current_position == 0:
            self._open_position(signal, position_size, current_price, current_date, signal_strength)
        
        # Update hold time
        if self.current_position != 0:
            self.current_hold_time += 1
    
    def _open_position(self, signal: int, size: float, price: float, date: str, signal_strength: float = 0.0):
        """Open new position"""
        self.current_position = signal
        self.entry_price = price
        self.position_size_actual = size
        self.current_hold_time = 0
        self.entry_date = date
        self.entry_signal_strength = signal_strength
    
    def _close_position(self, exit_price: float, exit_date: str):
        """Close current position and record trade"""
        if self.current_position == 0:
            return
        
        # Calculate P&L
        pnl_pct = (exit_price - self.entry_price) / self.entry_price * self.current_position
        pnl_dollars = self.current_equity * self.position_size_actual * pnl_pct
        
        # Update equity
        self.current_equity += pnl_dollars
        self.peak_equity = max(self.peak_equity, self.current_equity)
        
        # Record trade
        trade = Trade(
            entry_date=self.entry_date or "unknown",
            entry_price=self.entry_price,
            entry_signal=self.entry_signal_strength,
            exit_date=exit_date,
            exit_price=exit_price,
            direction=self.current_position,
            position_size=self.position_size_actual,
            pnl=pnl_dollars,
            return_pct=pnl_pct,
            hold_time=self.current_hold_time
        )
        self.trades.append(trade)
        
        # Reset position
        self.current_position = 0
        self.entry_price = 0.0
        self.position_size_actual = 0.0
        self.current_hold_time = 0
    
    def evaluate_fitness(self, 
                        historical_data: pd.DataFrame, 
                        predictions: np.ndarray) -> float:
        """
        Backtest specialist on historical data and calculate fitness
        
        Args:
            historical_data: DataFrame with OHLCV + indicators
            predictions: Array of ML predictions (same length as data)
        
        Returns:
            fitness: Combined performance score (higher is better)
        """
        # Reset state
        self.current_position = 0
        self.trades = []
        self.equity_curve = [self.current_equity]
        
        # Simulate trading
        for i in range(len(historical_data)):
            current_data = historical_data.iloc[:i+1]
            current_predictions = predictions[:i+1]
            current_price = historical_data['close'].iloc[i]
            current_date = historical_data.index[i] if isinstance(historical_data.index[i], str) else str(historical_data.index[i])
            
            # Generate signal
            signal, size = self.generate_signal(current_data, current_predictions)
            
            # Get prediction strength for entry tracking
            pred_strength = abs(current_predictions[-1]) if len(current_predictions) > 0 else 0.0
            
            # Execute signal
            if signal != 0 and self.current_position == 0:
                # Opening new position - track signal strength
                self.execute_signal(signal, size, current_price, current_date, pred_strength)
            else:
                self.execute_signal(signal, size, current_price, current_date)
            
            # Track equity
            if self.current_position != 0:
                # Mark-to-market
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price * self.current_position
                unrealized_pnl_dollars = self.current_equity * self.position_size_actual * unrealized_pnl
                mtm_equity = self.current_equity + unrealized_pnl_dollars
            else:
                mtm_equity = self.current_equity
            
            self.equity_curve.append(mtm_equity)
        
        # Close any open position at end
        if self.current_position != 0:
            final_price = historical_data['close'].iloc[-1]
            final_date = str(historical_data.index[-1])
            self._close_position(final_price, final_date)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics()
        
        return metrics.fitness
    
    def _calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        metrics = PerformanceMetrics()
        
        if len(self.trades) == 0:
            return metrics
        
        # Total return
        metrics.total_return = (self.current_equity - 10000) / 10000
        
        # Sharpe ratio
        returns = np.array([t.return_pct for t in self.trades])
        if len(returns) > 1 and np.std(returns) > 0:
            metrics.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 / max(1, np.mean([t.hold_time for t in self.trades])))
        
        # Win rate
        wins = sum(1 for t in self.trades if t.pnl > 0)
        metrics.win_rate = wins / len(self.trades)
        metrics.num_trades = len(self.trades)
        metrics.avg_trade_return = np.mean(returns)
        
        # Max drawdown
        equity_array = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        metrics.max_drawdown = np.max(drawdown)
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        # Combined fitness score
        metrics.fitness = (
            metrics.sharpe_ratio * 10.0 +          # Sharpe is key
            metrics.total_return * 20.0 +          # Total return matters
            metrics.win_rate * 5.0 -               # Win rate bonus
            metrics.max_drawdown * 15.0 +          # Drawdown penalty
            min(metrics.profit_factor, 3.0) * 3.0  # Profit factor (capped)
        )
        
        return metrics
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self._calculate_metrics()
    
    def __repr__(self):
        return (f"TradingSpecialist(regime={self.regime_type}, "
                f"stop_loss={self.stop_loss:.3f}, "
                f"take_profit={self.take_profit:.3f}, "
                f"position_size={self.position_size:.3f})")


if __name__ == '__main__':
    # Test specialist with random genome
    print("Testing TradingSpecialist class...\n")
    
    # Create random genome
    genome = np.array([
        0.02,   # stop_loss (2%)
        0.05,   # take_profit (5%)
        0.05,   # position_size (5%)
        0.6,    # entry_threshold
        0.4,    # exit_threshold
        5,      # max_hold_time
        1.0,    # volatility_scaling
        0.7     # momentum_weight (trend-following)
    ])
    
    specialist = TradingSpecialist(genome, regime_type='trending')
    print(f"Created: {specialist}\n")
    
    # Test with dummy data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    dummy_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(100)),
        'high': 100 + np.cumsum(np.random.randn(100)) + 2,
        'low': 100 + np.cumsum(np.random.randn(100)) - 2,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    dummy_data['returns'] = dummy_data['close'].pct_change()
    dummy_data['atr'] = (dummy_data['high'] - dummy_data['low']).rolling(14).mean()
    
    # Dummy predictions (trend-following signal)
    predictions = (dummy_data['returns'].rolling(5).mean() * 10).fillna(0).values
    
    # Evaluate fitness
    print("Running backtest on 100 days of dummy data...")
    fitness = specialist.evaluate_fitness(dummy_data, predictions)
    
    metrics = specialist.get_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Fitness:       {metrics.fitness:.2f}")
    print(f"  Total Return:  {metrics.total_return*100:.2f}%")
    print(f"  Sharpe Ratio:  {metrics.sharpe_ratio:.2f}")
    print(f"  Win Rate:      {metrics.win_rate*100:.1f}%")
    print(f"  Max Drawdown:  {metrics.max_drawdown*100:.2f}%")
    print(f"  Num Trades:    {metrics.num_trades}")
    print(f"  Avg Trade:     {metrics.avg_trade_return*100:.2f}%")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    
    print("\n✅ TradingSpecialist class working!")
