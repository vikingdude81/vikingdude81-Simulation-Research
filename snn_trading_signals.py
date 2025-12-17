"""
SNN Trading Signals
Generate trading signals using Spiking Neural Networks
"""

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from models.snn_trading_agent import SpikingTradingAgent
from utils.snn_utils import encode_to_spikes, decode_trading_signals


class SNNTradingSignalGenerator:
    """
    Generate trading signals using trained SNN agents.
    """
    
    def __init__(
        self,
        model_path: str = None,
        input_dim: int = 50,
        device: str = 'cpu'
    ):
        """
        Initialize SNN signal generator.
        
        Args:
            model_path: Path to trained SNN model (optional)
            input_dim: Number of input features
            device: Device to run inference on
        """
        self.input_dim = input_dim
        self.device = torch.device(device)
        
        # Create or load SNN agent
        self.agent = SpikingTradingAgent(
            input_dim=input_dim,
            hidden_dim=100,
            output_dim=3,  # Buy, Hold, Sell
            num_pathways=3
        ).to(self.device)
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained SNN model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.agent.eval()
        print(f"Loaded SNN model from: {model_path}")
    
    def save_model(self, model_path: str):
        """Save trained SNN model."""
        checkpoint = {
            'model_state_dict': self.agent.state_dict(),
            'input_dim': self.input_dim,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, model_path)
        print(f"Saved SNN model to: {model_path}")
    
    def generate_signal(
        self,
        features: np.ndarray,
        num_steps: int = 10
    ) -> Dict:
        """
        Generate trading signal from features.
        
        Args:
            features: Input features [input_dim]
            num_steps: Number of SNN simulation steps
            
        Returns:
            signal: Dictionary with signal information
        """
        self.agent.eval()
        
        # Ensure features are 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            # Get SNN output
            output, info = self.agent(features_tensor, num_steps=num_steps)
            
            # Get action probabilities
            probs = torch.softmax(output, dim=-1).squeeze()
            action = output.argmax(dim=-1).item()
            
            # Convert to signal
            signal_map = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}
            signal = signal_map[action]
            
            # Get confidence
            confidence = probs[action].item()
        
        return {
            'signal': signal,
            'action': action,
            'confidence': confidence,
            'probabilities': {
                'BUY': probs[0].item(),
                'HOLD': probs[1].item(),
                'SELL': probs[2].item()
            },
            'spike_rate': info['mean_spike_rate'].mean().item(),
            'num_pathways': info['num_active_pathways'],
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_signals_batch(
        self,
        features_batch: np.ndarray,
        num_steps: int = 10
    ) -> List[Dict]:
        """
        Generate trading signals for batch of features.
        
        Args:
            features_batch: Batch of features [batch_size, input_dim]
            num_steps: Number of SNN simulation steps
            
        Returns:
            signals: List of signal dictionaries
        """
        signals = []
        
        for i in range(len(features_batch)):
            signal = self.generate_signal(features_batch[i], num_steps)
            signals.append(signal)
        
        return signals
    
    def backtest_signals(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        initial_capital: float = 10000.0
    ) -> Dict:
        """
        Backtest SNN trading signals.
        
        Args:
            features: Historical features [num_samples, input_dim]
            prices: Historical prices [num_samples]
            initial_capital: Starting capital
            
        Returns:
            backtest_results: Dictionary with backtest metrics
        """
        capital = initial_capital
        position = 0.0
        trades = []
        
        for i in range(len(features)):
            # Generate signal
            signal_dict = self.generate_signal(features[i])
            signal = signal_dict['signal']
            price = prices[i]
            
            # Execute trade
            if signal == 'BUY' and position <= 0:
                # Buy (close short if any, open long)
                shares_to_buy = capital / price
                cost = shares_to_buy * price
                capital -= cost
                position = shares_to_buy
                
                trades.append({
                    'type': 'BUY',
                    'price': price,
                    'shares': shares_to_buy,
                    'time': i
                })
            
            elif signal == 'SELL' and position >= 0:
                # Sell (close long if any, open short)
                if position > 0:
                    revenue = position * price
                    capital += revenue
                    
                    trades.append({
                        'type': 'SELL',
                        'price': price,
                        'shares': position,
                        'time': i
                    })
                    
                    position = 0.0
        
        # Close any open position
        if position != 0:
            final_value = position * prices[-1]
            capital += final_value
        
        # Calculate metrics
        total_return = ((capital - initial_capital) / initial_capital) * 100
        num_trades = len(trades)
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return_pct': total_return,
            'num_trades': num_trades,
            'trades': trades
        }


def demo_snn_signals():
    """Demonstrate SNN signal generation."""
    print("=" * 80)
    print("SNN TRADING SIGNALS DEMO")
    print("=" * 80)
    
    # Create generator
    generator = SNNTradingSignalGenerator(input_dim=50)
    
    print("\nGenerating sample signals...")
    
    # Generate sample features
    sample_features = np.random.randn(5, 50) * 10 + 100
    
    # Generate signals
    signals = generator.generate_signals_batch(sample_features)
    
    print("\nGenerated Signals:")
    for i, signal in enumerate(signals):
        print(f"\nSignal {i+1}:")
        print(f"  Action: {signal['signal']}")
        print(f"  Confidence: {signal['confidence']:.2%}")
        print(f"  Probabilities:")
        for action, prob in signal['probabilities'].items():
            print(f"    {action}: {prob:.2%}")
    
    # Simple backtest
    print("\n" + "=" * 80)
    print("BACKTESTING")
    print("=" * 80)
    
    num_samples = 100
    features = np.random.randn(num_samples, 50) * 10 + 100
    prices = 100 + np.cumsum(np.random.randn(num_samples) * 0.5)
    
    results = generator.backtest_signals(features, prices)
    
    print(f"\nBacktest Results:")
    print(f"  Initial Capital: ${results['initial_capital']:.2f}")
    print(f"  Final Capital: ${results['final_capital']:.2f}")
    print(f"  Total Return: {results['total_return_pct']:.2f}%")
    print(f"  Number of Trades: {results['num_trades']}")
    
    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"snn_signals_demo_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'signals': signals[:3],  # Save first 3 signals
            'backtest': {
                'initial_capital': results['initial_capital'],
                'final_capital': results['final_capital'],
                'total_return_pct': results['total_return_pct'],
                'num_trades': results['num_trades']
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    demo_snn_signals()
