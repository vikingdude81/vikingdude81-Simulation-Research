"""
Market as Conscious Agent
Model market as hierarchical conscious agent network
Traders = micro-agents, market = macro-agent
Apply consciousness metrics to market
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime


class TraderAgent:
    """Micro-agent representing individual trader."""
    
    def __init__(self, agent_id: int, perception_dim: int = 10):
        self.agent_id = agent_id
        self.perception_dim = perception_dim
        self.state = np.random.randn(perception_dim)
        self.action_history = []
    
    def perceive(self, market_state: np.ndarray) -> np.ndarray:
        """Process market information through perceptual interface."""
        # Simple perceptual filter
        noise = np.random.randn(*market_state.shape) * 0.1
        perception = market_state + noise
        return perception[:self.perception_dim]
    
    def decide(self, perception: np.ndarray) -> float:
        """Make trading decision based on perception."""
        # Simple decision: weighted sum of perceptions
        action = np.dot(perception, np.random.randn(self.perception_dim))
        action = np.tanh(action)  # Normalize to [-1, 1]
        self.action_history.append(action)
        return action


class MarketAgent:
    """Macro-agent representing emergent market behavior."""
    
    def __init__(self, num_traders: int = 100):
        self.num_traders = num_traders
        self.traders = [TraderAgent(i) for i in range(num_traders)]
        self.market_state = np.random.randn(50)
        self.coherence_history = []
    
    def update(self):
        """Update market state based on trader actions."""
        # Each trader perceives and acts
        actions = []
        for trader in self.traders:
            perception = trader.perceive(self.market_state)
            action = trader.decide(perception)
            actions.append(action)
        
        actions = np.array(actions)
        
        # Market state evolves based on aggregate actions
        aggregate_action = actions.mean()
        self.market_state = self.market_state * 0.9 + aggregate_action * 0.1
        
        # Calculate coherence (how aligned are traders)
        coherence = 1.0 - np.std(actions)
        self.coherence_history.append(coherence)
        
        return aggregate_action, coherence


def calculate_consciousness_metrics(market_agent: MarketAgent):
    """Calculate consciousness-like metrics for market."""
    
    # 1. Integration (phi): How integrated is the market?
    # Measure mutual information between traders
    actions = np.array([
        trader.action_history[-10:] if len(trader.action_history) >= 10 
        else [0] * 10
        for trader in market_agent.traders
    ])
    
    # Simplified phi: variance of correlations
    correlations = np.corrcoef(actions)
    phi = np.var(correlations)
    
    # 2. Coherence: How synchronized are traders?
    coherence = np.mean(market_agent.coherence_history[-10:]) if market_agent.coherence_history else 0
    
    # 3. Information: Shannon entropy of actions
    action_dist = actions.flatten()
    hist, _ = np.histogram(action_dist, bins=10, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))
    
    return {
        'phi': float(phi),
        'coherence': float(coherence),
        'entropy': float(entropy)
    }


def main():
    """Main experiment execution."""
    print("=" * 80)
    print("MARKET AS CONSCIOUS AGENT")
    print("=" * 80)
    
    # Configuration
    num_traders = 100
    num_steps = 1000
    
    print(f"\nSimulating market with {num_traders} trader agents...")
    
    # Create market
    market = MarketAgent(num_traders=num_traders)
    
    # Simulate trading
    actions_over_time = []
    coherence_over_time = []
    consciousness_metrics = []
    
    for step in range(num_steps):
        action, coherence = market.update()
        actions_over_time.append(action)
        coherence_over_time.append(coherence)
        
        # Calculate consciousness metrics every 100 steps
        if (step + 1) % 100 == 0:
            metrics = calculate_consciousness_metrics(market)
            consciousness_metrics.append(metrics)
            
            print(f"\nStep {step + 1}:")
            print(f"  Phi (integration): {metrics['phi']:.4f}")
            print(f"  Coherence: {metrics['coherence']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
    
    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    avg_coherence = np.mean(coherence_over_time)
    avg_phi = np.mean([m['phi'] for m in consciousness_metrics])
    avg_entropy = np.mean([m['entropy'] for m in consciousness_metrics])
    
    print(f"\nAverage Metrics:")
    print(f"  Coherence: {avg_coherence:.4f}")
    print(f"  Phi: {avg_phi:.4f}")
    print(f"  Entropy: {avg_entropy:.4f}")
    
    # Determine if market exhibits conscious-like behavior
    consciousness_score = (avg_coherence + avg_phi) / 2
    
    print(f"\nConsciousness Score: {consciousness_score:.4f}")
    
    if consciousness_score > 0.5:
        print("✅ Market exhibits emergent conscious-like behavior")
    else:
        print("⚠️  Market shows weak conscious-like properties")
    
    # Save results
    results = {
        'num_traders': num_traders,
        'num_steps': num_steps,
        'avg_coherence': float(avg_coherence),
        'avg_phi': float(avg_phi),
        'avg_entropy': float(avg_entropy),
        'consciousness_score': float(consciousness_score),
        'consciousness_metrics_history': consciousness_metrics,
        'coherence_history': [float(c) for c in coherence_over_time]
    }
    
    output_dir = Path("outputs/market_consciousness")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"market_consciousness_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
