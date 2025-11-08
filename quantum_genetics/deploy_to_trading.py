"""
🚀 Deploy Quantum Genetic Champion to Crypto Trading System

Integrates the champion genome as an adaptive parameter controller for:
- Feature selection weights
- Model hyperparameter tuning
- Risk management adjustments
- Portfolio rebalancing decisions
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_genetics.deploy_champion import ChampionGenome
from quantum_genetics.quantum_genetic_agents import QuantumAgent


class QuantumTradingController:
    """
    Adaptive trading controller using quantum genetic champion.
    
    The genome parameters control trading behavior:
    - μ (mutation): Exploration of new strategies
    - ω (oscillation): Reaction speed to market changes
    - d (decoherence): Memory decay / strategy persistence
    - φ (phase): Market cycle alignment
    """
    
    def __init__(self, environment='standard', timesteps_per_decision=100):
        """
        Initialize quantum trading controller.
        
        Args:
            environment: Market condition ('standard', 'volatile', 'trending', 'ranging')
            timesteps_per_decision: Evolution steps before making decision
        """
        self.champion = ChampionGenome()
        self.genome = self.champion.get_genome()
        self.timesteps = timesteps_per_decision
        self.environment = environment
        
        # Create quantum agent
        self.agent = self.champion.create_agent(agent_id=0, environment=environment)
        
        # Tracking
        self.decisions = []
        self.fitness_history = []
        
    def evolve_and_decide(self, market_state):
        """
        Evolve agent and generate trading decision.
        
        Args:
            market_state: dict with keys:
                - 'volatility': float [0-1]
                - 'trend': float [-1, 1]
                - 'volume': float [0-1]
                - 'momentum': float [-1, 1]
        
        Returns:
            dict with trading parameters:
                - 'position_size': float [0, 1]
                - 'risk_multiplier': float [0.5, 2.0]
                - 'feature_weights': dict
                - 'rebalance': bool
        """
        # Evolve agent for specified timesteps
        for t in range(self.timesteps):
            self.agent.evolve(t)
        
        # Get final fitness (represents strategy quality)
        fitness = self.agent.get_final_fitness()
        self.fitness_history.append(fitness)
        
        # Extract quantum traits for decision making
        traits = self.agent.traits
        creativity = traits[0]  # Range: [0, 1]
        coherence = traits[1]   # Range: [0, 1]
        longevity = traits[2]   # Range: [0, 1]
        
        # Map traits to trading parameters
        decision = {
            'position_size': self._calculate_position_size(creativity, coherence, market_state),
            'risk_multiplier': self._calculate_risk_multiplier(longevity, market_state),
            'feature_weights': self._calculate_feature_weights(creativity, coherence),
            'rebalance': self._should_rebalance(coherence, longevity),
            'confidence': self._calculate_confidence(fitness, traits),
            'fitness': fitness,
            'traits': {
                'creativity': float(creativity),
                'coherence': float(coherence),
                'longevity': float(longevity)
            }
        }
        
        self.decisions.append(decision)
        return decision
    
    def _calculate_position_size(self, creativity, coherence, market_state):
        """
        Calculate position size based on quantum traits and market state.
        
        High creativity + high coherence + low volatility = larger positions
        Low creativity + low coherence + high volatility = smaller positions
        """
        volatility = market_state.get('volatility', 0.5)
        
        # Normalize traits to [0, 1] - handle negative values
        creativity_norm = max(0.0, min(1.0, creativity / 10.0))
        coherence_norm = max(0.0, min(1.0, coherence / 10.0))
        
        # Base size from quantum traits
        quantum_confidence = (creativity_norm * coherence_norm) ** 0.5
        
        # Adjust for volatility
        volatility_factor = 1.0 - (volatility * 0.8)  # Reduce size in high volatility
        
        # Final position size [0, 1]
        position_size = quantum_confidence * volatility_factor
        return float(np.clip(position_size, 0.0, 1.0))
    
    def _calculate_risk_multiplier(self, longevity, market_state):
        """
        Calculate risk multiplier based on strategy persistence.
        
        High longevity = lower risk (stable strategy)
        Low longevity = higher risk (exploratory phase)
        """
        trend_strength = abs(market_state.get('trend', 0.0))
        
        # Base multiplier from longevity
        base_multiplier = 0.5 + longevity  # Range: [0.5, 1.5]
        
        # Increase risk in strong trends
        trend_bonus = trend_strength * 0.5
        
        multiplier = base_multiplier + trend_bonus
        return float(np.clip(multiplier, 0.5, 2.0))
    
    def _calculate_feature_weights(self, creativity, coherence):
        """
        Calculate feature importance weights.
        
        High creativity = more technical indicators
        High coherence = more fundamental features
        """
        return {
            'technical': float(0.3 + creativity * 0.5),       # [0.3, 0.8]
            'fundamental': float(0.3 + coherence * 0.5),      # [0.3, 0.8]
            'sentiment': float(0.2 + (1 - creativity) * 0.3), # [0.2, 0.5]
            'volume': float(0.2 + coherence * 0.3)            # [0.2, 0.5]
        }
    
    def _should_rebalance(self, coherence, longevity):
        """Decide if portfolio should be rebalanced."""
        # Rebalance if coherence is low (unstable) or longevity is low (short-term focus)
        stability = (coherence + longevity) / 2
        return stability < 0.4
    
    def _calculate_confidence(self, fitness, traits):
        """Calculate overall decision confidence."""
        # Normalize fitness to [0, 1]
        fitness_confidence = np.clip(fitness / 50000, 0, 1)
        
        # Average trait quality
        trait_confidence = np.mean([traits[0], traits[1], traits[2]])
        
        # Combined confidence
        confidence = (fitness_confidence * 0.6 + trait_confidence * 0.4)
        return float(confidence)
    
    def reset(self):
        """Reset agent to initial state."""
        self.agent = self.champion.create_agent(agent_id=0, environment=self.environment)
        self.decisions = []
        self.fitness_history = []
    
    def get_statistics(self):
        """Get controller statistics."""
        if not self.fitness_history:
            return {}
        
        return {
            'total_decisions': len(self.decisions),
            'avg_fitness': float(np.mean(self.fitness_history)),
            'fitness_std': float(np.std(self.fitness_history)),
            'fitness_trend': float(np.polyfit(range(len(self.fitness_history)), 
                                              self.fitness_history, 1)[0]),
            'avg_position_size': float(np.mean([d['position_size'] for d in self.decisions])),
            'avg_confidence': float(np.mean([d['confidence'] for d in self.decisions])),
            'rebalance_frequency': sum(d['rebalance'] for d in self.decisions) / len(self.decisions)
        }


def demo_trading_integration():
    """Demonstrate quantum controller for trading."""
    print("\n" + "="*70)
    print("🚀 QUANTUM GENETIC CONTROLLER - TRADING DEMO")
    print("="*70)
    
    # Initialize controller
    controller = QuantumTradingController(environment='standard')
    
    print(f"\n📊 Champion Genome: {controller.genome}")
    print(f"   μ (mutation):    {controller.genome[0]}")
    print(f"   ω (oscillation): {controller.genome[1]}")
    print(f"   d (decoherence): {controller.genome[2]}")
    print(f"   φ (phase):       {controller.genome[3]:.6f} (2π)")
    
    # Simulate market conditions
    market_scenarios = [
        {'name': 'Low Volatility Uptrend', 'volatility': 0.2, 'trend': 0.7, 'volume': 0.6, 'momentum': 0.5},
        {'name': 'High Volatility', 'volatility': 0.9, 'trend': 0.1, 'volume': 0.8, 'momentum': -0.2},
        {'name': 'Strong Downtrend', 'volatility': 0.6, 'trend': -0.8, 'volume': 0.5, 'momentum': -0.6},
        {'name': 'Ranging Market', 'volatility': 0.3, 'trend': 0.0, 'volume': 0.4, 'momentum': 0.1},
        {'name': 'Breakout', 'volatility': 0.7, 'trend': 0.9, 'volume': 0.9, 'momentum': 0.8}
    ]
    
    print("\n" + "="*70)
    print("📈 MARKET SCENARIO TESTING")
    print("="*70)
    
    results = []
    
    for scenario in market_scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   Volatility: {scenario['volatility']:.2f} | Trend: {scenario['trend']:+.2f}")
        
        # Reset agent for each scenario
        controller.reset()
        
        # Get decision
        decision = controller.evolve_and_decide(scenario)
        
        print(f"\n   🎯 Decision:")
        print(f"      Position Size:   {decision['position_size']:.3f}")
        print(f"      Risk Multiplier: {decision['risk_multiplier']:.3f}")
        print(f"      Confidence:      {decision['confidence']:.3f}")
        print(f"      Rebalance:       {decision['rebalance']}")
        print(f"      Fitness:         {decision['fitness']:.0f}")
        
        print(f"\n   🧬 Quantum Traits:")
        print(f"      Creativity:      {decision['traits']['creativity']:.3f}")
        print(f"      Coherence:       {decision['traits']['coherence']:.3f}")
        print(f"      Longevity:       {decision['traits']['longevity']:.3f}")
        
        print(f"\n   📊 Feature Weights:")
        for feature, weight in decision['feature_weights'].items():
            print(f"      {feature:12s}: {weight:.3f}")
        
        results.append({
            'scenario': scenario['name'],
            'market_state': scenario,
            'decision': decision
        })
    
    # Save results (convert numpy types to Python types for JSON)
    output_file = Path(__file__).parent / "trading_controller_demo.json"
    
    # Convert results to JSON-serializable format
    json_results = []
    for result in results:
        json_result = {
            'scenario': result['scenario'],
            'market_state': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in result['market_state'].items()},
            'decision': {k: (float(v) if isinstance(v, (np.floating, np.integer, np.bool_)) else 
                           (bool(v) if isinstance(v, (bool, np.bool_)) else v))
                        for k, v in result['decision'].items() if k != 'feature_weights'}
        }
        # Handle feature_weights separately
        json_result['decision']['feature_weights'] = {
            k: float(v) for k, v in result['decision']['feature_weights'].items()
        }
        json_results.append(json_result)
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print(f"\n💾 Results saved to: {output_file.name}")
    print(f"\n📚 Integration Guide:")
    print(f"   1. Import: from quantum_genetics.deploy_to_trading import QuantumTradingController")
    print(f"   2. Initialize: controller = QuantumTradingController()")
    print(f"   3. Use: decision = controller.evolve_and_decide(market_state)")
    print(f"   4. Apply decision['position_size'], decision['risk_multiplier'], etc.")
    print(f"\n✨ Ready for production integration!")


if __name__ == "__main__":
    demo_trading_integration()
