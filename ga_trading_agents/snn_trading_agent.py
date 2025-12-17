"""
SNN-Based Trading Agent for GA Framework
Combines SNN brain with GA evolution strategy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from typing import Dict, List

# Import existing trading agent if available
try:
    from trading_agent import TradingAgent
    HAS_BASE_AGENT = True
except ImportError:
    HAS_BASE_AGENT = False
    # Define minimal base class
    class TradingAgent:
        def __init__(self, chromosome=None):
            self.chromosome = chromosome or {}
            self.fitness = 0.0

from models.snn_trading_agent import SpikingTradingAgent


class SNNTradingAgent(TradingAgent if HAS_BASE_AGENT else object):
    """
    Trading agent with SNN brain instead of rule-based chromosome.
    Combines GA evolution with SNN learning.
    """
    
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 100,
        chromosome: Dict = None
    ):
        """
        Initialize SNN trading agent.
        
        Args:
            input_dim: Number of input features
            hidden_dim: SNN hidden dimension
            chromosome: Optional chromosome for hybrid approach
        """
        if HAS_BASE_AGENT:
            super().__init__(chromosome)
        else:
            self.chromosome = chromosome or {}
            self.fitness = 0.0
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # SNN brain
        device = torch.device('cpu')  # CPU for GA compatibility
        self.snn_brain = SpikingTradingAgent(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=3,  # Buy, Hold, Sell
            num_pathways=3
        ).to(device)
        
        # Genome for evolution
        self.genome = {
            'hidden_dim': hidden_dim,
            'num_pathways': 3,
            'threshold': 1.0,
            'decay': 0.9,
            'risk_tolerance': 0.5
        }
    
    def perceive(self, market_state: np.ndarray) -> int:
        """
        Make trading decision using SNN.
        
        Args:
            market_state: Current market features
            
        Returns:
            action: 0=Hold, 1=Buy, 2=Sell
        """
        # Convert to tensor
        if len(market_state.shape) == 1:
            market_state = market_state.reshape(1, -1)
        
        state_tensor = torch.FloatTensor(market_state)
        
        # Get SNN decision
        with torch.no_grad():
            output, _ = self.snn_brain(state_tensor, num_steps=10)
            action = output.argmax(dim=-1).item()
        
        return action
    
    def act(self, market_state: np.ndarray, current_position: float = 0.0) -> Dict:
        """
        Execute trading action.
        
        Args:
            market_state: Current market features
            current_position: Current position size
            
        Returns:
            action_dict: Dictionary with action details
        """
        action = self.perceive(market_state)
        
        # Convert action to position change
        if action == 0:  # Hold
            position_change = 0.0
        elif action == 1:  # Buy
            position_change = self.genome['risk_tolerance']
        else:  # Sell
            position_change = -self.genome['risk_tolerance']
        
        return {
            'action': action,
            'position_change': position_change,
            'new_position': current_position + position_change
        }
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate agent genome."""
        if np.random.random() < mutation_rate:
            self.genome['hidden_dim'] = max(
                50,
                int(self.genome['hidden_dim'] + np.random.randint(-20, 21))
            )
        
        if np.random.random() < mutation_rate:
            self.genome['num_pathways'] = max(
                1,
                self.genome['num_pathways'] + np.random.choice([-1, 0, 1])
            )
        
        if np.random.random() < mutation_rate:
            self.genome['threshold'] = max(
                0.5,
                min(2.0, self.genome['threshold'] + np.random.randn() * 0.1)
            )
        
        if np.random.random() < mutation_rate:
            self.genome['decay'] = max(
                0.5,
                min(0.99, self.genome['decay'] + np.random.randn() * 0.05)
            )
        
        if np.random.random() < mutation_rate:
            self.genome['risk_tolerance'] = max(
                0.1,
                min(1.0, self.genome['risk_tolerance'] + np.random.randn() * 0.1)
            )
        
        # Recreate SNN with new parameters
        self._recreate_snn()
    
    def _recreate_snn(self):
        """Recreate SNN with current genome parameters."""
        device = torch.device('cpu')
        self.snn_brain = SpikingTradingAgent(
            input_dim=self.input_dim,
            hidden_dim=self.genome['hidden_dim'],
            output_dim=3,
            num_pathways=self.genome['num_pathways'],
            threshold=self.genome['threshold'],
            decay=self.genome['decay']
        ).to(device)
    
    def crossover(self, other: 'SNNTradingAgent') -> 'SNNTradingAgent':
        """Create offspring through crossover."""
        offspring = SNNTradingAgent(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Mix genomes
        for gene_name in self.genome:
            if np.random.random() < 0.5:
                offspring.genome[gene_name] = self.genome[gene_name]
            else:
                offspring.genome[gene_name] = other.genome[gene_name]
        
        # Recreate SNN
        offspring._recreate_snn()
        
        return offspring
    
    def train_on_experience(
        self,
        experiences: List[Dict],
        num_epochs: int = 5
    ):
        """
        Train SNN on collected experiences.
        
        Args:
            experiences: List of {state, action, reward} dicts
            num_epochs: Number of training epochs
        """
        if len(experiences) == 0:
            return
        
        optimizer = torch.optim.Adam(self.snn_brain.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Prepare data
        states = np.array([exp['state'] for exp in experiences])
        actions = np.array([exp['action'] for exp in experiences])
        rewards = np.array([exp['reward'] for exp in experiences])
        
        # Weight by rewards
        weights = torch.FloatTensor(rewards)
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        
        for epoch in range(num_epochs):
            # Forward pass
            states_tensor = torch.FloatTensor(states)
            outputs, _ = self.snn_brain(states_tensor, num_steps=10)
            
            # Loss
            targets = torch.LongTensor(actions)
            loss = criterion(outputs, targets)
            
            # Weight by rewards (reward-weighted behavioral cloning)
            loss = (loss * weights).mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def save(self, filepath: str):
        """Save agent state."""
        state = {
            'genome': self.genome,
            'fitness': self.fitness,
            'snn_state': self.snn_brain.state_dict()
        }
        torch.save(state, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        state = torch.load(filepath)
        self.genome = state['genome']
        self.fitness = state['fitness']
        self._recreate_snn()
        self.snn_brain.load_state_dict(state['snn_state'])


def demo_snn_agent():
    """Demonstrate SNN trading agent."""
    print("=" * 80)
    print("SNN TRADING AGENT DEMO")
    print("=" * 80)
    
    # Create agent
    agent = SNNTradingAgent(input_dim=50, hidden_dim=100)
    
    print("\nAgent initialized:")
    print(f"  Input dim: {agent.input_dim}")
    print(f"  Hidden dim: {agent.hidden_dim}")
    print(f"  Genome: {agent.genome}")
    
    # Simulate market state
    market_state = np.random.randn(50) * 10 + 100
    
    print("\nTesting perception...")
    action = agent.perceive(market_state)
    print(f"  Action: {action} ({'Hold' if action == 0 else 'Buy' if action == 1 else 'Sell'})")
    
    # Test action
    print("\nTesting action execution...")
    action_dict = agent.act(market_state, current_position=0.0)
    print(f"  Action dict: {action_dict}")
    
    # Test mutation
    print("\nTesting mutation...")
    old_genome = agent.genome.copy()
    agent.mutate(mutation_rate=1.0)  # Force mutations
    print(f"  Old genome: {old_genome}")
    print(f"  New genome: {agent.genome}")
    
    # Test training
    print("\nTesting training on experience...")
    experiences = [
        {'state': np.random.randn(50), 'action': np.random.randint(0, 3), 'reward': np.random.rand()}
        for _ in range(10)
    ]
    agent.train_on_experience(experiences, num_epochs=2)
    print("  Training complete")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demo_snn_agent()
