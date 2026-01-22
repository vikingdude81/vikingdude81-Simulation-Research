"""
Agent-based models for complex systems simulation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all agents in the simulation"""
    
    def __init__(self, agent_id: int):
        """
        Initialize base agent
        
        Args:
            agent_id: Unique identifier for the agent
        """
        self.id = agent_id
        self.state: Dict[str, Any] = {}
        self.neighbors: List['BaseAgent'] = []
    
    @abstractmethod
    def decide(self, environment: Dict[str, Any]) -> Any:
        """
        Make a decision based on current state and environment
        
        Args:
            environment: Current environment state
            
        Returns:
            Agent's decision
        """
        pass
    
    @abstractmethod
    def update(self, decision: Any, feedback: Dict[str, Any]) -> None:
        """
        Update agent state based on decision and feedback
        
        Args:
            decision: The decision made by the agent
            feedback: Feedback from the environment
        """
        pass
    
    def add_neighbor(self, agent: 'BaseAgent') -> None:
        """Add a neighboring agent"""
        if agent not in self.neighbors:
            self.neighbors.append(agent)
    
    def interact_with_neighbors(self) -> List[Tuple[int, Any]]:
        """
        Interact with neighboring agents
        
        Returns:
            List of (neighbor_id, interaction_result) tuples
        """
        interactions = []
        for neighbor in self.neighbors:
            result = self._interact(neighbor)
            interactions.append((neighbor.id, result))
        return interactions
    
    def _interact(self, other: 'BaseAgent') -> Any:
        """Define interaction behavior with another agent"""
        return None


class EconomicAgent(BaseAgent):
    """Agent that makes economic decisions"""
    
    def __init__(self, agent_id: int, initial_wealth: float = 1000.0):
        """
        Initialize economic agent
        
        Args:
            agent_id: Unique identifier
            initial_wealth: Starting wealth
        """
        super().__init__(agent_id)
        self.state = {
            'wealth': initial_wealth,
            'consumption': 0.0,
            'savings': initial_wealth * 0.2,
            'risk_aversion': np.random.uniform(0.3, 0.9)
        }
    
    def decide(self, environment: Dict[str, Any]) -> Dict[str, float]:
        """
        Make economic decision based on environment
        
        Args:
            environment: Economic indicators and opportunities
            
        Returns:
            Decision dictionary with consumption and investment
        """
        interest_rate = environment.get('interest_rate', 0.05)
        inflation = environment.get('inflation', 0.02)
        
        # Calculate optimal consumption/savings split
        real_return = interest_rate - inflation
        
        if real_return > 0 and self.state['risk_aversion'] < 0.7:
            # Invest more when returns are positive and not too risk-averse
            consumption_rate = 0.6
        else:
            # Consume more when returns are poor
            consumption_rate = 0.8
        
        consumption = self.state['wealth'] * consumption_rate
        savings = self.state['wealth'] * (1 - consumption_rate)
        
        return {
            'consumption': consumption,
            'savings': savings,
            'investment': savings * (1 - self.state['risk_aversion'])
        }
    
    def update(self, decision: Dict[str, float], feedback: Dict[str, Any]) -> None:
        """
        Update wealth and state based on decision outcomes
        
        Args:
            decision: The economic decision made
            feedback: Market outcomes and returns
        """
        investment_return = feedback.get('investment_return', 0.0)
        income = feedback.get('income', 0.0)
        
        # Update wealth
        self.state['wealth'] += income
        self.state['wealth'] -= decision['consumption']
        self.state['wealth'] += decision['investment'] * (1 + investment_return)
        self.state['wealth'] = max(0, self.state['wealth'])  # Cannot go negative
        
        # Update consumption and savings
        self.state['consumption'] = decision['consumption']
        self.state['savings'] = decision['savings']
    
    def _interact(self, other: 'BaseAgent') -> float:
        """
        Economic interaction with another agent (trading)
        
        Args:
            other: Another agent to trade with
            
        Returns:
            Value of trade
        """
        if not isinstance(other, EconomicAgent):
            return 0.0
        
        # Simple trade: exchange based on wealth difference
        trade_propensity = 0.1
        wealth_diff = self.state['wealth'] - other.state['wealth']
        
        if abs(wealth_diff) > 100:
            trade_value = wealth_diff * trade_propensity
            return trade_value
        
        return 0.0


class SocialAgent(BaseAgent):
    """Agent that participates in social networks"""
    
    def __init__(self, agent_id: int):
        """
        Initialize social agent
        
        Args:
            agent_id: Unique identifier
        """
        super().__init__(agent_id)
        self.state = {
            'opinion': np.random.uniform(-1, 1),  # -1 to 1 spectrum
            'confidence': np.random.uniform(0.5, 1.0),
            'influence': 1.0,
            'openness': np.random.uniform(0.3, 0.9)
        }
    
    def decide(self, environment: Dict[str, Any]) -> float:
        """
        Form opinion based on environment and neighbors
        
        Args:
            environment: Social context and information
            
        Returns:
            Updated opinion value
        """
        # Gather neighbor opinions
        neighbor_opinions = [
            n.state.get('opinion', 0) for n in self.neighbors 
            if isinstance(n, SocialAgent)
        ]
        
        if not neighbor_opinions:
            return self.state['opinion']
        
        # Calculate weighted average of neighbor opinions
        avg_neighbor_opinion = np.mean(neighbor_opinions)
        
        # Update opinion based on openness to others
        new_opinion = (
            self.state['opinion'] * (1 - self.state['openness']) +
            avg_neighbor_opinion * self.state['openness']
        )
        
        # Add some random variation
        new_opinion += np.random.normal(0, 0.05)
        new_opinion = np.clip(new_opinion, -1, 1)
        
        return new_opinion
    
    def update(self, decision: float, feedback: Dict[str, Any]) -> None:
        """
        Update opinion and confidence
        
        Args:
            decision: New opinion value
            feedback: Social feedback
        """
        self.state['opinion'] = decision
        
        # Update confidence based on agreement with neighbors
        agreement_rate = feedback.get('agreement_rate', 0.5)
        self.state['confidence'] = (
            self.state['confidence'] * 0.8 + agreement_rate * 0.2
        )
    
    def _interact(self, other: 'BaseAgent') -> float:
        """
        Social interaction measuring opinion alignment
        
        Args:
            other: Another agent
            
        Returns:
            Interaction strength (agreement level)
        """
        if not isinstance(other, SocialAgent):
            return 0.0
        
        opinion_diff = abs(self.state['opinion'] - other.state['opinion'])
        agreement = 1.0 - opinion_diff / 2.0  # Normalized to [0, 1]
        
        return agreement


class AdaptiveAgent(BaseAgent):
    """Agent that learns and adapts behavior over time"""
    
    def __init__(self, agent_id: int):
        """
        Initialize adaptive agent
        
        Args:
            agent_id: Unique identifier
        """
        super().__init__(agent_id)
        self.state = {
            'strategy': np.random.choice(['cooperative', 'competitive', 'neutral']),
            'success_rate': 0.5,
            'learning_rate': 0.1,
            'memory': []
        }
    
    def decide(self, environment: Dict[str, Any]) -> str:
        """
        Choose strategy based on past experience
        
        Args:
            environment: Current environment state
            
        Returns:
            Strategy choice
        """
        # Exploit if success rate is high, explore otherwise
        if self.state['success_rate'] > 0.7:
            return self.state['strategy']
        else:
            # Explore: occasionally try different strategies
            if np.random.random() < 0.2:
                return np.random.choice(['cooperative', 'competitive', 'neutral'])
            return self.state['strategy']
    
    def update(self, decision: str, feedback: Dict[str, Any]) -> None:
        """
        Learn from experience and adapt strategy
        
        Args:
            decision: Strategy used
            feedback: Outcome of strategy
        """
        success = feedback.get('success', False)
        
        # Update success rate using exponential moving average
        if success:
            self.state['success_rate'] = (
                self.state['success_rate'] * (1 - self.state['learning_rate']) +
                1.0 * self.state['learning_rate']
            )
        else:
            self.state['success_rate'] = (
                self.state['success_rate'] * (1 - self.state['learning_rate']) +
                0.0 * self.state['learning_rate']
            )
        
        # Store experience in memory (keep last 10)
        self.state['memory'].append({
            'strategy': decision,
            'success': success,
            'step': feedback.get('step', 0)
        })
        if len(self.state['memory']) > 10:
            self.state['memory'].pop(0)
        
        # Adapt strategy if consistently failing
        if self.state['success_rate'] < 0.3 and len(self.state['memory']) >= 5:
            # Change strategy
            strategies = ['cooperative', 'competitive', 'neutral']
            strategies.remove(self.state['strategy'])
            self.state['strategy'] = np.random.choice(strategies)


if __name__ == "__main__":
    # Example: Create a small network of agents
    print("Creating agent network...")
    
    economic_agents = [EconomicAgent(i, initial_wealth=np.random.uniform(500, 1500)) 
                      for i in range(5)]
    social_agents = [SocialAgent(i + 5) for i in range(5)]
    
    # Create connections
    for i in range(len(economic_agents) - 1):
        economic_agents[i].add_neighbor(economic_agents[i + 1])
    
    for i in range(len(social_agents) - 1):
        social_agents[i].add_neighbor(social_agents[i + 1])
    
    print("Agent network created successfully")
    print(f"Economic agents: {len(economic_agents)}")
    print(f"Social agents: {len(social_agents)}")
