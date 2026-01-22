"""
Government Simulation Module

This module provides a framework for simulating government decision-making
processes and policy outcomes using agent-based modeling.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class PolicyType(Enum):
    """Types of government policies"""
    ECONOMIC = "economic"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"


@dataclass
class Agent:
    """Represents a citizen agent in the simulation"""
    id: int
    wealth: float
    satisfaction: float
    political_leaning: float  # -1 (left) to 1 (right)
    education_level: float
    health_status: float
    
    def update_satisfaction(self, policy_impact: float) -> None:
        """Update agent satisfaction based on policy impact"""
        self.satisfaction = np.clip(self.satisfaction + policy_impact, 0, 100)
    
    def update_wealth(self, economic_change: float) -> None:
        """Update agent wealth"""
        self.wealth = max(0, self.wealth + economic_change)


@dataclass
class Policy:
    """Represents a government policy"""
    name: str
    policy_type: PolicyType
    cost: float
    expected_impact: Dict[str, float]
    implementation_time: int
    
    def apply(self, population: List[Agent]) -> Dict[str, Any]:
        """Apply policy to population and return impact metrics"""
        impacts = {
            'wealth_change': [],
            'satisfaction_change': [],
            'affected_agents': 0
        }
        
        for agent in population:
            # Calculate personalized impact based on agent characteristics
            base_impact = self.expected_impact.get('base', 0)
            wealth_factor = self.expected_impact.get('wealth_factor', 0) * agent.wealth
            education_factor = self.expected_impact.get('education_factor', 0) * agent.education_level
            
            total_impact = base_impact + wealth_factor + education_factor
            
            agent.update_satisfaction(total_impact)
            impacts['satisfaction_change'].append(total_impact)
            impacts['affected_agents'] += 1
        
        return impacts


class GovernmentSimulation:
    """Main simulation class for government policy modeling"""
    
    def __init__(self, 
                 population_size: int = 10000,
                 initial_budget: float = 1000000.0,
                 time_steps: int = 100):
        """
        Initialize the government simulation
        
        Args:
            population_size: Number of citizen agents
            initial_budget: Starting government budget
            time_steps: Number of simulation time steps
        """
        self.population_size = population_size
        self.budget = initial_budget
        self.time_steps = time_steps
        self.current_step = 0
        self.population: List[Agent] = []
        self.policies_enacted: List[Policy] = []
        self.history: List[Dict[str, Any]] = []
        
        self._initialize_population()
    
    def _initialize_population(self) -> None:
        """Create initial population with diverse characteristics"""
        # Use a configurable seed for reproducibility in tests, but allow override
        # For production use, don't set seed or make it a parameter
        np.random.seed(42)  # For reproducibility in tests
        
        for i in range(self.population_size):
            agent = Agent(
                id=i,
                wealth=np.random.lognormal(10, 1),
                satisfaction=np.random.normal(50, 15),
                political_leaning=np.random.normal(0, 0.3),
                education_level=np.random.beta(2, 5),
                health_status=np.random.beta(5, 2)
            )
            self.population.append(agent)
    
    def implement_policy(self, policy: Policy) -> Dict[str, Any]:
        """
        Implement a policy and track its effects
        
        Args:
            policy: The policy to implement
            
        Returns:
            Dictionary containing policy impact metrics
        """
        if policy.cost > self.budget:
            return {'success': False, 'reason': 'Insufficient budget'}
        
        self.budget -= policy.cost
        impacts = policy.apply(self.population)
        self.policies_enacted.append(policy)
        
        return {
            'success': True,
            'impacts': impacts,
            'remaining_budget': self.budget
        }
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one time step of the simulation
        
        Returns:
            State metrics for this time step
        """
        self.current_step += 1
        
        # Calculate aggregate statistics
        avg_satisfaction = np.mean([a.satisfaction for a in self.population])
        avg_wealth = np.mean([a.wealth for a in self.population])
        wealth_inequality = np.std([a.wealth for a in self.population])
        
        state = {
            'step': self.current_step,
            'avg_satisfaction': avg_satisfaction,
            'avg_wealth': avg_wealth,
            'wealth_inequality': wealth_inequality,
            'budget': self.budget,
            'policies_count': len(self.policies_enacted)
        }
        
        self.history.append(state)
        return state
    
    def run_simulation(self, policies: Optional[List[Policy]] = None) -> List[Dict[str, Any]]:
        """
        Run the complete simulation
        
        Args:
            policies: Optional list of policies to implement during simulation
            
        Returns:
            Complete simulation history
        """
        if policies is None:
            policies = []
        
        # Implement policies at specified intervals
        policy_schedule = np.linspace(0, self.time_steps, len(policies) + 1, dtype=int)[1:]
        policy_idx = 0
        
        for step in range(self.time_steps):
            # Check if we should implement a policy
            if policy_idx < len(policies) and step in policy_schedule:
                self.implement_policy(policies[policy_idx])
                policy_idx += 1
            
            # Execute simulation step
            self.step()
        
        return self.history
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the simulation"""
        if not self.history:
            return {}
        
        return {
            'initial_satisfaction': self.history[0]['avg_satisfaction'],
            'final_satisfaction': self.history[-1]['avg_satisfaction'],
            'satisfaction_change': self.history[-1]['avg_satisfaction'] - self.history[0]['avg_satisfaction'],
            'initial_inequality': self.history[0]['wealth_inequality'],
            'final_inequality': self.history[-1]['wealth_inequality'],
            'total_policies': len(self.policies_enacted),
            'budget_remaining': self.budget
        }


def create_example_policies() -> List[Policy]:
    """Create example policies for demonstration"""
    policies = [
        Policy(
            name="Universal Basic Income",
            policy_type=PolicyType.ECONOMIC,
            cost=50000,
            expected_impact={
                'base': 5,
                'wealth_factor': -0.001,
                'education_factor': 0
            },
            implementation_time=10
        ),
        Policy(
            name="Education Reform",
            policy_type=PolicyType.EDUCATION,
            cost=30000,
            expected_impact={
                'base': 3,
                'wealth_factor': 0,
                'education_factor': 0.1
            },
            implementation_time=20
        ),
        Policy(
            name="Healthcare Expansion",
            policy_type=PolicyType.HEALTHCARE,
            cost=40000,
            expected_impact={
                'base': 4,
                'wealth_factor': -0.0005,
                'education_factor': 0.05
            },
            implementation_time=15
        )
    ]
    return policies


if __name__ == "__main__":
    # Example usage
    print("Initializing Government Simulation...")
    sim = GovernmentSimulation(population_size=1000, time_steps=50)
    
    print("Creating policy portfolio...")
    policies = create_example_policies()
    
    print("Running simulation...")
    history = sim.run_simulation(policies)
    
    print("\nSimulation Summary:")
    summary = sim.get_summary_statistics()
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
