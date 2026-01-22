"""
Tests for government simulation module
"""

import pytest
import numpy as np
from src.simulations.government_simulation import (
    GovernmentSimulation, 
    Policy, 
    PolicyType, 
    Agent,
    create_example_policies
)


class TestAgent:
    """Test Agent class"""
    
    def test_agent_initialization(self):
        """Test agent is properly initialized"""
        agent = Agent(
            id=1,
            wealth=1000.0,
            satisfaction=50.0,
            political_leaning=0.0,
            education_level=0.5,
            health_status=0.8
        )
        
        assert agent.id == 1
        assert agent.wealth == 1000.0
        assert agent.satisfaction == 50.0
    
    def test_update_satisfaction(self):
        """Test satisfaction update is bounded"""
        agent = Agent(id=1, wealth=1000, satisfaction=50, 
                     political_leaning=0, education_level=0.5, health_status=0.8)
        
        # Test increase
        agent.update_satisfaction(30)
        assert agent.satisfaction == 80
        
        # Test upper bound
        agent.update_satisfaction(30)
        assert agent.satisfaction == 100
        
        # Test decrease
        agent.update_satisfaction(-150)
        assert agent.satisfaction == 0
    
    def test_update_wealth(self):
        """Test wealth update with floor at zero"""
        agent = Agent(id=1, wealth=1000, satisfaction=50,
                     political_leaning=0, education_level=0.5, health_status=0.8)
        
        agent.update_wealth(500)
        assert agent.wealth == 1500
        
        # Test floor at zero
        agent.update_wealth(-2000)
        assert agent.wealth == 0


class TestPolicy:
    """Test Policy class"""
    
    def test_policy_creation(self):
        """Test policy is created with correct attributes"""
        policy = Policy(
            name="Test Policy",
            policy_type=PolicyType.ECONOMIC,
            cost=1000,
            expected_impact={'base': 5},
            implementation_time=10
        )
        
        assert policy.name == "Test Policy"
        assert policy.policy_type == PolicyType.ECONOMIC
        assert policy.cost == 1000
    
    def test_policy_apply(self):
        """Test policy application to population"""
        population = [
            Agent(id=i, wealth=1000, satisfaction=50, political_leaning=0,
                 education_level=0.5, health_status=0.8)
            for i in range(10)
        ]
        
        policy = Policy(
            name="Test",
            policy_type=PolicyType.ECONOMIC,
            cost=100,
            expected_impact={'base': 10},
            implementation_time=5
        )
        
        impacts = policy.apply(population)
        
        assert impacts['affected_agents'] == 10
        assert len(impacts['satisfaction_change']) == 10


class TestGovernmentSimulation:
    """Test GovernmentSimulation class"""
    
    def test_initialization(self):
        """Test simulation initializes correctly"""
        sim = GovernmentSimulation(
            population_size=100,
            initial_budget=10000,
            time_steps=10
        )
        
        assert len(sim.population) == 100
        assert sim.budget == 10000
        assert sim.time_steps == 10
        assert sim.current_step == 0
    
    def test_population_diversity(self):
        """Test population has diverse characteristics"""
        sim = GovernmentSimulation(population_size=100)
        
        wealths = [a.wealth for a in sim.population]
        satisfactions = [a.satisfaction for a in sim.population]
        
        # Check that there's variation
        assert np.std(wealths) > 0
        assert np.std(satisfactions) > 0
    
    def test_implement_policy_success(self):
        """Test successful policy implementation"""
        sim = GovernmentSimulation(population_size=100, initial_budget=10000)
        
        policy = Policy(
            name="Test",
            policy_type=PolicyType.ECONOMIC,
            cost=1000,
            expected_impact={'base': 5},
            implementation_time=5
        )
        
        result = sim.implement_policy(policy)
        
        assert result['success'] == True
        assert sim.budget == 9000
        assert len(sim.policies_enacted) == 1
    
    def test_implement_policy_insufficient_budget(self):
        """Test policy implementation fails with insufficient budget"""
        sim = GovernmentSimulation(population_size=100, initial_budget=100)
        
        policy = Policy(
            name="Expensive",
            policy_type=PolicyType.ECONOMIC,
            cost=1000,
            expected_impact={'base': 5},
            implementation_time=5
        )
        
        result = sim.implement_policy(policy)
        
        assert result['success'] == False
        assert 'Insufficient budget' in result['reason']
    
    def test_simulation_step(self):
        """Test single simulation step"""
        sim = GovernmentSimulation(population_size=100)
        
        state = sim.step()
        
        assert state['step'] == 1
        assert 'avg_satisfaction' in state
        assert 'avg_wealth' in state
        assert 'wealth_inequality' in state
        assert len(sim.history) == 1
    
    def test_run_simulation(self):
        """Test complete simulation run"""
        sim = GovernmentSimulation(
            population_size=100,
            time_steps=10
        )
        
        history = sim.run_simulation()
        
        assert len(history) == 10
        assert sim.current_step == 10
    
    def test_run_simulation_with_policies(self):
        """Test simulation with policy implementation"""
        sim = GovernmentSimulation(
            population_size=100,
            time_steps=20
        )
        
        policies = create_example_policies()
        history = sim.run_simulation(policies[:2])
        
        assert len(history) == 20
        assert len(sim.policies_enacted) == 2
    
    def test_get_summary_statistics(self):
        """Test summary statistics generation"""
        sim = GovernmentSimulation(population_size=100, time_steps=10)
        sim.run_simulation()
        
        summary = sim.get_summary_statistics()
        
        assert 'initial_satisfaction' in summary
        assert 'final_satisfaction' in summary
        assert 'satisfaction_change' in summary
        assert 'total_policies' in summary


class TestExamplePolicies:
    """Test example policy creation"""
    
    def test_create_example_policies(self):
        """Test example policies are created correctly"""
        policies = create_example_policies()
        
        assert len(policies) > 0
        assert all(isinstance(p, Policy) for p in policies)
        
        # Check that policies have different types
        policy_types = {p.policy_type for p in policies}
        assert len(policy_types) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
