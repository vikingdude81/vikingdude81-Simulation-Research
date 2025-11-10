"""
🚀 Deploy Quantum Genetic Champion to Prisoner's Dilemma / Economic Simulation

Integrates the champion genome as agent behavior controller for:
- Cooperation/defection decisions
- Resource allocation strategies
- Government intervention parameters
- Economic policy adaptation
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "prisoner_dilemma_64gene"))

from quantum_genetics.deploy_champion import ChampionGenome
from quantum_genetics.quantum_genetic_agents import QuantumAgent


class QuantumEconomicAgent:
    """
    Economic agent controlled by quantum genetic champion.
    
    The genome parameters control economic behavior:
    - μ (mutation): Strategy exploration / innovation
    - ω (oscillation): Response to economic cycles
    - d (decoherence): Memory of past interactions
    - φ (phase): Synchronization with market cycles
    """
    
    def __init__(self, agent_id, environment='standard', initial_wealth=100.0):
        """
        Initialize quantum economic agent.
        
        Args:
            agent_id: Unique agent identifier
            environment: Economic condition ('standard', 'boom', 'recession', 'volatile')
            initial_wealth: Starting wealth
        """
        self.agent_id = agent_id
        self.champion = ChampionGenome()
        self.genome = self.champion.get_genome()
        self.environment = environment
        
        # Create quantum agent
        self.quantum_agent = self.champion.create_agent(agent_id=agent_id, environment=environment)
        
        # Economic state
        self.wealth = initial_wealth
        self.cooperation_history = []
        self.interaction_count = 0
        
    def decide_cooperation(self, opponent_history, current_round):
        """
        Decide whether to cooperate or defect in prisoner's dilemma.
        
        Args:
            opponent_history: List of opponent's past actions (True=cooperate, False=defect)
            current_round: Current round number
        
        Returns:
            bool: True to cooperate, False to defect
        """
        # Evolve quantum agent
        self.quantum_agent.evolve(current_round)
        
        # Get quantum traits
        creativity = self.quantum_agent.traits[0]
        coherence = self.quantum_agent.traits[1]
        longevity = self.quantum_agent.traits[2]
        
        # Calculate cooperation probability
        if not opponent_history:
            # First interaction - base decision on creativity
            cooperation_prob = creativity
        else:
            # Consider opponent's cooperation rate
            opponent_coop_rate = sum(opponent_history) / len(opponent_history)
            
            # High coherence = reciprocate opponent's behavior
            # Low coherence = more independent decision
            reciprocity_factor = coherence * opponent_coop_rate
            
            # High longevity = maintain consistent strategy
            # Low longevity = more adaptive
            consistency_factor = longevity * (sum(self.cooperation_history[-10:]) / min(10, len(self.cooperation_history)) if self.cooperation_history else 0.5)
            
            # High creativity = explore new strategies
            exploration_factor = creativity * np.random.random()
            
            # Combined probability
            cooperation_prob = (reciprocity_factor * 0.4 + 
                              consistency_factor * 0.4 + 
                              exploration_factor * 0.2)
        
        # Make decision
        cooperate = np.random.random() < cooperation_prob
        self.cooperation_history.append(cooperate)
        self.interaction_count += 1
        
        return cooperate
    
    def allocate_resources(self, total_resources, num_recipients):
        """
        Decide how to allocate resources among recipients.
        
        Args:
            total_resources: Total resources to allocate
            num_recipients: Number of recipients
        
        Returns:
            np.array: Resource allocation for each recipient
        """
        # Evolve agent
        self.quantum_agent.evolve(self.interaction_count)
        
        # Get traits
        creativity = self.quantum_agent.traits[0]
        coherence = self.quantum_agent.traits[1]
        
        # High creativity = more unequal distribution (risk-taking)
        # Low creativity = more equal distribution (safe)
        inequality_factor = creativity
        
        # High coherence = structured allocation
        # Low coherence = random allocation
        randomness = 1.0 - coherence
        
        # Generate allocation
        if inequality_factor > 0.7:
            # Concentrated allocation (winner-take-most)
            allocation = np.random.dirichlet([0.5] * num_recipients)
        else:
            # More balanced allocation
            alpha = 2.0 + (1.0 - inequality_factor) * 8.0
            allocation = np.random.dirichlet([alpha] * num_recipients)
        
        # Add randomness
        noise = np.random.random(num_recipients) * randomness * 0.2
        allocation = allocation + noise
        allocation = allocation / allocation.sum()  # Renormalize
        
        return allocation * total_resources
    
    def government_policy_weights(self):
        """
        Generate government policy weights based on quantum traits.
        
        Returns:
            dict: Policy weights for different government interventions
        """
        # Evolve agent
        self.quantum_agent.evolve(self.interaction_count)
        
        # Get fitness and traits
        fitness = self.quantum_agent.get_final_fitness()
        creativity = self.quantum_agent.traits[0]
        coherence = self.quantum_agent.traits[1]
        longevity = self.quantum_agent.traits[2]
        
        # Map quantum traits to policy preferences
        policies = {
            # High creativity = more welfare spending (redistributive)
            'welfare_weight': float(0.2 + creativity * 0.6),
            
            # High coherence = more regulation (structured economy)
            'regulation_weight': float(0.3 + coherence * 0.5),
            
            # High longevity = more long-term investment
            'infrastructure_weight': float(0.2 + longevity * 0.6),
            
            # Balanced by fitness (successful strategies get more weight)
            'tax_rate': float(np.clip(0.15 + (1 - fitness/50000) * 0.35, 0.15, 0.5)),
            
            # Innovation spending scales with creativity
            'innovation_budget': float(0.05 + creativity * 0.15),
            
            # Emergency reserves scale with coherence (preparedness)
            'reserve_ratio': float(0.1 + coherence * 0.2)
        }
        
        return policies
    
    def get_statistics(self):
        """Get agent statistics."""
        if not self.cooperation_history:
            return {}
        
        return {
            'agent_id': self.agent_id,
            'wealth': float(self.wealth),
            'interactions': self.interaction_count,
            'cooperation_rate': sum(self.cooperation_history) / len(self.cooperation_history),
            'genome': self.genome,
            'environment': self.environment,
            'quantum_traits': {
                'creativity': float(self.quantum_agent.traits[0]),
                'coherence': float(self.quantum_agent.traits[1]),
                'longevity': float(self.quantum_agent.traits[2])
            }
        }


class QuantumGovernmentController:
    """
    Government controller using quantum genetic champion for policy decisions.
    """
    
    def __init__(self, environment='standard'):
        """Initialize quantum government controller."""
        self.champion = ChampionGenome()
        self.genome = self.champion.get_genome()
        self.environment = environment
        
        # Create quantum agent for government
        self.agent = self.champion.create_agent(agent_id=0, environment=environment)
        
        # State tracking
        self.interventions = []
        self.timestep = 0
        
    def decide_intervention(self, economic_state):
        """
        Decide government intervention based on economic state.
        
        Args:
            economic_state: dict with keys:
                - 'avg_wealth': float
                - 'gini_coefficient': float [0, 1]
                - 'cooperation_rate': float [0, 1]
                - 'growth_rate': float
        
        Returns:
            dict: Intervention decision
        """
        # Evolve agent
        self.agent.evolve(self.timestep)
        self.timestep += 1
        
        # Get traits
        creativity = self.agent.traits[0]
        coherence = self.agent.traits[1]
        longevity = self.agent.traits[2]
        
        # Analyze economic state
        wealth = economic_state.get('avg_wealth', 100)
        gini = economic_state.get('gini_coefficient', 0.3)
        coop_rate = economic_state.get('cooperation_rate', 0.5)
        growth = economic_state.get('growth_rate', 0.0)
        
        # Decide intervention type
        intervention = {
            'type': None,
            'magnitude': 0.0,
            'target': None,
            'reasoning': []
        }
        
        # High inequality + high coherence = redistribute
        if gini > 0.4 and coherence > 0.6:
            intervention['type'] = 'wealth_redistribution'
            intervention['magnitude'] = float((gini - 0.4) * coherence)
            intervention['target'] = 'bottom_20_percent'
            intervention['reasoning'].append(f"High inequality (Gini={gini:.2f}) + structured governance (coherence={coherence:.2f})")
        
        # Low cooperation + high creativity = stimulus
        elif coop_rate < 0.4 and creativity > 0.6:
            intervention['type'] = 'economic_stimulus'
            intervention['magnitude'] = float((0.4 - coop_rate) * creativity * 2)
            intervention['target'] = 'all_agents'
            intervention['reasoning'].append(f"Low cooperation ({coop_rate:.2f}) + innovative approach (creativity={creativity:.2f})")
        
        # Negative growth + high longevity = infrastructure
        elif growth < -0.05 and longevity > 0.5:
            intervention['type'] = 'infrastructure_investment'
            intervention['magnitude'] = float(abs(growth) * longevity * 3)
            intervention['target'] = 'public_goods'
            intervention['reasoning'].append(f"Negative growth ({growth:.2%}) + long-term focus (longevity={longevity:.2f})")
        
        # Low wealth + balanced traits = welfare
        elif wealth < 80:
            intervention['type'] = 'welfare_program'
            intervention['magnitude'] = float((80 - wealth) / 80 * (creativity + coherence) / 2)
            intervention['target'] = 'bottom_30_percent'
            intervention['reasoning'].append(f"Low avg wealth ({wealth:.0f}) + balanced governance")
        
        else:
            intervention['type'] = 'no_intervention'
            intervention['reasoning'].append(f"Economy stable (wealth={wealth:.0f}, Gini={gini:.2f}, coop={coop_rate:.2f})")
        
        self.interventions.append(intervention)
        return intervention


def demo_economic_integration():
    """Demonstrate quantum controller for economic simulation."""
    print("\n" + "="*70)
    print("🚀 QUANTUM GENETIC CONTROLLER - ECONOMIC SIMULATION DEMO")
    print("="*70)
    
    # Test 1: Quantum Economic Agent
    print("\n" + "="*70)
    print("👤 TEST 1: QUANTUM ECONOMIC AGENT")
    print("="*70)
    
    agent = QuantumEconomicAgent(agent_id=1, environment='standard')
    print(f"\n📊 Agent Genome: {agent.genome}")
    
    # Simulate prisoner's dilemma
    print("\n🎮 Prisoner's Dilemma Simulation (10 rounds):")
    opponent_history = []
    
    for round_num in range(10):
        cooperate = agent.decide_cooperation(opponent_history, round_num)
        opponent_cooperates = np.random.random() > 0.5
        opponent_history.append(opponent_cooperates)
        
        print(f"   Round {round_num+1}: Agent={'COOP' if cooperate else 'DEFECT':6s} | "
              f"Opponent={'COOP' if opponent_cooperates else 'DEFECT':6s}")
    
    stats = agent.get_statistics()
    print(f"\n📈 Statistics:")
    print(f"   Cooperation Rate: {stats['cooperation_rate']:.2%}")
    print(f"   Quantum Traits:")
    print(f"      Creativity: {stats['quantum_traits']['creativity']:.3f}")
    print(f"      Coherence:  {stats['quantum_traits']['coherence']:.3f}")
    print(f"      Longevity:  {stats['quantum_traits']['longevity']:.3f}")
    
    # Test resource allocation
    print("\n💰 Resource Allocation (100 units to 5 recipients):")
    allocation = agent.allocate_resources(total_resources=100, num_recipients=5)
    for i, amount in enumerate(allocation):
        print(f"   Recipient {i+1}: {amount:.2f}")
    
    # Test 2: Quantum Government Controller
    print("\n" + "="*70)
    print("🏛️ TEST 2: QUANTUM GOVERNMENT CONTROLLER")
    print("="*70)
    
    gov = QuantumGovernmentController(environment='standard')
    print(f"\n📊 Government Genome: {gov.genome}")
    
    # Test different economic scenarios
    scenarios = [
        {'name': 'High Inequality', 'avg_wealth': 120, 'gini_coefficient': 0.55, 
         'cooperation_rate': 0.6, 'growth_rate': 0.02},
        {'name': 'Low Cooperation', 'avg_wealth': 100, 'gini_coefficient': 0.35, 
         'cooperation_rate': 0.25, 'growth_rate': 0.01},
        {'name': 'Recession', 'avg_wealth': 75, 'gini_coefficient': 0.40, 
         'cooperation_rate': 0.5, 'growth_rate': -0.08},
        {'name': 'Stable Economy', 'avg_wealth': 110, 'gini_coefficient': 0.32, 
         'cooperation_rate': 0.65, 'growth_rate': 0.03}
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   Avg Wealth: {scenario['avg_wealth']:.0f} | Gini: {scenario['gini_coefficient']:.2f} | "
              f"Coop: {scenario['cooperation_rate']:.2f} | Growth: {scenario['growth_rate']:+.2%}")
        
        intervention = gov.decide_intervention(scenario)
        
        print(f"\n   🏛️ Government Decision:")
        print(f"      Type:      {intervention['type']}")
        print(f"      Magnitude: {intervention['magnitude']:.3f}")
        print(f"      Target:    {intervention['target']}")
        print(f"      Reasoning: {', '.join(intervention['reasoning'])}")
        
        results.append({
            'scenario': scenario,
            'intervention': intervention
        })
    
    # Save results
    output_file = Path(__file__).parent / "economic_controller_demo.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print(f"\n💾 Results saved to: {output_file.name}")
    print(f"\n📚 Integration Guide:")
    print(f"\n   For Agent Behavior:")
    print(f"   1. Import: from quantum_genetics.deploy_to_simulation import QuantumEconomicAgent")
    print(f"   2. Create: agent = QuantumEconomicAgent(agent_id=1)")
    print(f"   3. Use: cooperate = agent.decide_cooperation(opponent_history, round)")
    print(f"\n   For Government:")
    print(f"   1. Import: from quantum_genetics.deploy_to_simulation import QuantumGovernmentController")
    print(f"   2. Create: gov = QuantumGovernmentController()")
    print(f"   3. Use: intervention = gov.decide_intervention(economic_state)")
    print(f"\n✨ Ready for integration into prisoner_dilemma_64gene!")


if __name__ == "__main__":
    demo_economic_integration()
