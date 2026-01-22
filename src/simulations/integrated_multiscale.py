"""
Integration module for combining government simulation with multiscale dynamics

This module provides tools to integrate agent-based government simulations
with multiscale modeling approaches.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from src.simulations.government_simulation import GovernmentSimulation, Policy, Agent
from src.simulations.multiscale_dynamics import (
    MultiscaleModel, GovernmentMultiscaleModel, ScaleLevel, ScaleState
)


class IntegratedGovernmentModel:
    """
    Integrated model combining government simulation with multiscale dynamics
    
    This model runs government policy simulations while tracking dynamics
    at multiple scales simultaneously.
    """
    
    def __init__(self,
                 population_size: int = 10000,
                 n_communities: int = 20,
                 initial_budget: float = 1000000.0,
                 time_steps: int = 100):
        """
        Initialize integrated government model
        
        Args:
            population_size: Number of citizen agents
            n_communities: Number of meso-level communities
            initial_budget: Government starting budget
            time_steps: Number of simulation steps
        """
        self.population_size = population_size
        self.n_communities = n_communities
        self.time_steps = time_steps
        
        # Initialize government simulation
        self.gov_sim = GovernmentSimulation(
            population_size=population_size,
            initial_budget=initial_budget,
            time_steps=time_steps
        )
        
        # Initialize multiscale model
        self.multiscale = GovernmentMultiscaleModel(
            n_agents=population_size,
            n_communities=n_communities
        )
        
        # Synchronize initial states
        self._synchronize_states()
        
        self.history: List[Dict[str, Any]] = []
    
    def _synchronize_states(self) -> None:
        """Synchronize states between government sim and multiscale model"""
        # Copy agent states from government sim to multiscale model
        gov_agents = self.gov_sim.population
        multi_agents = self.multiscale.micro_state.get_variable('agents', [])
        
        for i, (gov_agent, multi_agent) in enumerate(zip(gov_agents, multi_agents)):
            multi_agent['state'] = gov_agent.satisfaction
            multi_agent['wealth'] = gov_agent.wealth
            multi_agent['id'] = gov_agent.id
    
    def implement_policy_multiscale(self, policy: Policy) -> Dict[str, Any]:
        """
        Implement a policy and track effects across scales
        
        Args:
            policy: The policy to implement
            
        Returns:
            Policy impact metrics across all scales
        """
        # Implement in government simulation
        gov_result = self.gov_sim.implement_policy(policy)
        
        if not gov_result['success']:
            return gov_result
        
        # Synchronize changes to multiscale model
        self._synchronize_states()
        
        # Analyze impact at each scale
        micro_impact = self._measure_micro_impact()
        meso_impact = self._measure_meso_impact()
        macro_impact = self._measure_macro_impact()
        
        return {
            'success': True,
            'government_result': gov_result,
            'micro_impact': micro_impact,
            'meso_impact': meso_impact,
            'macro_impact': macro_impact
        }
    
    def _measure_micro_impact(self) -> Dict[str, float]:
        """Measure policy impact at micro (individual) level"""
        agents = self.multiscale.micro_state.get_variable('agents', [])
        
        if not agents:
            return {}
        
        satisfactions = [a['state'] for a in agents]
        wealths = [a['wealth'] for a in agents]
        
        return {
            'avg_satisfaction': np.mean(satisfactions),
            'satisfaction_std': np.std(satisfactions),
            'avg_wealth': np.mean(wealths),
            'wealth_std': np.std(wealths),
            'affected_individuals': len(agents)
        }
    
    def _measure_meso_impact(self) -> Dict[str, float]:
        """Measure policy impact at meso (community) level"""
        communities = self.multiscale.meso_state.get_variable('groups', [])
        
        if not communities:
            return {}
        
        cohesions = [c.get('cohesion', 0) for c in communities]
        resources = [c.get('resources', 0) for c in communities]
        
        return {
            'avg_cohesion': np.mean(cohesions),
            'cohesion_std': np.std(cohesions),
            'total_resources': sum(resources),
            'resource_inequality': np.std(resources) / np.mean(resources) if np.mean(resources) > 0 else 0,
            'affected_communities': len(communities)
        }
    
    def _measure_macro_impact(self) -> Dict[str, float]:
        """Measure policy impact at macro (national) level"""
        return {
            'policy_effectiveness': self.multiscale.macro_state.get_variable('policy_effectiveness', 0),
            'inequality': self.multiscale.macro_state.get_variable('inequality', 0),
            'budget': self.gov_sim.budget,
            'policies_enacted': len(self.gov_sim.policies_enacted)
        }
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one integrated time step
        
        Returns:
            Complete state across all scales
        """
        # Step government simulation
        gov_state = self.gov_sim.step()
        
        # Synchronize and step multiscale model
        self._synchronize_states()
        multi_state = self.multiscale.step()
        
        # Combine states
        integrated_state = {
            'step': self.gov_sim.current_step,
            'government': gov_state,
            'multiscale': multi_state,
            'micro_metrics': self._measure_micro_impact(),
            'meso_metrics': self._measure_meso_impact(),
            'macro_metrics': self._measure_macro_impact()
        }
        
        self.history.append(integrated_state)
        return integrated_state
    
    def run(self, policies: Optional[List[Policy]] = None) -> List[Dict[str, Any]]:
        """
        Run complete integrated simulation
        
        Args:
            policies: Optional list of policies to implement
            
        Returns:
            Complete simulation history
        """
        if policies is None:
            policies = []
        
        # Schedule policy implementations
        if policies:
            policy_schedule = np.linspace(0, self.time_steps, len(policies) + 1, dtype=int)[1:]
        else:
            policy_schedule = []
        
        policy_idx = 0
        
        for step in range(self.time_steps):
            # Check if we should implement a policy
            if policy_idx < len(policies) and step in policy_schedule:
                self.implement_policy_multiscale(policies[policy_idx])
                policy_idx += 1
            
            # Execute simulation step
            self.step()
        
        return self.history
    
    def analyze_cross_scale_dynamics(self) -> Dict[str, Any]:
        """
        Analyze dynamics across scales
        
        Returns:
            Cross-scale analysis metrics
        """
        if not self.history:
            return {}
        
        # Extract time series for each scale
        micro_satisfaction = [h['micro_metrics']['avg_satisfaction'] for h in self.history]
        meso_cohesion = [h['meso_metrics']['avg_cohesion'] for h in self.history]
        macro_effectiveness = [h['macro_metrics']['policy_effectiveness'] for h in self.history]
        
        # Calculate correlations
        corr_micro_meso = np.corrcoef(micro_satisfaction, meso_cohesion)[0, 1] if len(micro_satisfaction) > 1 else 0
        corr_meso_macro = np.corrcoef(meso_cohesion, macro_effectiveness)[0, 1] if len(meso_cohesion) > 1 else 0
        corr_micro_macro = np.corrcoef(micro_satisfaction, macro_effectiveness)[0, 1] if len(micro_satisfaction) > 1 else 0
        
        # Calculate volatility at each scale
        micro_volatility = np.std(np.diff(micro_satisfaction)) if len(micro_satisfaction) > 1 else 0
        meso_volatility = np.std(np.diff(meso_cohesion)) if len(meso_cohesion) > 1 else 0
        macro_volatility = np.std(np.diff(macro_effectiveness)) if len(macro_effectiveness) > 1 else 0
        
        return {
            'correlations': {
                'micro_meso': corr_micro_meso,
                'meso_macro': corr_meso_macro,
                'micro_macro': corr_micro_macro
            },
            'volatility': {
                'micro': micro_volatility,
                'meso': meso_volatility,
                'macro': macro_volatility
            },
            'trends': {
                'micro_satisfaction': {
                    'initial': micro_satisfaction[0],
                    'final': micro_satisfaction[-1],
                    'change': micro_satisfaction[-1] - micro_satisfaction[0]
                },
                'meso_cohesion': {
                    'initial': meso_cohesion[0],
                    'final': meso_cohesion[-1],
                    'change': meso_cohesion[-1] - meso_cohesion[0]
                },
                'macro_effectiveness': {
                    'initial': macro_effectiveness[0],
                    'final': macro_effectiveness[-1],
                    'change': macro_effectiveness[-1] - macro_effectiveness[0]
                }
            }
        }
    
    def identify_scale_transitions(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Identify moments when dynamics transition between scales
        
        Args:
            threshold: Threshold for detecting significant transitions (in std devs)
            
        Returns:
            List of identified scale transition events
        """
        if len(self.history) < 3:
            return []
        
        transitions = []
        
        # Extract metrics
        micro_values = [h['micro_metrics']['avg_satisfaction'] for h in self.history]
        macro_values = [h['macro_metrics']['policy_effectiveness'] for h in self.history]
        
        # Calculate derivatives
        micro_changes = np.diff(micro_values)
        macro_changes = np.diff(macro_values)
        
        micro_std = np.std(micro_changes)
        macro_std = np.std(macro_changes)
        
        # Detect transitions
        for i in range(len(micro_changes)):
            micro_significant = abs(micro_changes[i]) > threshold * micro_std if micro_std > 0 else False
            macro_significant = abs(macro_changes[i]) > threshold * macro_std if macro_std > 0 else False
            
            if micro_significant and not macro_significant:
                transitions.append({
                    'step': i + 1,
                    'type': 'micro_only',
                    'description': 'Micro-level change without macro response'
                })
            elif macro_significant and not micro_significant:
                transitions.append({
                    'step': i + 1,
                    'type': 'macro_only',
                    'description': 'Macro-level change without micro origin'
                })
            elif micro_significant and macro_significant:
                transitions.append({
                    'step': i + 1,
                    'type': 'cross_scale',
                    'description': 'Simultaneous change across scales'
                })
        
        return transitions


def compare_single_vs_multiscale(
    policies: List[Policy],
    population_size: int = 1000,
    time_steps: int = 50
) -> Dict[str, Any]:
    """
    Compare single-scale vs multiscale modeling results
    
    Args:
        policies: List of policies to test
        population_size: Number of agents
        time_steps: Simulation duration
        
    Returns:
        Comparison of both approaches
    """
    # Run single-scale simulation
    single_scale = GovernmentSimulation(
        population_size=population_size,
        time_steps=time_steps
    )
    single_history = single_scale.run_simulation(policies)
    
    # Run multiscale simulation
    multiscale = IntegratedGovernmentModel(
        population_size=population_size,
        time_steps=time_steps
    )
    multi_history = multiscale.run(policies)
    
    # Compare final outcomes
    single_final = single_history[-1]
    multi_final = multi_history[-1]
    
    return {
        'single_scale': {
            'final_satisfaction': single_final['avg_satisfaction'],
            'final_inequality': single_final['wealth_inequality'],
            'computation_time': 'baseline'
        },
        'multiscale': {
            'final_satisfaction': multi_final['micro_metrics']['avg_satisfaction'],
            'final_inequality': multi_final['macro_metrics']['inequality'],
            'cross_scale_insights': multiscale.analyze_cross_scale_dynamics(),
            'scale_transitions': len(multiscale.identify_scale_transitions())
        },
        'insights': {
            'additional_information': 'Multiscale provides community-level and cross-scale dynamics',
            'use_multiscale_when': [
                'Need to understand meso-level (community) dynamics',
                'Want to track how micro changes affect macro outcomes',
                'Studying emergence and scale-dependent phenomena'
            ]
        }
    }


if __name__ == "__main__":
    from src.simulations.government_simulation import create_example_policies
    
    print("Testing Integrated Multiscale Government Model...")
    
    # Create integrated model
    model = IntegratedGovernmentModel(
        population_size=500,
        n_communities=10,
        time_steps=30
    )
    
    # Get example policies
    policies = create_example_policies()[:2]
    
    print(f"Running simulation with {len(policies)} policies...")
    history = model.run(policies)
    
    print(f"\nSimulation completed: {len(history)} steps")
    
    # Analyze cross-scale dynamics
    analysis = model.analyze_cross_scale_dynamics()
    
    print("\nCross-Scale Analysis:")
    print(f"  Correlations:")
    for scale_pair, corr in analysis['correlations'].items():
        print(f"    {scale_pair}: {corr:.3f}")
    
    print(f"\n  Volatility:")
    for scale, vol in analysis['volatility'].items():
        print(f"    {scale}: {vol:.4f}")
    
    # Identify transitions
    transitions = model.identify_scale_transitions()
    print(f"\n  Scale transitions detected: {len(transitions)}")
    
    if transitions:
        print("  Key transitions:")
        for t in transitions[:3]:
            print(f"    Step {t['step']}: {t['description']}")
