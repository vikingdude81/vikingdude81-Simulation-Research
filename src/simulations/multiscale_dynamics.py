"""
Multiscale Dynamics Module

This module implements multiscale modeling approaches for complex systems,
allowing simulation at multiple levels (micro, meso, macro) simultaneously.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum


class ScaleLevel(Enum):
    """Enumeration of scale levels in the system"""
    MICRO = "micro"      # Individual agents
    MESO = "meso"        # Local groups/communities
    MACRO = "macro"      # System-wide aggregates


@dataclass
class ScaleState:
    """State at a particular scale level"""
    level: ScaleLevel
    variables: Dict[str, Any]
    timestamp: int
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable value with optional default"""
        return self.variables.get(name, default)
    
    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable value"""
        self.variables[name] = value


class MultiscaleModel:
    """
    Base class for multiscale modeling
    
    Coordinates interactions between different scale levels and ensures
    consistency across scales through upscaling and downscaling operations.
    """
    
    def __init__(self, n_agents: int = 1000):
        """
        Initialize multiscale model
        
        Args:
            n_agents: Number of agents at micro scale
        """
        self.n_agents = n_agents
        self.current_step = 0
        
        # Initialize states at each scale
        self.micro_state = ScaleState(
            level=ScaleLevel.MICRO,
            variables={'agents': []},
            timestamp=0
        )
        
        self.meso_state = ScaleState(
            level=ScaleLevel.MESO,
            variables={'groups': [], 'local_averages': {}},
            timestamp=0
        )
        
        self.macro_state = ScaleState(
            level=ScaleLevel.MACRO,
            variables={'global_mean': 0, 'global_variance': 0},
            timestamp=0
        )
        
        self.history: Dict[ScaleLevel, List[ScaleState]] = {
            ScaleLevel.MICRO: [],
            ScaleLevel.MESO: [],
            ScaleLevel.MACRO: []
        }
    
    def upscale(self, from_level: ScaleLevel, to_level: ScaleLevel) -> Dict[str, Any]:
        """
        Aggregate information from finer to coarser scale
        
        Args:
            from_level: Source scale level
            to_level: Target scale level
            
        Returns:
            Aggregated variables at target scale
        """
        if from_level == ScaleLevel.MICRO and to_level == ScaleLevel.MESO:
            return self._micro_to_meso()
        elif from_level == ScaleLevel.MESO and to_level == ScaleLevel.MACRO:
            return self._meso_to_macro()
        elif from_level == ScaleLevel.MICRO and to_level == ScaleLevel.MACRO:
            meso_data = self._micro_to_meso()
            return self._aggregate_to_macro(meso_data)
        else:
            raise ValueError(f"Invalid upscaling from {from_level} to {to_level}")
    
    def downscale(self, from_level: ScaleLevel, to_level: ScaleLevel) -> Dict[str, Any]:
        """
        Disaggregate information from coarser to finer scale
        
        Args:
            from_level: Source scale level
            to_level: Target scale level
            
        Returns:
            Disaggregated variables at target scale
        """
        if from_level == ScaleLevel.MACRO and to_level == ScaleLevel.MESO:
            return self._macro_to_meso()
        elif from_level == ScaleLevel.MESO and to_level == ScaleLevel.MICRO:
            return self._meso_to_micro()
        elif from_level == ScaleLevel.MACRO and to_level == ScaleLevel.MICRO:
            meso_data = self._macro_to_meso()
            return self._disaggregate_to_micro(meso_data)
        else:
            raise ValueError(f"Invalid downscaling from {from_level} to {to_level}")
    
    def _micro_to_meso(self) -> Dict[str, Any]:
        """Aggregate micro-level agents into meso-level groups"""
        agents = self.micro_state.get_variable('agents', [])
        
        if not agents:
            return {'groups': [], 'local_averages': {}}
        
        # Simple spatial grouping (can be overridden)
        group_size = int(np.sqrt(len(agents)))
        groups = []
        
        for i in range(0, len(agents), group_size):
            group = agents[i:i + group_size]
            if group:
                groups.append({
                    'size': len(group),
                    'agent_ids': [a.get('id', i+j) for j, a in enumerate(group)],
                    'avg_state': np.mean([a.get('state', 0) for a in group])
                })
        
        return {'groups': groups, 'local_averages': {}}
    
    def _meso_to_macro(self) -> Dict[str, Any]:
        """Aggregate meso-level groups to macro-level statistics"""
        groups = self.meso_state.get_variable('groups', [])
        
        if not groups:
            return {'global_mean': 0, 'global_variance': 0}
        
        group_means = [g['avg_state'] for g in groups]
        
        return {
            'global_mean': np.mean(group_means),
            'global_variance': np.var(group_means),
            'total_groups': len(groups)
        }
    
    def _aggregate_to_macro(self, meso_data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to aggregate meso data to macro"""
        groups = meso_data.get('groups', [])
        if not groups:
            return {'global_mean': 0, 'global_variance': 0}
        
        group_means = [g['avg_state'] for g in groups]
        return {
            'global_mean': np.mean(group_means),
            'global_variance': np.var(group_means)
        }
    
    def _macro_to_meso(self) -> Dict[str, Any]:
        """Disaggregate macro statistics to meso-level expectations"""
        global_mean = self.macro_state.get_variable('global_mean', 0)
        global_variance = self.macro_state.get_variable('global_variance', 1)
        
        # Create synthetic groups based on macro statistics
        n_groups = max(10, self.n_agents // 100)
        groups = []
        
        for i in range(n_groups):
            groups.append({
                'size': self.n_agents // n_groups,
                'expected_avg': global_mean + np.random.normal(0, np.sqrt(global_variance))
            })
        
        return {'groups': groups}
    
    def _meso_to_micro(self) -> Dict[str, Any]:
        """Disaggregate meso-level groups to individual agents"""
        groups = self.meso_state.get_variable('groups', [])
        agents = []
        
        for group in groups:
            group_mean = group.get('avg_state', 0)
            group_size = group.get('size', 1)
            
            # Create agents with states around group mean
            for i in range(group_size):
                agents.append({
                    'id': len(agents),
                    'state': group_mean + np.random.normal(0, 0.1),
                    'group_id': len(agents) // group_size
                })
        
        return {'agents': agents}
    
    def _disaggregate_to_micro(self, meso_data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to disaggregate meso to micro"""
        groups = meso_data.get('groups', [])
        agents = []
        
        for group_idx, group in enumerate(groups):
            expected_avg = group.get('expected_avg', 0)
            size = group.get('size', 10)
            
            for i in range(size):
                agents.append({
                    'id': len(agents),
                    'state': expected_avg + np.random.normal(0, 0.1),
                    'group_id': group_idx
                })
        
        return {'agents': agents}
    
    def couple_scales(self) -> None:
        """
        Perform scale coupling: update all scales consistently
        
        This ensures information flows both up (aggregation) and down (disaggregation)
        to maintain consistency across scales.
        """
        # Upscale from micro to meso
        meso_data = self.upscale(ScaleLevel.MICRO, ScaleLevel.MESO)
        for key, value in meso_data.items():
            self.meso_state.set_variable(key, value)
        
        # Upscale from meso to macro
        macro_data = self.upscale(ScaleLevel.MESO, ScaleLevel.MACRO)
        for key, value in macro_data.items():
            self.macro_state.set_variable(key, value)
        
        # Store history
        self.history[ScaleLevel.MICRO].append(self.micro_state)
        self.history[ScaleLevel.MESO].append(self.meso_state)
        self.history[ScaleLevel.MACRO].append(self.macro_state)
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one time step of multiscale simulation
        
        Returns:
            Current state across all scales
        """
        self.current_step += 1
        
        # Update states at each scale
        self._update_micro()
        self._update_meso()
        self._update_macro()
        
        # Couple scales to maintain consistency
        self.couple_scales()
        
        return {
            'step': self.current_step,
            'micro': self.micro_state.variables,
            'meso': self.meso_state.variables,
            'macro': self.macro_state.variables
        }
    
    def _update_micro(self) -> None:
        """Update micro-level dynamics (to be overridden)"""
        agents = self.micro_state.get_variable('agents', [])
        for agent in agents:
            # Simple random walk dynamics
            agent['state'] = agent.get('state', 0) + np.random.normal(0, 0.1)
    
    def _update_meso(self) -> None:
        """Update meso-level dynamics (to be overridden)"""
        pass
    
    def _update_macro(self) -> None:
        """Update macro-level dynamics (to be overridden)"""
        pass
    
    def run(self, n_steps: int) -> List[Dict[str, Any]]:
        """
        Run complete multiscale simulation
        
        Args:
            n_steps: Number of time steps to simulate
            
        Returns:
            Complete simulation history
        """
        history = []
        
        for _ in range(n_steps):
            state = self.step()
            history.append(state)
        
        return history


class GovernmentMultiscaleModel(MultiscaleModel):
    """
    Multiscale model for government policy impact
    
    Integrates individual citizen behavior (micro), community dynamics (meso),
    and national-level outcomes (macro).
    """
    
    def __init__(self, n_agents: int = 1000, n_communities: int = 10):
        """
        Initialize government multiscale model
        
        Args:
            n_agents: Number of individual citizens
            n_communities: Number of communities
        """
        super().__init__(n_agents)
        self.n_communities = n_communities
        
        # Initialize micro-level agents
        agents = []
        for i in range(n_agents):
            agents.append({
                'id': i,
                'state': np.random.normal(50, 10),  # Initial satisfaction
                'wealth': np.random.lognormal(10, 1),
                'community_id': i % n_communities
            })
        
        self.micro_state.set_variable('agents', agents)
        
        # Initialize meso-level communities
        communities = []
        for i in range(n_communities):
            communities.append({
                'id': i,
                'cohesion': np.random.uniform(0.5, 1.0),
                'resources': np.random.uniform(1000, 5000)
            })
        
        self.meso_state.set_variable('groups', communities)
        
        # Initialize macro-level state
        self.macro_state.set_variable('policy_effectiveness', 0.5)
        self.macro_state.set_variable('inequality', 0.3)
    
    def _update_micro(self) -> None:
        """Update individual citizen states"""
        agents = self.micro_state.get_variable('agents', [])
        communities = self.meso_state.get_variable('groups', [])
        policy_effect = self.macro_state.get_variable('policy_effectiveness', 0.5)
        
        for agent in agents:
            community_id = agent.get('community_id', 0)
            
            if community_id < len(communities):
                community = communities[community_id]
                
                # Agent state influenced by:
                # 1. Community cohesion
                # 2. National policy
                # 3. Individual wealth
                community_effect = community['cohesion'] * 5
                policy_effect_individual = policy_effect * 10
                wealth_effect = 0.001 * agent['wealth']
                
                state_change = (community_effect + policy_effect_individual + 
                               wealth_effect + np.random.normal(0, 2))
                
                agent['state'] = np.clip(agent['state'] + state_change, 0, 100)
    
    def _update_meso(self) -> None:
        """Update community-level dynamics"""
        communities = self.meso_state.get_variable('groups', [])
        agents = self.micro_state.get_variable('agents', [])
        
        for community in communities:
            community_id = community['id']
            
            # Get agents in this community
            community_agents = [a for a in agents if a.get('community_id') == community_id]
            
            if community_agents:
                # Update cohesion based on agent satisfaction variance
                satisfactions = [a['state'] for a in community_agents]
                variance = np.var(satisfactions)
                
                # Lower variance -> higher cohesion
                community['cohesion'] = np.clip(
                    community['cohesion'] + 0.01 * (50 - variance) / 50,
                    0, 1
                )
    
    def _update_macro(self) -> None:
        """Update national-level indicators"""
        agents = self.micro_state.get_variable('agents', [])
        
        if agents:
            # Calculate inequality
            wealths = [a['wealth'] for a in agents]
            sorted_wealth = sorted(wealths)
            n = len(sorted_wealth)
            
            # Simplified Gini coefficient
            gini = sum((2 * (i + 1) - n - 1) * w for i, w in enumerate(sorted_wealth)) / (n * sum(sorted_wealth))
            self.macro_state.set_variable('inequality', gini)
            
            # Update policy effectiveness based on average satisfaction
            avg_satisfaction = np.mean([a['state'] for a in agents])
            self.macro_state.set_variable('policy_effectiveness', avg_satisfaction / 100)


def analyze_scale_separation(model: MultiscaleModel, 
                            variable: str = 'state') -> Dict[str, float]:
    """
    Analyze scale separation in the model
    
    Quantifies how much variance exists at each scale level.
    
    Args:
        model: Multiscale model to analyze
        variable: Variable to analyze across scales
        
    Returns:
        Variance decomposition across scales
    """
    # Get micro-level data
    agents = model.micro_state.get_variable('agents', [])
    if not agents:
        return {}
    
    values = [a.get(variable, 0) for a in agents]
    total_variance = np.var(values)
    
    # Calculate within-group and between-group variance
    groups = model.meso_state.get_variable('groups', [])
    
    if not groups:
        return {'total_variance': total_variance}
    
    group_means = []
    within_group_vars = []
    
    for group in groups:
        agent_ids = group.get('agent_ids', [])
        group_values = [agents[i].get(variable, 0) for i in agent_ids if i < len(agents)]
        
        if group_values:
            group_means.append(np.mean(group_values))
            within_group_vars.append(np.var(group_values))
    
    within_variance = np.mean(within_group_vars) if within_group_vars else 0
    between_variance = np.var(group_means) if group_means else 0
    
    return {
        'total_variance': total_variance,
        'within_group_variance': within_variance,
        'between_group_variance': between_variance,
        'variance_ratio': between_variance / total_variance if total_variance > 0 else 0
    }


if __name__ == "__main__":
    print("Testing Multiscale Dynamics...")
    
    # Create and run government multiscale model
    model = GovernmentMultiscaleModel(n_agents=500, n_communities=10)
    
    print(f"Initial state:")
    print(f"  Agents: {len(model.micro_state.get_variable('agents', []))}")
    print(f"  Communities: {len(model.meso_state.get_variable('groups', []))}")
    
    # Run simulation
    history = model.run(n_steps=20)
    
    print(f"\nSimulation completed: {len(history)} steps")
    
    # Analyze final state
    final_state = history[-1]
    print(f"\nFinal macro state:")
    print(f"  Policy effectiveness: {final_state['macro']['policy_effectiveness']:.3f}")
    print(f"  Inequality: {final_state['macro']['inequality']:.3f}")
    
    # Analyze scale separation
    separation = analyze_scale_separation(model)
    print(f"\nScale separation analysis:")
    for key, value in separation.items():
        print(f"  {key}: {value:.3f}")
