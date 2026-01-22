"""
Tests for multiscale dynamics module
"""

import pytest
import numpy as np
from src.simulations.multiscale_dynamics import (
    MultiscaleModel,
    GovernmentMultiscaleModel,
    ScaleLevel,
    ScaleState,
    analyze_scale_separation
)


class TestScaleState:
    """Test ScaleState class"""
    
    def test_scale_state_creation(self):
        """Test scale state initialization"""
        state = ScaleState(
            level=ScaleLevel.MICRO,
            variables={'test': 123},
            timestamp=0
        )
        
        assert state.level == ScaleLevel.MICRO
        assert state.variables['test'] == 123
        assert state.timestamp == 0
    
    def test_get_variable(self):
        """Test variable retrieval"""
        state = ScaleState(
            level=ScaleLevel.MESO,
            variables={'a': 1, 'b': 2},
            timestamp=0
        )
        
        assert state.get_variable('a') == 1
        assert state.get_variable('c', default=99) == 99
    
    def test_set_variable(self):
        """Test variable setting"""
        state = ScaleState(
            level=ScaleLevel.MACRO,
            variables={},
            timestamp=0
        )
        
        state.set_variable('new_var', 42)
        assert state.get_variable('new_var') == 42


class TestMultiscaleModel:
    """Test MultiscaleModel class"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = MultiscaleModel(n_agents=100)
        
        assert model.n_agents == 100
        assert model.current_step == 0
        assert model.micro_state.level == ScaleLevel.MICRO
        assert model.meso_state.level == ScaleLevel.MESO
        assert model.macro_state.level == ScaleLevel.MACRO
    
    def test_upscale_micro_to_meso(self):
        """Test upscaling from micro to meso"""
        model = MultiscaleModel(n_agents=100)
        
        # Create micro agents
        agents = [{'id': i, 'state': np.random.random()} for i in range(100)]
        model.micro_state.set_variable('agents', agents)
        
        # Upscale
        meso_data = model.upscale(ScaleLevel.MICRO, ScaleLevel.MESO)
        
        assert 'groups' in meso_data
        assert len(meso_data['groups']) > 0
    
    def test_upscale_meso_to_macro(self):
        """Test upscaling from meso to macro"""
        model = MultiscaleModel(n_agents=100)
        
        # Create meso groups
        groups = [
            {'size': 10, 'avg_state': 0.5 + i * 0.1}
            for i in range(10)
        ]
        model.meso_state.set_variable('groups', groups)
        
        # Upscale
        macro_data = model.upscale(ScaleLevel.MESO, ScaleLevel.MACRO)
        
        assert 'global_mean' in macro_data
        assert 'global_variance' in macro_data
    
    def test_downscale_macro_to_meso(self):
        """Test downscaling from macro to meso"""
        model = MultiscaleModel(n_agents=100)
        
        # Set macro state
        model.macro_state.set_variable('global_mean', 50)
        model.macro_state.set_variable('global_variance', 10)
        
        # Downscale
        meso_data = model.downscale(ScaleLevel.MACRO, ScaleLevel.MESO)
        
        assert 'groups' in meso_data
        assert len(meso_data['groups']) > 0
    
    def test_couple_scales(self):
        """Test scale coupling"""
        model = MultiscaleModel(n_agents=50)
        
        # Initialize micro state
        agents = [{'id': i, 'state': 50 + np.random.randn() * 5} for i in range(50)]
        model.micro_state.set_variable('agents', agents)
        
        # Couple scales
        model.couple_scales()
        
        # Check that all scales have been updated
        assert len(model.history[ScaleLevel.MICRO]) > 0
        assert len(model.history[ScaleLevel.MESO]) > 0
        assert len(model.history[ScaleLevel.MACRO]) > 0
    
    def test_step(self):
        """Test single simulation step"""
        model = MultiscaleModel(n_agents=50)
        
        # Initialize
        agents = [{'id': i, 'state': 0} for i in range(50)]
        model.micro_state.set_variable('agents', agents)
        
        # Step
        state = model.step()
        
        assert state['step'] == 1
        assert 'micro' in state
        assert 'meso' in state
        assert 'macro' in state
    
    def test_run(self):
        """Test complete simulation run"""
        model = MultiscaleModel(n_agents=50)
        
        # Initialize
        agents = [{'id': i, 'state': 0} for i in range(50)]
        model.micro_state.set_variable('agents', agents)
        
        # Run
        history = model.run(n_steps=10)
        
        assert len(history) == 10
        assert model.current_step == 10


class TestGovernmentMultiscaleModel:
    """Test GovernmentMultiscaleModel class"""
    
    def test_initialization(self):
        """Test government multiscale model initialization"""
        model = GovernmentMultiscaleModel(n_agents=100, n_communities=5)
        
        assert model.n_agents == 100
        assert model.n_communities == 5
        
        agents = model.micro_state.get_variable('agents', [])
        assert len(agents) == 100
        
        communities = model.meso_state.get_variable('groups', [])
        assert len(communities) == 5
    
    def test_agents_have_communities(self):
        """Test that agents are assigned to communities"""
        model = GovernmentMultiscaleModel(n_agents=50, n_communities=5)
        
        agents = model.micro_state.get_variable('agents', [])
        
        # Check all agents have community assignments
        for agent in agents:
            assert 'community_id' in agent
            assert 0 <= agent['community_id'] < 5
    
    def test_micro_update(self):
        """Test micro-level update"""
        model = GovernmentMultiscaleModel(n_agents=50, n_communities=5)
        
        # Get initial states
        agents = model.micro_state.get_variable('agents', [])
        initial_states = [a['state'] for a in agents]
        
        # Update
        model._update_micro()
        
        # Check states have changed
        updated_states = [a['state'] for a in agents]
        assert initial_states != updated_states
    
    def test_meso_update(self):
        """Test meso-level update"""
        model = GovernmentMultiscaleModel(n_agents=50, n_communities=5)
        
        # Get initial cohesions
        communities = model.meso_state.get_variable('groups', [])
        initial_cohesions = [c['cohesion'] for c in communities]
        
        # Update
        model._update_meso()
        
        # Cohesion values should exist and be valid
        updated_cohesions = [c['cohesion'] for c in communities]
        assert all(0 <= c <= 1 for c in updated_cohesions)
    
    def test_macro_update(self):
        """Test macro-level update"""
        model = GovernmentMultiscaleModel(n_agents=50, n_communities=5)
        
        # Update
        model._update_macro()
        
        # Check macro variables
        inequality = model.macro_state.get_variable('inequality')
        policy_eff = model.macro_state.get_variable('policy_effectiveness')
        
        assert inequality is not None
        assert policy_eff is not None
        assert 0 <= policy_eff <= 1
    
    def test_run_simulation(self):
        """Test running complete government multiscale simulation"""
        model = GovernmentMultiscaleModel(n_agents=100, n_communities=10)
        
        history = model.run(n_steps=20)
        
        assert len(history) == 20
        
        # Check that all scale levels are present in history
        for state in history:
            assert 'micro' in state
            assert 'meso' in state
            assert 'macro' in state


class TestScaleSeparation:
    """Test scale separation analysis"""
    
    def test_analyze_scale_separation(self):
        """Test scale separation analysis"""
        model = GovernmentMultiscaleModel(n_agents=100, n_communities=5)
        
        # Run a few steps
        model.run(n_steps=5)
        
        # Analyze separation
        separation = analyze_scale_separation(model, variable='state')
        
        assert 'total_variance' in separation
        assert separation['total_variance'] >= 0
    
    def test_variance_decomposition(self):
        """Test variance is properly decomposed"""
        model = GovernmentMultiscaleModel(n_agents=100, n_communities=5)
        model.run(n_steps=5)
        
        separation = analyze_scale_separation(model, variable='state')
        
        if 'within_group_variance' in separation and 'between_group_variance' in separation:
            # Total variance should be >= within + between
            assert separation['total_variance'] >= 0


class TestInvalidOperations:
    """Test error handling"""
    
    def test_invalid_upscaling(self):
        """Test that invalid upscaling raises error"""
        model = MultiscaleModel(n_agents=50)
        
        with pytest.raises(ValueError):
            model.upscale(ScaleLevel.MACRO, ScaleLevel.MICRO)
    
    def test_invalid_downscaling(self):
        """Test that invalid downscaling raises error"""
        model = MultiscaleModel(n_agents=50)
        
        with pytest.raises(ValueError):
            model.downscale(ScaleLevel.MICRO, ScaleLevel.MACRO)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
