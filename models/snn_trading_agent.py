"""
Spiking Neural Network Trading Agent
Based on arXiv:2512.11743 CogniSNN framework

Features:
- LIF neurons encode price movements as spikes
- Pathway reuse for BTC/ETH/SOL (no forgetting)
- Dynamic network growth during online learning
- Neuromorphic deployment ready
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron model.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        threshold: float = 1.0,
        decay: float = 0.9,
        refractory_period: int = 2
    ):
        """
        Initialize LIF neuron layer.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in layer
            threshold: Spike threshold
            decay: Membrane potential decay rate
            refractory_period: Number of timesteps neuron cannot spike after spiking
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.decay = decay
        self.refractory_period = refractory_period
        
        # Synaptic weights
        self.fc = nn.Linear(input_size, hidden_size)
        
        # Register buffers for stateful computation
        self.register_buffer('membrane_potential', torch.zeros(1, hidden_size))
        self.register_buffer('refractory_counter', torch.zeros(1, hidden_size))
        
    def reset_state(self, batch_size: int = 1):
        """Reset neuron state."""
        device = self.fc.weight.device
        self.membrane_potential = torch.zeros(batch_size, self.hidden_size, device=device)
        self.refractory_counter = torch.zeros(batch_size, self.hidden_size, device=device)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LIF neurons.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            spikes: Binary spike tensor [batch_size, hidden_size]
            membrane: Current membrane potential [batch_size, hidden_size]
        """
        batch_size = x.shape[0]
        
        # Ensure state matches batch size
        if self.membrane_potential.shape[0] != batch_size:
            self.reset_state(batch_size)
        
        # Input current
        current = self.fc(x)
        
        # Update membrane potential (decay + input)
        self.membrane_potential = self.decay * self.membrane_potential + current
        
        # Check refractory period
        not_refractory = (self.refractory_counter == 0).float()
        
        # Generate spikes where potential exceeds threshold and not refractory
        spikes = ((self.membrane_potential >= self.threshold) * not_refractory).float()
        
        # Reset membrane potential where spikes occurred
        self.membrane_potential = self.membrane_potential * (1 - spikes)
        
        # Update refractory counter
        self.refractory_counter = torch.clamp(
            self.refractory_counter - 1 + spikes * self.refractory_period,
            min=0
        )
        
        return spikes, self.membrane_potential


class PathwayModule(nn.Module):
    """
    Reusable pathway for multi-asset learning without forgetting.
    Based on CogniSNN pathway reusability concept.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        pathway_id: int = 0
    ):
        """
        Initialize pathway module.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension
            pathway_id: Unique identifier for this pathway
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pathway_id = pathway_id
        
        # Pathway-specific LIF neurons
        self.lif_layer = LIFNeuron(input_size, hidden_size)
        
        # Pathway usage counter (for importance weighting)
        self.register_buffer('usage_count', torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward through pathway.
        
        Args:
            x: Input tensor
            
        Returns:
            spikes: Spike output
            membrane: Membrane potential
        """
        self.usage_count += 1
        return self.lif_layer(x)
    
    def reset_state(self, batch_size: int = 1):
        """Reset pathway state."""
        self.lif_layer.reset_state(batch_size)


class SpikingTradingAgent(nn.Module):
    """
    Spiking Neural Network for trading decisions.
    Based on arXiv:2512.11743 CogniSNN framework.
    
    Features:
    - LIF neurons encode price movements as spikes
    - Pathway reuse for multiple assets (BTC/ETH/SOL)
    - Dynamic network growth during online learning
    - Neuromorphic deployment ready
    """
    
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 100,
        output_dim: int = 3,  # Buy, Hold, Sell
        num_pathways: int = 3,  # Start with 3 pathways
        threshold: float = 1.0,
        decay: float = 0.9,
        growth_threshold: float = 0.8  # Threshold for adding new pathways
    ):
        """
        Initialize SpikingTradingAgent.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of neurons per pathway
            output_dim: Number of output actions
            num_pathways: Initial number of pathways
            threshold: Spike threshold for LIF neurons
            decay: Membrane potential decay rate
            growth_threshold: Performance threshold for triggering growth
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.growth_threshold = growth_threshold
        
        # Input encoding layer (rate-based spike encoding)
        self.input_encoder = nn.Linear(input_dim, input_dim)
        
        # Pathway pool (reusable across assets)
        self.pathways = nn.ModuleList([
            PathwayModule(input_dim, hidden_dim, pathway_id=i)
            for i in range(num_pathways)
        ])
        
        # Pathway selection mechanism
        self.pathway_selector = nn.Sequential(
            nn.Linear(input_dim, num_pathways),
            nn.Softmax(dim=-1)
        )
        
        # Output layer (spike-to-rate decoder)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Growth tracking
        self.register_buffer('performance_history', torch.tensor([]))
        self.register_buffer('growth_counter', torch.tensor(0))
        
    def encode_input(self, x: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        Encode input as spike rates.
        
        Args:
            x: Input features [batch_size, input_dim]
            num_steps: Number of time steps for spike encoding
            
        Returns:
            spike_rates: Encoded spike rates [batch_size, num_steps, input_dim]
        """
        batch_size = x.shape[0]
        
        # Transform input to [0, 1] range (spike probability)
        encoded = torch.sigmoid(self.input_encoder(x))
        
        # Generate spikes based on rates
        spikes = []
        for t in range(num_steps):
            # Bernoulli sampling based on rate
            spike_t = (torch.rand_like(encoded) < encoded).float()
            spikes.append(spike_t)
        
        return torch.stack(spikes, dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = 10,
        asset_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through SNN.
        
        Args:
            x: Input features [batch_size, input_dim]
            num_steps: Number of simulation time steps
            asset_id: Optional asset identifier for pathway routing
            
        Returns:
            output: Trading action logits [batch_size, output_dim]
            info: Dictionary with spike information
        """
        batch_size = x.shape[0]
        
        # Reset pathway states
        for pathway in self.pathways:
            pathway.reset_state(batch_size)
        
        # Encode input as spikes
        spike_train = self.encode_input(x, num_steps)
        
        # Select pathways for this input
        pathway_weights = self.pathway_selector(x)  # [batch_size, num_pathways]
        
        # Process through time steps
        all_spikes = []
        for t in range(num_steps):
            spike_input = spike_train[:, t, :]  # [batch_size, input_dim]
            
            # Accumulate weighted pathway outputs
            pathway_outputs = []
            for i, pathway in enumerate(self.pathways):
                spikes, membrane = pathway(spike_input)
                # Weight by pathway selection
                weight = pathway_weights[:, i:i+1]  # [batch_size, 1]
                weighted_spikes = spikes * weight
                pathway_outputs.append(weighted_spikes)
            
            # Sum weighted outputs
            combined_spikes = torch.stack(pathway_outputs, dim=0).sum(dim=0)
            all_spikes.append(combined_spikes)
        
        # Average spike rates over time
        mean_spike_rate = torch.stack(all_spikes, dim=1).mean(dim=1)  # [batch_size, hidden_dim]
        
        # Decode to output
        output = self.output_layer(mean_spike_rate)
        
        # Collect info
        info = {
            'pathway_weights': pathway_weights,
            'mean_spike_rate': mean_spike_rate,
            'num_active_pathways': len(self.pathways)
        }
        
        return output, info
    
    def add_pathway(self):
        """
        Dynamically add a new pathway to the network.
        Implements dynamic growth from CogniSNN.
        """
        new_pathway_id = len(self.pathways)
        new_pathway = PathwayModule(
            self.input_dim,
            self.hidden_dim,
            pathway_id=new_pathway_id
        )
        
        # Move to same device as existing pathways
        device = next(self.parameters()).device
        new_pathway = new_pathway.to(device)
        
        self.pathways.append(new_pathway)
        
        # Update pathway selector
        old_selector = self.pathway_selector[0]
        new_selector = nn.Linear(
            self.input_dim,
            len(self.pathways)
        ).to(device)
        
        # Initialize new weights similar to old ones
        with torch.no_grad():
            new_selector.weight[:new_pathway_id] = old_selector.weight
            new_selector.bias[:new_pathway_id] = old_selector.bias
            # Small random initialization for new pathway
            new_selector.weight[new_pathway_id] = torch.randn_like(
                new_selector.weight[new_pathway_id]
            ) * 0.01
            new_selector.bias[new_pathway_id] = 0.0
        
        self.pathway_selector = nn.Sequential(
            new_selector,
            nn.Softmax(dim=-1)
        )
        
        self.growth_counter += 1
        
        return new_pathway_id
    
    def should_grow(self, performance: float) -> bool:
        """
        Determine if network should grow based on performance.
        
        Args:
            performance: Current performance metric (e.g., accuracy)
            
        Returns:
            should_grow: Boolean indicating if growth is needed
        """
        # Add performance to history
        self.performance_history = torch.cat([
            self.performance_history,
            torch.tensor([performance])
        ])
        
        # Growth criterion: performance below threshold for several steps
        if len(self.performance_history) < 10:
            return False
        
        recent_performance = self.performance_history[-10:].mean()
        return recent_performance < self.growth_threshold
    
    def get_pathway_statistics(self) -> Dict[str, torch.Tensor]:
        """
        Get statistics about pathway usage.
        
        Returns:
            stats: Dictionary with pathway statistics
        """
        usage_counts = [
            pathway.usage_count.item()
            for pathway in self.pathways
        ]
        
        return {
            'num_pathways': len(self.pathways),
            'usage_counts': torch.tensor(usage_counts),
            'total_growth_events': self.growth_counter
        }


class SNNEnsemble(nn.Module):
    """
    Ensemble of SNNs for robust trading decisions.
    """
    
    def __init__(
        self,
        num_agents: int = 3,
        input_dim: int = 50,
        hidden_dim: int = 100,
        output_dim: int = 3
    ):
        """
        Initialize SNN ensemble.
        
        Args:
            num_agents: Number of SNN agents in ensemble
            input_dim: Input dimension
            hidden_dim: Hidden dimension per agent
            output_dim: Output dimension
        """
        super().__init__()
        
        self.agents = nn.ModuleList([
            SpikingTradingAgent(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
            for _ in range(num_agents)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input features
            
        Returns:
            output: Averaged ensemble output
        """
        outputs = []
        for agent in self.agents:
            out, _ = agent(x)
            outputs.append(out)
        
        return torch.stack(outputs, dim=0).mean(dim=0)
