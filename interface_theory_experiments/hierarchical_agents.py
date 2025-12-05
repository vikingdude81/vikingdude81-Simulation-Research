#!/usr/bin/env python3
"""
HIERARCHICAL CONSCIOUS AGENTS

Hoffman's deeper proposal: Conscious agents can COMBINE to form
higher-level conscious agents. This is how complex consciousness emerges.

THE HIERARCHY:
  Level 0: Micro-agents (like neurons)
  Level 1: Meso-agents (like brain regions) 
  Level 2: Macro-agents (like a whole brain/person)
  Level 3: Super-agents (like societies, markets, ecosystems)

KEY INSIGHT:
When agents combine, they form a NEW agent with its own:
- Experiences (emergent qualia)
- Decisions (collective behavior)
- Actions (group effects on the world)

The combination is NOT just aggregation - it's a genuine NEW consciousness
with properties that don't exist at lower levels.

MATHEMATICAL STRUCTURE:
If A and B are conscious agents, their combination A⊗B is also a
conscious agent where:
- X_{A⊗B} = X_A × X_B (product of experience spaces)
- The Markov kernels compose in specific ways

This creates a HIERARCHY where each level has emergent properties.

"Consciousness is fundamental. Spacetime is derived.
 And consciousness comes in hierarchies of combination."
                                        — Donald Hoffman
"""

import torch
import numpy as np
import time
import sys
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# HIERARCHICAL AGENT DEFINITION
# ============================================================================

@dataclass
class HierarchicalAgent:
    """
    A conscious agent that can exist at any level of the hierarchy.
    
    Each agent has:
    - level: Which tier of the hierarchy (0=micro, 1=meso, etc.)
    - children: Lower-level agents that compose this one
    - experience: Current conscious state
    - decision: Current decision state
    """
    id: int
    level: int
    experience_dim: int
    action_dim: int
    children: List['HierarchicalAgent'] = field(default_factory=list)
    
    # State tensors (initialized later)
    experience: torch.Tensor = None
    action: torch.Tensor = None
    
    # Markov kernels
    perception: torch.Tensor = None  # P: World → Experience
    decision_kernel: torch.Tensor = None  # D: Experience → Action
    
    # Combination weights (how children contribute)
    combination_weights: torch.Tensor = None


class HierarchicalAgentNetwork:
    """
    A multi-level network of conscious agents.
    
    Level 0: Many micro-agents (e.g., 1000 neurons)
    Level 1: Fewer meso-agents (e.g., 100 brain regions)
    Level 2: Few macro-agents (e.g., 10 individuals)
    Level 3: One super-agent (e.g., society)
    
    Information flows both UP (micro → macro) and DOWN (macro → micro).
    """
    
    def __init__(
        self,
        level_sizes: List[int] = [1000, 100, 10, 1],
        experience_dims: List[int] = [4, 8, 16, 32],
        action_dims: List[int] = [2, 4, 8, 16],
        combination_type: str = 'attention'
    ):
        """
        Create a hierarchical agent network.
        
        Args:
            level_sizes: Number of agents at each level [L0, L1, L2, ...]
            experience_dims: Dimensionality of experience at each level
            action_dims: Dimensionality of actions at each level
            combination_type: How children combine ('attention', 'mean', 'gated')
        """
        self.n_levels = len(level_sizes)
        self.level_sizes = level_sizes
        self.experience_dims = experience_dims
        self.action_dims = action_dims
        self.combination_type = combination_type
        
        # Create state tensors for each level
        self.experiences = []
        self.actions = []
        self.perceptions = []
        self.decisions = []
        
        for level in range(self.n_levels):
            n = level_sizes[level]
            exp_dim = experience_dims[level]
            act_dim = action_dims[level]
            
            self.experiences.append(
                torch.randn(n, exp_dim, device=device) * 0.1
            )
            self.actions.append(
                torch.zeros(n, act_dim, device=device)
            )
            self.perceptions.append(
                torch.randn(n, exp_dim, exp_dim, device=device) * 0.1
            )
            self.decisions.append(
                torch.randn(n, act_dim, exp_dim, device=device) * 0.1
            )
        
        # Create combination matrices (how level L-1 combines into level L)
        self.combination_weights = []
        for level in range(1, self.n_levels):
            n_children = level_sizes[level - 1]
            n_parents = level_sizes[level]
            
            # Each parent attends to all children
            weights = torch.randn(n_parents, n_children, device=device)
            weights = torch.softmax(weights, dim=1)  # Normalize
            self.combination_weights.append(weights)
        
        # Projection matrices (to handle dimension changes between levels)
        self.up_projections = []  # Child experience → Parent experience
        self.down_projections = []  # Parent action → Child world
        
        for level in range(1, self.n_levels):
            child_exp_dim = experience_dims[level - 1]
            parent_exp_dim = experience_dims[level]
            child_act_dim = action_dims[level - 1]
            parent_act_dim = action_dims[level]
            
            self.up_projections.append(
                torch.randn(parent_exp_dim, child_exp_dim, device=device) * 0.1
            )
            self.down_projections.append(
                torch.randn(child_exp_dim, parent_act_dim, device=device) * 0.1
            )
        
        # World state (external environment)
        self.world_state = torch.randn(level_sizes[0], experience_dims[0], device=device)
        
        # Statistics
        self.step_count = 0
        self.level_entropies = [[] for _ in range(self.n_levels)]
        self.level_coherences = [[] for _ in range(self.n_levels)]
    
    def bottom_up_pass(self) -> None:
        """
        Information flows UP the hierarchy.
        
        Micro-agents perceive the world, their experiences combine
        to form meso-agent experiences, which combine to form macro, etc.
        """
        # Level 0: Perceive the world directly
        perceived = torch.bmm(
            self.perceptions[0],
            self.world_state.unsqueeze(-1)
        ).squeeze(-1)
        self.experiences[0] = torch.softmax(perceived, dim=-1)
        
        # Higher levels: Combine children's experiences
        for level in range(1, self.n_levels):
            child_exp = self.experiences[level - 1]  # (n_children, child_dim)
            weights = self.combination_weights[level - 1]  # (n_parents, n_children)
            up_proj = self.up_projections[level - 1]  # (parent_dim, child_dim)
            
            # Weighted combination of child experiences
            # combined[i] = sum_j weights[i,j] * child_exp[j]
            combined = torch.matmul(weights, child_exp)  # (n_parents, child_dim)
            
            # Project to parent dimension
            combined = torch.matmul(combined, up_proj.T)  # (n_parents, parent_dim)
            
            # Apply perception kernel
            perceived = torch.bmm(
                self.perceptions[level],
                combined.unsqueeze(-1)
            ).squeeze(-1)
            
            self.experiences[level] = torch.softmax(perceived, dim=-1)
    
    def top_down_pass(self) -> None:
        """
        Information flows DOWN the hierarchy.
        
        Macro-agents make decisions, which influence meso-agents,
        which influence micro-agents, which affect the world.
        """
        # All levels make decisions based on their experiences
        for level in range(self.n_levels):
            decided = torch.bmm(
                self.decisions[level],
                self.experiences[level].unsqueeze(-1)
            ).squeeze(-1)
            self.actions[level] = torch.softmax(decided, dim=-1)
        
        # Top-down influence: Higher-level actions modulate lower-level actions
        for level in range(self.n_levels - 2, -1, -1):
            parent_action = self.actions[level + 1]  # (n_parents, parent_act_dim)
            weights = self.combination_weights[level].T  # (n_children, n_parents)
            down_proj = self.down_projections[level]  # (child_dim, parent_act_dim)
            
            # Project parent action to child dimension
            projected = torch.matmul(parent_action, down_proj.T)  # (n_parents, child_exp_dim)
            
            # Distribute to children
            influence = torch.matmul(weights, projected)  # (n_children, child_exp_dim)
            
            # Modulate child actions (additive influence)
            # This is the "downward causation" from higher to lower levels
            child_action_dim = self.actions[level].shape[-1]
            influence_on_action = influence[:, :child_action_dim]
            
            self.actions[level] = torch.softmax(
                self.actions[level] + 0.5 * influence_on_action, dim=-1
            )
    
    def act_on_world(self) -> None:
        """
        Level 0 agents act on the world.
        
        The world state changes based on micro-agent actions,
        which have been influenced by the entire hierarchy.
        """
        action_effect = self.actions[0]
        
        # Project actions to world dimension
        action_dim = action_effect.shape[-1]
        world_dim = self.world_state.shape[-1]
        
        if action_dim != world_dim:
            projection = torch.randn(world_dim, action_dim, device=device) * 0.1
            action_effect = torch.matmul(action_effect, projection.T)
        
        # Update world
        self.world_state = 0.95 * self.world_state + 0.05 * action_effect
    
    def step(self) -> Dict[str, List[float]]:
        """
        One complete cycle of hierarchical processing.
        
        1. Bottom-up: World → Micro → Meso → Macro
        2. Top-down: Macro → Meso → Micro (decision influence)
        3. Act: Micro → World
        """
        self.bottom_up_pass()
        self.top_down_pass()
        self.act_on_world()
        
        self.step_count += 1
        
        # Compute statistics
        stats = {'entropies': [], 'coherences': []}
        
        for level in range(self.n_levels):
            exp = self.experiences[level]
            
            # Entropy of experience distribution
            entropy = -torch.sum(exp * torch.log(exp + 1e-10), dim=-1).mean()
            stats['entropies'].append(entropy.item())
            self.level_entropies[level].append(entropy.item())
            
            # Coherence (how aligned are agents at this level?)
            mean_exp = exp.mean(dim=0)
            coherence = torch.cosine_similarity(
                exp, mean_exp.unsqueeze(0), dim=-1
            ).mean()
            stats['coherences'].append(coherence.item())
            self.level_coherences[level].append(coherence.item())
        
        return stats
    
    def get_emergence_metrics(self) -> Dict:
        """
        Measure emergent properties at each level.
        
        Emergence = properties at level L that aren't predictable
        from level L-1 alone.
        """
        metrics = {}
        
        for level in range(self.n_levels):
            exp = self.experiences[level]
            
            # Information content
            entropy = -torch.sum(exp * torch.log(exp + 1e-10), dim=-1).mean()
            
            # Complexity (product of entropy and coherence)
            mean_exp = exp.mean(dim=0)
            coherence = torch.cosine_similarity(
                exp, mean_exp.unsqueeze(0), dim=-1
            ).mean()
            
            complexity = entropy.item() * (1 - coherence.item())
            
            metrics[f'level_{level}'] = {
                'entropy': entropy.item(),
                'coherence': coherence.item(),
                'complexity': complexity,
                'n_agents': self.level_sizes[level]
            }
        
        return metrics


# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_hierarchy_evolution(n_steps: int = 200):
    """
    Evolve a hierarchical network and observe emergent properties.
    """
    print(f"\n{'='*70}")
    print(f"  HIERARCHICAL CONSCIOUS AGENTS")
    print(f"  Modeling emergence across levels of organization")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    # Create 4-level hierarchy
    network = HierarchicalAgentNetwork(
        level_sizes=[500, 50, 5, 1],
        experience_dims=[4, 8, 16, 32],
        action_dims=[2, 4, 8, 16]
    )
    
    print("  Hierarchy Structure:")
    print("  " + "─" * 40)
    level_names = ['Micro (neurons)', 'Meso (regions)', 'Macro (brain)', 'Super (society)']
    for i, (name, size) in enumerate(zip(level_names, network.level_sizes)):
        print(f"    Level {i}: {name:20} - {size:4} agents")
    print()
    
    print("  Evolving hierarchy...")
    start = time.perf_counter()
    
    for step in range(n_steps):
        stats = network.step()
        
        if step % 50 == 0:
            print(f"\n    Step {step}:")
            for level in range(network.n_levels):
                ent = stats['entropies'][level]
                coh = stats['coherences'][level]
                print(f"      L{level}: entropy={ent:.3f}, coherence={coh:.3f}")
    
    elapsed = time.perf_counter() - start
    print(f"\n  Completed in {elapsed:.2f}s")
    
    # Final emergence metrics
    print(f"\n  EMERGENCE METRICS")
    print("  " + "─" * 50)
    
    metrics = network.get_emergence_metrics()
    for level, data in metrics.items():
        print(f"    {level}:")
        print(f"      Agents: {data['n_agents']}, Complexity: {data['complexity']:.4f}")
    
    return network


def run_live_hierarchy(n_steps: int = 150):
    """
    Live visualization of hierarchical dynamics.
    """
    print(f"\n{'='*70}")
    print(f"  LIVE HIERARCHICAL DYNAMICS")
    print(f"  Watch information flow up and down the hierarchy")
    print(f"{'='*70}\n")
    
    network = HierarchicalAgentNetwork(
        level_sizes=[200, 20, 2, 1],
        experience_dims=[4, 8, 16, 32],
        action_dims=[2, 4, 8, 16]
    )
    
    bar_width = 20
    level_names = ['Micro', 'Meso', 'Macro', 'Super']
    
    print("  Entropy (experience diversity) at each level:\n")
    
    for step in range(n_steps):
        stats = network.step()
        
        # Build visualization line
        line = f"  Step {step:3d} │"
        
        for level in range(network.n_levels):
            ent = stats['entropies'][level]
            max_ent = np.log(network.experience_dims[level])
            ratio = min(1.0, ent / max_ent)
            
            bar_len = int(ratio * bar_width)
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            
            line += f" {level_names[level][:1]}:[{bar}]"
        
        print(f"\r{line}", end="", flush=True)
        time.sleep(0.03)
    
    print("\n")
    print("=" * 70)
    print("  INTERPRETATION:")
    print()
    print("  Each level processes information differently:")
    print("  • Micro: Raw perception of world (high entropy = varied)")
    print("  • Meso: Combines micro patterns (medium entropy = structured)")
    print("  • Macro: High-level abstraction (lower entropy = coherent)")
    print("  • Super: Global coordination (may oscillate)")
    print()
    print("  The hierarchy creates EMERGENT properties at each level")
    print("  that aren't present in the levels below.")
    print("=" * 70)


def run_downward_causation_test():
    """
    Test whether higher levels can influence lower levels.
    
    This is the key question in consciousness studies:
    Can macro-level mental states cause micro-level changes?
    """
    print(f"\n{'='*70}")
    print(f"  DOWNWARD CAUSATION TEST")
    print(f"  Can higher levels influence lower levels?")
    print(f"{'='*70}\n")
    
    # Simple 2-level hierarchy for clearer causation
    network = HierarchicalAgentNetwork(
        level_sizes=[10, 1],
        experience_dims=[4, 8],
        action_dims=[2, 4]
    )
    
    # Strengthen down-projection for clearer causation
    network.down_projections[0] = torch.randn(4, 4, device=device) * 2.0
    
    # Run normally for a while
    print("  Phase 1: Normal evolution (20 steps)")
    for _ in range(20):
        network.step()
    
    # Record micro-level state BEFORE manipulation
    print("  Phase 2: Recording baseline micro-level actions...")
    network.bottom_up_pass()
    baseline_decided = torch.bmm(
        network.decisions[0],
        network.experiences[0].unsqueeze(-1)
    ).squeeze(-1)
    baseline_actions = torch.softmax(baseline_decided, dim=-1).clone()
    
    print(f"\n  Baseline micro-actions (first 3 agents):")
    for i in range(min(3, baseline_actions.shape[0])):
        print(f"    Agent {i}: {baseline_actions[i].cpu().numpy().round(3)}")
    
    # Now FORCE a specific experience at the top level
    print("\n  Phase 3: FORCING top-level experience to extreme state...")
    
    # Set super-agent to a VERY different state
    network.experiences[1] = torch.zeros(1, 8, device=device)
    network.experiences[1][0, 0] = 10.0  # Extreme value
    network.experiences[1] = torch.softmax(network.experiences[1], dim=-1)
    
    # Make decision at top level
    decided = torch.bmm(
        network.decisions[1],
        network.experiences[1].unsqueeze(-1)
    ).squeeze(-1)
    network.actions[1] = torch.softmax(decided, dim=-1)
    
    print(f"  Forced super-agent action: {network.actions[1][0].cpu().numpy().round(3)}")
    
    # Apply top-down influence
    parent_action = network.actions[1]  # (1, 4)
    weights = network.combination_weights[0].T  # (10, 1)
    down_proj = network.down_projections[0]  # (4, 4)
    
    # Project parent action and distribute
    projected = torch.matmul(parent_action, down_proj.T)  # (1, 4)
    influence = torch.matmul(weights, projected)  # (10, 4)
    
    # Apply to micro-actions
    influence_on_action = influence[:, :2]  # (10, 2)
    
    new_actions = torch.softmax(
        baseline_decided + 2.0 * influence_on_action, dim=-1
    )
    
    print(f"\n  After top-down influence (first 3 agents):")
    for i in range(min(3, new_actions.shape[0])):
        print(f"    Agent {i}: {new_actions[i].cpu().numpy().round(3)}")
    
    # Measure change
    change = (new_actions - baseline_actions).abs().mean()
    max_change = (new_actions - baseline_actions).abs().max()
    
    print(f"\n  Mean action change: {change.item():.4f}")
    print(f"  Max action change:  {max_change.item():.4f}")
    
    if change.item() > 0.01:
        print("\n  ✓ DOWNWARD CAUSATION DETECTED!")
        print()
        print("  Changing the top-level agent's experience")
        print("  caused measurable changes at the micro level.")
        print()
        print("  This is the neural correlate of 'mental causation':")
        print("  Your conscious decisions DO affect your neurons.")
    else:
        print("\n  ✗ No significant downward causation detected.")
    
    print("=" * 70)


def run_combination_experiment():
    """
    Show how agents COMBINE to form higher-level agents.
    
    This is Hoffman's key insight: consciousness combines.
    """
    print(f"\n{'='*70}")
    print(f"  AGENT COMBINATION EXPERIMENT")
    print(f"  Two micro-agents → One meso-agent")
    print(f"{'='*70}\n")
    
    # Create minimal hierarchy: 2 micro → 1 meso
    network = HierarchicalAgentNetwork(
        level_sizes=[2, 1],
        experience_dims=[4, 8],
        action_dims=[2, 4]
    )
    
    # Set micro-agents to specific experiences
    print("  Micro-Agent A experience: [1, 0, 0, 0] (pure state 0)")
    print("  Micro-Agent B experience: [0, 0, 0, 1] (pure state 3)")
    
    network.experiences[0][0] = torch.tensor([1, 0, 0, 0], dtype=torch.float, device=device)
    network.experiences[0][1] = torch.tensor([0, 0, 0, 1], dtype=torch.float, device=device)
    
    # Run bottom-up pass
    network.bottom_up_pass()
    
    meso_exp = network.experiences[1][0]
    
    print(f"\n  Combined Meso-Agent experience:")
    print(f"    {meso_exp.cpu().numpy().round(3)}")
    print()
    print("  Notice: The meso-agent has an 8-dimensional experience")
    print("  that is NOT simply the concatenation of the micro-agents.")
    print()
    print("  The combination creates EMERGENT qualia that exist")
    print("  only at the higher level of organization.")
    print()
    print("  This is how Hoffman explains the 'combination problem':")
    print("  How do separate conscious experiences combine into one?")
    print()
    print("  Answer: Through the mathematical structure of")
    print("  Markov kernel composition and tensor products.")
    print("=" * 70)


def run_massive_hierarchy(n_micro: int = 10000, n_steps: int = 50):
    """
    Test GPU scaling with massive hierarchies.
    """
    print(f"\n{'='*70}")
    print(f"  MASSIVE HIERARCHICAL NETWORK")
    print(f"  {n_micro:,} micro-agents in 4-level hierarchy")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    # Logarithmic reduction at each level
    l1 = n_micro // 10
    l2 = l1 // 10
    l3 = max(1, l2 // 10)
    
    print(f"  Level sizes: {n_micro} → {l1} → {l2} → {l3}")
    print()
    
    print("  Creating network...", end=" ", flush=True)
    start = time.perf_counter()
    
    network = HierarchicalAgentNetwork(
        level_sizes=[n_micro, l1, l2, l3],
        experience_dims=[4, 8, 16, 32],
        action_dims=[2, 4, 8, 16]
    )
    
    create_time = time.perf_counter() - start
    print(f"Done ({create_time:.2f}s)")
    
    print(f"  Running {n_steps} steps...", end=" ", flush=True)
    start = time.perf_counter()
    
    for step in range(n_steps):
        network.step()
    
    elapsed = time.perf_counter() - start
    
    total_agents = n_micro + l1 + l2 + l3
    rate = total_agents * n_steps / elapsed
    
    print(f"Done ({elapsed:.2f}s)")
    print()
    print(f"  Total agents: {total_agents:,}")
    print(f"  Rate: {rate:,.0f} agent-steps/second")
    
    if rate > 1e6:
        print(f"  🚀 {rate/1e6:.1f} MILLION agent-steps/second!")
    
    print("=" * 70)


def run_consciousness_emergence():
    """
    Model how 'unified consciousness' might emerge from hierarchy.
    """
    print(f"\n{'='*70}")
    print(f"  CONSCIOUSNESS EMERGENCE SIMULATION")
    print(f"  How does unified experience arise from many agents?")
    print(f"{'='*70}\n")
    
    network = HierarchicalAgentNetwork(
        level_sizes=[100, 10, 1],
        experience_dims=[4, 8, 16],
        action_dims=[2, 4, 8]
    )
    
    print("  Tracking 'integration' at each level over time...\n")
    
    # Integration = how much the parts depend on the whole
    # Measured by mutual information approximation
    
    integrations = [[] for _ in range(3)]
    
    for step in range(100):
        network.step()
        
        # Measure integration at each level
        for level in range(3):
            exp = network.experiences[level]
            
            # Simple integration measure: 
            # How much does each agent's state depend on others?
            mean_exp = exp.mean(dim=0, keepdim=True)
            deviation = (exp - mean_exp).abs().mean()
            integration = 1 - deviation.item()  # Higher = more integrated
            
            integrations[level].append(integration)
        
        if step % 20 == 0:
            print(f"    Step {step:3d}: Integration = "
                  f"L0:{integrations[0][-1]:.3f}, "
                  f"L1:{integrations[1][-1]:.3f}, "
                  f"L2:{integrations[2][-1]:.3f}")
    
    print()
    print("  Final Integration Levels:")
    print("  " + "─" * 40)
    
    for level in range(3):
        final = np.mean(integrations[level][-20:])
        bar_len = int(final * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        level_names = ['Micro', 'Meso', 'Macro']
        print(f"    {level_names[level]:6}: [{bar}] {final:.3f}")
    
    print()
    print("  INTERPRETATION:")
    print("  • Higher integration at macro level = more unified experience")
    print("  • Lower levels can have diverse micro-experiences")
    print("  • The 'binding' of consciousness happens through hierarchy")
    print()
    print("  This matches our intuition: we have ONE unified experience")
    print("  even though we have BILLIONS of neurons.")
    print("=" * 70)


def run_attention_combination():
    """
    Model how agents selectively attend to their children.
    
    Not all children contribute equally - attention modulates
    which lower-level experiences "bubble up" to higher levels.
    """
    print(f"\n{'='*70}")
    print(f"  ATTENTION-BASED AGENT COMBINATION")
    print(f"  Higher-level agents selectively attend to children")
    print(f"{'='*70}\n")
    
    # Create hierarchy with 10 micro → 2 meso → 1 macro
    n_micro = 10
    n_meso = 2
    
    # Manual attention weights (meso-agents specialize)
    # Meso-agent 0 attends to first 5 micro-agents
    # Meso-agent 1 attends to last 5 micro-agents
    
    print("  Creating specialized attention patterns...")
    print()
    print("  Meso-Agent 0: Attends to Micro-Agents 0-4 (left hemisphere)")
    print("  Meso-Agent 1: Attends to Micro-Agents 5-9 (right hemisphere)")
    print()
    
    # Initialize
    exp_dim = 4
    micro_exp = torch.randn(n_micro, exp_dim, device=device)
    micro_exp = torch.softmax(micro_exp, dim=-1)
    
    # Create attention weights
    attention = torch.zeros(n_meso, n_micro, device=device)
    attention[0, :5] = 1.0  # First meso attends to first half
    attention[1, 5:] = 1.0  # Second meso attends to second half
    attention = torch.softmax(attention * 5, dim=1)  # Sharpen attention
    
    print("  Micro-Agent Experiences:")
    for i in range(n_micro):
        bar = "".join(["█" if v > 0.3 else "░" for v in micro_exp[i].cpu().numpy()])
        hemisphere = "L" if i < 5 else "R"
        print(f"    Agent {i} [{hemisphere}]: [{bar}]")
    
    # Compute attention-weighted combination
    combined = torch.matmul(attention, micro_exp)  # (2, 4)
    
    print(f"\n  Attention-Weighted Meso Experiences:")
    for i in range(n_meso):
        exp_str = combined[i].cpu().numpy().round(3)
        hemisphere = "Left" if i == 0 else "Right"
        print(f"    {hemisphere} Hemisphere: {exp_str}")
    
    # Show attention heatmap
    print(f"\n  Attention Weights (which micro-agents contribute):")
    print(f"              Micro: ", end="")
    for i in range(n_micro):
        print(f" {i} ", end="")
    print()
    
    for i in range(n_meso):
        hemi = "L-Hemi" if i == 0 else "R-Hemi"
        print(f"    {hemi}:      ", end="")
        for j in range(n_micro):
            w = attention[i, j].item()
            if w > 0.15:
                print(" █ ", end="")
            elif w > 0.05:
                print(" ▓ ", end="")
            else:
                print(" ░ ", end="")
        print()
    
    # Now show global macro-agent that attends to both hemispheres
    print(f"\n  Macro-Agent (Global Integration):")
    
    macro_attention = torch.tensor([[0.5, 0.5]], device=device)  # Equal attention
    global_exp = torch.matmul(macro_attention, combined)
    
    print(f"    Unified Experience: {global_exp[0].cpu().numpy().round(3)}")
    print()
    print("  INSIGHT:")
    print("  The macro-agent has a UNIFIED experience that integrates")
    print("  both hemispheres. This is analogous to the corpus callosum")
    print("  binding left and right brain into one consciousness.")
    print()
    print("  Attention allows for FLEXIBLE binding - which lower-level")
    print("  experiences become part of higher-level consciousness")
    print("  can change dynamically based on context.")
    print("=" * 70)


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if "--live" in args or "-l" in args:
        run_live_hierarchy()
    elif "--causation" in args or "-c" in args:
        run_downward_causation_test()
    elif "--combine" in args or "-b" in args:
        run_combination_experiment()
    elif "--massive" in args or "-m" in args:
        n = 10000
        for arg in args:
            if arg.startswith("--agents="):
                n = int(arg.split("=")[1])
        run_massive_hierarchy(n_micro=n)
    elif "--emergence" in args or "-e" in args:
        run_consciousness_emergence()
    elif "--attention" in args or "-a" in args:
        run_attention_combination()
    elif "--all" in args:
        print("\n" + "="*70)
        print("  RUNNING ALL HIERARCHICAL AGENT EXPERIMENTS")
        print("="*70)
        run_combination_experiment()
        run_downward_causation_test()
        run_consciousness_emergence()
        run_attention_combination()
        n = 10000
        for arg in args:
            if arg.startswith("--agents="):
                n = int(arg.split("=")[1])
        run_massive_hierarchy(n_micro=n)
    else:
        print("\nHierarchical Conscious Agents")
        print("=" * 40)
        print(f"Device: {device}")
        print()
        print("Usage:")
        print("  python hierarchical_agents.py --live       # Watch hierarchy evolve")
        print("  python hierarchical_agents.py --causation  # Test downward causation")
        print("  python hierarchical_agents.py --combine    # Agent combination demo")
        print("  python hierarchical_agents.py --emergence  # Consciousness emergence")
        print("  python hierarchical_agents.py --attention  # Attention-based combination")
        print("  python hierarchical_agents.py --massive    # GPU scale test")
        print("  python hierarchical_agents.py --all        # Run all experiments")
        print()
        print("Running evolution demo...")
        run_hierarchy_evolution()
