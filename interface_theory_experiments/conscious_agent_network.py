#!/usr/bin/env python3
"""
CONSCIOUS AGENT NETWORK

Donald Hoffman's deepest proposal: Reality isn't made of particles or fields.
It's a network of CONSCIOUS AGENTS communicating with each other.

THE THEORY:
- A Conscious Agent (CA) is a six-tuple: (X, G, P, D, A, W)
  - X = experiences (what the agent perceives)
  - G = actions (what the agent can do)  
  - P = perception map (world → experiences)
  - D = decision map (experiences → actions)
  - A = action map (actions → world effects)
  - W = world states

- Agents form NETWORKS where one agent's output is another's input
- Spacetime and particles EMERGE from these networks
- Distance = communication latency between agents
- Time = sequence of agent state updates

THE INSIGHT:
If two agents share a "direct channel" (are adjacent in the network),
they can communicate INSTANTLY - no spacetime delay.
Bell violations occur because entangled particles are the SAME agent
appearing at two "locations" in the spacetime interface.

"Conscious agents are the ontological primitives.
 Spacetime is just the data format for their communication."
                                        — Donald Hoffman

GPU-accelerated for massive agent networks.
"""

import torch
import numpy as np
import time
import sys
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# CONSCIOUS AGENT DEFINITION
# ============================================================================

class ExperienceType(Enum):
    """Types of experiences an agent can have."""
    QUALIA_RED = 0
    QUALIA_BLUE = 1
    QUALIA_PLEASURE = 2
    QUALIA_PAIN = 3
    QUALIA_NEUTRAL = 4


@dataclass
class ConsciousAgent:
    """
    A Conscious Agent in Hoffman's framework.
    
    The agent has:
    - Internal state (experiences/qualia)
    - Perception function (world → experience)
    - Decision function (experience → action)
    - Action function (action → world effect)
    """
    id: int
    experience_dim: int = 8      # Dimensionality of experience space
    action_dim: int = 4          # Dimensionality of action space
    
    # Internal state
    experience: torch.Tensor = None  # Current qualia
    action: torch.Tensor = None      # Current action
    
    # Markov kernels (as weight matrices)
    perception: torch.Tensor = None  # P: World → Experience
    decision: torch.Tensor = None    # D: Experience → Action
    
    def __post_init__(self):
        if self.experience is None:
            self.experience = torch.randn(self.experience_dim, device=device)
        if self.action is None:
            self.action = torch.zeros(self.action_dim, device=device)
        if self.perception is None:
            self.perception = torch.randn(self.experience_dim, self.experience_dim, device=device) * 0.1
        if self.decision is None:
            self.decision = torch.randn(self.action_dim, self.experience_dim, device=device) * 0.1


class ConsciousAgentNetwork:
    """
    A network of Conscious Agents.
    
    Agents are connected by channels. When agent A acts, it affects
    the world state that agent B perceives. The network topology
    determines what we experience as "spacetime distance."
    
    Key insight: Agents that are DIRECTLY connected can communicate
    instantly. "Distance" in spacetime is really network hops.
    """
    
    def __init__(self, n_agents: int, connection_prob: float = 0.1):
        """
        Create a network of conscious agents.
        
        Args:
            n_agents: Number of agents in the network
            connection_prob: Probability of direct connection between any two agents
        """
        self.n_agents = n_agents
        self.experience_dim = 8
        self.action_dim = 4
        
        # Create agents as batched tensors for GPU efficiency
        self.experiences = torch.randn(n_agents, self.experience_dim, device=device)
        self.actions = torch.zeros(n_agents, self.action_dim, device=device)
        
        # Perception matrices: (n_agents, experience_dim, experience_dim)
        self.perceptions = torch.randn(n_agents, self.experience_dim, self.experience_dim, device=device) * 0.1
        
        # Decision matrices: (n_agents, action_dim, experience_dim)
        self.decisions = torch.randn(n_agents, self.action_dim, self.experience_dim, device=device) * 0.1
        
        # Connection matrix: who can communicate with whom
        # This defines the "hidden" reality beneath spacetime
        self.connections = (torch.rand(n_agents, n_agents, device=device) < connection_prob).float()
        # Make symmetric and remove self-connections
        self.connections = (self.connections + self.connections.T) / 2
        self.connections.fill_diagonal_(0)
        
        # World state: the "ontological substrate" that agents interact through
        self.world_state = torch.randn(n_agents, self.experience_dim, device=device)
        
        # Track network statistics
        self.step_count = 0
        self.communication_log = []
    
    def perceive(self) -> torch.Tensor:
        """
        All agents perceive the world simultaneously.
        
        P: World → Experience
        
        Each agent's perception is influenced by:
        1. Its own perception matrix
        2. The world states of connected agents
        """
        # Aggregate world states from connected agents
        # connected_states[i] = sum of world_state[j] for all connected j
        connected_states = torch.matmul(self.connections, self.world_state)
        
        # Apply perception map
        # experiences[i] = perception[i] @ connected_states[i]
        new_experiences = torch.bmm(
            self.perceptions,
            connected_states.unsqueeze(-1)
        ).squeeze(-1)
        
        # Add some noise (perceptual uncertainty)
        new_experiences += torch.randn_like(new_experiences) * 0.01
        
        # Apply softmax to normalize (experiences as probability distribution)
        self.experiences = torch.softmax(new_experiences, dim=-1)
        
        return self.experiences
    
    def decide(self) -> torch.Tensor:
        """
        All agents make decisions based on their experiences.
        
        D: Experience → Action
        """
        # Apply decision map
        new_actions = torch.bmm(
            self.decisions,
            self.experiences.unsqueeze(-1)
        ).squeeze(-1)
        
        # Actions are probabilistic choices
        self.actions = torch.softmax(new_actions, dim=-1)
        
        return self.actions
    
    def act(self) -> torch.Tensor:
        """
        All agents act on the world simultaneously.
        
        A: Action → World Effect
        
        Each agent's action modifies the world state for connected agents.
        """
        # Actions propagate through connections
        # world_effect[i] = sum of actions[j] for connected j, projected to experience dim
        action_effects = torch.matmul(self.connections, self.actions)
        
        # Project action effects to experience dimension
        # Use a simple linear projection
        projection = torch.randn(self.action_dim, self.experience_dim, device=device) * 0.1
        projected_effects = torch.matmul(action_effects, projection)
        
        # Update world state
        self.world_state = 0.9 * self.world_state + 0.1 * projected_effects
        
        return self.world_state
    
    def step(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One complete cycle: Perceive → Decide → Act
        
        This is the fundamental "tick" of conscious experience.
        """
        exp = self.perceive()
        act = self.decide()
        world = self.act()
        
        self.step_count += 1
        
        return exp, act, world
    
    def measure_correlation(self, agent_i: int, agent_j: int) -> float:
        """
        Measure the correlation between two agents' experiences.
        
        If agents are directly connected, correlation can be HIGH
        even if they are "far apart" in spacetime (Bell violation!).
        """
        exp_i = self.experiences[agent_i]
        exp_j = self.experiences[agent_j]
        
        # Correlation
        corr = torch.corrcoef(torch.stack([exp_i, exp_j]))[0, 1]
        return corr.item()
    
    def network_distance(self, agent_i: int, agent_j: int) -> int:
        """
        Compute the network distance (minimum hops) between two agents.
        
        This is what we EXPERIENCE as "spatial distance" in the
        spacetime interface. It's really just network topology.
        """
        if agent_i == agent_j:
            return 0
        
        # BFS to find shortest path
        visited = {agent_i}
        queue = [(agent_i, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            
            # Get neighbors
            neighbors = torch.nonzero(self.connections[current] > 0).squeeze(-1)
            
            for n in neighbors.tolist():
                if n == agent_j:
                    return dist + 1
                if n not in visited:
                    visited.add(n)
                    queue.append((n, dist + 1))
        
        return -1  # Not connected
    
    def get_network_stats(self) -> Dict:
        """Get statistics about the network structure."""
        n_connections = (self.connections > 0).sum().item() / 2
        avg_degree = n_connections * 2 / self.n_agents
        
        return {
            'n_agents': self.n_agents,
            'n_connections': int(n_connections),
            'avg_degree': avg_degree,
            'density': n_connections / (self.n_agents * (self.n_agents - 1) / 2)
        }


# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_network_evolution(n_agents: int = 1000, n_steps: int = 100):
    """
    Evolve a conscious agent network and observe emergent patterns.
    """
    print(f"\n{'='*70}")
    print(f"  CONSCIOUS AGENT NETWORK EVOLUTION")
    print(f"  Agents: {n_agents} | Steps: {n_steps}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    # Create network
    network = ConsciousAgentNetwork(n_agents, connection_prob=0.05)
    stats = network.get_network_stats()
    
    print(f"  Network Statistics:")
    print(f"    Agents: {stats['n_agents']}")
    print(f"    Connections: {stats['n_connections']}")
    print(f"    Avg Degree: {stats['avg_degree']:.2f}")
    print(f"    Density: {stats['density']:.4f}")
    print()
    
    # Track experience entropy over time
    entropies = []
    
    print("  Evolving network...")
    start = time.perf_counter()
    
    for step in range(n_steps):
        exp, act, world = network.step()
        
        # Compute entropy of experience distribution
        entropy = -torch.sum(exp * torch.log(exp + 1e-10), dim=-1).mean()
        entropies.append(entropy.item())
        
        if step % 20 == 0:
            print(f"    Step {step:4d} | Entropy: {entropy.item():.4f}")
    
    elapsed = time.perf_counter() - start
    
    print(f"\n  Completed {n_steps} steps in {elapsed:.2f}s")
    print(f"  Rate: {n_agents * n_steps / elapsed:,.0f} agent-steps/sec")
    
    # Check for emergent correlations
    print(f"\n  Emergent Correlations:")
    print(f"  " + "─" * 50)
    
    # Sample some pairs and measure correlation vs distance
    pairs = []
    for _ in range(20):
        i = np.random.randint(n_agents)
        j = np.random.randint(n_agents)
        if i != j:
            corr = network.measure_correlation(i, j)
            dist = network.network_distance(i, j)
            pairs.append((i, j, corr, dist))
    
    # Sort by distance
    pairs.sort(key=lambda x: x[3])
    
    print(f"    {'Agent i':>8} {'Agent j':>8} {'Correlation':>12} {'Distance':>10}")
    for i, j, corr, dist in pairs[:10]:
        dist_str = str(dist) if dist >= 0 else "∞"
        print(f"    {i:>8} {j:>8} {corr:>+12.4f} {dist_str:>10}")
    
    return network, entropies


def run_bell_test_on_network(n_agents: int = 100, n_trials: int = 1000):
    """
    Test for Bell-like violations in the conscious agent network.
    
    If two agents are DIRECTLY connected (distance = 1), they can
    exhibit correlations that violate classical locality bounds.
    """
    print(f"\n{'='*70}")
    print(f"  BELL TEST ON CONSCIOUS AGENT NETWORK")
    print(f"  Testing whether direct connections enable 'spooky' correlations")
    print(f"{'='*70}\n")
    
    # Create dense network to ensure connections
    network = ConsciousAgentNetwork(n_agents, connection_prob=0.3)
    
    # Warm up the network
    for _ in range(50):
        network.step()
    
    # Find directly connected pairs (distance = 1)
    direct_pairs = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if network.connections[i, j] > 0:
                direct_pairs.append((i, j))
    
    # Find indirectly connected pairs (distance > 1)
    indirect_pairs = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if network.connections[i, j] == 0:
                dist = network.network_distance(i, j)
                if dist > 1:
                    indirect_pairs.append((i, j, dist))
    
    print(f"  Found {len(direct_pairs)} directly connected pairs")
    print(f"  Found {len(indirect_pairs)} indirectly connected pairs\n")
    
    # Measure correlations
    direct_corrs = []
    indirect_corrs = []
    
    for _ in range(n_trials):
        network.step()
        
        # Sample correlations
        if direct_pairs:
            i, j = direct_pairs[np.random.randint(len(direct_pairs))]
            direct_corrs.append(abs(network.measure_correlation(i, j)))
        
        if indirect_pairs:
            i, j, _ = indirect_pairs[np.random.randint(len(indirect_pairs))]
            indirect_corrs.append(abs(network.measure_correlation(i, j)))
    
    direct_avg = np.mean(direct_corrs) if direct_corrs else 0
    indirect_avg = np.mean(indirect_corrs) if indirect_corrs else 0
    
    print(f"  Results over {n_trials} trials:")
    print(f"  " + "─" * 50)
    print(f"    Direct connections (distance=1):   |corr| = {direct_avg:.4f}")
    print(f"    Indirect connections (distance>1): |corr| = {indirect_avg:.4f}")
    print()
    
    violation = direct_avg > indirect_avg * 1.2
    
    if violation:
        print("  ⚡ BELL-LIKE VIOLATION DETECTED!")
        print()
        print("  Directly connected agents show HIGHER correlations")
        print("  than indirectly connected ones. In spacetime terms,")
        print("  this looks like 'action at a distance.'")
        print()
        print("  But in the network view, it's just: adjacent agents")
        print("  share information directly, no 'travel' needed.")
    else:
        print("  No significant violation detected in this run.")
        print("  (Network dynamics may need more steps to equilibrate)")
    
    return direct_avg, indirect_avg


def run_live_network(n_agents: int = 500, n_steps: int = 200):
    """
    Live visualization of network evolution.
    """
    print(f"\n{'='*70}")
    print(f"  LIVE CONSCIOUS AGENT NETWORK")
    print(f"  Agents: {n_agents} | Device: {device}")
    print(f"{'='*70}\n")
    
    network = ConsciousAgentNetwork(n_agents, connection_prob=0.05)
    bar_width = 50
    
    print("  Watching experience entropy and action coherence...\n")
    
    for step in range(n_steps):
        exp, act, world = network.step()
        
        # Entropy of experiences (high = diverse, low = coherent)
        entropy = -torch.sum(exp * torch.log(exp + 1e-10), dim=-1).mean().item()
        max_entropy = np.log(network.experience_dim)
        entropy_ratio = entropy / max_entropy
        
        # Action coherence (how aligned are agent actions?)
        mean_action = act.mean(dim=0)
        coherence = torch.cosine_similarity(act, mean_action.unsqueeze(0), dim=-1).mean().item()
        
        # Visualization
        ent_bar = int(entropy_ratio * bar_width)
        coh_bar = int((coherence + 1) / 2 * bar_width)  # Normalize to [0,1]
        
        print(f"\r  Step {step:3d} | "
              f"Entropy: [{'█' * ent_bar}{'░' * (bar_width - ent_bar)}] {entropy:.2f} | "
              f"Coherence: {coherence:+.2f}", end="", flush=True)
        
        time.sleep(0.02)
    
    print("\n\n" + "=" * 70)
    print("  INTERPRETATION:")
    print()
    print("  • Entropy: How diverse are agent experiences?")
    print("    Low = agents converging to similar qualia")
    print("    High = agents in diverse experiential states")
    print()
    print("  • Coherence: How aligned are agent actions?")
    print("    High = collective behavior emerging")
    print("    Low = independent/chaotic actions")
    print()
    print("  Spacetime 'emerges' when agents develop stable")
    print("  patterns of perception-decision-action cycles.")
    print("=" * 70)


def run_spacetime_emergence(n_agents: int = 200, n_steps: int = 500):
    """
    Demonstrate how 'spacetime' emerges from agent network dynamics.
    """
    print(f"\n{'='*70}")
    print(f"  SPACETIME EMERGENCE FROM CONSCIOUS AGENTS")
    print(f"  Showing how distance and time emerge from network topology")
    print(f"{'='*70}\n")
    
    # Create network with clusters (simulating 'spatial' regions)
    network = ConsciousAgentNetwork(n_agents, connection_prob=0.02)
    
    # Add extra connections within 'clusters' to simulate spatial locality
    cluster_size = 20
    n_clusters = n_agents // cluster_size
    
    for c in range(n_clusters):
        start = c * cluster_size
        end = start + cluster_size
        # Dense connections within cluster
        for i in range(start, end):
            for j in range(i + 1, end):
                if torch.rand(1).item() < 0.5:
                    network.connections[i, j] = 1
                    network.connections[j, i] = 1
    
    print(f"  Created {n_clusters} spatial 'clusters' of {cluster_size} agents each")
    print()
    
    # Evolve and measure intra-cluster vs inter-cluster correlations
    print("  Evolving network to reach equilibrium...")
    for _ in range(200):
        network.step()
    
    # Measure correlations
    intra_corrs = []
    inter_corrs = []
    
    print("  Measuring correlations...")
    
    for trial in range(500):
        network.step()
        
        # Intra-cluster pair
        c = np.random.randint(n_clusters)
        i = c * cluster_size + np.random.randint(cluster_size)
        j = c * cluster_size + np.random.randint(cluster_size)
        if i != j:
            intra_corrs.append(abs(network.measure_correlation(i, j)))
        
        # Inter-cluster pair
        c1, c2 = np.random.choice(n_clusters, 2, replace=False)
        i = c1 * cluster_size + np.random.randint(cluster_size)
        j = c2 * cluster_size + np.random.randint(cluster_size)
        inter_corrs.append(abs(network.measure_correlation(i, j)))
    
    intra_avg = np.mean(intra_corrs)
    inter_avg = np.mean(inter_corrs)
    
    print(f"\n  Results:")
    print(f"  " + "─" * 50)
    print(f"    Intra-cluster correlation: {intra_avg:.4f}")
    print(f"    Inter-cluster correlation: {inter_avg:.4f}")
    print(f"    Ratio: {intra_avg / (inter_avg + 1e-10):.2f}x")
    print()
    
    if intra_avg > inter_avg:
        print("  ✓ SPATIAL LOCALITY EMERGED!")
        print()
        print("  Agents in the same 'cluster' (nearby in network topology)")
        print("  show stronger correlations than distant agents.")
        print()
        print("  This is what we EXPERIENCE as 'spatial locality':")
        print("  nearby things are more correlated than distant things.")
        print()
        print("  But there's no actual 'space' - just network connections!")
    
    print()
    print("  THE INSIGHT:")
    print("  'Space' is the user interface rendering of network topology.")
    print("  'Time' is the sequence of perception-decision-action cycles.")
    print("  Neither is fundamental - both emerge from conscious agents.")
    print("=" * 70)


def run_massive_network(n_agents: int = 100000, n_steps: int = 50):
    """
    Massive network simulation to test GPU scaling.
    """
    print(f"\n{'='*70}")
    print(f"  MASSIVE CONSCIOUS AGENT NETWORK")
    print(f"  Agents: {n_agents:,} | Steps: {n_steps}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    print("  Creating network...", end=" ", flush=True)
    start = time.perf_counter()
    network = ConsciousAgentNetwork(n_agents, connection_prob=0.001)
    create_time = time.perf_counter() - start
    print(f"Done ({create_time:.2f}s)")
    
    stats = network.get_network_stats()
    print(f"\n  Network Statistics:")
    print(f"    Agents: {stats['n_agents']:,}")
    print(f"    Connections: {stats['n_connections']:,}")
    print(f"    Density: {stats['density']:.6f}")
    print()
    
    print("  Running simulation...")
    start = time.perf_counter()
    
    for step in range(n_steps):
        network.step()
        if step % 10 == 0:
            print(f"    Step {step}/{n_steps}", end="\r", flush=True)
    
    elapsed = time.perf_counter() - start
    rate = n_agents * n_steps / elapsed
    
    print(f"\n\n  Completed {n_steps} steps in {elapsed:.2f}s")
    print(f"  Rate: {rate:,.0f} agent-steps/second")
    print()
    
    if rate > 1e6:
        print(f"  🚀 Over {rate/1e6:.1f} MILLION agent-steps per second!")
    
    print("=" * 70)


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if "--live" in args or "-l" in args:
        n = 500
        for arg in args:
            if arg.startswith("--agents="):
                n = int(arg.split("=")[1])
        run_live_network(n_agents=n)
        
    elif "--bell" in args or "-b" in args:
        run_bell_test_on_network()
        
    elif "--emergence" in args or "-e" in args:
        run_spacetime_emergence()
        
    elif "--massive" in args or "-m" in args:
        n = 100000
        for arg in args:
            if arg.startswith("--agents="):
                n = int(arg.split("=")[1])
        run_massive_network(n_agents=n)
        
    else:
        print("\nConscious Agent Network: Reality as Agent Dynamics")
        print("=" * 55)
        print(f"Device: {device}")
        print()
        print("Usage:")
        print("  python conscious_agent_network.py --live       # Watch network evolve")
        print("  python conscious_agent_network.py --bell       # Bell-like violation test")
        print("  python conscious_agent_network.py --emergence  # Spacetime emergence")
        print("  python conscious_agent_network.py --massive    # 100k agents GPU test")
        print()
        print("Running evolution demo...")
        run_network_evolution(n_agents=500, n_steps=100)
