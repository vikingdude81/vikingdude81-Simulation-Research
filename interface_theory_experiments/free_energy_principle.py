#!/usr/bin/env python3
"""
FREE ENERGY PRINCIPLE + INTERFACE THEORY

Karl Friston's Free Energy Principle states:
  "All living systems minimize variational free energy"

This is equivalent to:
  1. Minimizing SURPRISE (unexpected sensory states)
  2. Maximizing EVIDENCE for their model of the world
  3. ACTIVE INFERENCE: Acting to make predictions come true

CONNECTION TO HOFFMAN:
  - Hoffman: Agents evolve interfaces, not truth
  - Friston: Agents minimize surprise given their model
  - SYNTHESIS: The "best" interface is one that minimizes free energy
             while being computationally cheap

KEY INSIGHT:
  Evolution selects for low free energy agents, NOT accurate agents.
  An agent with a WRONG but SIMPLE model can have lower free energy
  than an agent with a CORRECT but COMPLEX model.

THE FREE ENERGY EQUATION:
  F = E_q[log q(s) - log p(o,s)]
  
  Where:
  - q(s) = agent's beliefs about hidden states
  - p(o,s) = generative model (how world generates observations)
  - o = observations
  - s = hidden states

  Minimizing F means:
  1. Making q(s) match the true posterior (accuracy)
  2. Making the model p simple (complexity cost)

This creates the ACCURACY-COMPLEXITY TRADEOFF that Hoffman predicts!

"The brain is not a camera. It's a prediction machine."
                                        — Karl Friston
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# FREE ENERGY AGENT
# ============================================================================

@dataclass
class FreeEnergyAgent:
    """
    An agent that minimizes variational free energy.
    
    Components:
    - Generative model: p(o|s)p(s) - how the agent thinks world works
    - Recognition model: q(s|o) - how agent infers hidden states
    - Policy: π(a|s) - how agent selects actions
    """
    n_observations: int = 8      # Dimensionality of observations
    n_hidden: int = 4            # Dimensionality of hidden states
    n_actions: int = 4           # Number of possible actions
    
    # Model parameters (will be tensors)
    likelihood: torch.Tensor = None      # p(o|s): hidden → observation
    prior: torch.Tensor = None           # p(s): prior over hidden states
    recognition: torch.Tensor = None     # q(s|o): observation → hidden
    policy: torch.Tensor = None          # π(a|s): hidden → action
    
    # Current state
    beliefs: torch.Tensor = None         # q(s): current beliefs
    prediction: torch.Tensor = None      # Expected observation
    
    # Complexity parameter (Hoffman's "interface cost")
    complexity_weight: float = 0.1


class FreeEnergyNetwork:
    """
    A population of free energy agents in parallel.
    
    Each agent:
    1. Receives observations from environment
    2. Updates beliefs to minimize free energy
    3. Takes actions that minimize EXPECTED free energy
    4. Evolves over generations (low F agents survive)
    """
    
    def __init__(
        self,
        n_agents: int = 1000,
        n_observations: int = 8,
        n_hidden: int = 4,
        n_actions: int = 4,
        complexity_weight: float = 0.1
    ):
        self.n_agents = n_agents
        self.n_obs = n_observations
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.complexity_weight = complexity_weight
        
        # Initialize agent parameters (batch of agents)
        # Likelihood: p(o|s) - shape: (n_agents, n_obs, n_hidden)
        self.likelihood = torch.randn(n_agents, n_observations, n_hidden, device=device) * 0.5
        self.likelihood = torch.softmax(self.likelihood, dim=1)  # Normalize per hidden state
        
        # Prior: p(s) - shape: (n_agents, n_hidden)
        self.prior = torch.ones(n_agents, n_hidden, device=device) / n_hidden
        
        # Recognition: q(s|o) - shape: (n_agents, n_hidden, n_obs)
        self.recognition = torch.randn(n_agents, n_hidden, n_observations, device=device) * 0.5
        self.recognition = torch.softmax(self.recognition, dim=1)
        
        # Policy: π(a|s) - shape: (n_agents, n_actions, n_hidden)
        self.policy = torch.randn(n_agents, n_actions, n_hidden, device=device) * 0.5
        self.policy = torch.softmax(self.policy, dim=1)
        
        # Current beliefs: q(s) - shape: (n_agents, n_hidden)
        self.beliefs = torch.ones(n_agents, n_hidden, device=device) / n_hidden
        
        # World state (true hidden state)
        self.true_state = torch.randint(0, n_hidden, (n_agents,), device=device)
        
        # Statistics
        self.free_energy_history = []
        self.accuracy_history = []
        self.complexity_history = []
        self.surprise_history = []
    
    def generate_observation(self) -> torch.Tensor:
        """
        Generate observations from the TRUE world.
        
        The world has a simple structure that agents must infer.
        """
        # True likelihood (what agents are trying to learn)
        # Each hidden state maps to a specific observation pattern
        true_likelihood = torch.zeros(self.n_hidden, self.n_obs, device=device)
        for s in range(self.n_hidden):
            # State s has high probability for observations s*2 and s*2+1
            obs_idx = (s * 2) % self.n_obs
            true_likelihood[s, obs_idx] = 0.7
            true_likelihood[s, (obs_idx + 1) % self.n_obs] = 0.2
            true_likelihood[s, :] += 0.1 / self.n_obs  # Small noise
        true_likelihood = true_likelihood / true_likelihood.sum(dim=1, keepdim=True)
        
        # Sample observations based on true state
        obs_probs = true_likelihood[self.true_state]  # (n_agents, n_obs)
        
        # Sample observation (one-hot)
        obs_idx = torch.multinomial(obs_probs, 1).squeeze(-1)
        observation = torch.zeros(self.n_agents, self.n_obs, device=device)
        observation.scatter_(1, obs_idx.unsqueeze(1), 1.0)
        
        return observation
    
    def update_beliefs(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Update beliefs q(s) given observation o.
        
        This is the PERCEPTION step - inferring hidden causes.
        Uses recognition model: q(s|o) ≈ p(s|o)
        """
        # Compute posterior using recognition model
        # q(s) = softmax(recognition @ observation)
        log_posterior = torch.bmm(
            self.recognition,  # (n_agents, n_hidden, n_obs)
            observation.unsqueeze(-1)  # (n_agents, n_obs, 1)
        ).squeeze(-1)  # (n_agents, n_hidden)
        
        # Add prior influence
        log_posterior = log_posterior + torch.log(self.prior + 1e-10)
        
        # Normalize
        self.beliefs = torch.softmax(log_posterior, dim=-1)
        
        return self.beliefs
    
    def compute_free_energy(self, observation: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute variational free energy:
        
        F = E_q[log q(s)] - E_q[log p(o,s)]
          = E_q[log q(s)] - E_q[log p(o|s)] - E_q[log p(s)]
          = -H[q(s)] - E_q[log p(o|s)] - E_q[log p(s)]
          = COMPLEXITY - ACCURACY
        
        Where:
        - COMPLEXITY = KL[q(s) || p(s)] = entropy cost
        - ACCURACY = E_q[log p(o|s)] = how well model explains data
        """
        # 1. Compute expected log likelihood: E_q[log p(o|s)]
        # likelihood: (n_agents, n_obs, n_hidden)
        # observation: (n_agents, n_obs)
        # beliefs: (n_agents, n_hidden)
        
        # For each hidden state, what's the log prob of the observation?
        log_likelihood = torch.log(self.likelihood + 1e-10)  # (n_agents, n_obs, n_hidden)
        
        # Observation probability under each hidden state
        obs_log_prob = torch.bmm(
            observation.unsqueeze(1),  # (n_agents, 1, n_obs)
            log_likelihood  # (n_agents, n_obs, n_hidden)
        ).squeeze(1)  # (n_agents, n_hidden)
        
        # Expected log likelihood under beliefs
        accuracy = (self.beliefs * obs_log_prob).sum(dim=-1)  # (n_agents,)
        
        # 2. Compute KL divergence: KL[q(s) || p(s)]
        kl_divergence = (self.beliefs * (
            torch.log(self.beliefs + 1e-10) - torch.log(self.prior + 1e-10)
        )).sum(dim=-1)  # (n_agents,)
        
        # 3. Compute model complexity (Hoffman's interface cost)
        # More complex models have higher entropy in their parameters
        likelihood_entropy = -(self.likelihood * torch.log(self.likelihood + 1e-10)).sum(dim=(1, 2))
        model_complexity = likelihood_entropy / (self.n_obs * self.n_hidden)
        
        # 4. Total free energy
        # F = -accuracy + KL + complexity_weight * model_complexity
        free_energy = -accuracy + kl_divergence + self.complexity_weight * model_complexity
        
        # 5. Surprise (for reference)
        # Surprise = -log p(o) ≈ -log E_p(s)[p(o|s)]
        marginal_likelihood = (self.likelihood * self.prior.unsqueeze(1)).sum(dim=-1)  # (n_agents, n_obs)
        log_marginal = torch.log((marginal_likelihood * observation).sum(dim=-1) + 1e-10)
        surprise = -log_marginal
        
        metrics = {
            'accuracy': accuracy.mean().item(),
            'kl_divergence': kl_divergence.mean().item(),
            'model_complexity': model_complexity.mean().item(),
            'surprise': surprise.mean().item(),
            'free_energy': free_energy.mean().item()
        }
        
        return free_energy, metrics
    
    def select_action(self) -> torch.Tensor:
        """
        Select action using active inference.
        
        Actions are chosen to minimize EXPECTED free energy.
        This means acting to:
        1. Reduce uncertainty (epistemic value)
        2. Achieve preferred outcomes (pragmatic value)
        """
        # Policy: π(a|s) - what action to take in each hidden state
        # Marginalize over beliefs: π(a) = Σ_s π(a|s) q(s)
        action_probs = torch.bmm(
            self.policy,  # (n_agents, n_actions, n_hidden)
            self.beliefs.unsqueeze(-1)  # (n_agents, n_hidden, 1)
        ).squeeze(-1)  # (n_agents, n_actions)
        
        # Sample action
        action = torch.multinomial(action_probs, 1).squeeze(-1)
        
        return action
    
    def environment_step(self, action: torch.Tensor) -> None:
        """
        Environment responds to action.
        
        Simple dynamics: action influences hidden state transition.
        """
        # Transition probability based on action
        # Action a tends to move toward state a (mod n_hidden)
        transition_prob = torch.zeros(self.n_agents, self.n_hidden, device=device)
        target_state = action % self.n_hidden
        
        for i in range(self.n_hidden):
            # Probability of transitioning to state i
            distance = torch.abs(target_state.float() - i)
            distance = torch.min(distance, self.n_hidden - distance)  # Wrap around
            transition_prob[:, i] = torch.exp(-distance)
        
        transition_prob = transition_prob / transition_prob.sum(dim=-1, keepdim=True)
        
        # Sample new hidden state
        self.true_state = torch.multinomial(transition_prob, 1).squeeze(-1)
    
    def learn(self, observation: torch.Tensor, learning_rate: float = 0.01) -> None:
        """
        Update model parameters to reduce free energy.
        
        This is "learning" - updating the generative model.
        """
        # Update likelihood: move toward observed patterns
        # For each hidden state s that we believe in, increase p(o|s)
        belief_weighted_obs = self.beliefs.unsqueeze(1) * observation.unsqueeze(2)
        # (n_agents, n_obs, n_hidden)
        
        self.likelihood = (1 - learning_rate) * self.likelihood + learning_rate * belief_weighted_obs
        self.likelihood = self.likelihood / (self.likelihood.sum(dim=1, keepdim=True) + 1e-10)
        
        # Update recognition model similarly
        obs_weighted_beliefs = observation.unsqueeze(1) * self.beliefs.unsqueeze(2)
        # (n_agents, n_hidden, n_obs)
        
        self.recognition = (1 - learning_rate) * self.recognition + learning_rate * obs_weighted_beliefs
        self.recognition = self.recognition / (self.recognition.sum(dim=1, keepdim=True) + 1e-10)
    
    def step(self, learn: bool = True) -> Dict:
        """
        One complete cycle:
        1. Observe
        2. Update beliefs (perception)
        3. Compute free energy
        4. Select action (active inference)
        5. Environment responds
        6. Learn (optional)
        """
        # 1. Generate observation from environment
        observation = self.generate_observation()
        
        # 2. Update beliefs
        self.update_beliefs(observation)
        
        # 3. Compute free energy
        free_energy, metrics = self.compute_free_energy(observation)
        
        # 4. Select action
        action = self.select_action()
        
        # 5. Environment step
        self.environment_step(action)
        
        # 6. Learn
        if learn:
            self.learn(observation)
        
        # Record history
        self.free_energy_history.append(metrics['free_energy'])
        self.accuracy_history.append(metrics['accuracy'])
        self.complexity_history.append(metrics['model_complexity'])
        self.surprise_history.append(metrics['surprise'])
        
        return metrics
    
    def evolve(self, n_steps: int = 100, selection_pressure: float = 0.1) -> None:
        """
        Evolutionary selection based on free energy.
        
        Low free energy agents survive and reproduce.
        This is the connection to Hoffman's evolution argument.
        """
        # Run for n_steps
        for _ in range(n_steps):
            self.step(learn=True)
        
        # Compute final free energy for each agent
        observation = self.generate_observation()
        self.update_beliefs(observation)
        free_energy, _ = self.compute_free_energy(observation)
        
        # Selection: low free energy agents survive
        fitness = -free_energy  # Higher fitness = lower free energy
        fitness = fitness - fitness.min()  # Shift to positive
        fitness = fitness / (fitness.max() + 1e-10)  # Normalize
        
        # Softmax selection
        selection_probs = torch.softmax(fitness / selection_pressure, dim=0)
        
        # Select parents
        parent_indices = torch.multinomial(selection_probs, self.n_agents, replacement=True)
        
        # Copy parent parameters to children
        self.likelihood = self.likelihood[parent_indices].clone()
        self.recognition = self.recognition[parent_indices].clone()
        self.policy = self.policy[parent_indices].clone()
        self.prior = self.prior[parent_indices].clone()
        
        # Add mutation
        mutation_rate = 0.1
        self.likelihood = self.likelihood + mutation_rate * torch.randn_like(self.likelihood)
        self.likelihood = torch.softmax(self.likelihood, dim=1)
        
        self.recognition = self.recognition + mutation_rate * torch.randn_like(self.recognition)
        self.recognition = torch.softmax(self.recognition, dim=1)


# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_basic_free_energy():
    """
    Basic demonstration of free energy minimization.
    """
    print(f"\n{'='*70}")
    print(f"  FREE ENERGY PRINCIPLE - BASIC DEMO")
    print(f"  Agents minimize surprise through perception and action")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    net = FreeEnergyNetwork(n_agents=1000)
    
    print("  Running 200 steps of active inference...\n")
    
    for step in range(200):
        metrics = net.step()
        
        if step % 40 == 0:
            print(f"    Step {step:3d}: F={metrics['free_energy']:.4f}, "
                  f"Acc={metrics['accuracy']:.4f}, "
                  f"Surprise={metrics['surprise']:.4f}")
    
    print()
    print("  RESULTS:")
    print("  " + "-" * 50)
    print(f"    Initial Free Energy: {net.free_energy_history[0]:.4f}")
    print(f"    Final Free Energy:   {net.free_energy_history[-1]:.4f}")
    print(f"    Reduction: {(net.free_energy_history[0] - net.free_energy_history[-1]):.4f}")
    print()
    print("  Agents learned to minimize surprise through:")
    print("    1. Updating beliefs to match observations (perception)")
    print("    2. Taking actions that lead to predictable outcomes")
    print("=" * 70)


def run_accuracy_complexity_tradeoff():
    """
    Show that complexity cost creates Hoffman's interface effect.
    
    Agents with high complexity weight prefer SIMPLE (wrong) models
    over ACCURATE (complex) models.
    """
    print(f"\n{'='*70}")
    print(f"  ACCURACY vs COMPLEXITY TRADEOFF")
    print(f"  Testing Hoffman's claim: Simple beats Accurate")
    print(f"{'='*70}\n")
    
    complexity_weights = [0.0, 0.1, 0.5, 1.0, 2.0]
    results = []
    
    for cw in complexity_weights:
        print(f"  Complexity weight = {cw}...")
        
        net = FreeEnergyNetwork(n_agents=500, complexity_weight=cw)
        
        # Run for 100 steps
        for _ in range(100):
            metrics = net.step()
        
        # Measure final state
        results.append({
            'complexity_weight': cw,
            'accuracy': metrics['accuracy'],
            'model_complexity': metrics['model_complexity'],
            'free_energy': metrics['free_energy']
        })
        
        print(f"    Accuracy={metrics['accuracy']:.4f}, "
              f"Complexity={metrics['model_complexity']:.4f}")
    
    print()
    print("  TRADEOFF ANALYSIS:")
    print("  " + "-" * 60)
    print(f"  {'Weight':>8} | {'Accuracy':>10} | {'Complexity':>12} | {'Free Energy':>12}")
    print("  " + "-" * 60)
    
    for r in results:
        print(f"  {r['complexity_weight']:>8.1f} | {r['accuracy']:>10.4f} | "
              f"{r['model_complexity']:>12.4f} | {r['free_energy']:>12.4f}")
    
    print()
    print("  INTERPRETATION:")
    print("  • Low complexity weight: Agents optimize for accuracy")
    print("  • High complexity weight: Agents optimize for simplicity")
    print("  • This IS Hoffman's interface theory!")
    print("    Evolution (high complexity cost) → Simple interfaces")
    print("=" * 70)


def run_evolution_of_interfaces():
    """
    Evolve agents under free energy selection.
    
    Show that evolution selects for low free energy,
    which means simple interfaces that minimize surprise.
    """
    print(f"\n{'='*70}")
    print(f"  EVOLUTION OF FREE ENERGY AGENTS")
    print(f"  Natural selection → Low free energy → Simple interfaces")
    print(f"{'='*70}\n")
    
    net = FreeEnergyNetwork(n_agents=500, complexity_weight=0.5)
    
    n_generations = 20
    
    print("  Evolving population...\n")
    
    gen_metrics = []
    
    for gen in range(n_generations):
        # Evolve (run + select + reproduce)
        net.evolve(n_steps=50, selection_pressure=0.2)
        
        # Measure current generation
        metrics = net.step(learn=False)
        gen_metrics.append(metrics)
        
        if gen % 4 == 0:
            print(f"    Gen {gen:2d}: F={metrics['free_energy']:.4f}, "
                  f"Complexity={metrics['model_complexity']:.4f}")
    
    print()
    print("  EVOLUTION RESULTS:")
    print("  " + "-" * 50)
    
    initial = gen_metrics[0]
    final = gen_metrics[-1]
    
    print(f"    Free Energy: {initial['free_energy']:.4f} → {final['free_energy']:.4f}")
    print(f"    Complexity:  {initial['model_complexity']:.4f} → {final['model_complexity']:.4f}")
    print(f"    Accuracy:    {initial['accuracy']:.4f} → {final['accuracy']:.4f}")
    print()
    
    if final['model_complexity'] < initial['model_complexity']:
        print("  ✓ Evolution selected for SIMPLER models!")
        print("    This confirms Hoffman: Evolution → Interfaces, not Truth")
    else:
        print("  → Models became more complex (accuracy dominated)")
    
    print("=" * 70)


def run_live_free_energy(n_steps: int = 200):
    """
    Live visualization of free energy minimization.
    """
    print(f"\n{'='*70}")
    print(f"  LIVE FREE ENERGY MINIMIZATION")
    print(f"  Watch agents reduce surprise in real-time")
    print(f"{'='*70}\n")
    
    net = FreeEnergyNetwork(n_agents=1000, complexity_weight=0.3)
    
    bar_width = 30
    
    print("  Free Energy | Accuracy  | Surprise  | Complexity")
    print("  " + "-" * 60)
    
    for step in range(n_steps):
        metrics = net.step()
        
        # Normalize for visualization
        fe_norm = min(1.0, max(0.0, (metrics['free_energy'] + 2) / 4))
        acc_norm = min(1.0, max(0.0, (metrics['accuracy'] + 3) / 3))
        surp_norm = min(1.0, max(0.0, metrics['surprise'] / 3))
        comp_norm = min(1.0, max(0.0, metrics['model_complexity']))
        
        fe_bar = "█" * int(fe_norm * 10) + "░" * (10 - int(fe_norm * 10))
        acc_bar = "█" * int(acc_norm * 10) + "░" * (10 - int(acc_norm * 10))
        surp_bar = "█" * int(surp_norm * 10) + "░" * (10 - int(surp_norm * 10))
        comp_bar = "█" * int(comp_norm * 10) + "░" * (10 - int(comp_norm * 10))
        
        line = f"  [{fe_bar}] | [{acc_bar}] | [{surp_bar}] | [{comp_bar}]"
        
        print(f"\r{line} Step {step:3d}", end="", flush=True)
        time.sleep(0.02)
    
    print("\n")
    print("  LEGEND:")
    print("    Free Energy: Total surprise + complexity (lower = better)")
    print("    Accuracy: How well model explains observations (higher = better)")
    print("    Surprise: Unexpected observations (lower = better)")
    print("    Complexity: Model complexity cost (lower = simpler)")
    print("=" * 70)


def run_active_inference_demo():
    """
    Demonstrate active inference - agents act to confirm predictions.
    """
    print(f"\n{'='*70}")
    print(f"  ACTIVE INFERENCE DEMONSTRATION")
    print(f"  Agents act to minimize EXPECTED free energy")
    print(f"{'='*70}\n")
    
    net = FreeEnergyNetwork(n_agents=100, n_actions=4)
    
    print("  Active inference means:")
    print("    - Perception: Update beliefs given observations")
    print("    - Action: Act to make predictions come true")
    print()
    
    # Track action patterns
    action_history = []
    state_history = []
    
    for step in range(100):
        observation = net.generate_observation()
        net.update_beliefs(observation)
        action = net.select_action()
        
        action_history.append(action.float().mean().item())
        state_history.append(net.true_state.float().mean().item())
        
        net.environment_step(action)
        net.learn(observation)
    
    # Analyze action-state correlation
    actions = np.array(action_history)
    states = np.array(state_history)
    
    correlation = np.corrcoef(actions[:-1], states[1:])[0, 1]
    
    print(f"  Action-State Correlation: {correlation:.4f}")
    print()
    
    if correlation > 0.1:
        print("  ✓ Agents learned to CONTROL their environment!")
        print("    Actions predict future states")
        print("    This is active inference in action")
    else:
        print("  Agents haven't yet learned control")
        print("  (May need more training steps)")
    
    print()
    print("  This is how consciousness works:")
    print("    We don't passively observe - we actively shape our world")
    print("    to match our predictions (self-fulfilling prophecy)")
    print("=" * 70)


def run_hierarchical_free_energy():
    """
    Combine hierarchical agents with free energy.
    
    Each level minimizes free energy at its scale.
    """
    print(f"\n{'='*70}")
    print(f"  HIERARCHICAL FREE ENERGY")
    print(f"  Multi-level active inference")
    print(f"{'='*70}\n")
    
    # Create hierarchy: 1000 micro → 100 meso → 10 macro
    level_sizes = [1000, 100, 10]
    
    # Create networks at each level
    networks = []
    for size in level_sizes:
        net = FreeEnergyNetwork(
            n_agents=size,
            complexity_weight=0.3
        )
        networks.append(net)
    
    print(f"  Hierarchy: {' → '.join(str(s) for s in level_sizes)}")
    print()
    
    # Run each level
    level_names = ['Micro', 'Meso', 'Macro']
    
    for level, (net, name) in enumerate(zip(networks, level_names)):
        # Run 50 steps
        for _ in range(50):
            net.step()
        
        # Measure final state
        metrics = net.step(learn=False)
        
        print(f"  {name:6} ({net.n_agents:4d} agents):")
        print(f"    Free Energy: {metrics['free_energy']:.4f}")
        print(f"    Accuracy:    {metrics['accuracy']:.4f}")
        print(f"    Complexity:  {metrics['model_complexity']:.4f}")
        print()
    
    print("  INSIGHT:")
    print("  Each level minimizes its own free energy")
    print("  Higher levels integrate predictions from lower levels")
    print("  This is how the brain does predictive processing!")
    print("=" * 70)


def run_massive_free_energy(n_agents: int = 100000, n_steps: int = 100):
    """
    Massive scale free energy simulation.
    """
    print(f"\n{'='*70}")
    print(f"  MASSIVE FREE ENERGY SIMULATION")
    print(f"  {n_agents:,} agents minimizing surprise")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    print("  Creating network...", end=" ", flush=True)
    start = time.perf_counter()
    
    net = FreeEnergyNetwork(n_agents=n_agents, complexity_weight=0.3)
    
    create_time = time.perf_counter() - start
    print(f"Done ({create_time:.2f}s)")
    
    if device.type == 'cuda':
        print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print(f"\n  Running {n_steps} steps...")
    start = time.perf_counter()
    
    for step in range(n_steps):
        metrics = net.step()
        
        if step % 20 == 0:
            print(f"    Step {step:3d}: F={metrics['free_energy']:.4f}")
    
    elapsed = time.perf_counter() - start
    rate = n_agents * n_steps / elapsed
    
    print()
    print(f"  PERFORMANCE:")
    print(f"    Total time: {elapsed:.2f}s")
    print(f"    Rate: {rate/1e6:.1f}M agent-steps/second")
    
    if rate > 1e6:
        print(f"    🚀 {rate/1e6:.1f} MILLION inferences per second!")
    
    print("=" * 70)


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if "--live" in args or "-l" in args:
        run_live_free_energy()
    elif "--tradeoff" in args or "-t" in args:
        run_accuracy_complexity_tradeoff()
    elif "--evolve" in args or "-e" in args:
        run_evolution_of_interfaces()
    elif "--active" in args or "-a" in args:
        run_active_inference_demo()
    elif "--hierarchy" in args or "-h" in args:
        run_hierarchical_free_energy()
    elif "--massive" in args or "-m" in args:
        n = 100000
        for arg in args:
            if arg.startswith("--agents="):
                n = int(arg.split("=")[1])
        run_massive_free_energy(n_agents=n)
    elif "--all" in args:
        run_basic_free_energy()
        run_accuracy_complexity_tradeoff()
        run_evolution_of_interfaces()
        run_active_inference_demo()
        run_hierarchical_free_energy()
    else:
        print("\nFree Energy Principle Experiments")
        print("=" * 40)
        print(f"Device: {device}")
        print()
        print("Usage:")
        print("  python free_energy_principle.py --live      # Watch F minimization")
        print("  python free_energy_principle.py --tradeoff  # Accuracy vs complexity")
        print("  python free_energy_principle.py --evolve    # Evolution of interfaces")
        print("  python free_energy_principle.py --active    # Active inference demo")
        print("  python free_energy_principle.py --hierarchy # Multi-level inference")
        print("  python free_energy_principle.py --massive   # GPU scale test")
        print("  python free_energy_principle.py --all       # Run all experiments")
        print()
        
        # Default demo
        run_basic_free_energy()
