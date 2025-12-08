#!/usr/bin/env python3
"""
GAME THEORY OF CONSCIOUSNESS
=============================

Connects Interface Theory to game-theoretic foundations:
1. Prisoner's Dilemma - Do conscious agents cooperate?
2. Hawk-Dove Game - Aggression vs interface perception
3. Stag Hunt - Collective consciousness emergence
4. Public Goods Game - Free energy minimization as cooperation

Key Insight: If perception is an interface (not truth), then
STRATEGIC perception becomes evolutionarily optimal.

Author: Interface Theory Experiments
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import sys

# Game theory payoff matrices (row player, column player)
# Classic Prisoner's Dilemma: T > R > P > S
PRISONERS_DILEMMA = torch.tensor([
    [[3, 3], [0, 5]],  # Row cooperates: (C,C)=3,3  (C,D)=0,5
    [[5, 0], [1, 1]]   # Row defects:    (D,C)=5,0  (D,D)=1,1
], dtype=torch.float32)

# Hawk-Dove (Chicken): V=2, C=4
HAWK_DOVE = torch.tensor([
    [[1, 1], [0, 2]],  # Dove: (D,D)=1,1  (D,H)=0,2
    [[2, 0], [-1, -1]] # Hawk: (H,D)=2,0  (H,H)=-1,-1
], dtype=torch.float32)

# Stag Hunt: Coordination game
STAG_HUNT = torch.tensor([
    [[4, 4], [0, 3]],  # Stag: (S,S)=4,4  (S,H)=0,3
    [[3, 0], [2, 2]]   # Hare: (H,S)=3,0  (H,H)=2,2
], dtype=torch.float32)


class ConsciousGameTheory:
    """
    Conscious agents playing games.
    
    Key hypothesis: Interface perception (not truth perception)
    allows for strategic flexibility that leads to cooperation
    and higher collective fitness.
    """
    
    def __init__(
        self,
        num_agents: int = 10000,
        dim: int = 16,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_agents = num_agents
        self.dim = dim
        
        # Agent internal states (consciousness)
        self.beliefs = torch.randn(num_agents, dim, device=self.device) * 0.1
        self.intentions = torch.zeros(num_agents, 2, device=self.device)  # [cooperate_prob, defect_prob]
        self.intentions[:, 0] = 0.5  # Start neutral
        self.intentions[:, 1] = 0.5
        
        # Interface vs truth perception
        self.perception_type = torch.rand(num_agents, device=self.device)  # 0=truth, 1=interface
        
        # Fitness/energy
        self.fitness = torch.ones(num_agents, device=self.device) * 10.0
        
        # Reputation (for repeated games)
        self.reputation = torch.zeros(num_agents, device=self.device)
        
        # Free energy (Friston)
        self.free_energy = torch.zeros(num_agents, device=self.device)
        
    def perceive_opponent(
        self,
        agent_idx: torch.Tensor,
        opponent_idx: torch.Tensor,
        opponent_intentions: torch.Tensor
    ) -> torch.Tensor:
        """
        How does agent perceive opponent's intentions?
        
        Truth perceivers: See exact intentions
        Interface perceivers: See fitness-relevant signal (reputation + noise)
        """
        # Get perception types
        is_interface = self.perception_type[agent_idx] > 0.5
        
        # Truth perception: exact opponent intentions
        truth_perception = opponent_intentions[:, 0]  # Cooperation probability
        
        # Interface perception: reputation-based + uncertainty
        interface_perception = torch.sigmoid(
            self.reputation[opponent_idx] + 
            torch.randn_like(truth_perception) * 0.2
        )
        
        # Mix based on perception type
        perception = torch.where(is_interface, interface_perception, truth_perception)
        
        return perception
    
    def decide_action(
        self,
        perceived_opponent: torch.Tensor,
        game_type: str = "prisoners_dilemma"
    ) -> torch.Tensor:
        """
        Decide to cooperate (0) or defect (1) based on perception.
        
        This is where consciousness matters:
        - Beliefs about the game
        - Prediction of opponent
        - Free energy minimization
        """
        batch_size = perceived_opponent.shape[0]
        
        # Compute expected payoffs
        if game_type == "prisoners_dilemma":
            payoff_matrix = PRISONERS_DILEMMA.to(self.device)
        elif game_type == "hawk_dove":
            payoff_matrix = HAWK_DOVE.to(self.device)
        elif game_type == "stag_hunt":
            payoff_matrix = STAG_HUNT.to(self.device)
        else:
            payoff_matrix = PRISONERS_DILEMMA.to(self.device)
        
        # Expected payoff for cooperating vs defecting
        # Given perception of opponent's cooperation probability
        coop_prob = perceived_opponent.unsqueeze(1)  # [batch, 1]
        
        # E[payoff | cooperate] = p(opp_coop) * R + (1-p) * S
        expected_coop = coop_prob * payoff_matrix[0, 0, 0] + (1 - coop_prob) * payoff_matrix[0, 1, 0]
        
        # E[payoff | defect] = p(opp_coop) * T + (1-p) * P
        expected_defect = coop_prob * payoff_matrix[1, 0, 0] + (1 - coop_prob) * payoff_matrix[1, 1, 0]
        
        # Add exploration noise (conscious uncertainty)
        noise = torch.randn(batch_size, 1, device=self.device) * 0.3
        
        # Softmax decision
        logits = torch.cat([expected_coop, expected_defect], dim=1) + noise
        probs = F.softmax(logits * 2.0, dim=1)  # Temperature
        
        # Sample action
        actions = torch.bernoulli(probs[:, 1])  # 0=cooperate, 1=defect
        
        return actions.long()
    
    def play_round(
        self,
        game_type: str = "prisoners_dilemma",
        random_matching: bool = True
    ) -> Dict[str, float]:
        """
        All agents play one round against random opponents.
        """
        # Random matching
        if random_matching:
            perm = torch.randperm(self.num_agents, device=self.device)
            agent1_idx = torch.arange(self.num_agents, device=self.device)
            agent2_idx = perm
        else:
            # Sequential pairing
            agent1_idx = torch.arange(0, self.num_agents, 2, device=self.device)
            agent2_idx = torch.arange(1, self.num_agents, 2, device=self.device)
        
        # Perception phase (consciousness!)
        perceived1 = self.perceive_opponent(agent1_idx, agent2_idx, self.intentions[agent2_idx])
        perceived2 = self.perceive_opponent(agent2_idx, agent1_idx, self.intentions[agent1_idx])
        
        # Decision phase
        actions1 = self.decide_action(perceived1, game_type)
        actions2 = self.decide_action(perceived2, game_type)
        
        # Get payoff matrix
        if game_type == "prisoners_dilemma":
            payoff_matrix = PRISONERS_DILEMMA.to(self.device)
        elif game_type == "hawk_dove":
            payoff_matrix = HAWK_DOVE.to(self.device)
        elif game_type == "stag_hunt":
            payoff_matrix = STAG_HUNT.to(self.device)
        else:
            payoff_matrix = PRISONERS_DILEMMA.to(self.device)
        
        # Calculate payoffs
        payoffs1 = payoff_matrix[actions1, actions2, 0]
        payoffs2 = payoff_matrix[actions1, actions2, 1]
        
        # Update fitness
        self.fitness[agent1_idx] += payoffs1
        self.fitness[agent2_idx] += payoffs2
        
        # Update reputation (cooperation increases it)
        self.reputation[agent1_idx] += (1 - actions1.float()) * 0.1 - actions1.float() * 0.1
        self.reputation[agent2_idx] += (1 - actions2.float()) * 0.1 - actions2.float() * 0.1
        
        # Update intentions based on outcomes (learning)
        alpha = 0.1
        # If cooperation led to good outcome, increase cooperation tendency
        coop_success1 = ((actions1 == 0) & (payoffs1 > 2)).float()
        defect_success1 = ((actions1 == 1) & (payoffs1 > 2)).float()
        
        self.intentions[agent1_idx, 0] += alpha * (coop_success1 - defect_success1)
        self.intentions[agent1_idx, 1] += alpha * (defect_success1 - coop_success1)
        self.intentions[agent1_idx] = F.softmax(self.intentions[agent1_idx], dim=1)
        
        # Compute free energy (surprise = prediction error)
        prediction_error = (perceived1 - actions2.float()).abs().mean()
        self.free_energy = prediction_error
        
        # Statistics
        coop_rate = 1 - (actions1.float().mean() + actions2.float().mean()) / 2
        interface_coop = (1 - actions1[self.perception_type[agent1_idx] > 0.5].float()).mean()
        truth_coop = (1 - actions1[self.perception_type[agent1_idx] <= 0.5].float()).mean()
        
        return {
            "cooperation_rate": coop_rate.item(),
            "interface_coop_rate": interface_coop.item() if not torch.isnan(interface_coop) else 0.5,
            "truth_coop_rate": truth_coop.item() if not torch.isnan(truth_coop) else 0.5,
            "mean_fitness": self.fitness.mean().item(),
            "fitness_std": self.fitness.std().item(),
            "free_energy": self.free_energy.item(),
        }
    
    def evolve(self, selection_strength: float = 0.1):
        """
        Evolution step: fitter agents reproduce, unfit die.
        """
        # Selection
        fitness_probs = F.softmax(self.fitness * selection_strength, dim=0)
        
        # Sample parents for bottom 20%
        bottom_20 = torch.argsort(self.fitness)[:self.num_agents // 5]
        parents = torch.multinomial(fitness_probs, len(bottom_20), replacement=True)
        
        # Inherit with mutation
        self.beliefs[bottom_20] = self.beliefs[parents] + torch.randn_like(self.beliefs[bottom_20]) * 0.01
        self.intentions[bottom_20] = self.intentions[parents]
        self.perception_type[bottom_20] = self.perception_type[parents] + torch.randn(len(bottom_20), device=self.device) * 0.05
        self.perception_type = self.perception_type.clamp(0, 1)
        
        # Reset fitness for new generation
        self.fitness = torch.ones_like(self.fitness) * 10.0
        self.reputation[bottom_20] = 0


def experiment_1_prisoners_dilemma():
    """
    Do interface perceivers cooperate more than truth perceivers?
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: PRISONER'S DILEMMA")
    print("  Interface vs Truth Perception")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    
    game = ConsciousGameTheory(num_agents=10000, device=device)
    
    # Force half to be interface, half truth
    game.perception_type[:5000] = 0.0  # Truth
    game.perception_type[5000:] = 1.0  # Interface
    
    print(f"  Agents: {game.num_agents:,} (50% truth, 50% interface)")
    print("\n  Running 100 rounds...\n")
    
    history = []
    for round_num in range(100):
        stats = game.play_round("prisoners_dilemma")
        history.append(stats)
        
        if round_num % 20 == 0:
            print(f"    Round {round_num:3d}: Coop={stats['cooperation_rate']:.3f} | "
                  f"Interface={stats['interface_coop_rate']:.3f} | "
                  f"Truth={stats['truth_coop_rate']:.3f}")
    
    # Final analysis
    final_stats = history[-1]
    print("\n  RESULTS:")
    print("  " + "-" * 50)
    print(f"    Overall Cooperation: {final_stats['cooperation_rate']:.1%}")
    print(f"    Interface Agents:    {final_stats['interface_coop_rate']:.1%}")
    print(f"    Truth Agents:        {final_stats['truth_coop_rate']:.1%}")
    print(f"    Mean Fitness:        {final_stats['mean_fitness']:.2f}")
    print(f"    Free Energy:         {final_stats['free_energy']:.4f}")
    
    # Who wins?
    interface_fitness = game.fitness[game.perception_type > 0.5].mean()
    truth_fitness = game.fitness[game.perception_type <= 0.5].mean()
    
    print(f"\n  FITNESS COMPARISON:")
    print(f"    Interface Agents: {interface_fitness:.2f}")
    print(f"    Truth Agents:     {truth_fitness:.2f}")
    
    if interface_fitness > truth_fitness:
        print(f"\n  ✓ INTERFACE PERCEIVERS WIN by {interface_fitness - truth_fitness:.2f}")
        print("    (Supports Hoffman's thesis)")
    else:
        print(f"\n  ✗ Truth perceivers win by {truth_fitness - interface_fitness:.2f}")
    
    print("=" * 70)
    return history


def experiment_2_evolution_of_cooperation():
    """
    Does evolution favor interface perception for cooperation?
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: EVOLUTION OF COOPERATION")
    print("  Does interface perception evolve?")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    game = ConsciousGameTheory(num_agents=10000, device=device)
    
    # Start with mixed population
    game.perception_type = torch.rand(10000, device=game.device)
    initial_interface_ratio = (game.perception_type > 0.5).float().mean().item()
    
    print(f"\n  Initial interface ratio: {initial_interface_ratio:.1%}")
    print("  Running 50 generations, 20 rounds each...\n")
    
    generations = []
    for gen in range(50):
        # Play 20 rounds
        for _ in range(20):
            game.play_round("prisoners_dilemma")
        
        interface_ratio = (game.perception_type > 0.5).float().mean().item()
        mean_fitness = game.fitness.mean().item()
        generations.append({
            "generation": gen,
            "interface_ratio": interface_ratio,
            "mean_fitness": mean_fitness
        })
        
        # Evolve
        game.evolve(selection_strength=0.2)
        
        if gen % 10 == 0:
            print(f"    Gen {gen:3d}: Interface={interface_ratio:.1%} | Fitness={mean_fitness:.2f}")
    
    final_ratio = (game.perception_type > 0.5).float().mean().item()
    
    print("\n  EVOLUTION RESULTS:")
    print("  " + "-" * 50)
    print(f"    Initial Interface Ratio: {initial_interface_ratio:.1%}")
    print(f"    Final Interface Ratio:   {final_ratio:.1%}")
    print(f"    Change: {(final_ratio - initial_interface_ratio)*100:+.1f}%")
    
    if final_ratio > initial_interface_ratio + 0.1:
        print("\n  ✓ INTERFACE PERCEPTION EVOLVED!")
        print("    Evolution favors fitness-interfaces over truth.")
    elif final_ratio < initial_interface_ratio - 0.1:
        print("\n  ✗ Truth perception evolved (unexpected!)")
    else:
        print("\n  ~ Mixed equilibrium reached")
    
    print("=" * 70)
    return generations


def experiment_3_stag_hunt_coordination():
    """
    Stag Hunt: Does shared consciousness help coordination?
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: STAG HUNT - COLLECTIVE CONSCIOUSNESS")
    print("  Can agents coordinate on the better equilibrium?")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    game = ConsciousGameTheory(num_agents=10000, device=device)
    
    # In Stag Hunt: Both cooperate (stag) = 4,4
    #               Both defect (hare) = 2,2
    # The challenge: (stag, hare) = 0,3 punishes trust
    
    print("\n  Stag Hunt Payoffs:")
    print("    (Stag, Stag) = (4, 4)  <- Pareto optimal")
    print("    (Hare, Hare) = (2, 2)  <- Risk-dominant")
    print("    (Stag, Hare) = (0, 3)  <- Sucker's payoff")
    
    print("\n  Question: Can conscious agents achieve (4,4)?")
    print("  Running 100 rounds...\n")
    
    history = []
    for round_num in range(100):
        stats = game.play_round("stag_hunt")
        history.append(stats)
        
        if round_num % 20 == 0:
            # In stag hunt, "cooperation" means hunting stag
            stag_rate = stats['cooperation_rate']
            print(f"    Round {round_num:3d}: Stag Hunt={stag_rate:.1%} | "
                  f"Mean Fitness={stats['mean_fitness']:.2f}")
    
    final_stag_rate = history[-1]['cooperation_rate']
    
    print("\n  COORDINATION RESULTS:")
    print("  " + "-" * 50)
    print(f"    Final Stag Hunt Rate: {final_stag_rate:.1%}")
    
    if final_stag_rate > 0.7:
        print("\n  ✓ COORDINATION ACHIEVED!")
        print("    Conscious agents reached the Pareto-optimal equilibrium.")
        print("    This requires predicting others' intentions - consciousness!")
    elif final_stag_rate < 0.3:
        print("\n  ✗ Risk-dominant equilibrium prevailed")
        print("    Agents couldn't trust each other enough.")
    else:
        print("\n  ~ Mixed strategies, some coordination")
    
    print("=" * 70)
    return history


def experiment_4_public_goods_free_energy():
    """
    Public Goods Game: Free energy minimization as cooperation mechanism.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: PUBLIC GOODS & FREE ENERGY")
    print("  Does minimizing surprise lead to cooperation?")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_agents = 10000
    
    # State: contribution level (0 to 1)
    contributions = torch.rand(num_agents, device=device) * 0.5
    beliefs_about_others = torch.ones(num_agents, device=device) * 0.5  # Expect 50% contribution
    
    multiplier = 2.0  # Public goods multiplier
    
    print(f"\n  {num_agents:,} agents in public goods game")
    print(f"  Multiplier: {multiplier}x")
    print("  Each agent predicts others' contributions (Bayesian brain)")
    print("\n  Free Energy = Prediction Error + Complexity")
    print("  Running 50 rounds...\n")
    
    for round_num in range(50):
        # Public pool
        total_contribution = contributions.sum()
        public_good = total_contribution * multiplier / num_agents
        
        # Individual payoff: kept + public good share
        kept = 1.0 - contributions
        payoffs = kept + public_good
        
        # Prediction error (surprise)
        mean_contribution = contributions.mean()
        prediction_errors = (beliefs_about_others - mean_contribution).abs()
        
        # Free energy
        free_energy = prediction_errors + 0.1 * (contributions - 0.5).abs()  # Complexity penalty
        
        # Update contributions to minimize free energy
        # Gradient: if others contribute more than expected, contribute more
        gradient = (mean_contribution - beliefs_about_others) * 0.1
        contributions = (contributions + gradient + torch.randn_like(contributions) * 0.02).clamp(0, 1)
        
        # Update beliefs (Bayesian update)
        beliefs_about_others = 0.9 * beliefs_about_others + 0.1 * mean_contribution
        
        if round_num % 10 == 0:
            print(f"    Round {round_num:3d}: Mean Contribution={mean_contribution:.1%} | "
                  f"Free Energy={free_energy.mean():.4f} | "
                  f"Payoff={payoffs.mean():.2f}")
    
    final_contribution = contributions.mean().item()
    
    print("\n  RESULTS:")
    print("  " + "-" * 50)
    print(f"    Final Mean Contribution: {final_contribution:.1%}")
    print(f"    Final Free Energy:       {free_energy.mean():.4f}")
    
    if final_contribution > 0.6:
        print("\n  ✓ HIGH COOPERATION through free energy minimization!")
        print("    Predictive processing → Social coordination")
    elif final_contribution > 0.4:
        print("\n  ~ Moderate cooperation")
    else:
        print("\n  ✗ Free riding prevailed")
    
    print("\n  KEY INSIGHT:")
    print("  Friston's free energy principle provides a mechanism")
    print("  for cooperation: agents that accurately predict others")
    print("  and minimize complexity naturally coordinate.")
    print("=" * 70)


def experiment_5_massive_scale():
    """
    Massive scale game theory with consciousness.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5: MASSIVE SCALE GAME THEORY")
    print("  1 Million conscious agents playing games")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("\n  ⚠ CUDA not available, using reduced scale")
        num_agents = 100000
    else:
        num_agents = 1000000
    
    print(f"\n  Device: {device}")
    print(f"  Agents: {num_agents:,}")
    
    # Initialize
    start = time.time()
    game = ConsciousGameTheory(num_agents=num_agents, device=device)
    init_time = time.time() - start
    print(f"  Initialization: {init_time:.2f}s")
    
    # Run multiple rounds
    print("\n  Running 20 rounds of Prisoner's Dilemma...\n")
    
    start = time.time()
    total_decisions = 0
    
    for round_num in range(20):
        stats = game.play_round("prisoners_dilemma")
        total_decisions += num_agents * 2  # Each agent makes 2 perceptions + decisions
        
        if round_num % 5 == 0:
            elapsed = time.time() - start
            rate = total_decisions / elapsed / 1e6
            print(f"    Round {round_num:3d}: Coop={stats['cooperation_rate']:.1%} | "
                  f"Rate={rate:.1f}M decisions/sec")
    
    total_time = time.time() - start
    final_rate = total_decisions / total_time / 1e6
    
    print("\n  PERFORMANCE:")
    print("  " + "-" * 50)
    print(f"    Total Decisions: {total_decisions:,}")
    print(f"    Total Time:      {total_time:.2f}s")
    print(f"    Rate:            {final_rate:.1f} MILLION decisions/second")
    print(f"    🚀 {final_rate:.1f}M conscious game decisions per second!")
    print("=" * 70)


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("  GAME THEORY OF CONSCIOUSNESS")
    print("  Connecting Interface Theory to Strategic Interaction")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "--pd" or arg == "--prisoners":
            experiment_1_prisoners_dilemma()
        elif arg == "--evolve":
            experiment_2_evolution_of_cooperation()
        elif arg == "--stag":
            experiment_3_stag_hunt_coordination()
        elif arg == "--public":
            experiment_4_public_goods_free_energy()
        elif arg == "--massive":
            experiment_5_massive_scale()
        elif arg == "--all":
            experiment_1_prisoners_dilemma()
            experiment_2_evolution_of_cooperation()
            experiment_3_stag_hunt_coordination()
            experiment_4_public_goods_free_energy()
            experiment_5_massive_scale()
        else:
            print("\n  Usage:")
            print("    python game_theory_consciousness.py --pd      # Prisoner's Dilemma")
            print("    python game_theory_consciousness.py --evolve  # Evolution of cooperation")
            print("    python game_theory_consciousness.py --stag    # Stag Hunt coordination")
            print("    python game_theory_consciousness.py --public  # Public goods + free energy")
            print("    python game_theory_consciousness.py --massive # 1M agents")
            print("    python game_theory_consciousness.py --all     # All experiments")
    else:
        # Default: Run Prisoner's Dilemma demo
        experiment_1_prisoners_dilemma()
        print("\n  Run with --all for all experiments")


if __name__ == "__main__":
    main()
