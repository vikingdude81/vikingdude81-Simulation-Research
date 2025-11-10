"""
🤖 ML-BASED GOVERNANCE - Reinforcement Learning God AI
======================================================

This implements Todo #4: ML-based God (learning controller)

Uses the data from our 5 government experiments to train an RL agent
that learns optimal governance policies.

Architecture:
1. State Space: World metrics (population, cooperation, wealth, gini, diversity)
2. Action Space: Government interventions (welfare, stimulus, enforcement, etc.)
3. Reward Function: Weighted combination of objectives
4. Algorithm: PPO (Proximal Policy Optimization) or SAC (Soft Actor-Critic)

Research Questions:
- Can ML discover better policies than human-designed ones?
- Does RL learn the Mixed Economy strategy naturally?
- Can it beat 99.9% cooperation while maintaining diversity?
- What novel strategies emerge?

Training Data:
- Use deep_dive_analysis_data.json from 5 government experiments
- 5×300=1,500 state-action-reward samples
- Augment with additional exploration runs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict
import json
from collections import deque
import random

# RL imports (will need: pip install stable-baselines3)
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_checker import check_env
    import gymnasium as gym
    from gymnasium import spaces
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("⚠️  stable-baselines3 not installed. Install with:")
    print("   pip install stable-baselines3 gymnasium")

from ultimate_echo_simulation import UltimateEchoSimulation
from government_styles import GovernmentStyle


class GovernanceEnvironment(gym.Env):
    """
    Gymnasium environment for training RL governance agent.
    
    State space (10D):
    - Population (normalized)
    - Cooperation rate (0-1)
    - Average wealth (normalized)
    - Gini coefficient (0-1)
    - Diversity score (0-1)
    - Population growth rate
    - Cooperation change rate
    - Wealth change rate
    - Generation number (normalized)
    - Time since last intervention
    
    Action space (discrete):
    0: Do nothing (laissez-faire)
    1: Welfare redistribution (tax rich, help poor)
    2: Universal stimulus (boost all)
    3: Targeted stimulus (boost poor only)
    4: Remove worst defectors
    5: Tax defectors specifically
    6: Boost cooperators
    
    Reward:
    r = w1*cooperation + w2*population_health + w3*diversity - w4*inequality - w5*intervention_cost
    """
    
    def __init__(self, max_generations=300):
        super().__init__()
        
        self.max_generations = max_generations
        
        # State space: 10 continuous values
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )
        
        # Action space: 7 discrete actions
        self.action_space = spaces.Discrete(7)
        
        # Initialize simulation
        self.sim = None
        self.generation = 0
        self.last_state = None
        
        # Reward weights (tunable hyperparameters)
        self.reward_weights = {
            'cooperation': 2.0,      # High weight on cooperation
            'population': 1.0,       # Population health
            'diversity': 1.5,        # Genetic diversity
            'inequality': -1.0,      # Penalize high Gini
            'intervention_cost': -0.1  # Small cost to intervene
        }
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Create new simulation
        self.sim = UltimateEchoSimulation(
            initial_size=200,
            grid_size=(75, 75),
            government_style=GovernmentStyle.LAISSEZ_FAIRE  # Start with no intervention
        )
        
        self.generation = 0
        self.time_since_intervention = 0
        
        # Get initial state
        state = self._get_state()
        self.last_state = state
        
        return state, {}
    
    def _get_state(self) -> np.ndarray:
        """Extract state vector from simulation"""
        if not self.sim.agents:
            return np.zeros(10, dtype=np.float32)
        
        # Calculate metrics
        population = len(self.sim.agents)
        cooperators = sum(1 for a in self.sim.agents if a.traits.strategy == 1)
        cooperation_rate = cooperators / population if population > 0 else 0
        
        wealth_values = [a.wealth for a in self.sim.agents]
        avg_wealth = np.mean(wealth_values) if wealth_values else 0
        
        # Gini coefficient
        sorted_wealth = np.sort(wealth_values)
        n = len(sorted_wealth)
        if n > 0 and np.sum(sorted_wealth) > 0:
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_wealth)) / (n * np.sum(sorted_wealth)) - (n + 1) / n
        else:
            gini = 0
        
        # Diversity
        chromosomes = [tuple(a.chromosome.flatten()) for a in self.sim.agents]
        unique = len(set(chromosomes))
        diversity = unique / population if population > 0 else 0
        
        # Calculate rates of change
        if self.last_state is not None:
            pop_growth = (population - self.last_state[0] * 5000) / 5000  # Normalized
            coop_change = cooperation_rate - self.last_state[1]
            wealth_change = (avg_wealth - self.last_state[2] * 100) / 100
        else:
            pop_growth = 0
            coop_change = 0
            wealth_change = 0
        
        # Construct state vector
        state = np.array([
            population / 5000,  # Normalize to [0, 1]
            cooperation_rate,
            avg_wealth / 100,  # Assume max ~100 wealth
            gini,
            diversity,
            np.clip(pop_growth, -1, 1),  # Clip to [-1, 1], then shift to [0, 1]
            np.clip(coop_change, -1, 1),
            np.clip(wealth_change, -1, 1),
            self.generation / self.max_generations,
            min(self.time_since_intervention / 50, 1.0)  # Normalize time
        ], dtype=np.float32)
        
        # Shift negative values to [0, 1] range
        state[5:8] = (state[5:8] + 1) / 2
        
        return state
    
    def _apply_action(self, action: int):
        """Apply governance action to simulation"""
        intervention_cost = 0
        
        if action == 0:
            # Do nothing
            pass
        
        elif action == 1:
            # Welfare redistribution
            rich = [a for a in self.sim.agents if a.wealth > 50]
            poor = [a for a in self.sim.agents if a.wealth < 10]
            if rich and poor:
                tax_rate = 0.3
                total_tax = sum(a.wealth * tax_rate for a in rich)
                for a in rich:
                    a.wealth *= (1 - tax_rate)
                per_person = total_tax / len(poor)
                for a in poor:
                    a.wealth += per_person
                intervention_cost = 0.1
        
        elif action == 2:
            # Universal stimulus
            for a in self.sim.agents:
                a.wealth += 10
            intervention_cost = 0.2
        
        elif action == 3:
            # Targeted stimulus (poor only)
            poor = [a for a in self.sim.agents if a.wealth < 15]
            for a in poor:
                a.wealth += 15
            intervention_cost = 0.1
        
        elif action == 4:
            # Remove worst defectors
            defectors = [a for a in self.sim.agents if a.traits.strategy == 0]
            if defectors:
                # Remove bottom 10%
                defectors.sort(key=lambda a: a.wealth)
                to_remove = defectors[:max(1, len(defectors) // 10)]
                for a in to_remove:
                    a.wealth = -9999  # Mark for removal
                intervention_cost = 0.3
        
        elif action == 5:
            # Tax defectors
            defectors = [a for a in self.sim.agents if a.traits.strategy == 0]
            for a in defectors:
                a.wealth *= 0.7
            intervention_cost = 0.1
        
        elif action == 6:
            # Boost cooperators
            cooperators = [a for a in self.sim.agents if a.traits.strategy == 1]
            for a in cooperators:
                a.wealth += 5
            intervention_cost = 0.1
        
        return intervention_cost
    
    def step(self, action):
        """Execute one step in the environment"""
        # Apply action
        intervention_cost = self._apply_action(action)
        self.time_since_intervention = 0 if action > 0 else self.time_since_intervention + 1
        
        # Step simulation forward
        self.sim.step()
        self.generation += 1
        
        # Get new state
        state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward(state, action, intervention_cost)
        
        # Check if done
        done = (self.generation >= self.max_generations) or (len(self.sim.agents) < 10)
        truncated = False
        
        self.last_state = state
        
        return state, reward, done, truncated, {}
    
    def _calculate_reward(self, state, action, intervention_cost):
        """Calculate reward for current state"""
        population = state[0]
        cooperation = state[1]
        wealth = state[2]
        gini = state[3]
        diversity = state[4]
        
        # Multi-objective reward
        reward = (
            self.reward_weights['cooperation'] * cooperation +
            self.reward_weights['population'] * population +
            self.reward_weights['diversity'] * diversity +
            self.reward_weights['inequality'] * (1 - gini) +  # Reward low inequality
            self.reward_weights['intervention_cost'] * intervention_cost
        )
        
        # Bonus for very high cooperation with high diversity
        if cooperation > 0.8 and diversity > 0.5:
            reward += 1.0
        
        # Penalty for extinction risk
        if population < 0.1:  # Less than 500 agents
            reward -= 5.0
        
        return reward


def train_rl_agent(
    total_timesteps=100000,
    algorithm='PPO',
    save_path='ml_governance_model'
):
    """
    Train RL agent to learn optimal governance.
    
    Args:
        total_timesteps: Number of training steps
        algorithm: 'PPO' or 'SAC'
        save_path: Where to save trained model
    """
    if not RL_AVAILABLE:
        print("❌ Cannot train: stable-baselines3 not installed")
        return None
    
    print("\n" + "="*80)
    print("🤖 TRAINING ML-BASED GOVERNANCE AGENT")
    print("="*80)
    print(f"\nAlgorithm: {algorithm}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Expected time: {total_timesteps // 1000} minutes")
    print("\nThis will train an RL agent to discover optimal governance policies!")
    print("="*80 + "\n")
    
    # Create environment
    env = GovernanceEnvironment(max_generations=300)
    
    # Check environment
    print("🔍 Checking environment...")
    check_env(env)
    print("✅ Environment valid!\n")
    
    # Create RL agent
    if algorithm == 'PPO':
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            tensorboard_log="./ml_governance_logs/"
        )
    else:  # SAC
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            tensorboard_log="./ml_governance_logs/"
        )
    
    print(f"\n🚀 Starting training...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save model
    model.save(save_path)
    print(f"\n✅ Model saved to: {save_path}")
    
    return model


def evaluate_ml_agent(model_path='ml_governance_model', n_episodes=5):
    """Evaluate trained ML agent"""
    if not RL_AVAILABLE:
        print("❌ Cannot evaluate: stable-baselines3 not installed")
        return
    
    print("\n" + "="*80)
    print("📊 EVALUATING ML GOVERNANCE AGENT")
    print("="*80 + "\n")
    
    # Load model
    model = PPO.load(model_path)
    env = GovernanceEnvironment()
    
    results = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        actions_taken = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            actions_taken.append(action)
        
        final_coop = obs[1] * 100  # Cooperation rate
        final_div = obs[4]  # Diversity
        
        results.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'cooperation': final_coop,
            'diversity': final_div,
            'actions': actions_taken
        })
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Coop={final_coop:.1f}%, Div={final_div:.3f}")
    
    # Summary
    avg_coop = np.mean([r['cooperation'] for r in results])
    avg_div = np.mean([r['diversity'] for r in results])
    
    print("\n" + "-"*80)
    print(f"📊 AVERAGE RESULTS:")
    print(f"   Cooperation: {avg_coop:.1f}%")
    print(f"   Diversity: {avg_div:.3f}")
    print("="*80 + "\n")
    
    return results


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🤖 ML-BASED GOVERNANCE - Option C")
    print("="*80)
    print("\nThis will train an RL agent to learn optimal governance!\n")
    print("Options:")
    print("  1. Train new agent (100k steps, ~1-2 hours)")
    print("  2. Train quick agent (10k steps, ~10 minutes)")
    print("  3. Evaluate existing agent")
    print("  4. Compare ML vs Human-designed governments")
    print("="*80 + "\n")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        train_rl_agent(total_timesteps=100000)
    elif choice == '2':
        train_rl_agent(total_timesteps=10000)
    elif choice == '3':
        evaluate_ml_agent()
    elif choice == '4':
        print("\n🔜 Coming soon: Full comparison of ML vs all 5 governments!")
    else:
        print("\n❌ Invalid choice")


if __name__ == '__main__':
    if not RL_AVAILABLE:
        print("\n⚠️  Please install stable-baselines3 first:")
        print("   pip install stable-baselines3 gymnasium")
    else:
        main()
