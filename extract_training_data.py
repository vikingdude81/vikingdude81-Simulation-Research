"""
Extract Training Data from Baseline GA Runs
Parse the training history from baseline specialists to create training dataset
"""

import json
import numpy as np
from pathlib import Path


def extract_training_data_from_baseline(specialist_file: str, regime_type: str):
    """
    Extract training data from a baseline specialist training run
    
    For each generation, we extract:
    - State: population statistics + configuration
    - Action: mutation/crossover rates used (fixed in baseline)
    - Reward: fitness improvement
    """
    print(f"\n{'='*70}")
    print(f"📊 Extracting training data from: {specialist_file}")
    print(f"{'='*70}\n")
    
    # Load baseline results
    with open(specialist_file, 'r') as f:
        data = json.load(f)
    
    training_history = data['training_history']
    best_fitness = training_history['best_fitness']
    avg_fitness = training_history['avg_fitness']
    diversity = training_history['diversity']
    
    n_generations = len(best_fitness)
    
    # Baseline configuration (fixed)
    baseline_config = {
        'population_size': 200,
        'mutation_rate': 0.1,
        'crossover_rate': 0.7,
        'selection_pressure': 5 / 200,  # tournament_size / population
    }
    
    print(f"Regime: {regime_type.upper()}")
    print(f"Generations: {n_generations}")
    print(f"Baseline config: {baseline_config}")
    print()
    
    states = []
    actions = []
    rewards = []
    
    for gen in range(n_generations):
        # Calculate derived metrics
        if gen > 0:
            fitness_improvement = best_fitness[gen] - best_fitness[gen-1]
            convergence_speed = (best_fitness[gen] - best_fitness[0]) / (gen + 1)
        else:
            fitness_improvement = 0.0
            convergence_speed = 0.0
        
        # Time since improvement
        time_since_improvement = 0
        for i in range(gen-1, -1, -1):
            if best_fitness[i] < best_fitness[gen]:
                break
            time_since_improvement += 1
        
        # Stagnation indicator
        stagnation = 1.0 if time_since_improvement > 20 else 0.0
        
        # Calculate worst and std from avg and best (approximation)
        worst_fitness = avg_fitness[gen] - (best_fitness[gen] - avg_fitness[gen])
        fitness_std = abs(best_fitness[gen] - avg_fitness[gen]) / 2.0
        
        # Build 13-feature state for Enhanced ML Predictor
        state_13 = np.array([
            # Population stats (6)
            avg_fitness[gen],
            best_fitness[gen],
            worst_fitness,
            fitness_std,
            diversity[gen],
            baseline_config['population_size'] / 500.0,  # Normalized
            
            # Progress metrics (4)
            gen / n_generations,  # Normalized generation
            time_since_improvement / 50.0,  # Normalized
            convergence_speed,
            stagnation,
            
            # Configuration context (3)
            baseline_config['crossover_rate'],
            baseline_config['mutation_rate'],
            baseline_config['selection_pressure']
        ], dtype=np.float32)
        
        # Build 25-feature state for GA Conductor
        # For baseline, we don't have all 25 features, so we'll approximate
        state_25 = np.array([
            # Population stats (10)
            avg_fitness[gen],
            best_fitness[gen],
            worst_fitness,
            fitness_std,
            diversity[gen],
            gen / n_generations,
            time_since_improvement / 50.0,
            convergence_speed,
            stagnation,
            baseline_config['population_size'] / 500.0,
            
            # Wealth percentiles (6) - approximate from fitness
            worst_fitness * 0.8,  # bottom 10%
            worst_fitness * 0.9,  # bottom 25%
            avg_fitness[gen],     # median
            best_fitness[gen] * 0.9,  # top 25%
            best_fitness[gen] * 0.95, # top 10%
            0.5,  # gini coefficient (assume moderate)
            
            # Age metrics (3) - approximate
            gen / 2.0,  # avg age
            gen,        # oldest age
            0.3,        # young agents % (assume constant)
            
            # Strategy diversity (2) - approximate from genome diversity
            diversity[gen] * 20.0,  # unique strategies
            1.0 - diversity[gen],   # dominant strategy %
            
            # Configuration (4)
            baseline_config['population_size'] / 500.0,
            baseline_config['crossover_rate'],
            baseline_config['mutation_rate'],
            baseline_config['selection_pressure']
        ], dtype=np.float32)
        
        # Action (baseline used fixed rates)
        action = np.array([
            baseline_config['mutation_rate'],
            baseline_config['crossover_rate']
        ], dtype=np.float32)
        
        # Reward (fitness improvement + diversity bonus)
        reward = fitness_improvement * 10.0 + diversity[gen] * 2.0
        
        states.append({'state_13': state_13, 'state_25': state_25})
        actions.append(action)
        rewards.append(reward)
    
    print(f"✅ Extracted {len(states)} training samples")
    print(f"   Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
    print(f"   Avg reward: {np.mean(rewards):.2f}")
    print()
    
    return states, actions, rewards, regime_type


def combine_all_specialists():
    """Combine training data from all baseline specialist runs"""
    print("\n" + "="*70)
    print("🔄 COMBINING TRAINING DATA FROM ALL SPECIALISTS")
    print("="*70 + "\n")
    
    specialists = [
        ('outputs/specialist_volatile_20251105_191229.json', 'volatile'),
        ('outputs/specialist_trending_20251105_203753.json', 'trending'),
        ('outputs/specialist_ranging_20251105_235216.json', 'ranging')
    ]
    
    all_states_13 = []
    all_states_25 = []
    all_actions = []
    all_rewards = []
    all_metadata = []
    
    for spec_file, regime in specialists:
        if Path(spec_file).exists():
            states, actions, rewards, regime_type = extract_training_data_from_baseline(
                spec_file, regime
            )
            
            for i, (state, action, reward) in enumerate(zip(states, actions, rewards)):
                all_states_13.append(state['state_13'])
                all_states_25.append(state['state_25'])
                all_actions.append(action)
                all_rewards.append(reward)
                all_metadata.append({
                    'regime': regime_type,
                    'generation': i,
                    'specialist_file': spec_file
                })
    
    # Convert to arrays
    states_13 = np.array(all_states_13)
    states_25 = np.array(all_states_25)
    actions = np.array(all_actions)
    rewards = np.array(all_rewards)
    
    print(f"{'='*70}")
    print(f"📊 COMBINED DATASET SUMMARY")
    print(f"{'='*70}\n")
    print(f"Total samples: {len(states_13)}")
    print(f"Volatile samples: {sum(1 for m in all_metadata if m['regime'] == 'volatile')}")
    print(f"Trending samples: {sum(1 for m in all_metadata if m['regime'] == 'trending')}")
    print(f"Ranging samples: {sum(1 for m in all_metadata if m['regime'] == 'ranging')}")
    print(f"\nState_13 shape: {states_13.shape}")
    print(f"State_25 shape: {states_25.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"\nReward statistics:")
    print(f"  Min: {rewards.min():.2f}")
    print(f"  Max: {rewards.max():.2f}")
    print(f"  Mean: {rewards.mean():.2f}")
    print(f"  Std: {rewards.std():.2f}")
    
    # Save combined dataset
    output_file = 'outputs/baseline_training_data.npz'
    np.savez(
        output_file,
        states_13=states_13,
        states_25=states_25,
        actions=actions,
        rewards=rewards
    )
    print(f"\n✅ Saved combined dataset to: {output_file}")
    
    # Save metadata as JSON
    metadata_file = 'outputs/baseline_training_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"✅ Saved metadata to: {metadata_file}")
    
    print(f"\n{'='*70}\n")
    
    return states_13, states_25, actions, rewards


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 BASELINE TRAINING DATA EXTRACTION")
    print("="*70)
    
    states_13, states_25, actions, rewards = combine_all_specialists()
    
    print("✅ Training data ready for:")
    print("   1. Enhanced ML Predictor (13 inputs)")
    print("   2. GA Conductor (25 inputs)")
    print("\nNext: Train models with behavioral cloning")
    print("="*70 + "\n")
