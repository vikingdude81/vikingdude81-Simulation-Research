"""
Train Enhanced ML Predictor on Baseline Data

This script:
1. Loads baseline training history from all 3 specialists
2. Extracts state-action pairs (state → optimal mutation rate)
3. Trains EnhancedMLPredictor model
4. Tests on volatile specialist (compare vs baseline)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import pandas as pd
from pathlib import Path
from enhanced_trainer import EnhancedMLPredictor, EnhancedSpecialistTrainer
import matplotlib.pyplot as plt


def load_baseline_data():
    """Load baseline training results to extract training data"""
    baseline_file = 'outputs/all_specialists_baseline_20251105_235217.json'
    
    with open(baseline_file, 'r') as f:
        data = json.load(f)
        
    return data['specialists']


def extract_training_examples(specialist_data, regime_type, config):
    """
    Extract state-action pairs from baseline training history
    
    For each generation, we know:
    - State: [fitness, diversity, generation, trends, CONFIG]
    - Action: mutation_rate (fixed at 0.1 in baseline)
    - Outcome: fitness improvement in next few generations
    
    We'll create synthetic labels based on:
    - If fitness improved soon after → mutation_rate was good
    - If stagnated → maybe need higher mutation
    - If fitness dropped → maybe too much mutation
    """
    
    history = specialist_data['training_history']
    best_fitness_history = history['best_fitness']
    diversity_history = history['diversity']
    
    population_size = config['population_size']
    crossover_rate = config['crossover_rate']
    mutation_rate = config['mutation_rate']
    
    X = []  # States (13 features)
    y = []  # Target mutation rates
    
    # Process each generation
    for gen in range(len(best_fitness_history)):
        # Extract current state
        current_fitness = best_fitness_history[gen]
        current_diversity = diversity_history[gen] if gen < len(diversity_history) else 0.1
        
        # Calculate features
        if gen == 0:
            avg_fitness_norm = 1.0
            best_fitness_norm = 1.0
            worst_fitness_norm = 0.8
            fitness_trend = 0.0
            diversity_trend = 0.0
            stagnation = 0.0
            convergence_speed = 1.0
            improvement_rate = 0.0
        else:
            # Normalize by max fitness seen so far
            max_fitness = max(best_fitness_history[:gen+1])
            avg_fitness_norm = current_fitness / max(max_fitness, 1.0)
            best_fitness_norm = 1.0
            worst_fitness_norm = 0.8  # Assume some variance
            
            # Fitness trend (last 10 gens)
            lookback = min(10, gen)
            if lookback > 0:
                past_fitness = best_fitness_history[gen-lookback]
                fitness_trend = (current_fitness - past_fitness) / (abs(past_fitness) + 1e-8)
            else:
                fitness_trend = 0.0
                
            # Diversity trend
            if gen > 0 and gen < len(diversity_history):
                past_diversity = diversity_history[max(0, gen-10)]
                diversity_trend = (current_diversity - past_diversity) / (past_diversity + 1e-8)
            else:
                diversity_trend = 0.0
                
            # Stagnation (generations since last big improvement)
            stagnation = 0.0
            for i in range(gen-1, max(0, gen-30), -1):
                if best_fitness_history[i] < current_fitness * 0.95:
                    break
                stagnation += 1
            stagnation = min(stagnation / 30.0, 1.0)
            
            # Convergence speed
            if gen >= 5:
                recent_fitness = best_fitness_history[gen-5:gen+1]
                convergence_speed = np.std(recent_fitness) / (abs(np.mean(recent_fitness)) + 1e-8)
            else:
                convergence_speed = 1.0
                
            # Calculate improvement rate (for labeling)
            future_window = min(10, len(best_fitness_history) - gen - 1)
            if future_window > 0:
                future_fitness = best_fitness_history[gen+future_window]
                improvement_rate = (future_fitness - current_fitness) / (abs(current_fitness) + 1e-8)
            else:
                improvement_rate = 0.0
        
        # Build state vector (13 features)
        state = np.array([
            avg_fitness_norm,
            best_fitness_norm,
            worst_fitness_norm,
            current_diversity,
            gen / 300.0,  # Normalize by max generations
            stagnation,
            fitness_trend,
            diversity_trend,
            stagnation,
            convergence_speed,
            population_size / 500.0,  # NEW
            crossover_rate,  # NEW
            mutation_rate / 2.0  # NEW
        ], dtype=np.float32)
        
        # Label: Determine optimal mutation rate based on improvement
        # Heuristic labeling:
        # - High improvement → current rate was good (keep it)
        # - Stagnation + low diversity → need more exploration (higher rate)
        # - Good fitness + high diversity → keep exploring (moderate rate)
        # - Near optimal + low diversity → fine-tune (lower rate)
        
        if improvement_rate > 0.01:  # Good improvement
            target_mutation = mutation_rate * 1.0  # Keep current rate
        elif stagnation > 0.5 and current_diversity < 0.05:  # Stuck + converged
            target_mutation = mutation_rate * 2.0  # Need more exploration
        elif current_diversity < 0.03:  # Very converged
            target_mutation = mutation_rate * 0.5  # Fine-tuning
        else:
            target_mutation = mutation_rate * 1.0  # Default
            
        target_mutation = np.clip(target_mutation, 0.01, 2.0)
        
        X.append(state)
        y.append(target_mutation)
        
    return np.array(X), np.array(y)


def train_ml_model(X_train, y_train, X_val, y_val, epochs=100):
    """Train EnhancedMLPredictor model"""
    
    # Create model
    model = EnhancedMLPredictor(input_dim=13, hidden_dims=[128, 256, 128])
    
    # Fit normalization
    model.fit_normalization(X_train)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print("\n" + "="*70)
    print("TRAINING ENHANCED ML PREDICTOR")
    print("="*70)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Epochs: {epochs}")
    print("="*70 + "\n")
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t)
            val_loss = criterion(val_predictions, y_val_t)
            val_losses.append(val_loss.item())
            
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")
            
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss: {val_losses[-1]:.6f}")
    print("="*70 + "\n")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Enhanced ML Predictor Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/ml_predictor_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved training plot to: outputs/ml_predictor_training.png")
    
    return model


def test_on_volatile():
    """Test enhanced trainer on volatile specialist"""
    
    # Load data
    print("\n" + "="*70)
    print("TESTING ENHANCED TRAINER ON VOLATILE SPECIALIST")
    print("="*70 + "\n")
    
    # Load training data
    df = pd.read_csv('DATA/yf_btc_1d_volatile.csv')
    training_data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values
    
    # Generate predictions (same as baseline)
    close = df['close'].values
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])
    
    sma_short = pd.Series(close).rolling(10).mean().fillna(close).values
    sma_long = pd.Series(close).rolling(50).mean().fillna(close).values
    crossover = (sma_short - sma_long) / close
    momentum = pd.Series(returns).rolling(5).mean().fillna(0).values
    predictions = crossover * 5 + momentum * 10
    
    print(f"✅ Loaded {len(training_data)} days of volatile market data")
    print(f"📊 Generated momentum-based trading signals")
    print()
    
    # Create enhanced trainer WITHOUT ML first (baseline comparison)
    print("1️⃣  Training with BASELINE method (fixed mutation rate)...")
    baseline_trainer = EnhancedSpecialistTrainer(
        regime_type='volatile',
        training_data=training_data,
        predictions=predictions,
        population_size=100,  # Smaller for faster testing
        generations=50,
        initial_mutation_rate=0.1,
        crossover_rate=0.7,
        use_ml_adaptation=False  # Disable ML
    )
    
    baseline_genome, baseline_fitness, baseline_history = baseline_trainer.train()
    print(f"\n✅ BASELINE: Final Fitness = {baseline_fitness:.2f}")
    
    # Now with ML model loaded (we'll load pre-trained model)
    # For now, let's just compare baseline
    print("\n" + "="*70)
    print("ML-ENHANCED COMPARISON (coming after model training)")
    print("="*70)
    
    return baseline_fitness


if __name__ == '__main__':
    print("="*70)
    print("ENHANCED ML PREDICTOR - TRAINING PIPELINE")
    print("="*70)
    print("\nPhase 2A: Build and train ML model that learns from baseline data")
    print("\nSteps:")
    print("  1. Load baseline training history")
    print("  2. Extract state-action pairs with synthetic labels")
    print("  3. Train EnhancedMLPredictor model")
    print("  4. Save trained model")
    print("  5. Test on volatile specialist")
    print("="*70 + "\n")
    
    # Load baseline data
    print("Step 1: Loading baseline training data...")
    specialists = load_baseline_data()
    print(f"✅ Loaded 3 specialists: {list(specialists.keys())}")
    
    # Extract training examples from all specialists
    print("\nStep 2: Extracting training examples...")
    all_X = []
    all_y = []
    
    config_volatile = {'population_size': 200, 'crossover_rate': 0.7, 'mutation_rate': 0.1}
    config_trending = {'population_size': 200, 'crossover_rate': 0.7, 'mutation_rate': 0.1}
    config_ranging = {'population_size': 200, 'crossover_rate': 0.7, 'mutation_rate': 0.1}
    
    for regime, config in [('volatile', config_volatile), ('trending', config_trending), ('ranging', config_ranging)]:
        X, y = extract_training_examples(specialists[regime], regime, config)
        all_X.append(X)
        all_y.append(y)
        print(f"  {regime:10s}: {len(X):4d} examples")
        
    # Combine all data
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    print(f"\n✅ Total: {len(X_all)} training examples from 3 specialists")
    
    # Split train/val
    split_idx = int(len(X_all) * 0.8)
    indices = np.random.permutation(len(X_all))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_val = X_all[val_indices]
    y_val = y_all[val_indices]
    
    print(f"  Train: {len(X_train)} examples")
    print(f"  Val:   {len(X_val)} examples")
    
    # Train model
    print("\nStep 3: Training ML model...")
    model = train_ml_model(X_train, y_train, X_val, y_val, epochs=100)
    
    # Save model
    print("\nStep 4: Saving trained model...")
    torch.save(model.state_dict(), 'outputs/enhanced_ml_predictor.pth')
    print("✅ Saved model to: outputs/enhanced_ml_predictor.pth")
    
    # Test on volatile
    print("\nStep 5: Testing on volatile specialist...")
    test_fitness = test_on_volatile()
    
    print("\n" + "="*70)
    print("ENHANCED ML PREDICTOR TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run full comparison (baseline vs enhanced)")
    print("  2. Build full GA Conductor (25 inputs, multi-output)")
    print("  3. Test on all specialists")
    print("="*70 + "\n")
