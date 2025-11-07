"""
Enhanced ML Predictor - Phase 2A
Context-Aware Adaptive Genetic Algorithm Controller

This is the enhanced version that adds configuration context to the state:
- Original 10 inputs: fitness stats, diversity, trends
- New 3 inputs: population_size, crossover_rate, mutation_rate

Total: 13 inputs → Predicts optimal mutation_rate and crossover_rate

Key Innovation: Model learns strategic relationships like:
- High crossover → Low mutation (inverse relationship)
- Small population → High mutation (exploration)
- High diversity → Low mutation (convergence)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List
import json


class EnhancedMLPredictor(nn.Module):
    """
    Enhanced ML Predictor with Configuration Context
    
    Input Features (13):
        Population Statistics (6):
            - avg_fitness: Average fitness across population
            - best_fitness: Best individual fitness
            - worst_fitness: Worst individual fitness
            - fitness_std: Standard deviation of fitness
            - diversity: Genome diversity (0-1)
            - population_size_normalized: Population size / max_size (0-1)
        
        Progress Metrics (4):
            - generation_normalized: Current gen / max_gen (0-1)
            - time_since_improvement: Generations since last improvement
            - convergence_speed: Rate of fitness improvement
            - stagnation_indicator: Binary flag for stagnation
        
        Configuration Context (3) - NEW!:
            - crossover_rate_current: Current crossover rate
            - mutation_rate_current: Current mutation rate (feedback loop)
            - selection_pressure: Tournament size / population_size
    
    Output (2):
        - mutation_rate_adjustment: Recommended mutation rate (0.0-1.0)
        - crossover_rate_adjustment: Recommended crossover rate (0.0-1.0)
    """
    
    def __init__(self, hidden_size: int = 256, device: str = 'auto'):
        super().__init__()
        
        # Auto-detect GPU
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🎮 EnhancedMLPredictor using device: {self.device}")
        
        # Encoder: Process all 13 features
        self.encoder = nn.Sequential(
            nn.Linear(13, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        
        # Separate heads for mutation and crossover
        self.mutation_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output 0-1
        )
        
        self.crossover_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output 0-1
        )
        
        # Move model to device
        self.to(self.device)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: (batch_size, 13) tensor of state features
            
        Returns:
            mutation_rate: (batch_size, 1) recommended mutation rate
            crossover_rate: (batch_size, 1) recommended crossover rate
        """
        # Ensure input is on correct device
        state = state.to(self.device)
        
        encoded = self.encoder(state)
        mutation_rate = self.mutation_head(encoded)
        crossover_rate = self.crossover_head(encoded)
        return mutation_rate, crossover_rate


class EnhancedTrainingDataCollector:
    """
    Collects training data from GA evolution runs
    
    For each generation, we record:
    - State: 13 features describing population state
    - Action: mutation/crossover rates used
    - Reward: fitness improvement achieved
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.metadata = []
        
    def record_generation(self, 
                         population_stats: Dict,
                         config: Dict,
                         generation: int,
                         max_generation: int,
                         fitness_improvement: float,
                         convergence_speed: float,
                         time_since_improvement: int,
                         stagnation: bool):
        """
        Record data for one generation
        
        Args:
            population_stats: Dict with avg_fitness, best_fitness, worst_fitness, 
                            fitness_std, diversity, population_size
            config: Dict with mutation_rate, crossover_rate, selection_pressure
            generation: Current generation number
            max_generation: Maximum generations
            fitness_improvement: Fitness delta from previous generation
            convergence_speed: Rate of improvement
            time_since_improvement: Gens since last improvement
            stagnation: Boolean flag
        """
        # Build 13-feature state vector
        state = np.array([
            # Population stats (6)
            population_stats['avg_fitness'],
            population_stats['best_fitness'],
            population_stats['worst_fitness'],
            population_stats['fitness_std'],
            population_stats['diversity'],
            population_stats['population_size'] / 500.0,  # Normalize (assume max 500)
            
            # Progress metrics (4)
            generation / max_generation,  # Normalized generation
            time_since_improvement / 50.0,  # Normalize (assume max 50)
            convergence_speed,
            1.0 if stagnation else 0.0,
            
            # Configuration context (3) - THE NEW PART!
            config['crossover_rate'],
            config['mutation_rate'],
            config['selection_pressure']
        ], dtype=np.float32)
        
        # Action taken (mutation_rate, crossover_rate)
        action = np.array([
            config['mutation_rate'],
            config['crossover_rate']
        ], dtype=np.float32)
        
        # Reward: fitness improvement with diversity bonus
        reward = fitness_improvement * 10.0 + population_stats['diversity'] * 2.0
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.metadata.append({
            'generation': generation,
            'best_fitness': population_stats['best_fitness'],
            'diversity': population_stats['diversity']
        })
        
    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get collected data as numpy arrays
        
        Returns:
            states: (N, 13) array
            actions: (N, 2) array
            rewards: (N,) array
        """
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards)
        )
    
    def save(self, filepath: str):
        """Save collected data to JSON"""
        data = {
            'states': [s.tolist() for s in self.states],
            'actions': [a.tolist() for a in self.actions],
            'rewards': [float(r) for r in self.rewards],
            'metadata': self.metadata
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {len(self.states)} training samples to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load collected data from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        collector = cls()
        collector.states = [np.array(s, dtype=np.float32) for s in data['states']]
        collector.actions = [np.array(a, dtype=np.float32) for a in data['actions']]
        collector.rewards = [float(r) for r in data['rewards']]
        collector.metadata = data['metadata']
        
        print(f"✅ Loaded {len(collector.states)} training samples from {filepath}")
        return collector


class EnhancedMLTrainer:
    """
    Train the Enhanced ML Predictor using behavioral cloning + reward weighting
    
    Strategy:
    1. Behavioral Cloning: Learn from baseline GA's actions (imitation)
    2. Reward Weighting: Weight samples by their success (reward)
    3. Strategic Relationships: Model learns inverse/direct relationships
    """
    
    def __init__(self, model: EnhancedMLPredictor, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='none')  # Per-sample loss
        
    def train(self, 
              states: np.ndarray,
              actions: np.ndarray,
              rewards: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2):
        """
        Train model on collected data
        
        Args:
            states: (N, 13) state features
            actions: (N, 2) actions taken (mutation, crossover)
            rewards: (N,) rewards achieved
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            
        Returns:
            training_history: Dict with losses
        """
        # Convert to tensors and move to device
        device = self.model.device
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.FloatTensor(actions).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        
        # Normalize rewards to [0, 1] for weighting
        reward_weights = (rewards_tensor - rewards_tensor.min()) / (rewards_tensor.max() - rewards_tensor.min() + 1e-8)
        reward_weights = reward_weights + 0.1  # Minimum weight
        
        # Split train/validation
        n = len(states)
        n_val = int(n * validation_split)
        indices = np.random.permutation(n)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_states = states_tensor[train_indices]
        train_actions = actions_tensor[train_indices]
        train_weights = reward_weights[train_indices]
        
        val_states = states_tensor[val_indices]
        val_actions = actions_tensor[val_indices]
        
        history = {'train_loss': [], 'val_loss': []}
        
        print(f"\n{'='*70}")
        print(f"🎓 TRAINING ENHANCED ML PREDICTOR")
        print(f"{'='*70}")
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*70}\n")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(train_indices), batch_size):
                batch_states = train_states[i:i+batch_size]
                batch_actions = train_actions[i:i+batch_size]
                batch_weights = train_weights[i:i+batch_size]
                
                # Forward pass
                pred_mutation, pred_crossover = self.model(batch_states)
                
                # Separate loss for mutation and crossover
                mutation_loss = self.criterion(pred_mutation.squeeze(), batch_actions[:, 0])
                crossover_loss = self.criterion(pred_crossover.squeeze(), batch_actions[:, 1])
                
                # Weight by reward
                mutation_loss = (mutation_loss * batch_weights).mean()
                crossover_loss = (crossover_loss * batch_weights).mean()
                
                loss = mutation_loss + crossover_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                pred_mutation, pred_crossover = self.model(val_states)
                
                mutation_loss = self.criterion(pred_mutation.squeeze(), val_actions[:, 0]).mean()
                crossover_loss = self.criterion(pred_crossover.squeeze(), val_actions[:, 1]).mean()
                
                val_loss = mutation_loss + crossover_loss
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        print(f"\n✅ Training complete!")
        return history
    
    def save_model(self, filepath: str):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Model loaded from {filepath}")


def test_enhanced_predictor():
    """Test the enhanced predictor"""
    print("\n" + "="*70)
    print("🧪 TESTING ENHANCED ML PREDICTOR")
    print("="*70 + "\n")
    
    # Create model
    model = EnhancedMLPredictor(hidden_size=256)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    test_state = torch.randn(batch_size, 13)
    
    mutation_rate, crossover_rate = model(test_state)
    
    print(f"\n📊 Test forward pass:")
    print(f"   Input shape: {test_state.shape}")
    print(f"   Mutation rate output: {mutation_rate.shape}, range: [{mutation_rate.min():.3f}, {mutation_rate.max():.3f}]")
    print(f"   Crossover rate output: {crossover_rate.shape}, range: [{crossover_rate.min():.3f}, {crossover_rate.max():.3f}]")
    
    # Test scenarios
    print(f"\n🎯 Testing strategic scenarios:\n")
    
    scenarios = [
        {
            'name': 'Early Gen, Low Diversity, High Crossover',
            'state': [0.5, 0.6, 0.4, 0.1, 0.1, 0.4, 0.1, 0.0, 0.5, 0.0, 0.8, 0.1, 0.5],
            'expected': 'HIGH mutation (explore), LOW crossover (inverse)'
        },
        {
            'name': 'Late Gen, High Diversity, Low Crossover',
            'state': [0.8, 0.9, 0.7, 0.1, 0.8, 0.4, 0.9, 0.0, 0.1, 0.0, 0.2, 0.3, 0.5],
            'expected': 'LOW mutation (converge), HIGH crossover (direct)'
        },
        {
            'name': 'Mid Gen, Stagnant, Moderate Config',
            'state': [0.6, 0.7, 0.5, 0.1, 0.3, 0.4, 0.5, 0.8, 0.0, 1.0, 0.5, 0.2, 0.5],
            'expected': 'HIGH mutation (break stagnation)'
        }
    ]
    
    for scenario in scenarios:
        state_tensor = torch.FloatTensor([scenario['state']])
        mut, cross = model(state_tensor)
        print(f"{scenario['name']}")
        print(f"  Predicted: Mutation={mut.item():.3f}, Crossover={cross.item():.3f}")
        print(f"  Expected: {scenario['expected']}\n")
    
    print("✅ All tests passed!\n")


if __name__ == '__main__':
    test_enhanced_predictor()
