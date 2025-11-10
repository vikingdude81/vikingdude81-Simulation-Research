"""
Train Enhanced ML Predictor and GA Conductor on GPU
Using baseline training data with behavioral cloning + reward weighting
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from datetime import datetime


# Import models from phase2_complete_implementation
import sys
sys.path.append('.')
from phase2_complete_implementation import EnhancedMLPredictor, GAConductor


class ModelTrainer:
    """Train models with behavioral cloning + reward weighting"""
    
    def __init__(self, model, model_name: str, learning_rate: float = 0.001):
        self.model = model
        self.model_name = model_name
        self.device = model.device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='none')
        
    def train_predictor(self, states, actions, rewards, 
                       epochs=100, batch_size=32, validation_split=0.2):
        """Train Enhanced ML Predictor (2 outputs)"""
        
        print(f"\n{'='*70}")
        print(f"🎓 TRAINING {self.model_name}")
        print(f"{'='*70}\n")
        
        # Convert to tensors on GPU
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        
        # Normalize rewards for weighting
        reward_weights = (rewards_tensor - rewards_tensor.min()) / \
                        (rewards_tensor.max() - rewards_tensor.min() + 1e-8)
        reward_weights = reward_weights + 0.1  # Minimum weight
        
        # Split train/val
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
        
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(train_indices), batch_size):
                batch_states = train_states[i:i+batch_size]
                batch_actions = train_actions[i:i+batch_size]
                batch_weights = train_weights[i:i+batch_size]
                
                # Forward
                pred_mutation, pred_crossover = self.model(batch_states)
                
                # Loss for each output
                mutation_loss = self.criterion(pred_mutation.squeeze(), batch_actions[:, 0])
                crossover_loss = self.criterion(pred_crossover.squeeze(), batch_actions[:, 1])
                
                # Weight by reward
                mutation_loss = (mutation_loss * batch_weights).mean()
                crossover_loss = (crossover_loss * batch_weights).mean()
                
                loss = mutation_loss + crossover_loss
                
                # Backward
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
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f'outputs/{self.model_name}_best.pth')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")
        
        print(f"\n✅ Training complete! Best val loss: {best_val_loss:.6f}")
        return history
    
    def train_conductor(self, states, rewards, epochs=50, batch_size=32):
        """Train GA Conductor (12 outputs) - simplified for demo"""
        
        print(f"\n{'='*70}")
        print(f"🎓 TRAINING {self.model_name}")
        print(f"{'='*70}\n")
        
        # For conductor, we'll train it to predict optimal parameters
        # using the same behavioral cloning approach but with synthetic targets
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        
        # Create synthetic targets based on state patterns
        # (In full RL training, these would come from policy gradient)
        targets = self._create_synthetic_targets(states_tensor, rewards_tensor)
        
        # Split train/val
        n = len(states)
        n_val = int(n * 0.2)
        indices = np.random.permutation(n)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(train_indices), batch_size):
                idx = train_indices[i:i+batch_size]
                batch_states = states_tensor[idx]
                batch_targets = {k: v[idx] for k, v in targets.items()}
                
                # Forward
                outputs = self.model(batch_states)
                
                # Calculate loss for all outputs
                loss = 0
                for key in outputs.keys():
                    if key in batch_targets:
                        loss += self.criterion(outputs[key].squeeze(), batch_targets[key]).mean()
                
                # Backward
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
                val_outputs = self.model(states_tensor[val_indices])
                val_loss = 0
                for key in val_outputs.keys():
                    if key in targets:
                        val_loss += self.criterion(
                            val_outputs[key].squeeze(), 
                            targets[key][val_indices]
                        ).mean()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss.item())
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f'outputs/{self.model_name}_best.pth')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        print(f"\n✅ Training complete! Best val loss: {best_val_loss:.6f}")
        return history
    
    def _create_synthetic_targets(self, states, rewards):
        """Create synthetic targets for conductor training"""
        n = len(states)
        
        # Extract key features
        diversity = states[:, 4]  # diversity feature
        generation = states[:, 5]  # generation progress
        fitness_trend = states[:, 7]  # convergence speed
        
        # Create targets based on heuristics
        targets = {
            # Evolution: inverse relationships
            'mutation_rate': torch.clamp(1.0 - diversity + 0.1 * torch.randn(n, device=self.device), 0, 1),
            'crossover_rate': torch.clamp(diversity + 0.5 + 0.1 * torch.randn(n, device=self.device), 0, 1),
            'selection_pressure': torch.clamp(0.5 + generation * 0.3, 0, 1),
            
            # Population: dynamic adjustment
            'population_delta': torch.clamp((0.5 - diversity) * 0.2, -0.5, 0.5),
            'immigration_rate': torch.clamp(1.0 - diversity, 0, 1),
            'culling_rate': torch.clamp(diversity - 0.3, 0, 1),
            'diversity_injection': torch.clamp(1.0 - diversity, 0, 1),
            
            # Crisis: threshold-based
            'extinction_trigger': (diversity < 0.1).float(),
            'elite_preservation': (generation > 0.5).float(),
            'restart_signal': (fitness_trend < -0.1).float(),
            
            # Institutional: moderate values
            'welfare_amount': torch.clamp(0.5 - diversity, 0, 1),
            'tax_rate': torch.clamp(diversity - 0.3, 0, 1)
        }
        
        return targets
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def main():
    print("\n" + "="*70)
    print("🚀 TRAINING ENHANCED ML PREDICTOR AND GA CONDUCTOR")
    print("="*70)
    
    # Load training data
    print("\n📂 Loading training data...")
    data = np.load('outputs/baseline_training_data.npz')
    states_13 = data['states_13']
    states_25 = data['states_25']
    actions = data['actions']
    rewards = data['rewards']
    
    print(f"✅ Loaded {len(states_13)} samples")
    print(f"   States_13 shape: {states_13.shape}")
    print(f"   States_25 shape: {states_25.shape}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Rewards range: [{rewards.min():.2f}, {rewards.max():.2f}]")
    
    # Train Enhanced ML Predictor
    print("\n" + "="*70)
    print("PART 1: ENHANCED ML PREDICTOR (13 inputs → 2 outputs)")
    print("="*70)
    
    predictor = EnhancedMLPredictor(hidden_size=256)
    predictor_trainer = ModelTrainer(predictor, 'enhanced_ml_predictor', learning_rate=0.001)
    
    predictor_history = predictor_trainer.train_predictor(
        states_13, actions, rewards,
        epochs=100, batch_size=32
    )
    
    # Save final model
    predictor_trainer.save_model('outputs/enhanced_ml_predictor_final.pth')
    print(f"\n✅ Enhanced ML Predictor saved to outputs/")
    
    # Train GA Conductor
    print("\n" + "="*70)
    print("PART 2: GA CONDUCTOR (25 inputs → 12 outputs)")
    print("="*70)
    
    conductor = GAConductor(hidden_size=512)
    conductor_trainer = ModelTrainer(conductor, 'ga_conductor', learning_rate=0.0005)
    
    conductor_history = conductor_trainer.train_conductor(
        states_25, rewards,
        epochs=50, batch_size=32
    )
    
    # Save final model
    conductor_trainer.save_model('outputs/ga_conductor_final.pth')
    print(f"\n✅ GA Conductor saved to outputs/")
    
    # Save training histories
    history_file = 'outputs/training_history.json'
    with open(history_file, 'w') as f:
        json.dump({
            'predictor': predictor_history,
            'conductor': conductor_history,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"\n✅ Training histories saved to {history_file}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\nModels trained:")
    print("  1. ✅ Enhanced ML Predictor (349K params)")
    print("  2. ✅ GA Conductor (1.7M params)")
    print("\nSaved files:")
    print("  - outputs/enhanced_ml_predictor_best.pth")
    print("  - outputs/enhanced_ml_predictor_final.pth")
    print("  - outputs/ga_conductor_best.pth")
    print("  - outputs/ga_conductor_final.pth")
    print("  - outputs/training_history.json")
    print("\nNext step: Test on out-of-sample data and compare to baseline!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
