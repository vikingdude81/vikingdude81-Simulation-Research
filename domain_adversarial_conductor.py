"""
Domain-Adversarial Neural Network (DANN) Conductor
===================================================

Based on "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2015)
https://arxiv.org/abs/1505.07818

Architecture:
    Input (13 market features)
            ↓
    Feature Extractor (G_f) ← Learns regime-invariant features
            ↓
        Features
       ↙       ↘
      ↓         ↓ (Gradient Reversal Layer)
    Label      Domain
    Predictor  Classifier
    (G_y)      (G_d)
      ↓          ↓
    12 GA      Regime
    Params     Class

Training objective:
    Total Loss = L_y + (λ * L_d)
    
    where:
    - L_y = MSE(predicted_params, true_params)  ← Minimize
    - L_d = CrossEntropy(regime_pred, true_regime) ← Maximize (via GRL)
    - λ = domain adaptation strength (0.1 - 1.0)
    
The adversarial mechanism:
    - G_d tries to classify regime from features
    - G_f tries to fool G_d while predicting parameters
    - Equilibrium: Features that predict well but hide regime info
    
Success criteria:
    - Parameter prediction MSE < 0.05
    - Domain classifier accuracy ~35-45% (near random 33% = good!)
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import os


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL)
    
    Forward pass: identity (y = x)
    Backward pass: reverses gradient (dy/dx = -λ * gradient)
    
    This makes the feature extractor learn features that:
    1. Are useful for parameter prediction (minimize L_y)
    2. Confuse the domain classifier (maximize L_d via reversed gradient)
    """
    
    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient and scale by lambda
        return grad_output.neg() * ctx.lambda_param, None


class FeatureExtractor(nn.Module):
    """
    Shared feature extractor (G_f)
    
    Learns regime-invariant representations from market features.
    These features are used by both:
    - Parameter predictor (to predict GA parameters)
    - Domain classifier (to classify regime - but we want to fool it!)
    """
    
    def __init__(self, input_size=13, hidden_size=128, feature_size=64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, feature_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)


class ParameterPredictor(nn.Module):
    """
    Label predictor (G_y)
    
    Predicts 12 GA parameters from extracted features.
    This is the main task - we want accurate parameter predictions.
    """
    
    def __init__(self, feature_size=64, hidden_size=64, output_size=12):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Parameters are in [0, 1]
        )
    
    def forward(self, features):
        return self.network(features)


class RegimeClassifier(nn.Module):
    """
    Domain classifier (G_d)
    
    Tries to classify regime from features (after GRL).
    During training:
    - G_d wants to classify correctly (minimize classification error)
    - G_f wants to fool G_d (via reversed gradient)
    
    Success = G_d accuracy near random (33% for 3 classes)
    This means features are regime-invariant!
    """
    
    def __init__(self, feature_size=64, hidden_size=64, num_regimes=3):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_regimes)
        )
    
    def forward(self, features):
        return self.network(features)


class DANNDataset(Dataset):
    """Dataset for DANN training"""
    
    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        features = torch.tensor(sample['features'], dtype=torch.float32)
        parameters = torch.tensor(sample['parameters'], dtype=torch.float32)
        regime = torch.tensor(sample['regime'], dtype=torch.long)
        
        return features, parameters, regime


class DANNConductor:
    """
    Domain-Adversarial Neural Network Conductor
    
    Single universal conductor that works across all regimes
    by learning regime-invariant features.
    """
    
    def __init__(
        self,
        input_size: int = 13,
        feature_size: int = 64,
        hidden_size: int = 128,
        output_size: int = 12,
        num_regimes: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Build networks
        self.feature_extractor = FeatureExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            feature_size=feature_size
        ).to(device)
        
        self.parameter_predictor = ParameterPredictor(
            feature_size=feature_size,
            hidden_size=hidden_size,
            output_size=output_size
        ).to(device)
        
        self.regime_classifier = RegimeClassifier(
            feature_size=feature_size,
            hidden_size=hidden_size,
            num_regimes=num_regimes
        ).to(device)
        
        # Optimizers
        self.optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.parameter_predictor.parameters()) +
            list(self.regime_classifier.parameters()),
            lr=0.001
        )
        
        # Loss functions
        self.param_loss_fn = nn.MSELoss()
        self.regime_loss_fn = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'epoch': [],
            'train_param_loss': [],
            'train_regime_loss': [],
            'train_total_loss': [],
            'val_param_loss': [],
            'val_regime_loss': [],
            'val_regime_accuracy': [],
            'lambda': []
        }
    
    def predict(self, market_features: np.ndarray) -> np.ndarray:
        """
        Predict GA parameters for given market features.
        
        Args:
            market_features: (13,) array of market features
            
        Returns:
            (12,) array of GA parameters
        """
        self.feature_extractor.eval()
        self.parameter_predictor.eval()
        
        with torch.no_grad():
            features_tensor = torch.tensor(
                market_features,
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            features = self.feature_extractor(features_tensor)
            parameters = self.parameter_predictor(features)
            
            return parameters.cpu().numpy().flatten()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        lambda_param: float = 0.5
    ) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            lambda_param: Domain adaptation strength (0.0 - 1.0)
            
        Returns:
            (param_loss, regime_loss, total_loss)
        """
        self.feature_extractor.train()
        self.parameter_predictor.train()
        self.regime_classifier.train()
        
        total_param_loss = 0.0
        total_regime_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for features, parameters, regimes in train_loader:
            features = features.to(self.device)
            parameters = parameters.to(self.device)
            regimes = regimes.to(self.device)
            
            # Forward pass
            # 1. Extract features
            extracted_features = self.feature_extractor(features)
            
            # 2. Predict parameters (main task)
            pred_parameters = self.parameter_predictor(extracted_features)
            param_loss = self.param_loss_fn(pred_parameters, parameters)
            
            # 3. Classify regime (adversarial task)
            # Apply gradient reversal before regime classification
            reversed_features = GradientReversalLayer.apply(
                extracted_features,
                lambda_param
            )
            regime_logits = self.regime_classifier(reversed_features)
            regime_loss = self.regime_loss_fn(regime_logits, regimes)
            
            # Total loss
            loss = param_loss + (lambda_param * regime_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track losses
            total_param_loss += param_loss.item()
            total_regime_loss += regime_loss.item()
            total_loss += loss.item()
            num_batches += 1
        
        return (
            total_param_loss / num_batches,
            total_regime_loss / num_batches,
            total_loss / num_batches
        )
    
    def validate(
        self,
        val_loader: DataLoader,
        lambda_param: float = 0.5
    ) -> Tuple[float, float, float]:
        """
        Validate on validation set.
        
        Returns:
            (param_loss, regime_loss, regime_accuracy)
        """
        self.feature_extractor.eval()
        self.parameter_predictor.eval()
        self.regime_classifier.eval()
        
        total_param_loss = 0.0
        total_regime_loss = 0.0
        correct_regime = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, parameters, regimes in val_loader:
                features = features.to(self.device)
                parameters = parameters.to(self.device)
                regimes = regimes.to(self.device)
                
                # Extract features
                extracted_features = self.feature_extractor(features)
                
                # Predict parameters
                pred_parameters = self.parameter_predictor(extracted_features)
                param_loss = self.param_loss_fn(pred_parameters, parameters)
                
                # Classify regime
                regime_logits = self.regime_classifier(extracted_features)
                regime_loss = self.regime_loss_fn(regime_logits, regimes)
                
                # Regime accuracy
                regime_preds = torch.argmax(regime_logits, dim=1)
                correct_regime += (regime_preds == regimes).sum().item()
                
                total_param_loss += param_loss.item()
                total_regime_loss += regime_loss.item()
                total_samples += regimes.size(0)
        
        num_batches = len(val_loader)
        regime_accuracy = correct_regime / total_samples
        
        return (
            total_param_loss / num_batches,
            total_regime_loss / num_batches,
            regime_accuracy
        )
    
    def train(
        self,
        train_data_path: str,
        val_data_path: str,
        epochs: int = 100,
        batch_size: int = 32,
        lambda_schedule: str = 'constant',  # 'constant', 'linear', or 'progressive'
        lambda_max: float = 0.5,
        save_dir: str = 'outputs'
    ) -> Dict:
        """
        Train the DANN conductor.
        
        Args:
            train_data_path: Path to training data JSON
            val_data_path: Path to validation data JSON
            epochs: Number of training epochs
            batch_size: Batch size
            lambda_schedule: How to schedule lambda parameter
                - 'constant': Fixed lambda throughout
                - 'linear': Linearly increase from 0 to lambda_max
                - 'progressive': 2 / (1 + exp(-10 * p)) - 1, where p = epoch/epochs
            lambda_max: Maximum lambda value
            save_dir: Directory to save results
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print("TRAINING DANN CONDUCTOR")
        print(f"{'='*60}\n")
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Load data
        train_dataset = DANNDataset(train_data_path)
        val_dataset = DANNDataset(val_data_path)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Calculate lambda for this epoch
            if lambda_schedule == 'constant':
                lambda_param = lambda_max
            elif lambda_schedule == 'linear':
                lambda_param = lambda_max * (epoch / epochs)
            else:  # progressive
                p = epoch / epochs
                lambda_param = lambda_max * (2 / (1 + np.exp(-10 * p)) - 1)
            
            # Train
            train_param_loss, train_regime_loss, train_total_loss = \
                self.train_epoch(train_loader, lambda_param)
            
            # Validate
            val_param_loss, val_regime_loss, val_regime_acc = \
                self.validate(val_loader, lambda_param)
            
            # Record history
            self.history['epoch'].append(epoch)
            self.history['train_param_loss'].append(train_param_loss)
            self.history['train_regime_loss'].append(train_regime_loss)
            self.history['train_total_loss'].append(train_total_loss)
            self.history['val_param_loss'].append(val_param_loss)
            self.history['val_regime_loss'].append(val_regime_loss)
            self.history['val_regime_accuracy'].append(val_regime_acc)
            self.history['lambda'].append(lambda_param)
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}/{epochs}")
                print(f"  Train - Param Loss: {train_param_loss:.6f}, "
                      f"Regime Loss: {train_regime_loss:.6f}, "
                      f"Total: {train_total_loss:.6f}")
                print(f"  Val   - Param Loss: {val_param_loss:.6f}, "
                      f"Regime Acc: {val_regime_acc:.2%} (λ={lambda_param:.3f})")
                print()
            
            # Early stopping
            if val_param_loss < best_val_loss:
                best_val_loss = val_param_loss
                patience_counter = 0
                
                # Save best model
                self.save(os.path.join(save_dir, 'dann_conductor_best.pth'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(save_dir, f'dann_conductor_{timestamp}.json')
        
        results = {
            'timestamp': timestamp,
            'epochs_trained': epoch + 1,
            'best_val_param_loss': best_val_loss,
            'final_val_regime_accuracy': val_regime_acc,
            'training_history': self.history,
            'config': {
                'batch_size': batch_size,
                'lambda_schedule': lambda_schedule,
                'lambda_max': lambda_max,
                'epochs': epochs
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("✅ TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Best validation param loss: {best_val_loss:.6f}")
        print(f"Final regime accuracy: {val_regime_acc:.2%}")
        print(f"  (Target: 35-45% = regime-invariant features)")
        print(f"Saved results: {results_path}")
        
        return results
    
    def save(self, path: str):
        """Save model state"""
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'parameter_predictor': self.parameter_predictor.state_dict(),
            'regime_classifier': self.regime_classifier.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.parameter_predictor.load_state_dict(checkpoint['parameter_predictor'])
        self.regime_classifier.load_state_dict(checkpoint['regime_classifier'])


if __name__ == '__main__':
    # Find the most recent training/validation data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Run extract_dann_training_data.py first.")
        exit(1)
    
    data_files = [f for f in os.listdir(data_dir) if f.startswith('dann_training_data')]
    if not data_files:
        print("❌ No training data found! Run extract_dann_training_data.py first.")
        exit(1)
    
    # Get latest file and extract full timestamp (format: dann_training_data_YYYYMMDD_HHMMSS.json)
    latest_file = sorted(data_files)[-1]
    # Extract timestamp: dann_training_data_20251108_144720.json -> 20251108_144720
    latest_timestamp = '_'.join(latest_file.split('_')[3:]).replace('.json', '')
    
    train_path = os.path.join(data_dir, f'dann_training_data_{latest_timestamp}.json')
    val_path = os.path.join(data_dir, f'dann_validation_data_{latest_timestamp}.json')
    
    print(f"Using training data: {train_path}")
    print(f"Using validation data: {val_path}")
    
    # Create conductor
    conductor = DANNConductor(
        input_size=13,
        feature_size=64,
        hidden_size=128,
        output_size=12,
        num_regimes=3
    )
    
    # Train
    outputs_dir = os.path.join(script_dir, 'outputs')
    results = conductor.train(
        train_data_path=train_path,
        val_data_path=val_path,
        epochs=100,
        batch_size=32,
        lambda_schedule='progressive',  # Gradually increase domain adaptation
        lambda_max=0.5,
        save_dir=outputs_dir
    )
    
    print("\n✅ DANN conductor ready for specialist training!")
