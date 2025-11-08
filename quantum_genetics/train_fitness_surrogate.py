"""
ML Fitness Surrogate Trainer
============================
Train a neural network to predict quantum genome fitness instantly.
Uses the mega-long evolution data (1000 generations) as training data.

Expected speedup: 10x (instant ML prediction vs ~30ms simulation per genome)
"""

import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime

class GenomeFitnessDataset(Dataset):
    """Dataset of genome parameters → fitness mappings"""
    def __init__(self, genomes, fitnesses, generation_nums=None):
        """
        Args:
            genomes: List/array of [μ, ω, d, φ] genomes
            fitnesses: Corresponding fitness values
            generation_nums: Optional (not used for random samples)
        """
        self.genomes = np.array(genomes, dtype=np.float32)
        self.fitnesses = np.array(fitnesses, dtype=np.float32).reshape(-1, 1)
        
        # Use genomes directly as features (no generation)
        self.features = self.genomes
        
        print(f"📊 Dataset: {len(self.genomes)} genomes")
        print(f"   Features: {self.features.shape[1]} dimensions")
        print(f"   Fitness range: [{self.fitnesses.min():.6f}, {self.fitnesses.max():.6f}]")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.fitnesses[idx])

class FitnessSurrogate(nn.Module):
    """
    Neural network that predicts fitness from genome parameters.
    
    Architecture:
        Input: [μ, ω, d, φ] (4 features)
        Hidden: 128 → 64 → 32 neurons with ReLU activation
        Output: fitness (1 value)
    """
    def __init__(self, input_dim=4):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)

def load_training_data(json_path):
    """
    Load training data from generated genome-fitness pairs.
    
    Returns:
        genomes: List of [μ, ω, d, φ] arrays
        fitnesses: Corresponding fitness values
        generations: Generation numbers (dummy values since random sampling)
    """
    print(f"\n📁 Loading data from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    genomes = []
    fitnesses = []
    
    # Extract genome-fitness pairs from the data
    for entry in data['data']:
        genome = entry['genome']
        fitness = entry['fitness']
        
        genomes.append(genome)
        fitnesses.append(fitness)
    
    # Use dummy generation numbers (0 for all since these are random samples)
    generations = np.zeros(len(genomes))
    
    print(f"✓ Loaded {len(genomes)} genome-fitness pairs")
    print(f"✓ Fitness range: [{min(fitnesses):.6f}, {max(fitnesses):.6f}]")
    print(f"✓ Metadata: {data['metadata']['num_samples']} samples, {data['metadata']['timesteps']} timesteps")
    
    return np.array(genomes), np.array(fitnesses), generations

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for features, targets in train_loader:
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        predictions = model(features)
        loss = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            predictions = model(features)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    predictions = np.array(all_predictions).flatten()
    targets = np.array(all_targets).flatten()
    
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return total_loss / len(val_loader), mae, rmse, r2

def test_on_champions(model, scaler, device):
    """Test model predictions on known champion genomes"""
    print("\n" + "="*70)
    print("Testing on Champion Genomes")
    print("="*70)
    
    champions = [
        ('Gentle', [2.7668, 0.1853, 0.0050, 0.6798]),
        ('Standard', [2.9460, 0.1269, 0.0050, 0.2996]),
        ('Chaotic', [3.0000, 0.5045, 0.0050, 0.4108]),
        ('Oscillating', [3.0000, 1.8126, 0.0050, 0.0000]),
        ('Harsh', [3.0000, 2.0000, 0.0050, 0.5713])
    ]
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for name, genome in champions:
            # Use just the genome (no generation feature)
            features = np.array([genome], dtype=np.float32)
            features_scaled = scaler.transform(features)
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
            
            pred = model(features_tensor).cpu().item()
            predictions.append((name, genome, pred))
            
            print(f"  {name:12s}: genome={genome}, predicted_fitness={pred:.6f}")
    
    return predictions

def plot_training_history(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(epochs, history['val_mae'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].set_title('Validation MAE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE
    axes[1, 0].plot(epochs, history['val_rmse'], 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Root Mean Squared Error')
    axes[1, 0].set_title('Validation RMSE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # R² Score
    axes[1, 1].plot(epochs, history['val_r2'], 'c-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_title('Validation R² Score')
    axes[1, 1].axhline(y=0.9, color='r', linestyle='--', label='R²=0.9', alpha=0.5)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to: {save_path}")
    plt.close()

def main():
    print("\n" + "="*70)
    print("ML FITNESS SURROGATE TRAINER")
    print("="*70)
    
    # Configuration
    script_dir = Path(__file__).parent
    # Find the most recent training data file
    training_files = sorted(script_dir.glob('ml_training_data_*.json'), key=lambda x: x.stat().st_mtime, reverse=True)
    if not training_files:
        raise FileNotFoundError("No ml_training_data_*.json files found. Run generate_training_data.py first.")
    DATA_PATH = training_files[0]
    print(f"Using training data: {DATA_PATH.name}")
    BATCH_SIZE = 256
    EPOCHS = 150
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    genomes, fitnesses, generations = load_training_data(DATA_PATH)
    
    # Create dataset
    dataset = GenomeFitnessDataset(genomes, fitnesses, generations)
    
    # Split into train/val
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Feature scaling
    all_features = dataset.features
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Scale dataset features
    dataset.features = scaler.transform(dataset.features).astype(np.float32)
    
    print(f"\n✓ Feature scaling applied (StandardScaler)")
    
    # Create model
    model = FitnessSurrogate(input_dim=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print(f"\n🧠 Model Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n   Total parameters: {total_params:,}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': []
    }
    
    best_val_loss = float('inf')
    best_model_path = script_dir / 'fitness_surrogate_best.pth'
    
    print(f"\n{'='*70}")
    print("Training Started")
    print(f"{'='*70}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Val MAE':>12} {'Val RMSE':>12} {'Val R²':>12}")
    print("-" * 70)
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, val_rmse, val_r2 = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            }, best_model_path)
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:6d} {train_loss:12.6f} {val_loss:12.6f} {val_mae:12.6f} {val_rmse:12.6f} {val_r2:12.4f}")
    
    print("=" * 70)
    print(f"✓ Training complete!")
    print(f"✓ Best validation loss: {best_val_loss:.6f}")
    print(f"✓ Best model saved to: {best_model_path}")
    
    # Load best model (weights_only=False for PyTorch 2.6 compatibility)
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test on champions
    champion_predictions = test_on_champions(model, scaler, device)
    
    # Plot training history
    plot_path = script_dir / 'training_history.png'
    plot_training_history(history, plot_path)
    
    # Save scaler
    scaler_path = script_dir / 'fitness_surrogate_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'training_samples': train_size,
        'validation_samples': val_size,
        'epochs_trained': EPOCHS,
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': float(checkpoint['val_loss']),
        'best_val_mae': float(checkpoint['val_mae']),
        'best_val_rmse': float(checkpoint['val_rmse']),
        'best_val_r2': float(checkpoint['val_r2']),
        'champion_predictions': [
            {'name': name, 'genome': genome, 'predicted_fitness': pred}
            for name, genome, pred in champion_predictions
        ],
        'model_architecture': {
            'input_dim': 4,
            'hidden_layers': [128, 64, 32],
            'output_dim': 1,
            'total_parameters': total_params
        },
        'training_config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'optimizer': 'Adam',
            'loss_function': 'MSE'
        }
    }
    
    summary_path = script_dir / 'fitness_surrogate_training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Training summary saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"  Best Validation Loss: {checkpoint['val_loss']:.6f}")
    print(f"  Best Validation MAE: {checkpoint['val_mae']:.6f}")
    print(f"  Best Validation RMSE: {checkpoint['val_rmse']:.6f}")
    print(f"  Best Validation R²: {checkpoint['val_r2']:.4f}")
    print(f"\n  Model: {best_model_path}")
    print(f"  Scaler: {scaler_path}")
    print(f"  Plot: {plot_path}")
    print(f"  Summary: {summary_path}")
    print("="*70)
    
    return model, scaler

if __name__ == '__main__':
    model, scaler = main()
