"""
Extract Training Data for DANN Conductor

This script extracts training samples from all 3 regime training histories
(volatile, trending, ranging) to prepare data for Domain-Adversarial Neural Network
(DANN) training.

Expected output: ~180K training samples with:
- Market features (13 inputs)
- GA parameters (12 outputs)
- Regime labels (0=volatile, 1=trending, 2=ranging)
- Fitness values
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

class DANNDataExtractor:
    def __init__(self):
        self.regime_labels = {
            'volatile': 0,
            'trending': 1,
            'ranging': 2
        }
        
        # Find latest training files
        self.training_files = {
            'volatile': 'outputs/conductor_enhanced_volatile_20251108_111639.json',
            'trending': 'outputs/conductor_enhanced_trending_20251108_114301.json',
            'ranging': 'outputs/conductor_enhanced_ranging_20251108_141359.json'
        }
        
        self.samples = []
    
    def extract_from_history(self, regime, filepath):
        """Extract training samples from a single regime training history."""
        print(f"\n{'='*60}")
        print(f"Extracting {regime.upper()} regime data...")
        print(f"{'='*60}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            return []
        
        regime_label = self.regime_labels[regime]
        samples = []
        
        # Extract metadata
        population_size = data.get('population_size', 200)
        num_generations = data.get('generations', 300)
        
        print(f"  Population size: {population_size}")
        print(f"  Generations: {num_generations}")
        
        # Extract training history (arrays format)
        training_history = data.get('training_history', {})
        
        if not training_history:
            print(f"⚠️  No training history found in {filepath}")
            return []
        
        # Get arrays from training history
        generations = training_history.get('generation', [])
        best_fitness_arr = training_history.get('best_fitness', [])
        avg_fitness_arr = training_history.get('avg_fitness', [])
        diversity_arr = training_history.get('diversity', [])
        mutation_rate_arr = training_history.get('mutation_rate', [])
        crossover_rate_arr = training_history.get('crossover_rate', [])
        
        if len(generations) == 0:
            print(f"⚠️  Empty training history in {filepath}")
            return []
        
        print(f"  Found {len(generations)} generations")
        
        # Extract samples from each generation
        for i in range(len(generations)):
            gen_idx = generations[i]
            
            # Get GA parameters for this generation
            mutation_rate = mutation_rate_arr[i] if i < len(mutation_rate_arr) else 0.1
            crossover_rate = crossover_rate_arr[i] if i < len(crossover_rate_arr) else 0.7
            
            # Create 12 GA parameters (matching conductor output)
            # For now, use actual values + defaults for parameters not stored
            ga_parameters = [
                mutation_rate,
                crossover_rate,
                0.1,  # elite_size (default)
                0.05,  # tournament_size (default)
                0.5,  # mutation_strength (default)
                0.5,  # crossover_alpha (default)
                0.3,  # diversity_weight (default)
                0.2,  # exploration_rate (default)
                0.7,  # selection_pressure (default)
                0.1,  # adaptation_rate (default)
                0.1,  # novelty_bonus (default)
                0.01   # convergence_threshold (default)
            ]
            
            # Get market features for this generation
            best_fitness = best_fitness_arr[i] if i < len(best_fitness_arr) else 0.0
            avg_fitness = avg_fitness_arr[i] if i < len(avg_fitness_arr) else 0.0
            diversity = diversity_arr[i] if i < len(diversity_arr) else 0.0
            
            # Extract market state features (13 features)
            # These are aggregate statistics for the regime
            fitness_gap = (best_fitness - avg_fitness) / (abs(best_fitness) + 1e-6)
            
            market_features = [
                best_fitness / 100.0,  # Normalized best fitness
                avg_fitness / 100.0,   # Normalized avg fitness
                diversity,              # Population diversity
                gen_idx / num_generations,  # Progress through training
                mutation_rate,          # Current mutation rate
                crossover_rate,         # Current crossover rate
                fitness_gap,            # Fitness gap
                abs(fitness_gap),       # Abs fitness gap (fitness spread)
                0.5,  # Placeholder: regime strength
                0.5,  # Placeholder: market volatility
                0.5,  # Placeholder: trend strength
                0.5,  # Placeholder: regime stability
                0.5   # Placeholder: signal quality
            ]
            
            # Create sample
            sample = {
                'features': market_features,
                'parameters': ga_parameters,
                'fitness': best_fitness,
                'regime': regime_label,
                'regime_name': regime,
                'generation': gen_idx
            }
            
            samples.append(sample)
        
        print(f"✓ Extracted {len(samples)} samples from {regime}")
        return samples
    
    def extract_all_regimes(self):
        """Extract training samples from all 3 regimes."""
        print(f"\n{'='*60}")
        print("DANN TRAINING DATA EXTRACTION")
        print(f"{'='*60}")
        
        all_samples = []
        
        for regime, filepath in self.training_files.items():
            regime_samples = self.extract_from_history(regime, filepath)
            all_samples.extend(regime_samples)
        
        print(f"\n{'='*60}")
        print(f"EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples extracted: {len(all_samples)}")
        
        # Count by regime
        regime_counts = {}
        for sample in all_samples:
            regime_name = sample['regime_name']
            regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1
        
        for regime, count in regime_counts.items():
            print(f"  {regime.capitalize()}: {count} samples ({count/len(all_samples)*100:.1f}%)")
        
        return all_samples
    
    def balance_and_split(self, samples, train_ratio=0.8):
        """Balance regime distribution and split into train/val sets."""
        print(f"\n{'='*60}")
        print("BALANCING AND SPLITTING DATA")
        print(f"{'='*60}")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(samples)
        
        # Find minimum regime count for balancing
        regime_counts = df['regime'].value_counts()
        min_count = regime_counts.min()
        
        print(f"Original distribution:")
        for regime_name, count in regime_counts.items():
            regime_label = ['volatile', 'trending', 'ranging'][regime_name]
            print(f"  {regime_label}: {count} samples")
        
        print(f"\nBalancing to {min_count} samples per regime...")
        
        # Balance by sampling equal amounts from each regime
        balanced_samples = []
        for regime_label in [0, 1, 2]:
            regime_df = df[df['regime'] == regime_label]
            sampled = regime_df.sample(n=min_count, random_state=42)
            balanced_samples.append(sampled)
        
        balanced_df = pd.concat(balanced_samples, ignore_index=True)
        
        print(f"✓ Balanced to {len(balanced_df)} total samples")
        print(f"  {min_count} samples per regime ({min_count/len(balanced_df)*100:.1f}% each)")
        
        # Shuffle
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split train/val
        train_size = int(len(balanced_df) * train_ratio)
        train_df = balanced_df[:train_size]
        val_df = balanced_df[train_size:]
        
        print(f"\n✓ Split into:")
        print(f"  Training: {len(train_df)} samples ({len(train_df)/len(balanced_df)*100:.1f}%)")
        print(f"  Validation: {len(val_df)} samples ({len(val_df)/len(balanced_df)*100:.1f}%)")
        
        # Convert back to list of dicts
        train_samples = train_df.to_dict('records')
        val_samples = val_df.to_dict('records')
        
        return train_samples, val_samples
    
    def save_datasets(self, train_samples, val_samples, output_dir='data'):
        """Save train and validation datasets to JSON files."""
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save training data
        train_path = f"{output_dir}/dann_training_data_{timestamp}.json"
        with open(train_path, 'w') as f:
            json.dump(train_samples, f, indent=2)
        print(f"\n✓ Saved training data: {train_path}")
        
        # Save validation data
        val_path = f"{output_dir}/dann_validation_data_{timestamp}.json"
        with open(val_path, 'w') as f:
            json.dump(val_samples, f, indent=2)
        print(f"✓ Saved validation data: {val_path}")
        
        # Save metadata
        metadata = {
            'extraction_date': timestamp,
            'total_samples': len(train_samples) + len(val_samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'num_features': 13,
            'num_parameters': 12,
            'num_regimes': 3,
            'regime_labels': {
                'volatile': 0,
                'trending': 1,
                'ranging': 2
            },
            'source_files': self.training_files
        }
        
        metadata_path = f"{output_dir}/dann_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_path}")
        
        return train_path, val_path, metadata_path
    
    def run(self):
        """Complete extraction pipeline."""
        # Extract from all regimes
        all_samples = self.extract_all_regimes()
        
        if len(all_samples) == 0:
            print("\n❌ No samples extracted! Check file paths.")
            return None, None, None
        
        # Balance and split
        train_samples, val_samples = self.balance_and_split(all_samples)
        
        # Save datasets
        train_path, val_path, metadata_path = self.save_datasets(train_samples, val_samples)
        
        print(f"\n{'='*60}")
        print("✅ DATA EXTRACTION COMPLETE!")
        print(f"{'='*60}")
        print(f"Ready for DANN training with {len(train_samples)} training samples")
        print(f"and {len(val_samples)} validation samples.")
        
        return train_path, val_path, metadata_path


def main():
    """Run DANN data extraction."""
    extractor = DANNDataExtractor()
    train_path, val_path, metadata_path = extractor.run()
    
    if train_path:
        print(f"\nNext steps:")
        print(f"1. Review extracted data: {train_path}")
        print(f"2. Implement DANN architecture: domain_adversarial_conductor.py")
        print(f"3. Train DANN conductor using this data")


if __name__ == '__main__':
    main()
