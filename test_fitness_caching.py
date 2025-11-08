"""Quick test of fitness caching implementation"""
import pandas as pd
import numpy as np
from conductor_enhanced_trainer import ConductorEnhancedTrainer

# Load volatile regime data
print("Loading volatile regime data...")
df = pd.read_csv('DATA/yf_btc_1d.csv', parse_dates=['time'], index_col='time')

# Simple predictions
df['predictions'] = (df['close'].pct_change().rolling(5).mean()).fillna(0).values

# Filter for volatile regime (simplified - just use volatility)
df['volatility'] = df['close'].pct_change().rolling(20).std()
volatile_threshold = df['volatility'].quantile(0.66)
volatile_data = df[df['volatility'] > volatile_threshold].copy()

print(f"✓ Loaded {len(volatile_data)} days of volatile data")

# Short training run to test caching
print("\n" + "="*60)
print("Testing Fitness Caching with 50 Generations")
print("="*60)

trainer = ConductorEnhancedTrainer(
    regime='volatile',
    regime_data=volatile_data,
    population_size=100,  # Smaller for faster test
    generations=50        # Shorter for faster test
)

trainer.train()

print("\n" + "="*60)
print("Fitness Caching Test Complete!")
print("="*60)
