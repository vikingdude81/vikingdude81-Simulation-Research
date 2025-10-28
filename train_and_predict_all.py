"""
Train All Assets and Generate Multi-Asset Future Predictions
Train BTC, ETH, SOL on fresh data and generate 7-day forecasts for each
"""

import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime

print("\n" + "="*80)
print("🚀 MULTI-ASSET TRAINING & PREDICTION PIPELINE")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Step 1: Train all assets
print("\n📊 STEP 1: Training All Assets (BTC, ETH, SOL)")
print("="*80)
print("\nThis will train models for all 3 assets on fresh data...")
print("Estimated time: ~15 minutes total\n")

# Run train_all_assets.py
result = subprocess.run([sys.executable, 'train_all_assets.py'], 
                       capture_output=False, text=True)

if result.returncode != 0:
    print("\n❌ Training failed!")
    sys.exit(1)

print("\n✅ All assets trained successfully!")

# Step 2: Generate predictions for each asset
print("\n" + "="*80)
print("🔮 STEP 2: Generating Future Predictions for All Assets")
print("="*80)

# Import prediction logic
exec(open('predict_future_prices.py').read())

print("\n" + "="*80)
print("✅ MULTI-ASSET PIPELINE COMPLETE!")
print("="*80)
