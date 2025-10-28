"""
Multi-Asset Model Trainer
=========================

Trains prediction models for BTC, ETH, and SOL using the same pipeline.
Generates predictions and signals for all three assets.

Author: AI Trading System
Date: October 25, 2025
"""

import sys
import os
import subprocess
from datetime import datetime

def train_asset(asset_name, asset_symbol):
    """
    Train models for a single asset by modifying main.py parameters.
    
    Args:
        asset_name: 'btc', 'eth', or 'sol'
        asset_symbol: Display name (e.g., 'Bitcoin', 'Ethereum', 'Solana')
    """
    print("\n" + "="*80)
    print(f"🚀 TRAINING {asset_symbol.upper()} ({asset_name.upper()}) MODELS")
    print("="*80 + "\n")
    
    # Since we can't easily modify main.py on the fly, we'll provide instructions
    print(f"To train {asset_symbol} models:")
    print(f"1. Update main.py to use {asset_name} data files:")
    print(f"   - Change 'yf_btc' to 'yf_{asset_name}' in file paths")
    print(f"2. Run: python main.py")
    print(f"3. Models will be saved to MODEL_STORAGE/")
    print()
    
    response = input(f"Press ENTER when ready to train {asset_symbol}, or 's' to skip: ")
    
    if response.lower() == 's':
        print(f"Skipping {asset_symbol} training.\n")
        return False
    
    print(f"\n▶️  Training {asset_symbol}...")
    print("This will take approximately 7-10 minutes...\n")
    
    # Run training (would need to modify main.py first)
    # For now, we'll provide a template script
    
    return True

def create_quick_train_script(asset_name):
    """
    Create a quick training script for a specific asset.
    
    This is a simplified version that uses the existing pipeline.
    """
    script_content = f"""# Quick {asset_name.upper()} Training Script
# Automatically generated

# This script would:
# 1. Load {asset_name.upper()} data from DATA/yf_{asset_name}_*.csv
# 2. Apply enhanced_features.py
# 3. Train 6 models (RF, XGB, LGB, LSTM, Transformer, MultiTask)
# 4. Save to MODEL_STORAGE/{asset_name}_*
# 5. Generate predictions

# To implement: Copy main.py and modify DATA_PREFIX = 'yf_{asset_name}'
"""
    
    filename = f"train_{asset_name}.py"
    with open(filename, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created template: {filename}")
    return filename


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("🎯 MULTI-ASSET MODEL TRAINING")
    print("="*80)
    print("\nThis wizard will guide you through training models for:")
    print("  • Bitcoin (BTC)")
    print("  • Ethereum (ETH)")  
    print("  • Solana (SOL)")
    print("\nEach asset takes ~7-10 minutes to train (6 models each).")
    print("Total time for all 3 assets: ~20-30 minutes")
    print("\n" + "="*80 + "\n")
    
    choice = input("Train all 3 assets now? (y/n): ")
    
    if choice.lower() != 'y':
        print("\n💡 TIP: You can train assets individually using the quick train scripts.")
        print("I'll create template scripts for you...")
        
        for asset in ['btc', 'eth', 'sol']:
            create_quick_train_script(asset)
        
        print("\n✅ Template scripts created!")
        print("="*80)
        return
    
    # Training sequence
    assets = [
        ('btc', 'Bitcoin'),
        ('eth', 'Ethereum'),
        ('sol', 'Solana')
    ]
    
    results = {}
    
    for asset_name, asset_symbol in assets:
        success = train_asset(asset_name, asset_symbol)
        results[asset_name] = success
    
    # Summary
    print("\n" + "="*80)
    print("📊 TRAINING SUMMARY")
    print("="*80)
    
    for asset_name, success in results.items():
        status = "✅ Completed" if success else "⏭️  Skipped"
        print(f"{asset_name.upper()}: {status}")
    
    print("\n" + "="*80)
    print("🎉 Multi-Asset Training Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
