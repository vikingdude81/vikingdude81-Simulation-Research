"""
Simplified Multi-Asset Trainer
==============================

Quick wrapper to train models for BTC, ETH, and SOL using existing pipeline.
Modifies data paths and trains each asset sequentially.

Usage:
    python train_all_assets.py              # Train all 3 assets
    python train_all_assets.py --asset sol  # Train only SOL
    python train_all_assets.py --asset eth  # Train only ETH

Author: AI Trading System
Date: October 25, 2025
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess

# Asset configurations
ASSETS = {
    'btc': {
        'name': 'Bitcoin',
        'symbol': 'BTC',
        'data_prefix': 'yf_btc',
        'price_col': 'close',
        'target_rmse': 0.45  # We know BTC achieves 0.45%
    },
    'eth': {
        'name': 'Ethereum',
        'symbol': 'ETH',
        'data_prefix': 'yf_eth',
        'price_col': 'close',
        'target_rmse': 0.80  # Target <1% RMSE
    },
    'sol': {
        'name': 'Solana',
        'symbol': 'SOL',
        'data_prefix': 'yf_sol',
        'price_col': 'close',
        'target_rmse': 0.80  # Target <1% RMSE
    }
}

def modify_main_for_asset(asset_name):
    """
    Create instructions for training a specific asset.
    
    Note: This provides manual steps since main.py is complex.
    A production version would programmatically modify main.py or use config files.
    """
    asset = ASSETS[asset_name]
    
    print(f"\n{'='*80}")
    print(f"📋 MANUAL TRAINING INSTRUCTIONS FOR {asset['name'].upper()}")
    print(f"{'='*80}\n")
    
    print(f"To train {asset['name']} models, follow these steps:\n")
    
    print(f"1️⃣  Edit main.py - Change data paths (around lines 66-70):")
    print(f"   From: YF_FILE_PATH_1H = SCRIPT_DIR / 'DATA' / 'yf_btc_1h.csv'")
    print(f"   To:   YF_FILE_PATH_1H = SCRIPT_DIR / 'DATA' / '{asset['data_prefix']}_1h.csv'")
    print(f"")
    print(f"   From: YF_FILE_PATH_4H = SCRIPT_DIR / 'DATA' / 'yf_btc_4h.csv'")
    print(f"   To:   YF_FILE_PATH_4H = SCRIPT_DIR / 'DATA' / '{asset['data_prefix']}_4h.csv'")
    print(f"")
    print(f"   From: YF_FILE_PATH_12H = SCRIPT_DIR / 'DATA' / 'yf_btc_12h.csv'")
    print(f"   To:   YF_FILE_PATH_12H = SCRIPT_DIR / 'DATA' / '{asset['data_prefix']}_12h.csv'")
    print(f"")
    print(f"   From: YF_FILE_PATH_1D = SCRIPT_DIR / 'DATA' / 'yf_btc_1d.csv'")
    print(f"   To:   YF_FILE_PATH_1D = SCRIPT_DIR / 'DATA' / '{asset['data_prefix']}_1d.csv'")
    print(f"")
    
    print(f"2️⃣  Run training:")
    print(f"   python main.py")
    print(f"")
    
    print(f"3️⃣  Training will output:")
    print(f"   • 6 models (RF, XGB, LGB, LSTM, Transformer, MultiTask)")
    print(f"   • Feature importance analysis")
    print(f"   • Predictions CSV")
    print(f"   • Performance metrics")
    print(f"")
    
    print(f"4️⃣  Expected results:")
    print(f"   • Target RMSE: <{asset['target_rmse']}%")
    print(f"   • Training time: ~7-10 minutes")
    print(f"")
    
    print(f"5️⃣  After training, rename outputs:")
    print(f"   • MODEL_STORAGE/best_model.pkl → {asset_name}_best_model.pkl")
    print(f"   • predictions_forecast.csv → {asset_name}_predictions.csv")
    print(f"")
    
    print(f"{'='*80}\n")
    
    return True


def create_automated_script(asset_name):
    """
    Create an automated training script for a specific asset.
    This is a better approach - creates a modified copy of main.py.
    """
    asset = ASSETS[asset_name]
    
    print(f"\n🔧 Creating automated training script for {asset['name']}...")
    
    # Read main.py
    try:
        with open('main.py', 'r', encoding='utf-8') as f:
            main_content = f.read()
    except Exception as e:
        print(f"❌ Error reading main.py: {e}")
        return False
    
    # Replace data paths
    modified_content = main_content.replace(
        "YF_FILE_PATH_1H = SCRIPT_DIR / 'DATA' / 'yf_btc_1h.csv'",
        f"YF_FILE_PATH_1H = SCRIPT_DIR / 'DATA' / '{asset['data_prefix']}_1h.csv'"
    )
    modified_content = modified_content.replace(
        "YF_FILE_PATH_4H = SCRIPT_DIR / 'DATA' / 'yf_btc_4h.csv'",
        f"YF_FILE_PATH_4H = SCRIPT_DIR / 'DATA' / '{asset['data_prefix']}_4h.csv'"
    )
    modified_content = modified_content.replace(
        "YF_FILE_PATH_12H = SCRIPT_DIR / 'DATA' / 'yf_btc_12h.csv'",
        f"YF_FILE_PATH_12H = SCRIPT_DIR / 'DATA' / '{asset['data_prefix']}_12h.csv'"
    )
    modified_content = modified_content.replace(
        "YF_FILE_PATH_1D = SCRIPT_DIR / 'DATA' / 'yf_btc_1d.csv'",
        f"YF_FILE_PATH_1D = SCRIPT_DIR / 'DATA' / '{asset['data_prefix']}_1d.csv'"
    )
    
    # Save to asset-specific file
    output_file = f"main_{asset_name}.py"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"✅ Created {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Error creating script: {e}")
        return False


def train_asset(asset_name, auto=False):
    """
    Train models for a specific asset.
    
    Args:
        asset_name: 'btc', 'eth', or 'sol'
        auto: If True, create and run automated script
    """
    asset = ASSETS[asset_name]
    
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING {asset['name'].upper()} ({asset['symbol']}) MODELS")
    print(f"{'='*80}\n")
    
    if auto:
        # Create automated script
        script_file = create_automated_script(asset_name)
        if not script_file:
            return False
        
        # Ask if user wants to run it now
        response = input(f"\nRun training for {asset['name']} now? (y/n): ")
        if response.lower() == 'y':
            print(f"\n▶️  Starting {asset['name']} training...")
            print("This will take approximately 7-10 minutes...\n")
            
            try:
                subprocess.run([sys.executable, script_file], check=True)
                print(f"\n✅ {asset['name']} training complete!")
                return True
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Error during training: {e}")
                return False
        else:
            print(f"\n💡 Run manually later with: python {script_file}")
            return True
    else:
        # Provide manual instructions
        modify_main_for_asset(asset_name)
        return True


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Train multi-asset prediction models')
    parser.add_argument('--asset', choices=['btc', 'eth', 'sol', 'all'], default='all',
                      help='Asset to train (default: all)')
    parser.add_argument('--auto', action='store_true',
                      help='Create automated training scripts')
    parser.add_argument('--manual', action='store_true',
                      help='Show manual instructions only')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎯 MULTI-ASSET MODEL TRAINER")
    print("="*80)
    print("\nThis tool helps you train prediction models for:")
    print("  • Bitcoin (BTC)  - Already trained (0.45% RMSE)")
    print("  • Ethereum (ETH) - New asset (target <1% RMSE)")
    print("  • Solana (SOL)   - New asset (target <1% RMSE)")
    print("\nEach asset takes ~7-10 minutes to train.")
    print("="*80 + "\n")
    
    # Determine mode
    if args.manual:
        auto_mode = False
    elif args.auto:
        auto_mode = True
    else:
        # Ask user
        response = input("Create automated scripts? (y/n, default=y): ").strip().lower()
        auto_mode = response != 'n'
    
    # Train assets
    if args.asset == 'all':
        assets_to_train = ['btc', 'eth', 'sol']
    else:
        assets_to_train = [args.asset]
    
    results = {}
    
    for asset_name in assets_to_train:
        success = train_asset(asset_name, auto=auto_mode)
        results[asset_name] = success
    
    # Summary
    print("\n" + "="*80)
    print("📊 TRAINING SUMMARY")
    print("="*80)
    
    for asset_name, success in results.items():
        asset = ASSETS[asset_name]
        status = "✅ Ready" if success else "❌ Failed"
        print(f"{asset['symbol']}: {status}")
    
    if auto_mode:
        print("\n💡 Automated scripts created:")
        for asset_name in assets_to_train:
            print(f"  • main_{asset_name}.py")
        print("\nRun each with: python main_[asset].py")
    
    print("\n" + "="*80)
    print("🎉 Setup Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
