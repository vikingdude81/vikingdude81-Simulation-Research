"""
Quick diagnostic to find which selected feature is missing during main.py execution
"""
import pandas as pd
import numpy as np
from enhanced_features import add_all_enhanced_features

# Load 1h data (like main.py does)
df = pd.read_csv('c:/Users/akbon/OneDrive/Documents/PRICE-DETECTION-TEST-1/PRICE-DETECTION-TEST-1/DATA/yf_btc_1h.csv')
print(f"Loaded {len(df)} rows")

# Add enhanced features (like main.py does)
print("\nAdding enhanced features...")
df_enhanced = add_all_enhanced_features(df)
print(f"Enhanced features: {len(df_enhanced.columns)} total columns")

# Load selected features list
selected_file = 'c:/Users/akbon/OneDrive/Documents/PRICE-DETECTION-TEST-1/PRICE-DETECTION-TEST-1/MODEL_STORAGE/feature_data/selected_features_with_interactions.txt'
with open(selected_file, 'r') as f:
    selected_features = [line.strip() for line in f if line.strip()]

print(f"\nSelected features in file: {len(selected_features)}")

# Find missing features
available_cols = set(df_enhanced.columns)
missing_features = [f for f in selected_features if f not in available_cols]

print(f"\n{'='*60}")
if missing_features:
    print(f"❌ MISSING FEATURES ({len(missing_features)}):")
    for feat in missing_features:
        print(f"   {feat}")
else:
    print("✅ All selected features are present!")
print(f"{'='*60}")

# Show which ones ARE present
present_features = [f for f in selected_features if f in available_cols]
print(f"\n✅ Present features: {len(present_features)}")

print(f"\nInteraction features check:")
interaction_features = [
    'momentum_vol_ratio', 'returns_vol_adjusted', 'trend_vol_adjusted',
    'round_level_flow', 'round_5k_imbalance', 'high_dist_flow', 'low_dist_buying',
    'spread_regime', 'spread_vol_regime', 'liquidity_trend', 'spread_trend_strength',
    'vol_accel_regime', 'vol_chaos_combo', 'vol_persistence',
    'momentum_scale_ratio', 'kurtosis_change',
    'volume_weighted_returns', 'imbalance_momentum', 'imbalance_trend',
    'fractal_vol_regime', 'chaos_trend',
    'flow_vol_ratio', 'intensity_spread_ratio'
]

interaction_in_selected = [f for f in selected_features if f in interaction_features]
interaction_missing = [f for f in interaction_in_selected if f not in available_cols]

print(f"  Interactions in selected list: {len(interaction_in_selected)}")
print(f"  Interactions present in df: {len(interaction_in_selected) - len(interaction_missing)}")
if interaction_missing:
    print(f"  ❌ Missing interactions:")
    for feat in interaction_missing:
        print(f"     {feat}")
