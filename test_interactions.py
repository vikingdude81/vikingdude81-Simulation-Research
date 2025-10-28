"""
Test interaction features
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Change to script directory
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)
sys.path.insert(0, str(script_dir))

from enhanced_features import add_all_enhanced_features

print("="*70)
print("🧪 TESTING INTERACTION FEATURES")
print("="*70)

# Load data
print("\n1️⃣ Loading 1h data...")
df = pd.read_csv('DATA/yf_btc_1h.csv')
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
print(f"   ✓ Loaded {len(df)} rows")
print(f"   Original columns: {len(df.columns)}")

# Add enhanced features (including interactions)
print("\n2️⃣ Adding enhanced features with interactions...")
df_enhanced = add_all_enhanced_features(df.copy())

# Check interaction features
print("\n3️⃣ Checking interaction features...")
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

found_features = [f for f in interaction_features if f in df_enhanced.columns]
missing_features = [f for f in interaction_features if f not in df_enhanced.columns]

print(f"\n   ✅ Found {len(found_features)} / {len(interaction_features)} interaction features")
if found_features:
    print("\n   Top 5 interaction features (latest values):")
    for feat in found_features[:5]:
        val = df_enhanced[feat].iloc[-1]
        print(f"      {feat}: {val:.6f}")

if missing_features:
    print(f"\n   ⚠️  Missing {len(missing_features)} features:")
    for feat in missing_features[:5]:
        print(f"      - {feat}")

# Check for NaN/Inf
print("\n4️⃣ Checking data quality...")
nan_counts = df_enhanced[found_features].isnull().sum()
features_with_nan = nan_counts[nan_counts > 0]

if len(features_with_nan) > 0:
    print(f"   ⚠️  {len(features_with_nan)} features have NaN values:")
    for feat, count in features_with_nan.head(5).items():
        pct = (count / len(df_enhanced)) * 100
        print(f"      {feat}: {count} ({pct:.1f}%)")
else:
    print("   ✅ No NaN values in interaction features!")

# Check for Inf
inf_check = np.isinf(df_enhanced[found_features].select_dtypes(include=[np.number]))
features_with_inf = inf_check.sum()[inf_check.sum() > 0]

if len(features_with_inf) > 0:
    print(f"   ⚠️  {len(features_with_inf)} features have Inf values:")
    for feat, count in features_with_inf.head(5).items():
        print(f"      {feat}: {count}")
else:
    print("   ✅ No Inf values in interaction features!")

# Summary statistics
print("\n5️⃣ Feature statistics (first 5 interaction features):")
for feat in found_features[:5]:
    data = df_enhanced[feat].dropna()
    if len(data) > 0:
        print(f"\n   {feat}:")
        print(f"      Mean: {data.mean():.6f}")
        print(f"      Std:  {data.std():.6f}")
        print(f"      Min:  {data.min():.6f}")
        print(f"      Max:  {data.max():.6f}")

print("\n" + "="*70)
print("✅ TEST COMPLETE!")
print(f"   Total features: {len(df_enhanced.columns)}")
print(f"   Interaction features: {len(found_features)}")
print(f"   Dataset rows: {len(df_enhanced)}")
print("="*70)
