"""
Feature Selection with Geometric MA Features
Re-run feature selection to see which GMA features should be included
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)

print("\n" + "="*70)
print("🔍 FEATURE SELECTION WITH GEOMETRIC MA")
print("="*70)

# Load data
df = pd.read_csv('DATA/yf_btc_1h.csv')
df['time'] = pd.to_datetime(df['time'])

print(f"\n✅ Loaded {len(df)} rows")

# Add enhanced features
from enhanced_features import add_all_enhanced_features
df = add_all_enhanced_features(df.copy())

# Clean data
df = df.dropna()
df['target_return'] = df['close'].pct_change().shift(-1)
df = df.dropna()

print(f"🧹 Clean dataset: {len(df)} rows")

# Prepare features
exclude_cols = ['time', 'target_return', 'close', 'open', 'high', 'low', 
                'volume', 'Dividends', 'Stock Splits', 'timestamp', 'price', 'next_price']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['target_return']

print(f"\n📊 Total features: {len(feature_cols)}")

# Feature importance with Random Forest
print("\n🌲 Training Random Forest for feature importance...")
rf = RandomForestRegressor(
    n_estimators=200, 
    max_depth=15, 
    random_state=42, 
    n_jobs=-1,
    min_samples_split=10
)
rf.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📈 Top 20 Features Overall:")
for idx, row in feature_importance.head(20).iterrows():
    is_gma = '🚀' if 'gma' in row['feature'].lower() else '  '
    print(f"{is_gma} {idx+1:3d}. {row['feature']:35s} {row['importance']:.6f}")

# GMA feature analysis
gma_features = feature_importance[feature_importance['feature'].str.contains('gma', case=False)]
print(f"\n📊 GMA Features Analysis:")
print(f"   Total GMA features: {len(gma_features)}")
print(f"   Best GMA rank: #{gma_features.index[0] + 1}")
print(f"   GMA in top 50: {len(gma_features[gma_features.index < 50])}")
print(f"   GMA in top 100: {len(gma_features[gma_features.index < 100])}")

print(f"\n🚀 All GMA Features (ranked):")
for idx, row in gma_features.iterrows():
    rank = idx + 1
    marker = '⭐' if rank <= 50 else '✓' if rank <= 100 else ' '
    print(f"{marker} #{rank:3d}. {row['feature']:30s} {row['importance']:.6f}")

# Select top features (using median importance as threshold)
selector = SelectFromModel(rf, threshold='median', prefit=True)
selected_mask = selector.get_support()
selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]

print(f"\n✂️  Feature Selection (median threshold):")
print(f"   Selected: {len(selected_features)} features")
print(f"   Removed: {len(feature_cols) - len(selected_features)} features")

# How many GMA features selected?
selected_gma = [f for f in selected_features if 'gma' in f.lower()]
print(f"\n🚀 GMA Features Selected: {len(selected_gma)} of {len(gma_features)}")
for f in selected_gma:
    imp = feature_importance[feature_importance['feature'] == f]['importance'].values[0]
    rank = feature_importance[feature_importance['feature'] == f].index[0] + 1
    print(f"   ✓ #{rank:3d}. {f:30s} {imp:.6f}")

# Force-include top GMA features if not already selected
force_include_gma = ['gma_200', 'gma_60', 'dist_to_gma_50', 'gma_75', 'dist_to_gma_200']
force_included_count = 0

for feat in force_include_gma:
    if feat in feature_cols and feat not in selected_features:
        selected_features.append(feat)
        force_included_count += 1
        imp = feature_importance[feature_importance['feature'] == feat]['importance'].values[0]
        rank = feature_importance[feature_importance['feature'] == feat].index[0] + 1
        print(f"\n   🔧 Force-included: {feat} (rank #{rank}, importance {imp:.6f})")

if force_included_count > 0:
    print(f"\n✅ Force-included {force_included_count} additional GMA features")

# K-Means features (always include)
kmeans_features = [
    'market_cluster',
    'cluster_0_ranging',
    'cluster_1_trending', 
    'cluster_2_choppy',
    'cluster_3_stable',
    'cluster_confidence'
]

for feat in kmeans_features:
    if feat in feature_cols and feat not in selected_features:
        selected_features.append(feat)
        print(f"   🔧 Force-included K-Means: {feat}")

# Save updated feature list
output_file = 'MODEL_STORAGE/feature_data/selected_features_with_gma.txt'
with open(output_file, 'w') as f:
    for feat in sorted(selected_features):
        f.write(f"{feat}\n")

print(f"\n💾 Saved {len(selected_features)} selected features to:")
print(f"   {output_file}")

# Summary
print("\n" + "="*70)
print("📊 FINAL FEATURE SELECTION SUMMARY")
print("="*70)
print(f"Total features available: {len(feature_cols)}")
print(f"Features selected: {len(selected_features)}")
print(f"   - By importance: {len(selected_features) - force_included_count - len(kmeans_features)}")
print(f"   - GMA force-included: {force_included_count}")
print(f"   - K-Means force-included: {len(kmeans_features)}")
print(f"\nGMA features in final set: {len([f for f in selected_features if 'gma' in f.lower()])}")
print("="*70)
