"""
Extract feature importance including NEW interaction features
Similar to quick_feature_importance.py but with interactions
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Change to script directory
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)
sys.path.insert(0, str(script_dir))

from enhanced_features import add_all_enhanced_features
from external_data import ExternalDataCollector

print("="*80)
print("🔬 FEATURE IMPORTANCE EXTRACTION WITH INTERACTIONS")
print("="*80)
print(f"   Goal: Extract importance of all features INCLUDING 23 new interactions")
print(f"   Method: RandomForest on 1h data")
print("="*80 + "\n")

# Step 1: Load 1h data
print("📊 Step 1: Loading 1h price data...")
df = pd.read_csv('DATA/yf_btc_1h.csv')
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
print(f"   ✓ Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")

# Create target
df['target'] = df['close'].shift(-1) / df['close'] - 1
print(f"   ✓ Created target (next hour return)")

# Step 2: Add enhanced features (including interactions!)
print(f"\n🔧 Step 2: Adding enhanced features WITH INTERACTIONS...")
import time
start_time = time.time()
df = add_all_enhanced_features(df)
elapsed = time.time() - start_time
print(f"   ✓ Enhanced features added in {elapsed:.1f}s")
print(f"   Total columns: {len(df.columns)}")

# Step 3: Clean NaN
print(f"\n🧹 Step 3: Cleaning NaN values...")
rows_before = len(df)
df = df.dropna()
rows_after = len(df)
dropped_pct = ((rows_before - rows_after) / rows_before) * 100
print(f"   ✓ Dropped {rows_before - rows_after} rows with NaN ({dropped_pct:.1f}%)")
print(f"   Clean dataset: {rows_after:,} rows")

# Step 4: Add external data (for completeness)
print(f"\n🌐 Step 4: Adding external data...")
try:
    collector = ExternalDataCollector()
    external_data = collector.get_current_data()
    
    # Add external features
    for key, value in external_data.items():
        if key not in ['timestamp', 'fear_greed_class']:
            df[f'ext_{key}'] = value
    
    print(f"   ✓ Added external data features")
    ext_features = [col for col in df.columns if col.startswith('ext_')]
    print(f"   External features: {len(ext_features)}")
except Exception as e:
    print(f"   ⚠️  Could not add external data: {e}")

# Step 5: Filter to numeric features only
print(f"\n🔢 Step 5: Filtering to numeric features...")
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in feature_cols if col != 'target']
print(f"   ✓ {len(feature_cols)} numeric features available")

# Check for non-numeric that were included
non_numeric = [col for col in df.columns if col not in feature_cols and col != 'target']
if non_numeric:
    print(f"   (Excluded {len(non_numeric)} non-numeric columns)")

# Prepare data
X = df[feature_cols]
y = df['target']

print(f"\n   Final feature matrix: {X.shape}")
print(f"   Target vector: {y.shape}")

# Step 6: Train RandomForest to get feature importance
print(f"\n🌲 Step 6: Training RandomForest for feature importance...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print(f"   Training on {len(X_train):,} samples...")
start_time = time.time()
rf.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"   ✓ RandomForest trained in {train_time:.1f}s")

# Get importance scores
importances = rf.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

# Step 7: Analyze and save
print(f"\n📊 Step 7: Analyzing feature importance...")

# Save full results
output_dir = Path('MODEL_STORAGE/feature_data')
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / 'feature_importance_with_interactions.csv'
feature_importance.to_csv(csv_path, index=False)
print(f"   ✓ Saved full importance to: {csv_path}")

# Calculate selection threshold
median_importance = feature_importance['importance'].median()
selected_features = feature_importance[feature_importance['importance'] >= median_importance]

# Save selected features
selected_path = output_dir / 'selected_features_with_interactions.txt'
with open(selected_path, 'w') as f:
    for feat in selected_features['feature']:
        f.write(f"{feat}\n")
print(f"   ✓ Saved {len(selected_features)} selected features to: {selected_path}")

# Step 8: Display results
print(f"\n" + "="*80)
print("📈 FEATURE IMPORTANCE RESULTS")
print("="*80)

print(f"\n🎯 SELECTION CRITERIA:")
print(f"   Median importance: {median_importance:.6f}")
print(f"   Features above median: {len(selected_features)} (selected)")
print(f"   Features below median: {len(feature_importance) - len(selected_features)} (removed)")
print(f"   Reduction: {((len(feature_importance) - len(selected_features)) / len(feature_importance) * 100):.1f}%")

print(f"\n🏆 TOP 20 FEATURES:")
print("-" * 80)
for i, row in feature_importance.head(20).iterrows():
    is_interaction = '🔥 ' if row['feature'] in [
        'momentum_vol_ratio', 'returns_vol_adjusted', 'trend_vol_adjusted',
        'round_level_flow', 'round_5k_imbalance', 'high_dist_flow', 'low_dist_buying',
        'spread_regime', 'spread_vol_regime', 'liquidity_trend', 'spread_trend_strength',
        'vol_accel_regime', 'vol_chaos_combo', 'vol_persistence',
        'momentum_scale_ratio', 'kurtosis_change',
        'volume_weighted_returns', 'imbalance_momentum', 'imbalance_trend',
        'fractal_vol_regime', 'chaos_trend', 'flow_vol_ratio', 'intensity_spread_ratio'
    ] else '   '
    print(f"{is_interaction}#{i+1:2d}  {row['feature']:30s}  {row['importance']:.6f}")

# Analyze interaction features specifically
interaction_features = [f for f in feature_cols if f in [
    'momentum_vol_ratio', 'returns_vol_adjusted', 'trend_vol_adjusted',
    'round_level_flow', 'round_5k_imbalance', 'high_dist_flow', 'low_dist_buying',
    'spread_regime', 'spread_vol_regime', 'liquidity_trend', 'spread_trend_strength',
    'vol_accel_regime', 'vol_chaos_combo', 'vol_persistence',
    'momentum_scale_ratio', 'kurtosis_change',
    'volume_weighted_returns', 'imbalance_momentum', 'imbalance_trend',
    'fractal_vol_regime', 'chaos_trend', 'flow_vol_ratio', 'intensity_spread_ratio'
]]

interaction_importance = feature_importance[feature_importance['feature'].isin(interaction_features)]
selected_interactions = interaction_importance[interaction_importance['importance'] >= median_importance]

print(f"\n🔥 INTERACTION FEATURES PERFORMANCE:")
print("-" * 80)
print(f"   Total interaction features: {len(interaction_features)}")
print(f"   Selected (above median): {len(selected_interactions)}")
print(f"   Success rate: {(len(selected_interactions) / len(interaction_features) * 100):.1f}%")

if len(selected_interactions) > 0:
    print(f"\n   Top interaction features:")
    for i, row in interaction_importance.head(10).iterrows():
        rank = feature_importance[feature_importance['feature'] == row['feature']].index[0] + 1
        status = "✅" if row['importance'] >= median_importance else "❌"
        print(f"   {status} #{rank:2d}  {row['feature']:30s}  {row['importance']:.6f}")

# Category breakdown of selected features
print(f"\n📊 SELECTED FEATURES BY CATEGORY:")
print("-" * 80)

categories = {
    'base': lambda f: f in ['returns', 'volume', 'volatility', 'volatility_lag1', 'volatility_lag2'],
    'microstructure': lambda f: 'spread' in f or 'illiquidity' in f or 'roll' in f or 'intensity' in f,
    'volatility': lambda f: 'vol' in f.lower() and 'flow' not in f,
    'price_levels': lambda f: 'dist_to' in f or 'round' in f,
    'order_flow': lambda f: 'flow' in f or 'imbalance' in f or 'buy' in f or 'sell' in f or 'pressure' in f,
    'fractal': lambda f: 'hurst' in f or 'fractal' in f or 'chaos' in f or 'skew' in f or 'kurtosis' in f,
    'regime': lambda f: 'regime' in f or 'trend' in f or 'adx' in f,
    'interaction': lambda f: f in interaction_features,
    'external': lambda f: f.startswith('ext_')
}

for cat_name, cat_filter in categories.items():
    cat_features = [f for f in selected_features['feature'] if cat_filter(f)]
    if cat_features:
        print(f"   {cat_name.upper():15s}: {len(cat_features):2d} features")

print(f"\n" + "="*80)
print("✅ FEATURE IMPORTANCE EXTRACTION COMPLETE!")
print("="*80)
print(f"\n📁 Output files:")
print(f"   • {csv_path}")
print(f"   • {selected_path}")
print(f"\n🎯 Next step: Review selected features and run training (Run 5)")
print("="*80)
