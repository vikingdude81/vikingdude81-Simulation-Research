"""
Simple Feature Importance Extraction
Quickly trains RandomForest on current feature set to get importance scores
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import os

# Change to script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

print("\n" + "="*80)
print("FEATURE IMPORTANCE EXTRACTION")
print("="*80)

# Step 1: Load base data
print("\n📂 Step 1: Loading 1h base data...")
df_1h = pd.read_csv('DATA/yf_btc_1h.csv')
df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
df_1h = df_1h.set_index('timestamp')
# Rename 'close' to 'price' for compatibility
df_1h['price'] = df_1h['close']
df_1h['target_return'] = df_1h['price'].pct_change().shift(-1)
df_1h['next_price'] = df_1h['price'].shift(-1)
print(f"   Loaded {len(df_1h):,} rows")

# Step 2: Add enhanced features
print("\n✨ Step 2: Adding enhanced features (~40s)...")
start = time.time()
from enhanced_features import add_all_enhanced_features
df_enhanced = add_all_enhanced_features(df_1h)
print(f"   Enhanced features added in {time.time()-start:.1f}s")

# Step 3: Clean NaN
print("\n🧹 Step 3: Cleaning NaN values...")
rows_before = len(df_enhanced)
df_enhanced = df_enhanced.dropna()
rows_after = len(df_enhanced)
print(f"   Dropped {rows_before - rows_after:,} rows ({(rows_before-rows_after)/rows_before*100:.1f}%)")
print(f"   Clean dataset: {rows_after:,} rows")

# Step 4: Add external data
print("\n🌍 Step 4: Adding external data...")
from external_data import ExternalDataCollector
collector = ExternalDataCollector()
external_data = collector.collect_all()
for key, value in external_data.items():
    df_enhanced[f'ext_{key}'] = value
print(f"   Added {len(external_data)} external features")

# Step 5: Prepare features
print("\n📊 Step 5: Preparing features...")
feature_cols = [col for col in df_enhanced.columns 
               if col not in ['price', 'target_return', 'next_price']]

# Remove any non-numeric columns
numeric_cols = df_enhanced[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < len(feature_cols):
    removed = set(feature_cols) - set(numeric_cols)
    print(f"   ⚠️  Removed {len(removed)} non-numeric columns: {removed}")
    feature_cols = numeric_cols

X = df_enhanced[feature_cols].values
y = df_enhanced['target_return'].values

print(f"   Total features: {len(feature_cols)}")
print(f"   Total samples: {len(X):,}")

# Train/test split
n_samples = len(X)
split_point = int(n_samples * 0.8)
X_train, y_train = X[:split_point], y[:split_point]
X_test, y_test = X[split_point:], y[split_point:]

print(f"   Training: {len(X_train):,} samples")
print(f"   Test: {len(X_test):,} samples")

# Step 6: Train RandomForest
print("\n🎯 Step 6: Training RandomForest...")
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=30,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', rf_model)
])

print("   Training (this will take a few minutes)...")
start = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - start
print(f"   ✅ Training completed in {train_time:.1f}s ({train_time/60:.2f} min)")

# Step 7: Extract importance
print("\n📈 Step 7: Extracting feature importance...")
importances = pipeline.named_steps['model'].feature_importances_

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

# Save
output_path = Path("MODEL_STORAGE/feature_data/feature_importance_extraction.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
importance_df.to_csv(output_path, index=False)
print(f"   ✅ Saved to: {output_path}")

# Step 8: Analyze
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

mean_imp = importance_df['importance'].mean()
median_imp = importance_df['importance'].median()

print(f"\n📊 STATISTICS:")
print(f"   Total features: {len(importance_df)}")
print(f"   Mean importance: {mean_imp:.6f}")
print(f"   Median importance: {median_imp:.6f}")
print(f"   Max: {importance_df['importance'].max():.6f}")
print(f"   Min: {importance_df['importance'].min():.6f}")

print(f"\n🏆 TOP 30 FEATURES:\n")
print(importance_df.head(30).to_string(index=False))

# Categorize
enhanced_keywords = ['hurst', 'regime', 'fractal', 'order_flow', 'microstructure', 
                     'dist_to', 'round_number', 'chaos', 'efficiency', 'illiquidity',
                     'spread', 'trade_intensity', 'percentile', 'acceleration', 
                     'parkinson', 'skew', 'kurtosis', 'dimension', 'pressure', 'imbalance']

top_30 = importance_df.head(30)
enhanced = [f for f in top_30['feature'].values 
           if any(kw in f.lower() for kw in enhanced_keywords)]
external = [f for f in top_30['feature'].values if f.startswith('ext_')]

print(f"\n📈 TOP 30 BREAKDOWN:")
print(f"   Base features: {30 - len(enhanced) - len(external)}")
print(f"   Enhanced features: {len(enhanced)}")
print(f"   External features: {len(external)}")

# Above median
above_median = importance_df[importance_df['importance'] >= median_imp]
print(f"\n🎯 FEATURES >= MEDIAN:")
print(f"   Count: {len(above_median)}/{len(importance_df)}")
print(f"   Percentage: {len(above_median)/len(importance_df)*100:.1f}%")

# Dominance
dominance = importance_df[importance_df['feature'].str.contains('dominance', case=False)]
if len(dominance) > 0:
    print(f"\n👑 DOMINANCE METRICS:")
    for _, row in dominance.iterrows():
        rank = list(importance_df['feature']).index(row['feature']) + 1
        print(f"   #{rank:3d}. {row['feature']}: {row['importance']:.6f}")

# Recommendation
print(f"\n💡 RECOMMENDATION:")
print(f"   Keep {len(above_median)} features with importance >= {median_imp:.6f}")
print(f"   Remove {len(importance_df) - len(above_median)} features ({(len(importance_df)-len(above_median))/len(importance_df)*100:.1f}%)")

# Save selected list
selected_features = above_median['feature'].tolist()
list_path = Path("MODEL_STORAGE/feature_data/selected_features.txt")
with open(list_path, 'w') as f:
    for feat in selected_features:
        f.write(f"{feat}\n")
print(f"   ✅ Selected features list saved: {list_path}")

print("\n" + "="*80)
print("✅ EXTRACTION COMPLETE")
print("="*80)
print(f"\nFiles created:")
print(f"  • {output_path} - Full importance scores")
print(f"  • {list_path} - List of {len(selected_features)} selected features")
print(f"\nNext: Train with selected features to improve performance!")
