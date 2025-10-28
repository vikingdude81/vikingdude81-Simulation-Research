"""
Analyze Geometric MA Features Impact
Quick analysis to see if GMA features improve the model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)

print("\n" + "="*70)
print("🔍 ANALYZING GEOMETRIC MA FEATURES")
print("="*70)

# Load recent training data
try:
    df = pd.read_csv('DATA/yf_btc_1h.csv')
    df['time'] = pd.to_datetime(df['time'])
    
    print(f"\n✅ Loaded {len(df)} rows")
    
    # Add enhanced features
    from enhanced_features import add_all_enhanced_features
    df = add_all_enhanced_features(df.copy())
    
    # Get GMA features
    gma_features = [col for col in df.columns if 'gma' in col.lower()]
    print(f"\n📊 Found {len(gma_features)} GMA features:")
    for feat in gma_features:
        print(f"   • {feat}")
    
    # Clean data
    df = df.dropna()
    df['target_return'] = df['close'].pct_change().shift(-1)
    df = df.dropna()
    
    print(f"\n🧹 Clean dataset: {len(df)} rows")
    
    # Prepare features (use last 1000 rows for speed)
    df_recent = df.tail(1000).copy()
    
    # Test 1: Model with GMA features
    all_features = [col for col in df_recent.columns if col not in ['time', 'target_return', 'close', 'open', 'high', 'low', 'volume', 'Dividends', 'Stock Splits', 'timestamp', 'price', 'next_price']]
    X_with_gma = df_recent[all_features]
    y = df_recent['target_return']
    
    X_train_gma, X_test_gma, y_train, y_test = train_test_split(
        X_with_gma, y, test_size=0.2, shuffle=False
    )
    
    print(f"\n📈 Training with GMA features...")
    print(f"   Features: {X_train_gma.shape[1]}")
    
    rf_with_gma = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_with_gma.fit(X_train_gma, y_train)
    score_with_gma = rf_with_gma.score(X_test_gma, y_test)
    
    print(f"   R² Score: {score_with_gma:.6f}")
    
    # Test 2: Model WITHOUT GMA features
    non_gma_features = [col for col in all_features if 'gma' not in col.lower()]
    X_without_gma = df_recent[non_gma_features]
    
    X_train_no_gma = X_without_gma.loc[X_train_gma.index]
    X_test_no_gma = X_without_gma.loc[X_test_gma.index]
    
    print(f"\n📉 Training without GMA features...")
    print(f"   Features: {X_train_no_gma.shape[1]}")
    
    rf_without_gma = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_without_gma.fit(X_train_no_gma, y_train)
    score_without_gma = rf_without_gma.score(X_test_no_gma, y_test)
    
    print(f"   R² Score: {score_without_gma:.6f}")
    
    # Compare
    print("\n" + "="*70)
    print("📊 COMPARISON")
    print("="*70)
    print(f"Without GMA: R² = {score_without_gma:.6f}")
    print(f"With GMA:    R² = {score_with_gma:.6f}")
    
    improvement = score_with_gma - score_without_gma
    pct_improvement = (improvement / abs(score_without_gma)) * 100
    
    if improvement > 0:
        print(f"\n✅ IMPROVEMENT: +{improvement:.6f} ({pct_improvement:+.2f}%)")
        print("   GMA features are helping!")
    else:
        print(f"\n⚠️  DECLINE: {improvement:.6f} ({pct_improvement:+.2f}%)")
        print("   GMA features not improving model")
    
    # Top GMA feature importance
    print("\n📊 Top 10 GMA Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': X_train_gma.columns,
        'importance': rf_with_gma.feature_importances_
    }).sort_values('importance', ascending=False)
    
    gma_importance = feature_importance[feature_importance['feature'].str.contains('gma', case=False)].head(10)
    for idx, row in gma_importance.iterrows():
        print(f"   {row['feature']:30s} {row['importance']:.6f}")
    
    # Overall feature ranking
    gma_rank = feature_importance[feature_importance['feature'].str.contains('gma', case=False)].iloc[0]
    overall_rank = feature_importance[feature_importance['feature'] == gma_rank['feature']].index[0] + 1
    
    print(f"\n🏆 Best GMA feature: {gma_rank['feature']}")
    print(f"   Importance: {gma_rank['importance']:.6f}")
    print(f"   Overall rank: #{overall_rank} of {len(feature_importance)}")
    
    print("\n" + "="*70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
