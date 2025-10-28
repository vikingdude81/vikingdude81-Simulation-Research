"""
Extract feature importance from a trained RandomForest model
This script can extract importance from either saved models or by re-running a quick training
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_from_saved_model(run_id='run_20251025_104443'):
    """Try to extract feature importance from a saved model"""
    model_dir = Path("MODEL_STORAGE/saved_models")
    
    # Look for RandomForest model (saved as .pkl)
    rf_files = list(model_dir.glob(f"{run_id}*randomforest*.pkl"))
    rf_files.extend(list(model_dir.glob(f"{run_id}*rf*.pkl")))
    
    if rf_files:
        logging.info(f"Found RF model: {rf_files[0]}")
        with open(rf_files[0], 'rb') as f:
            rf_pipeline = pickle.load(f)
        
        # Extract the model from pipeline
        if hasattr(rf_pipeline, 'named_steps'):
            rf_model = rf_pipeline.named_steps['model']
        else:
            rf_model = rf_pipeline
        
        # Get feature importances
        importances = rf_model.feature_importances_
        
        # We need feature names - load from metadata or reconstruct
        logging.info(f"Found {len(importances)} feature importances")
        
        # Try to get feature names from training data
        metadata_path = Path(f"MODEL_STORAGE/training_runs/{run_id}/metadata.json")
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            n_features = metadata['config']['n_features']
            logging.info(f"Expected {n_features} features from metadata")
        
        return importances
    else:
        logging.warning(f"No RF model found for {run_id}")
        return None

def extract_by_quick_training():
    """
    Run a quick training to get feature importance
    This loads the data and trains just RandomForest to get importance
    """
    logging.info("\n" + "="*80)
    logging.info("QUICK FEATURE IMPORTANCE EXTRACTION")
    logging.info("="*80)
    
    # Import necessary modules
    import sys
    sys.path.insert(0, '.')
    from enhanced_features import add_all_enhanced_features
    from external_data import ExternalDataCollector
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import ta  # Technical analysis library
    
    logging.info("\n📊 Loading data...")
    
    # Load all timeframes
    timeframes = {
        '1h': 'DATA/yf_btc_1h.csv',
        '4h': 'DATA/yf_btc_4h.csv',
        '12h': 'DATA/yf_btc_12h.csv',
        '1d': 'DATA/yf_btc_1d.csv',
        '1w': 'DATA/yf_btc_1w.csv'
    }
    
    dfs = {}
    for tf, path in timeframes.items():
        try:
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            dfs[tf] = df
            logging.info(f"   ✅ {tf}: {len(df)} rows")
        except Exception as e:
            logging.error(f"   ❌ Failed to load {tf}: {e}")
            return None
    
    # Use 1h as base
    combined_df = dfs['1h'].copy()
    
    # Add base features
    logging.info("\n🔧 Adding base features...")
    combined_df = add_advanced_features(combined_df, dfs)
    base_features = len([col for col in combined_df.columns if col not in ['price', 'target_return', 'next_price']])
    logging.info(f"   Base features: {base_features}")
    
    # Add enhanced features
    logging.info("\n✨ Adding enhanced features...")
    combined_df = add_all_enhanced_features(combined_df)
    after_enhanced = len([col for col in combined_df.columns if col not in ['price', 'target_return', 'next_price']])
    logging.info(f"   After enhanced: {after_enhanced} (+{after_enhanced - base_features})")
    
    # Clean NaN
    rows_before = len(combined_df)
    combined_df = combined_df.dropna()
    rows_after = len(combined_df)
    logging.info(f"\n🧹 Dropped {rows_before - rows_after} rows with NaN")
    
    # Add external data
    logging.info("\n🌍 Collecting external data...")
    collector = ExternalDataCollector()
    external_data = collector.collect_all()
    for key, value in external_data.items():
        combined_df[f'ext_{key}'] = value
    
    total_features = len([col for col in combined_df.columns if col not in ['price', 'target_return', 'next_price']])
    logging.info(f"   Total features: {total_features}")
    
    # Prepare data
    feature_cols = [col for col in combined_df.columns if col not in ['price', 'target_return', 'next_price']]
    X = combined_df[feature_cols].values
    y = combined_df['target_return'].values
    
    # Train/test split
    n_samples = len(X)
    split_point = int(n_samples * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    logging.info(f"\n🎯 Training RandomForest for feature importance...")
    logging.info(f"   Training samples: {len(X_train):,}")
    logging.info(f"   Test samples: {len(X_test):,}")
    logging.info(f"   Features: {len(feature_cols)}")
    
    # Train RandomForest with best params from Run 2
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
    
    logging.info("   Training...")
    pipeline.fit(X_train, y_train)
    
    # Get feature importance
    importances = pipeline.named_steps['model'].feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Save
    output_path = Path("MODEL_STORAGE/feature_data/quick_feature_importance.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False)
    
    logging.info(f"\n✅ Feature importance saved to: {output_path}")
    
    return importance_df

def analyze_importance(importance_df):
    """Analyze and display feature importance"""
    if importance_df is None:
        logging.error("No importance data to analyze")
        return
    
    logging.info("\n" + "="*80)
    logging.info("FEATURE IMPORTANCE ANALYSIS")
    logging.info("="*80)
    
    # Statistics
    mean_imp = importance_df['importance'].mean()
    median_imp = importance_df['importance'].median()
    
    print(f"\n📊 STATISTICS:")
    print(f"   Total features: {len(importance_df)}")
    print(f"   Mean importance: {mean_imp:.6f}")
    print(f"   Median importance: {median_imp:.6f}")
    print(f"   Max importance: {importance_df['importance'].max():.6f}")
    print(f"   Min importance: {importance_df['importance'].min():.6f}")
    
    # Top 30 features
    print(f"\n🏆 TOP 30 FEATURES:\n")
    print(importance_df.head(30).to_string(index=False))
    
    # Feature categories
    enhanced_keywords = ['hurst', 'regime', 'fractal', 'order_flow', 'microstructure', 
                         'dist_to', 'round_number', 'chaos', 'efficiency', 'illiquidity',
                         'spread_proxy', 'roll_spread', 'trade_intensity', 'percentile',
                         'acceleration', 'parkinson', 'skew', 'kurtosis', 'dimension',
                         'buy_pressure', 'sell_pressure', 'imbalance', 'cumulative_order']
    
    top_30 = importance_df.head(30)
    enhanced_in_top30 = [f for f in top_30['feature'].values 
                         if any(keyword in f.lower() for keyword in enhanced_keywords)]
    external_in_top30 = [f for f in top_30['feature'].values if f.startswith('ext_')]
    
    print(f"\n📈 TOP 30 BREAKDOWN:")
    print(f"   Enhanced features: {len(enhanced_in_top30)}")
    print(f"   External features: {len(external_in_top30)}")
    print(f"   Base features: {30 - len(enhanced_in_top30) - len(external_in_top30)}")
    
    # Features above median
    above_median = importance_df[importance_df['importance'] >= median_imp]
    print(f"\n🎯 FEATURES ABOVE MEDIAN IMPORTANCE:")
    print(f"   Count: {len(above_median)}/{len(importance_df)}")
    print(f"   Percentage: {len(above_median)/len(importance_df)*100:.1f}%")
    
    # Dominance metrics
    dominance_features = importance_df[importance_df['feature'].str.contains('dominance', case=False)]
    if len(dominance_features) > 0:
        print(f"\n👑 DOMINANCE METRICS:")
        for _, row in dominance_features.iterrows():
            rank = importance_df[importance_df['feature'] == row['feature']].index[0] + 1
            print(f"   • {row['feature']}: Rank #{rank}, Importance: {row['importance']:.6f}")
    
    # Recommendation
    print(f"\n💡 FEATURE SELECTION RECOMMENDATION:")
    print(f"   Keep features with importance >= {median_imp:.6f} (median)")
    print(f"   This would reduce features from {len(importance_df)} to {len(above_median)}")
    print(f"   Reduction: {len(importance_df) - len(above_median)} features ({(len(importance_df) - len(above_median))/len(importance_df)*100:.1f}%)")
    
    # Save selected features list
    selected_features = above_median['feature'].tolist()
    output_path = Path("MODEL_STORAGE/feature_data/selected_features_list.txt")
    with open(output_path, 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    
    print(f"\n✅ Selected features list saved to: {output_path}")
    
    return selected_features

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE EXTRACTION & ANALYSIS")
    print("="*80)
    
    # Try method 1: Extract from saved model
    print("\nMethod 1: Trying to extract from saved model...")
    importances = extract_from_saved_model()
    
    if importances is None:
        # Method 2: Quick training
        print("\nMethod 2: Running quick training to get importance...")
        importance_df = extract_by_quick_training()
    else:
        print("\n⚠️  Found importances but need feature names.")
        print("   Running quick training to get complete data...")
        importance_df = extract_by_quick_training()
    
    # Analyze
    if importance_df is not None:
        selected_features = analyze_importance(importance_df)
        
        print("\n" + "="*80)
        print("✅ EXTRACTION COMPLETE")
        print("="*80)
        print(f"\nNext steps:")
        print(f"1. Review top 30 features above")
        print(f"2. Selected features list saved ({len(selected_features)} features)")
        print(f"3. Ready to train with selected features (Run 4)")
    else:
        print("\n❌ Failed to extract feature importance")
