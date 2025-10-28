# Feature Selection Implementation Plan
**Goal: Reduce from 156 to ~80-100 features to beat 0.66% baseline**
*Created: October 25, 2025 - 12:00 PM*

---

## Current Status

### Step 1: Feature Importance Extraction ⏳ IN PROGRESS
- **Script**: `quick_feature_importance.py`  
- **Status**: Running (currently on fractal features calculation)
- **Expected completion**: 2-3 minutes
- **Output**: 
  - `MODEL_STORAGE/feature_data/feature_importance_extraction.csv` - Full importance scores
  - `MODEL_STORAGE/feature_data/selected_features.txt` - List of features >= median importance

---

## Implementation Steps

### ✅ Step 1: Extract Feature Importance (IN PROGRESS)
- Train RandomForest on current 156-feature dataset
- Extract feature importance scores for all features
- Calculate median importance threshold
- Identify features above median (~78-80 features expected)
- Save selected features list

### Step 2: Modify main.py for Feature Selection (NEXT)
Two options:

**Option A: Feature Selection Function** (Recommended - Flexible)
- Add `select_features()` function to main.py
- Read selected features list from file
- Filter dataframe to keep only selected features
- Easy to switch between full/selected feature sets
- Can test multiple thresholds quickly

**Option B: Command Line Flag** (More user-friendly)
- Add `--feature-selection` argument
- If enabled, load selected features list
- Filter features before training
- Allows easy A/B testing

### Step 3: Run Training with Selected Features (Run 4)
- Execute: `python main.py` (with feature selection enabled)
- Expected improvements:
  - Reduced training time (fewer features)
  - Better RMSE (less noise)
  - Cleaner feature space
- Target: <0.66% RMSE (beat Phase 4 baseline)

### Step 4: Analyze Results & Compare
- Compare Run 4 vs Run 2 vs Phase 4
- Validate which features were removed
- Check if dominance metrics were kept
- Document improvements

---

## Expected Feature Breakdown

### Current (156 features):
- **Base (Phase 4)**: 95 features
- **Enhanced**: 46 features
- **External**: 15 features

### After Selection (~78-80 features, 50%):
- **Base**: ~50-55 features (most valuable technical indicators)
- **Enhanced**: ~20-25 features (best microstructure, regime, order flow)
- **External**: ~5-8 features (highest-signal external data)

---

## Feature Selection Criteria

### Keep (Above Median Importance):
1. **Strong technical indicators**: Volume, volatility, momentum that rank high
2. **Valuable enhanced features**: Hurst exponents if ranked high, key microstructure
3. **Signal-rich external**: Dominance metrics (BTC.D/USDT.D), Fear & Greed if valuable
4. **Regime indicators**: If they provide predictive power

### Remove (Below Median Importance):
1. **Noisy features**: High variance, low predictive power
2. **Redundant features**: Highly correlated with kept features
3. **Low-signal external**: Google Trends, social sentiment if not predictive
4. **Complex calculations**: Fractal/chaos features if not adding value

---

## Code Changes Required

### Main.py Modifications:

```python
# Add at top of file
FEATURE_SELECTION = True  # Set to True to use selected features
SELECTED_FEATURES_PATH = 'MODEL_STORAGE/feature_data/selected_features.txt'

def load_selected_features():
    """Load list of selected features from file"""
    if not Path(SELECTED_FEATURES_PATH).exists():
        logging.warning(f"Selected features file not found: {SELECTED_FEATURES_PATH}")
        return None
    
    with open(SELECTED_FEATURES_PATH, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    
    logging.info(f"📋 Loaded {len(features)} selected features from {SELECTED_FEATURES_PATH}")
    return features

def apply_feature_selection(df, selected_features):
    """Filter dataframe to keep only selected features"""
    # Keep non-feature columns
    keep_cols = ['price', 'target_return', 'next_price']
    
    # Add selected features that exist in dataframe
    available_features = [f for f in selected_features if f in df.columns]
    missing_features = [f for f in selected_features if f not in df.columns]
    
    if missing_features:
        logging.warning(f"⚠️  {len(missing_features)} selected features not in dataframe")
    
    # Filter dataframe
    final_cols = keep_cols + available_features
    df_filtered = df[final_cols]
    
    logging.info(f"✂️  Feature selection applied:")
    logging.info(f"   Original features: {len(df.columns) - len(keep_cols)}")
    logging.info(f"   Selected features: {len(available_features)}")
    logging.info(f"   Reduction: {len(df.columns) - len(keep_cols) - len(available_features)} features removed")
    
    return df_filtered

# In main function, after NaN cleanup:
if FEATURE_SELECTION:
    selected_features = load_selected_features()
    if selected_features:
        combined_df = apply_feature_selection(combined_df, selected_features)
```

---

## Success Criteria

### Run 4 (Feature Selection) Goals:
1. **RMSE < 0.66%** - Beat Phase 4 baseline ✅ PRIMARY GOAL
2. **Training time < 15 min** - Faster than Run 2's ~20 min
3. **All models train successfully** - No NaN issues
4. **Feature count 75-85** - Roughly half of 156

### Quality Metrics:
- Individual model RMSEs should improve (LightGBM, XGBoost, RF)
- Neural networks should contribute positively to ensemble
- Ensemble weighting should be more balanced
- Predictions should have tighter confidence intervals

---

## Risk Mitigation

### Risk 1: Removing too many features
- **Mitigation**: Start with median threshold (50%), can adjust
- **Fallback**: Keep top 100 features instead of top 80

### Risk 2: Removing valuable features
- **Mitigation**: Review top 30 before running
- **Validation**: Check dominance metrics are included

### Risk 3: Still not beating 0.66%
- **Plan B**: Try different threshold (top 60%, top 70%)
- **Plan C**: Implement dynamic ensemble weighting (Tier 1)
- **Plan D**: Move to Tier 2 (stacking)

---

## Timeline

| Step | Duration | Status |
|------|----------|--------|
| 1. Extract importance | 3-5 min | ⏳ IN PROGRESS |
| 2. Modify main.py | 10 min | ⏳ PENDING |
| 3. Run training (Run 4) | ~15 min | ⏳ PENDING |
| 4. Analyze results | 5 min | ⏳ PENDING |
| **Total** | **~35 min** | **30% Complete** |

---

## Next Immediate Action

**After feature extraction completes:**
1. Review top 30 features
2. Verify dominance metrics (BTC.D, USDT.D) are in top 100
3. Check how many enhanced features made the cut
4. Modify main.py with feature selection code
5. Run training (Run 4)

---

*This plan will be updated as steps complete*
