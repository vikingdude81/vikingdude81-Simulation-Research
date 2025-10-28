# Run 4 - Feature Selection Training
**50% Feature Reduction: 68 → 34 Features**
*Started: October 25, 2025 - 12:11 PM*

---

## 🎯 Objective

Beat Phase 4 baseline (0.66% RMSE) by using only the most important features identified through RandomForest feature importance analysis.

---

## Feature Selection Results

### Extraction Summary
- **Method**: Trained RandomForest on all 68 features
- **Training time**: 23 seconds
- **Threshold**: Median importance (0.013599)
- **Features kept**: 34 (50% reduction)
- **Features removed**: 34

### Top 10 Most Important Features
1. **returns** (0.047) - Base feature
2. **spread_proxy** (0.045) - Enhanced microstructure
3. **dist_to_high_168h** (0.042) - Enhanced price levels
4. **volatility_acceleration** (0.042) - Enhanced volatility regime
5. **dist_to_high_24h** (0.040) - Enhanced price levels
6. **trend_strength** (0.032) - Enhanced market regime
7. **cumulative_order_flow** (0.030) - Enhanced order flow
8. **round_10k_dist** (0.030) - Enhanced price levels
9. **volume_imbalance_ma** (0.028) - Enhanced order flow
10. **hurst_48h** (0.028) - Enhanced fractal/chaos

### Feature Category Breakdown

**In Top 30 Features:**
- Base features: 6 (20%)
- Enhanced features: 24 (80%) ⭐
- External features: 0 (0%)

**Removed Features:**
- **All external data features** (17 features) - Zero predictive value!
  - ext_fear_greed, ext_google_trends, ext_social_sentiment
  - ext_btc_dominance, ext_usdt_dominance, ext_eth_dominance
  - ext_trading_volume, ext_market_cap, etc.
- **Low-importance base features** (some technical indicators)
- **Low-importance enhanced features** (noisy calculations)

---

## Key Insights

### ✅ Enhanced Features Validated!
- **24 of top 30** features are enhanced features we added in Phase 5
- Microstructure, price levels, volatility regime, and order flow features dominate
- Hurst exponents (fractal/chaos) made it into top 10

### ❌ External Data Failed
- **All 17 external features** ranked in bottom half (below median)
- Dominance metrics (BTC.D, USDT.D, ETH.D): Ranked #64-66 with 0.000000 importance
- Fear & Greed, Google Trends, social sentiment: No predictive value
- **Lesson**: External data adds noise, not signal

### 🎯 Feature Engineering Success
Enhanced features that matter most:
- **Price levels**: Distance to recent highs/lows, round number proximity
- **Microstructure**: Spread, efficiency, illiquidity
- **Order flow**: Cumulative flow, volume/order imbalance
- **Volatility regime**: Acceleration, percentile
- **Fractal/chaos**: Hurst exponents (48h especially)

---

## Run 4 Configuration

### Training Setup
- **Features**: 34 (down from 68 in Run 2/3)
- **Feature selection**: Enabled (USE_FEATURE_SELECTION = True)
- **Models**: 6 (RF, XGB, LGB, LSTM, Transformer, MultiTask)
- **Device**: CUDA (NVIDIA RTX 4070 Ti)
- **CV Folds**: 5

### Expected Improvements
1. **Better RMSE**: Target <0.66% (beat Phase 4)
2. **Faster training**: Fewer features = faster computation
3. **Less overfitting**: Removed noisy features
4. **Cleaner ensemble**: Better model contributions

---

## Comparison: Run 2 vs Run 4

| Aspect | Run 2 | Run 4 (Expected) |
|--------|-------|------------------|
| Features | 68 | 34 (-50%) |
| RMSE | 0.74% | <0.66% 🎯 |
| Price Error | $597 | <$520 |
| Training Time | ~20 min | ~15 min |
| External Data | 17 features | 0 features |
| Enhanced Used | All 46 | Best 24 |

---

## Training Status

### Phase 1: Data Loading ✅
- Multi-timeframe data loaded (0.1s)
- External data collected (0.7s) - but will be removed
- Feature engineering complete (0.1s)

### Phase 2: Enhanced Features ⏳ IN PROGRESS
- Currently adding fractal & chaos features
- Expected: ~45s total

### Phase 3: Feature Selection ⏳ PENDING
- Load selected_features.txt
- Filter to 34 features
- Remove all external data

### Phase 4: Model Training ⏳ PENDING
- RandomForest, XGBoost, LightGBM
- LSTM, Transformer, MultiTask
- Expected: ~12-15 min

### Phase 5: Results ⏳ PENDING
- Ensemble predictions
- Performance metrics
- Storage save

---

## Success Criteria

### Primary Goal
✅ **RMSE < 0.66%** - Beat Phase 4 baseline

### Secondary Goals
- Training time < 15 min (vs Run 2's 20 min)
- All neural networks train successfully
- Price error < $520
- Cleaner feature space with no external noise

---

## Next Steps After Completion

1. **Analyze Results**
   - Compare Run 4 vs Run 2 vs Phase 4
   - Validate RMSE improvement
   - Check individual model performance

2. **Document Findings**
   - Create Run 4 results document
   - Update Phase 5 summary
   - Feature selection lessons learned

3. **Decide Next Phase**
   - If successful (<0.66%): Consider Tier 2 enhancements (stacking)
   - If not successful: Adjust threshold, try top 50-60 features
   - Alternative: Dynamic ensemble weighting

---

## Feature List (34 Selected)

Will be confirmed after training completes. Expected to include:
- **Base**: returns, volume indicators, key technical indicators
- **Enhanced**: 
  - Microstructure: spread_proxy, amihud_illiquidity, roll_spread
  - Price levels: dist_to_high/low (24h/168h), round number distances
  - Order flow: cumulative_order_flow, volume_imbalance, order_imbalance
  - Volatility: volatility_acceleration, parkinson_vol, volatility_percentile
  - Fractal: hurst_48h, chaos_indicator, fractal_dimension
  - Market regime: trend_strength, adx_proxy, regime_duration

---

*Training in progress... Expected completion: ~12:26 PM*
*This document will be updated with final results*
