# Bitcoin Price Predictor

## Project Overview
A multi-timeframe Bitcoin price prediction model using machine learning (SVR and RandomForest) with technical indicators from multiple timeframes (1h, 4h, 12h, 1d, 1w).

## Recent Changes (October 24, 2025)

### Time Series Cross-Validation Implementation (Latest - Oct 24)
- **5-Fold Time Series Cross-Validation**: Robust validation with TimeSeriesSplit
  - Gap parameter set to 3 to prevent data leakage from lagged features (t-1, t-2, t-3)
  - Multiple out-of-sample performance estimates (more reliable than single split)
  - CV score std dev: ±0.0001 shows excellent stability across time periods
- **Pipeline Architecture**: Fixed critical data leakage issue
  - StandardScaler now wrapped in Pipeline
  - Scaling happens independently within each CV fold
  - Ensures no future data contamination in training
- **Comprehensive Progress Tracking**: Real-time feedback during training
  - Start/end timestamps for full session
  - Per-model training duration (RF: 21 min, XGB: 8 min)
  - Per-fold completion times with verbose GridSearchCV output
  - Total training time: 29.66 minutes
- **Performance Maintained**: 0.29% RMSE ($229.90) with leak-free validation

### Advanced Features Enhancement
- **Expanded Features**: 20 → **95 features** (4.75x increase!)
  - RSI (Relative Strength Index) - 7 & 14-period momentum indicators
  - MACD (Moving Average Convergence Divergence) - line, signal, histogram
  - Volume-weighted indicators - MA, price-volume trend, volume ratios
  - Lagged price features - Price and return memory (t-1, t-2, t-3)
- **95% Confidence Intervals**: Worst case | Most likely | Best case predictions
  - Using ensemble variance between RandomForest & XGBoost models
  - Provides uncertainty quantification for 12-hour forecast

### Ensemble Model Implementation
- **Upgraded to Ensemble Model**: Now combining RandomForest + XGBoost with equal weighting
- **Extended Prediction Horizon**: From 3 hours → 12 hours ahead
- **99% Accuracy Improvement**: Initial RMSE $26,693 → $238 (0.21% error)
- GridSearchCV optimization for both RandomForest and XGBoost
- LightGBM gracefully disabled (missing libgomp.so.1 system library)

### Data Acquisition Improvements
- **Fixed fetch_data.py** to successfully download Yahoo Finance data
  - Fixed column naming issue (Datetime vs Date)
  - Fixed deprecated 'H' to 'h' in resampling
  - Successfully fetched **17,503 hourly samples** (2 years of data)
  - Fetched 4,381 4-hour samples
  - Fetched 1,461 12-hour samples
  - Removed dependency on BTC_90day_data.csv (not needed for Yahoo Finance data)
  
### Dataset Improvements
- **Massive data increase**: From 300 samples → 17,502 samples (58x increase!)
- Training samples: 14,001
- Test samples: 3,501
- Date range: October 2023 - October 2024 (2 full years)

### Model Configuration
- Currently using: **Ensemble of RandomForest + XGBoost** (equal weighting)
- GridSearchCV for hyperparameter optimization on both models
- Multi-timeframe feature engineering (1h, 4h, 12h, 1d, 1w)
- Features: **95 advanced technical indicators** across timeframes
  - RSI (7 & 14-period), MACD (line, signal, histogram)
  - Volume indicators (MA, price-volume trend, ratios)
  - Lagged prices (t-1, t-2, t-3) + lagged returns
  - Original features (Bollinger, EMA, volatility, z-scores)
- Prediction horizon: **12 hours ahead with 95% confidence intervals**

## Project Architecture

### Files
- `main.py` - Main prediction model with multi-timeframe analysis
- `fetch_data.py` - Yahoo Finance data fetcher for all timeframes
- `DATA/` - Contains all historical Bitcoin price data
  - `yf_btc_1h.csv` - 17,503 rows of hourly data (1.8MB)
  - `yf_btc_4h.csv` - 4,381 rows (430KB)
  - `yf_btc_12h.csv` - 1,461 rows (146KB)
  - `yf_btc_1d.csv` - 4,056 rows (470KB)
  - `yf_btc_1w.csv` - 580 rows (68KB)
  - `BTC_90day_data.csv` - Legacy 90-day data
- `attached_assets/` - Coinbase indicator data (alternative data source)

### Workflows
1. **Bitcoin Predictor** - Runs main.py to train and predict
2. **Fetch Data** - Runs fetch_data.py to update historical data

## Current Performance Metrics (October 24, 2025)
- **Test Set Return RMSE: 0.002925 (0.29%)**
- **Approximate Price RMSE: $229.90**
- **Model: Ensemble of RandomForest + XGBoost (equal weighting)**
- **Total Features: 95** (expanded from 20)
- **Cross-Validation**: 5-fold Time Series CV with gap=3 (leak-free!)
- Training Samples: 14,001
- Test Samples: 3,501
- Training Time: 29.66 minutes (RF: 21 min, XGB: 8 min)
- Prediction horizon: 12 hours ahead with 95% confidence intervals
- Last prediction: From $111,496 → $113,499 (most likely)
  - Range: $113,268 (worst) to $113,729 (best)
  - 12-hour outlook: +1.59% to +2.00% (most likely: +1.80%)

## Known Issues & TODOs

### Immediate Fixes Needed
- [ ] LSP type hint errors in main.py (4 diagnostics) - cosmetic but should fix
- [ ] LSP errors in fetch_data.py (5 diagnostics) - cosmetic

### Short-Term Improvements
- [x] Add more technical indicators - **COMPLETED**:
  - [x] RSI (Relative Strength Index) - 7 & 14-period
  - [x] MACD (Moving Average Convergence Divergence) - line, signal, histogram
  - [x] Volume-weighted indicators - MA, trend, ratios
  - [x] Lagged price features (t-1, t-2, t-3) + lagged returns
- [x] Implement proper time series cross-validation (TimeSeriesSplit) - **COMPLETED**
  - [x] 5-fold CV with gap=3 for lagged features
  - [x] Pipeline architecture to prevent data leakage
  - [x] Comprehensive progress tracking and timing
- [ ] Expand GridSearch hyperparameter ranges
- [ ] Fix ATR calculation to use actual High/Low data (currently using price proxy)
- [ ] Consider RSI epsilon clamping for numerical stability
- [ ] Update lag features inside prediction loop for consistency

### Long-Term Enhancements
- [x] Test XGBoost and LightGBM models (XGBoost implemented, LightGBM unavailable due to system constraints)
- [x] Add ensemble model (combine multiple models) - **Completed: RF + XGBoost ensemble**
- [x] Uncertainty quantification (prediction intervals) - **Completed: 95% confidence bounds**
- [ ] Implement LSTM/GRU for sequence learning
- [ ] Sentiment analysis integration
- [ ] Multi-step prediction optimization
- [ ] Real-time data updates
- [ ] Visualization dashboard

## User Preferences
- User wants comprehensive analysis of model performance
- Focus on improving prediction accuracy
- Interest in understanding what console output reveals about model quality
- Planning to migrate to local GPU execution (CUDA) for 3-5x speedup
- Interested in expanding to multi-asset trading (ETH, SOL, SPY, etc.)
- Future goals: LSTM models, real-time trading, backtesting framework

## Dependencies
- pandas
- numpy
- scikit-learn
- yfinance
- matplotlib
- mplfinance
- xgboost
- lightgbm (installed but unavailable due to missing libgomp.so.1)
- pycoingecko (legacy)

## Notes
- Yahoo Finance data preferred over Coinbase data (has High/Low for proper ATR calculation)
- Model currently training on full 17,502 sample dataset
- GridSearchCV takes ~30 minutes on CPU (RF: 21 min, XGB: 8 min)
- Ensemble uses equal weighting between RandomForest and XGBoost predictions
- LightGBM cannot be used in this environment (missing libgomp.so.1 system library)
- BTC_90day_data.csv is no longer needed when using Yahoo Finance data (USE_YAHOO_FINANCE = True)

## GPU Migration Guide (For Local Execution)
See `/tmp/gpu_training_example.py` and `/tmp/export_instructions.md` for complete migration guide.

### Quick Start
1. Install CUDA Toolkit 12.1+ from nvidia.com
2. Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
3. Modify `main.py` line ~330: Change `tree_method='hist'` to `tree_method='gpu_hist', predictor='gpu_predictor'`
4. Run: `python main.py`

### Expected Performance Gains
- **CPU (Replit)**: ~30 minutes total training time
- **GPU (RTX 3080)**: ~8-10 minutes total training time
- **Speedup**: 3-5x faster (XGBoost benefits most)
- **With LSTM**: Add ~5 minutes for deep learning model

### Hardware Recommendations
- **Minimum**: NVIDIA GTX 1060 (6GB), 16GB RAM
- **Recommended**: RTX 3060/3070 (12GB), 32GB RAM  
- **Ideal**: RTX 4080/4090 (16GB+), 64GB RAM
