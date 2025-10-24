# Bitcoin Price Predictor

## Project Overview
A multi-timeframe Bitcoin price prediction model using machine learning (SVR and RandomForest) with technical indicators from multiple timeframes (1h, 4h, 12h, 1d, 1w).

## Recent Changes (October 24, 2025)

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
- Features: 20 technical indicators across timeframes
- Prediction horizon: **12 hours ahead**

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
- **Test Set Return RMSE: 0.003034 (0.30%)**
- **Approximate Price RMSE: $238.40**
- **Model: Ensemble of RandomForest + XGBoost (equal weighting)**
- Training Samples: 14,001
- Test Samples: 3,501
- Prediction horizon: 12 hours ahead
- Last prediction: From $111,496 → $113,248 (12-hour forecast)

## Known Issues & TODOs

### Immediate Fixes Needed
- [ ] LSP type hint errors in main.py (4 diagnostics) - cosmetic but should fix
- [ ] LSP errors in fetch_data.py (5 diagnostics) - cosmetic

### Short-Term Improvements
- [ ] Add more technical indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Volume-weighted indicators
  - Lagged price features (t-1, t-2, t-3)
- [ ] Implement proper time series cross-validation (TimeSeriesSplit)
- [ ] Add validation set monitoring
- [ ] Expand GridSearch hyperparameter ranges
- [ ] Fix ATR calculation to use actual High/Low data (currently using price proxy)

### Long-Term Enhancements
- [x] Test XGBoost and LightGBM models (XGBoost implemented, LightGBM unavailable due to system constraints)
- [x] Add ensemble model (combine multiple models) - **Completed: RF + XGBoost ensemble**
- [ ] Implement LSTM/GRU for sequence learning
- [ ] Sentiment analysis integration
- [ ] Multi-step prediction optimization
- [ ] Uncertainty quantification (prediction intervals)
- [ ] Real-time data updates
- [ ] Visualization dashboard

## User Preferences
- User wants comprehensive analysis of model performance
- Focus on improving prediction accuracy
- Interest in understanding what console output reveals about model quality

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
- GridSearchCV may take several minutes with large dataset (2-3 minutes per ensemble training)
- Ensemble uses equal weighting between RandomForest and XGBoost predictions
- LightGBM cannot be used in this environment (missing libgomp.so.1 system library)
- BTC_90day_data.csv is no longer needed when using Yahoo Finance data (USE_YAHOO_FINANCE = True)
