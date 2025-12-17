# Multiscale Dynamics Experiments

Experiments based on arXiv:2512.12462 - Multiscale framework for market analysis.

## Overview

This directory contains experiments for testing and validating the multiscale market encoder across different timeframes (1H, 4H, 12H, 1D, 1W).

## Experiments

### 1. Timeframe Encoder (`exp1_timeframe_encoder.py`)
**Goal**: Test encoder on BTC/ETH/SOL across all timeframes

Tests the `MultiscaleMarketEncoder` by:
- Loading historical data for multiple assets
- Encoding data from all timeframes simultaneously
- Comparing performance to separate single-timeframe models
- Measuring prediction accuracy improvement

**Success Metrics**:
- Prediction accuracy +10-15% vs current system
- Effective feature extraction across all timeframes

### 2. Missing Data Trading (`exp2_missing_data_trading.py`)
**Goal**: Simulate exchange outages and test robustness

Simulates real-world scenarios with:
- 10%, 20%, 30% data missing randomly
- Exchange outage patterns (consecutive missing periods)
- Performance degradation analysis

**Success Metrics**:
- Handle 30% missing data with <5% performance loss
- Robust trading decisions with incomplete information

### 3. Real-time Regime Detection (`exp3_realtime_regime_detection.py`)
**Goal**: Real-time regime shifts across timeframes

Tests integration with regime detection by:
- Detecting regime changes in real-time
- Multi-timeframe regime alignment
- Latency benchmarking

**Success Metrics**:
- Real-time regime detection <500ms latency
- Accurate regime identification across timeframes

### 4. Nonlinear Price Dynamics (`exp4_nonlinear_price_dynamics.py`)
**Goal**: Nonlinear temporal dynamics for price prediction

Explores nonlinear modeling:
- Compare linear vs nonlinear aggregation
- Test different activation functions
- Regime-specific nonlinearity optimization

**Success Metrics**:
- Outperform linear models (LSTM baseline)
- Identify optimal nonlinearity per regime

## Usage

Run individual experiments:
```bash
python multiscale_experiments/exp1_timeframe_encoder.py
python multiscale_experiments/exp2_missing_data_trading.py
python multiscale_experiments/exp3_realtime_regime_detection.py
python multiscale_experiments/exp4_nonlinear_price_dynamics.py
```

Or through the ML models menu (option 19).

## Dependencies

- PyTorch >= 2.0.0
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (for visualization)

## Results

Results are saved to:
- `outputs/multiscale_experiments/` - Experiment outputs
- `MODEL_STORAGE/multiscale_models/` - Trained models
- Logs printed to console during execution

## Integration

These experiments integrate with:
- `models/multiscale_predictor.py` - Core encoder
- `utils/multiscale_utils.py` - Helper functions
- `regime_detector.py` - Regime detection system
- `train_models.py` - Training pipeline
