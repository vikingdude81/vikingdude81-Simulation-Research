# Spiking Neural Network Trading Experiments

Experiments based on arXiv:2512.11743 - CogniSNN framework for efficient edge AI trading.

## Overview

This directory contains experiments for testing spiking neural networks (SNNs) in trading applications, including pathway reuse, dynamic growth, and neuromorphic deployment.

## Experiments

### 1. SNN Price Prediction (`exp1_snn_price_prediction.py`)
**Goal**: Replace LSTM with SNN for price forecasting

Tests the `SpikingTradingAgent` by:
- Encoding price data as spike rates
- Training SNN for next-price prediction
- Comparing accuracy to existing LSTM models

**Success Metrics**:
- SNN matches or exceeds LSTM accuracy
- Efficient spike-based encoding of market data

### 2. Pathway Reuse Multi-Asset (`exp2_pathway_reuse_multiasset.py`)
**Goal**: Train on BTC, reuse pathways for ETH/SOL

Demonstrates pathway-based learning without forgetting:
- Train SNN on BTC data
- Reuse learned pathways for ETH and SOL
- Measure transfer learning efficiency

**Success Metrics**:
- Pathway reuse enables 50%+ faster multi-asset training
- No catastrophic forgetting across assets

### 3. Dynamic Growth Adaptation (`exp3_dynamic_growth_adaptation.py`)
**Goal**: Start small, grow network based on complexity

Tests dynamic network growth:
- Initialize with 100 neurons
- Add pathways when performance plateaus
- Track accuracy during growth

**Success Metrics**:
- Dynamic growth improves online learning by 20%+
- Efficient adaptation to market regime changes

### 4. Neuromorphic Trading Bot (`exp4_neuromorphic_trading_bot.py`)
**Goal**: Deploy SNN to neuromorphic hardware

Simulates neuromorphic deployment:
- Measure power consumption vs GPU
- Real-time trading latency benchmarks
- Hardware compatibility analysis

**Success Metrics**:
- Neuromorphic deployment uses <5W power (vs 150W GPU)
- Maintains real-time performance

## Usage

Run individual experiments:
```bash
python snn_trading_experiments/exp1_snn_price_prediction.py
python snn_trading_experiments/exp2_pathway_reuse_multiasset.py
python snn_trading_experiments/exp3_dynamic_growth_adaptation.py
python snn_trading_experiments/exp4_neuromorphic_trading_bot.py
```

Or through the ML models menu (option 20).

## Dependencies

- PyTorch >= 2.0.0
- Norse >= 1.0.0 (SNN library)
- NumPy
- Pandas

## Results

Results are saved to:
- `outputs/snn_trading_experiments/` - Experiment outputs
- `MODEL_STORAGE/snn_models/` - Trained SNN models
- Logs printed to console during execution

## Integration

These experiments integrate with:
- `models/snn_trading_agent.py` - Core SNN agent
- `utils/snn_utils.py` - Spike encoding/decoding
- `ga_trading_agents/` - Genetic algorithm integration
- `snn_trading_signals.py` - Signal generation
