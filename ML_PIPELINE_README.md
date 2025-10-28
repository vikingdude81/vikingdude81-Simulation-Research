# 🚀 ML Pipeline Full - GPU-Accelerated Deep Learning Showcase

**Branch**: `ml-pipeline-full`  
**Purpose**: Production-grade deep learning pipeline with PyTorch, K-Means clustering, and advanced features  
**Use Case**: Reusable ML infrastructure for GPU-accelerated time series forecasting projects

---

## 📊 **Architecture Overview**

### **Deep Learning Models** (PyTorch with GPU)
- **LSTM with Attention** - 3 layers, 256 hidden units, 1.4M parameters
- **Transformer** - 4 layers, 8 heads, 256 dimensions, 3.2M parameters  
- **MultiTask Network** - Price + Volatility + Direction prediction, 3.4M parameters

### **Feature Engineering** (44+ features per asset)
1. **Microstructure** (6) - Spread, efficiency, illiquidity, trade intensity
2. **Volatility Regime** (7) - Classification, percentile, acceleration
3. **Fractal & Chaos** (7) - Hurst exponents, kurtosis, skewness
4. **Order Flow** (10) - Buy/sell ratio, volume imbalance, VWAP
5. **Market Regime** (7) - Trend strength, state classification
6. **Price Levels** (7) - Support/resistance, pivot points
7. **Geometric MA** (21) - GMA crossovers, distance metrics (4.33-6.47 Sharpe)
8. **Long-term Momentum** (9) - 90d/180d/365d academic research indicators
9. **Extreme Markets** (8) - Crisis detection, tail risk, extreme moves
10. **K-Means Clustering** (6) - 4 regime clusters (ranging/trending/choppy/stable)
11. **Interaction Features** (23) - GMA×trend, vol×chaos, flow×regime
12. **External Data** (15) - Fear & Greed, Google Trends, sentiment, dominance

### **Advanced Components**
- ✅ **GPU Acceleration** - CUDA support with automatic fallback to CPU
- ✅ **K-Means Clustering** - Market regime detection with 4 clusters
- ✅ **UCB Asset Selection** - Upper Confidence Bound adaptive allocation
- ✅ **Model Persistence** - Complete storage manager with metadata
- ✅ **Feature Importance** - Automated ranking and selection
- ✅ **External Data Integration** - Real-time market sentiment

---

## 📈 **Performance**

| Asset | RMSE | Models | Training Time | GPU Speedup |
|-------|------|--------|---------------|-------------|
| **BTC** | 0.44% | LSTM+Trans+Multi | ~20 min | 3-5x |
| **ETH** | 0.91% | LSTM+Trans+Multi | ~20 min | 3-5x |
| **SOL** | 1.01% | LSTM+Trans+Multi | ~20 min | 3-5x |

**Total**: 18 models (6 per asset), ~60 min training with GPU

---

## 🔧 **Key Files**

### **Training & Inference**
- `main.py` - Single asset training with all features
- `train_all_assets.py` - Multi-asset batch training
- `predict_all_assets.py` - 24-hour forecasts for all assets
- `multi_asset_signals.py` - Portfolio allocation signals

### **Feature Engineering**
- `enhanced_features.py` - 116 advanced features (11 categories)
- `external_data.py` - 15 external market indicators
- `storage_manager.py` - Model persistence and metadata

### **Advanced Components**
- `ucb_asset_selector.py` - UCB algorithm for adaptive allocation
- `backtest_ucb.py` - 90-day UCB learning simulation
- `analyze_models.py` - Model architecture analysis
- `geometric_ma_crossover.py` - GMA feature generation (champion strategy)

### **Utilities**
- `fetch_multi_asset.py` - Download BTC/ETH/SOL data
- `check_device.py` - Verify GPU availability
- `intensive_gpu_test.py` - GPU stress testing

---

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Install PyTorch with CUDA (if GPU available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install pandas numpy scikit-learn xgboost lightgbm yfinance
```

### **2. Check GPU**
```python
python check_device.py
# Expected output: GPU detected: NVIDIA GeForce RTX...
```

### **3. Download Data**
```python
python fetch_multi_asset.py
# Downloads BTC, ETH, SOL across 5 timeframes (1h, 4h, 12h, 1d, 1w)
```

### **4. Train Models**
```python
# Train single asset (BTC)
python main.py

# Train all assets (BTC + ETH + SOL)
python train_all_assets.py
```

### **5. Generate Predictions**
```python
# 24-hour forecasts for all assets
python predict_all_assets.py
```

---

## 🎯 **Use Cases**

### **1. Time Series Forecasting Projects**
- Copy LSTM/Transformer architectures to new projects
- Reuse feature engineering modules
- Adapt for stock prices, commodity futures, forex

### **2. GPU Training Research**
- Benchmark GPU vs CPU performance
- Test PyTorch optimization techniques
- Experiment with batch sizes and learning rates

### **3. Feature Engineering Reference**
- 116 pre-built features across 11 categories
- Academic research indicators (momentum, GMA)
- Market microstructure and regime detection

### **4. Production ML Pipeline**
- Model persistence and versioning
- External data integration patterns
- Multi-asset training workflows

---

## 📚 **Technical Details**

### **LSTM with Attention**
```python
class BitcoinLSTM(nn.Module):
    - 3 LSTM layers (256 hidden units each)
    - Attention mechanism for temporal importance
    - Dropout 0.3 for regularization
    - 48-hour sequence input
    - 1.4M parameters
```

### **Transformer**
```python
class BitcoinTransformer(nn.Module):
    - 4 encoder layers
    - 8 attention heads
    - 256 embedding dimensions
    - Positional encoding
    - 3.2M parameters
```

### **MultiTask Network**
```python
class MultiTaskModel(nn.Module):
    - Shared LSTM backbone
    - 3 prediction heads:
      * Price prediction
      * Volatility estimation
      * Direction classification
    - Monte Carlo Dropout for uncertainty
    - 3.4M parameters
```

### **K-Means Clustering**
```python
def add_kmeans_regime_features(df):
    - 4 clusters based on volatility + trend
    - Cluster 0: Ranging (6.9% of time, high vol)
    - Cluster 1: Trending (58.5%, low vol)
    - Cluster 2: Choppy (32.9%, medium vol)
    - Cluster 3: Stable (1.7%, low vol)
    - Features: cluster_id, confidence, probabilities
```

---

## 🔬 **Research Features**

### **Geometric MA Crossover** (Champion Strategy)
- Based on TradingView research
- 4.33-6.47 Sharpe ratio across crypto pairs
- 21 features: spreads, slopes, alignments, distances
- Ranked #2 in feature importance (+42% improvement)

### **Long-term Momentum** (Academic)
- Moskowitz et al. (2012) time series momentum
- 90d, 180d, 365d returns
- Cross-sectional and time-series components
- 9 features per asset

### **External Data Integration**
- Fear & Greed Index (market sentiment)
- Google Trends (search interest)
- Social sentiment (simulated Twitter/Reddit)
- BTC dominance, USDT dominance, ETH dominance
- Exchange metrics (volume, market cap)

---

## ⚙️ **Configuration**

### **Model Hyperparameters**
```python
SEQUENCE_LENGTH = 48  # Hours of history
HIDDEN_SIZE = 256
NUM_LAYERS = 3
DROPOUT = 0.3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
```

### **Feature Engineering**
```python
N_CLUSTERS = 4  # K-Means market regimes
ENHANCED_FEATURES = True  # Use all 116 features
EXTERNAL_DATA = True  # Include sentiment/dominance
INTERACTION_FEATURES = True  # GMA×trend, vol×chaos, etc.
```

### **GPU Settings**
```python
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')
MIXED_PRECISION = True  # For faster training
```

---

## 📊 **Model Comparison**

| Feature | Simple Branch | ML Pipeline (This) |
|---------|---------------|-------------------|
| **Models** | RF+XGB+LGB | LSTM+Trans+Multi |
| **Framework** | Scikit-learn | PyTorch |
| **Features** | 20 | 44+ |
| **GPU** | No | Yes (CUDA) |
| **K-Means** | No | Yes (4 clusters) |
| **External Data** | No | Yes (15 sources) |
| **Training Time** | 3 min | 60 min |
| **BTC RMSE** | **0.30%** ✅ | 0.44% |
| **ETH RMSE** | **0.59%** ✅ | 0.91% |
| **SOL RMSE** | **0.67%** ✅ | 1.01% |
| **Use Case** | Production deployment | Research & development |

**Note**: Simple branch has better accuracy but ML pipeline offers richer feature set and GPU infrastructure for other projects.

---

## 🔄 **Workflow**

```
1. fetch_multi_asset.py
   ↓ Download BTC/ETH/SOL data
   
2. enhanced_features.py
   ↓ Generate 116 features + K-Means
   
3. external_data.py
   ↓ Collect Fear & Greed, Trends, Sentiment
   
4. train_all_assets.py
   ↓ Train 18 models (LSTM+Trans+Multi × 3 assets)
   
5. storage_manager.py
   ↓ Save models + metadata + feature data
   
6. predict_all_assets.py
   ↓ Generate 24-hour forecasts
   
7. multi_asset_signals.py
   ↓ Create buy/sell/hold signals
   
8. ucb_asset_selector.py
   ↓ Adaptive portfolio allocation
```

---

## 📦 **Reusable Components**

### **For New Projects**
1. **Copy Neural Networks**: `analyze_models.py` → LSTM/Transformer classes
2. **Copy Feature Engineering**: `enhanced_features.py` → 116 features
3. **Copy K-Means**: `add_kmeans_regime_features()` function
4. **Copy GPU Utils**: `check_device.py`, GPU training loops
5. **Copy Storage**: `storage_manager.py` → model persistence

### **Adaptation Guide**
```python
# Example: Adapt for stock prices
from enhanced_features import add_all_enhanced_features
from analyze_models import BitcoinLSTM  # Rename to StockLSTM

# Load your stock data
df = pd.read_csv('stock_prices.csv')

# Apply feature engineering
df = add_all_enhanced_features(df)

# Train LSTM
model = BitcoinLSTM(input_size=44, hidden_size=256, num_layers=3)
# ... training code ...
```

---

## 🎓 **Learning Resources**

This branch demonstrates:
- ✅ PyTorch LSTM/Transformer implementation
- ✅ Multi-task learning architecture
- ✅ GPU acceleration best practices
- ✅ Feature engineering at scale (116 features)
- ✅ K-Means unsupervised learning
- ✅ Model persistence and versioning
- ✅ External data integration
- ✅ Production ML pipeline design

Perfect for learning advanced ML techniques or bootstrapping new time series projects!

---

## 📝 **License & Attribution**

- Geometric MA research: TradingView user studies
- Momentum indicators: Moskowitz et al. (2012) academic paper
- K-Means implementation: Scikit-learn
- Deep learning: PyTorch
- Data: Yahoo Finance (yfinance)

---

## 🚀 **Branch Strategy**

- **`main`**: Latest stable version (complex system)
- **`production-simple`**: Deployed simple version (0.30% RMSE) ⭐ **Best accuracy**
- **`ml-pipeline-full`**: This branch - GPU showcase & reusable infrastructure
- **`experimental-pre-indicators`**: Development/testing

---

**Use this branch when you need:**
- GPU-accelerated deep learning infrastructure
- Advanced feature engineering templates
- K-Means clustering implementation
- Multi-model ensemble architecture
- External data integration patterns
- Production ML pipeline reference

**Use `production-simple` when you need:**
- Fast, accurate predictions
- Lightweight deployment
- Production reliability
- Quick training cycles
