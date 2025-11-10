# 🤖 ML Models Menu System

**Individual Model Testing Without Breaking the Pipeline**

## Quick Start

```bash
python ml_models_menu.py
```

## Features

### 📊 Classical ML Models (Fast Training)
- **Random Forest** - Tree ensemble, good baseline (~2-5 min)
- **XGBoost** - Gradient boosting, fast and accurate (~3-7 min)
- **LightGBM** - Memory efficient, fast training (~3-7 min)
- **Classical Ensemble** - All three combined (~10-20 min)

### 🧠 Deep Learning Models (GPU Recommended)
- **LSTM with Attention** - Temporal patterns, 1.4M params (~10-15 min GPU)
- **Transformer** - Multi-head attention, 3.2M params (~10-15 min GPU)
- **MultiTask Network** - Price+Vol+Direction, 3.4M params (~10-15 min GPU)
- **Deep Learning Suite** - All three DL models (~30-45 min GPU)

### 🚀 Full Pipeline
- **Complete Pipeline** - All 6 models (~60 min GPU, ~120 min CPU)

### ⚙️ Utilities
- **Compare Models** - Side-by-side performance comparison
- **Feature Importance** - Analyze top features
- **GPU Check** - Verify CUDA availability

## How It Works

### Classical Models
The menu creates standalone training scripts for each classical ML model:
- `train_random_forest.py`
- `train_xgboost.py`
- `train_lightgbm.py`

These scripts:
1. Import data pipeline from `main.py`
2. Use the same feature engineering
3. Train only the selected model
4. Save to `MODEL_STORAGE/<model>_standalone.pkl`
5. Display results and feature importance

### Deep Learning Models
Deep learning models use the existing implementations in `main.py`:
- Share the same architecture
- Use GPU if available (automatic fallback to CPU)
- Can be trained individually or as a suite

### No Breaking Changes
- ✅ Original `main.py` pipeline unchanged
- ✅ All models use same feature set
- ✅ Results saved to separate files for standalone runs
- ✅ Full pipeline still works exactly as before

## Use Cases

### Quick Testing
```bash
# Test Random Forest quickly
python ml_models_menu.py
# Select: 1

# Compare with XGBoost
python ml_models_menu.py
# Select: 2
```

### Model-Specific Experiments
```bash
# Test different RF hyperparameters
# 1. Select option 1 to create train_random_forest.py
# 2. Edit train_random_forest.py parameters
# 3. Run again with new settings
```

### Progressive Training
```bash
# Train classical models first (fast)
python ml_models_menu.py
# Select: 4 (Classical Ensemble)

# Then add deep learning if satisfied
# Select: 8 (Deep Learning Suite)
```

### Comparison & Analysis
```bash
# After training individual models
python ml_models_menu.py
# Select: 10 (Compare Models)
# Select: 11 (Feature Importance)
```

## Output Files

### Standalone Models
```
MODEL_STORAGE/
├── random_forest_standalone.pkl
├── xgboost_standalone.pkl
├── lightgbm_standalone.pkl
└── (full pipeline models remain separate)
```

### Training Scripts
```
train_random_forest.py      # Generated on demand
train_xgboost.py            # Generated on demand
train_lightgbm.py           # Generated on demand
```

## Example Session

```bash
$ python ml_models_menu.py

================================================================================
🤖 ML MODELS MENU - Individual Model Testing
================================================================================

Test individual ML models or run the complete pipeline
All models use the same feature set for fair comparison

================================================================================
📊 CLASSICAL ML MODELS (Scikit-learn & Gradient Boosting)
================================================================================
1. Random Forest          - Tree ensemble with bagging
2. XGBoost               - Gradient boosting trees (fast)
3. LightGBM              - Microsoft gradient boosting (memory efficient)
4. Classical Ensemble    - RF + XGBoost + LightGBM combined

================================================================================
🧠 DEEP LEARNING MODELS (PyTorch with GPU)
================================================================================
5. LSTM with Attention   - 3 layers, 256 hidden, ~1.4M params
6. Transformer           - 4 layers, 8 heads, ~3.2M params
7. MultiTask Network     - Price + Vol + Direction, ~3.4M params
8. Deep Learning Suite   - LSTM + Transformer + MultiTask

================================================================================
🚀 FULL PIPELINE
================================================================================
9. Complete Pipeline     - All 6 models (RF, XGB, LGB, LSTM, Trans, Multi)

================================================================================
⚙️  UTILITIES
================================================================================
10. Compare Models       - Side-by-side performance comparison
11. Feature Importance   - Analyze which features matter most
12. GPU Check           - Verify CUDA availability

================================================================================
0. Exit
================================================================================

👉 Select an option (0-12): 1

================================================================================
🌲 RANDOM FOREST TRAINING
================================================================================

Creating training script for Random Forest only...

✅ Created: train_random_forest.py

▶️  Running Random Forest training...

================================================================================
🌲 RANDOM FOREST - Standalone Training
================================================================================
✅ Loaded data pipeline from main.py

📊 Loading data and engineering features...
✅ Features: 44
✅ Training samples: 8640
✅ Test samples: 2160

🌲 Training Random Forest model...
[Training progress...]

📊 RESULTS:
   Train RMSE: 0.0032
   Test RMSE:  0.0045

🔝 Top 10 Important Features:
   price_returns_24h             : 0.1234
   volatility_regime_percentile  : 0.0876
   gma_distance_fast            : 0.0654
   [...]

✅ Model saved to: MODEL_STORAGE/random_forest_standalone.pkl

================================================================================

✅ Press Enter to return to menu...
```

## Tips

### Speed Up Classical Models
- Use fewer estimators for testing: `n_estimators=100` instead of `200`
- Reduce max_depth: `max_depth=10` instead of `15`
- Use smaller feature subsets for initial testing

### Optimize Deep Learning
- Start with fewer epochs to verify setup
- Monitor GPU usage with option 12
- Use mixed precision for faster training
- Batch size affects speed vs memory tradeoff

### Best Practice Workflow
1. **Quick baseline**: Train Random Forest first (fastest)
2. **Compare boosting**: Test XGBoost and LightGBM
3. **Deep learning**: Add LSTM/Transformer if classical models look good
4. **Full pipeline**: Run complete pipeline for final production models

## Integration with Existing Code

The menu system is **completely non-invasive**:

- ✅ No changes to `main.py`
- ✅ No changes to existing training scripts
- ✅ Standalone scripts import from main.py (reuses all logic)
- ✅ Separate save paths prevent overwriting
- ✅ Original workflow still works identically

You can use both systems:
- Use **menu** for individual testing and experiments
- Use **main.py** for full production pipeline runs

## Requirements

Same as main pipeline:
- Python 3.8+
- PyTorch (with CUDA for GPU)
- scikit-learn
- xgboost
- lightgbm
- pandas, numpy, yfinance

---

**Happy Model Testing! 🚀**
