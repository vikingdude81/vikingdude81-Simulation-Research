# Fourier Transform Experiments for Crypto Trading

This directory contains experiments testing FTTF (Fourier Transform holographic memory) concepts on cryptocurrency trading data.

## 🎯 Purpose

Test whether Fourier transform techniques from the FTTF-fournier project can improve:
1. **Regime Detection** (+6-10% accuracy target)
2. **Signal Filtering** (+5-15% quality improvement)
3. **Data Reconstruction** (+20-30% robustness)
4. **Multi-Timeframe Analysis** (+10-20% performance)

## 📁 Directory Structure

```
fourier_experiments/
├── 01_basic_fft/              # ✅ ACTIVE - Foundation: FFT on crypto data
│   └── crypto_fft_basics.py   # Experiment 1.1: Identify dominant frequencies
├── 02_holographic_memory/     # NEXT - Test memory properties
├── 03_band_analysis/          # TODO - LOW/MID/HIGH band importance
├── 04_noise_filtering/        # TODO - Learned filter experiments
├── 05_regime_features/        # TODO - FFT features for regime detection
└── results/                   # Experiment outputs
    └── 01_basic_fft/          # Plots and JSON results

fourier_integration/           # Production-ready modules (after validation)
external/                      # FTTF-fournier project code reference
```

## 🚀 Quick Start

### Run First Experiment (Experiment 1.1)

```powershell
# From workspace root
cd PRICE-DETECTION-TEST-1

# Run basic FFT analysis on BTC and ETH
python fourier_experiments\01_basic_fft\crypto_fft_basics.py
```

**Expected Output:**
- Identify top 10 dominant frequencies for BTC/ETH
- Reconstruct price from top 50 frequency components
- Generate 4-panel visualization plots
- Save results to JSON

**Success Criteria:**
- ✅ Identify 3-5 dominant frequencies
- ✅ Reconstruction R² > 80%
- ✅ Visual patterns match market cycles

### View Results

```powershell
# Check results directory
ls fourier_experiments\results\01_basic_fft\

# View latest results
Get-Content (Get-ChildItem fourier_experiments\results\01_basic_fft\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

## 📋 Experiment Timeline

### Week 1: Foundation (Current)
- [x] **Experiment 1.1**: Basic FFT on crypto data (2-3 hours)
- [ ] **Experiment 1.2**: Holographic memory test (3-4 hours)
- [ ] **Experiment 1.3**: Band importance analysis (2-3 hours)

### Week 2: Trading Applications
- [ ] **Experiment 2.1**: FFT features for regime detection (1-2 days)
- [ ] **Experiment 2.2**: Learned noise filters (2-3 days)
- [ ] **Experiment 2.3**: Multi-timeframe synthesis (2-3 days)

### Week 3: Integration
- [ ] **Integration 3.1**: Enhanced regime detector (2-3 days)
- [ ] **Integration 3.2**: Data reconstruction (1-2 days)
- [ ] **Integration 3.3**: Learned filter network (2-3 days)

## 🎓 Key Concepts

### Holographic Memory Property
Information distributed across ALL frequency components. Damage 10-30% of FFT → still reconstruct signal!

**Trading Application**: Handle exchange outages gracefully - reconstruct missing data from available frequencies.

### Band-Specific Analysis
- **LOW frequencies** (0-25%): Long-term trends (daily/weekly cycles)
- **MID frequencies** (25-75%): Medium-term moves (4H-daily)
- **HIGH frequencies** (75-100%): Short-term noise (5min-1H)

**Trading Application**: Map frequency bands to regime-specific specialists.

### Learned Filters via Neuroevolution
Neural networks generate frequency-domain filters (not hand-tuned parameters).

**Trading Application**: Evolve optimal noise filters for each market condition.

## 📊 Success Criteria

| Experiment | Minimum Success | Target | Action if Failed |
|------------|----------------|--------|------------------|
| 1.1 FFT Basics | Identify frequencies | R² > 80% | Check data quality |
| 1.2 Holography | 20% damage ok | 30% damage ok | Adjust reconstruction |
| 2.1 Regime Features | +3% accuracy | +6% accuracy | Try different features |
| 2.2 Noise Filters | Beat SMA baseline | +10% vs SMA | More training generations |
| 2.3 Multi-Timeframe | +5% performance | +10% performance | Refine band mapping |

## 🔬 Development Workflow

### Daily Experiment Cycle
1. **Run experiment** → 2. **Analyze results** → 3. **Document findings** → 4. **Commit**

### Integration Path
**Experiment** → **Validate success** → **Create production module** → **Write tests** → **Merge**

## 📈 Expected Impact

| Application | Expected Improvement | Timeline |
|-------------|---------------------|----------|
| Enhanced Regime Detection | +6-10% accuracy | 1 week |
| Learned Noise Filters | +5-15% signal quality | 2-3 weeks |
| Data Reconstruction | +20-30% robustness | 1 week |
| Multi-Timeframe Synthesis | +10-20% performance | 3-4 weeks |

**Combined System Impact**: +30-50% overall improvement

## 🛠️ Dependencies

```bash
# Core scientific computing (already installed)
numpy
pandas
matplotlib

# JAX (for advanced experiments - Week 2+)
pip install jax jaxlib

# Our trading system modules
fetch_data.py          # Data fetching
regime_detector.py     # Current regime detection
```

## 📝 Logging & Tracking

Each experiment generates:
- **JSON results**: Metrics, dominant frequencies, reconstruction quality
- **Plots**: Visual analysis (4-panel figures)
- **Console output**: Real-time progress and success evaluation

All saved to: `fourier_experiments/results/<experiment_name>/`

## 🔄 Parallel Tracks

These experiments can run **in parallel** with:
- ✅ Phase 3D ensemble testing
- ✅ Other ML model training
- ✅ Backtest validation

**No risk to production system** - completely isolated experiments!

## 🎯 Next Steps After Experiment 1.1

If **SUCCESS** (R² > 80%):
1. Review plots - do dominant frequencies match known cycles (daily, weekly)?
2. Proceed to Experiment 1.2 (Holographic Memory Test)
3. Start thinking about regime detection features

If **PARTIAL** (60% < R² < 80%):
1. Analyze why - data quality? Too few components?
2. Try different reconstruction approaches
3. Still proceed but note limitations

If **FAILURE** (R² < 60%):
1. Document findings
2. Investigate data preprocessing needs
3. Consider alternative Fourier approaches

## 📚 References

- **FTTF Project**: `C:\Users\akbon\OneDrive\Documents\FTTF-fournier (1)`
- **Integration Analysis**: `FOURIER_INTEGRATION_ANALYSIS.md`
- **Experiment Plan**: `FOURIER_EXPERIMENT_PLAN.md`
- **Phase 3C Results**: `PHASE_3C_COMPLETE.md`

## ⚡ Quick Commands

```powershell
# Run current experiment
python fourier_experiments\01_basic_fft\crypto_fft_basics.py

# Check latest results
ls fourier_experiments\results -Recurse -File | Sort-Object LastWriteTime -Descending | Select-Object -First 5

# View experiment logs
Get-Content fourier_experiments\results\01_basic_fft\experiment_*.json | ConvertFrom-Json

# Commit progress
git add fourier_experiments/
git commit -m "Experiment 1.1: Basic FFT analysis results"
```

---

**Status**: 🟢 Week 1 Foundation Phase - Experiment 1.1 Ready!

**Last Updated**: 2025-11-09

**Branch**: `fourier-integration`
