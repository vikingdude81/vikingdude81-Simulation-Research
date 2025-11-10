# Fourier Transform Experiments & Integration Plan

**Branch**: fourier-integration  
**Status**: Experimental Phase  
**Goal**: Test FTTF concepts + Integrate into trading system

---

## 🧪 Experimental Sandbox Structure

### Directory Organization

```
PRICE-DETECTION-TEST-1/
├── fourier_experiments/           # NEW - Isolated experiments
│   ├── 01_basic_fft/             # Learn FFT basics on crypto data
│   ├── 02_holographic_memory/    # Test memory properties
│   ├── 03_band_analysis/         # LOW/MID/HIGH frequency bands
│   ├── 04_noise_filtering/       # Learned filter experiments
│   ├── 05_regime_features/       # FFT features for regime detection
│   └── results/                  # Experiment outputs
│
├── fourier_integration/           # NEW - Production integration
│   ├── frequency_analyzer.py     # FFT analysis tools
│   ├── learned_filters.py        # Neural filter networks
│   ├── holographic_reconstruction.py  # Data gap filling
│   └── regime_frequency_features.py   # Enhanced regime detection
│
└── FTTF-fournier/                # External reference (submodule)
    └── [their original code]
```

---

## 📋 Experiment Pipeline: Learn → Test → Integrate

### Phase 1: Foundation Experiments (Week 1)

**Goal**: Understand FTTF concepts with crypto data

#### Experiment 1.1: Basic FFT on Crypto Prices
```python
# File: fourier_experiments/01_basic_fft/crypto_fft_basics.py

Objective: Apply FFT to BTC/ETH price data
- Load hourly price data (last 1000 hours)
- Compute FFT, plot frequency spectrum
- Identify dominant frequencies
- Reconstruct signal from top N frequencies

Expected Insights:
- What frequencies dominate crypto markets?
- How many components needed for good reconstruction?
- Visual understanding of frequency domain

Timeline: 1 day
```

#### Experiment 1.2: Holographic Memory Test
```python
# File: fourier_experiments/02_holographic_memory/crypto_memory_test.py

Objective: Test holographic property on real market data
- Take price data, FFT to frequency domain
- Damage 10%, 20%, 30% of frequency components
- Reconstruct via IFFT
- Measure MSE, visualize degradation

Expected Insights:
- Does crypto data show holographic properties?
- How much damage before signal unusable?
- Which frequencies most critical?

Timeline: 1 day
```

#### Experiment 1.3: Frequency Band Analysis
```python
# File: fourier_experiments/03_band_analysis/band_importance.py

Objective: Identify which frequency bands matter most
- Define LOW (0-25%), MID (25-75%), HIGH (75-100%) bands
- Selectively damage each band
- Measure reconstruction quality
- Compare band importance

Expected Insights:
- Are low frequencies (trends) most important?
- Can we filter high frequencies (noise)?
- Band-specific strategies for trading?

Timeline: 1-2 days
```

---

### Phase 2: Trading-Specific Experiments (Week 1-2)

#### Experiment 2.1: Regime Detection with FFT Features
```python
# File: fourier_experiments/05_regime_features/fft_regime_features.py

Objective: Do frequency features improve regime detection?

Current RegimeDetector Features:
- volatility, trend_strength, volume_ratio, ...

New FFT Features to Test:
- low_freq_power (long-term trends)
- mid_freq_power (medium cycles)  
- high_freq_power (short-term noise)
- dominant_frequency (primary cycle)
- spectral_entropy (randomness measure)
- band_power_ratio (low/high)

Experiment Design:
1. Extract features from 5000 days of data
2. Train classifier with time-domain only (baseline)
3. Train classifier with time + frequency features
4. Compare accuracy on hold-out set

Success Metric: >5% accuracy improvement

Timeline: 2-3 days
```

#### Experiment 2.2: Learned Noise Filters
```python
# File: fourier_experiments/04_noise_filtering/evolve_crypto_filters.py

Objective: Can neural nets learn better filters than SMA/EMA?

Experiment Design:
1. Ground truth: Aggregate 5-min data → hourly (smooth signal)
2. Noisy input: Raw 5-min data with gaps, spikes
3. Evolve filter network (port FTTF JAX code)
4. Compare reconstructions:
   - No filter (raw)
   - SMA(20), EMA(20) 
   - Evolved neural filter

Metrics:
- MSE vs ground truth
- Signal-to-noise ratio
- Backtest performance with each filter

Timeline: 3-4 days
```

#### Experiment 2.3: Multi-Timeframe Synthesis
```python
# File: fourier_experiments/03_band_analysis/timeframe_synthesis.py

Objective: Use frequency bands for multi-timeframe trading

Mapping:
- LOW frequencies → Daily/Weekly trends (hold days-weeks)
- MID frequencies → 4H/Daily cycles (hold hours-days)
- HIGH frequencies → 5M/1H moves (hold minutes-hours)

Experiment:
1. Decompose price into 3 bands via FFT
2. Generate signals from each band independently
3. Test synthesis strategies:
   - Vote (majority of 3)
   - Weight by band power
   - Hierarchical (long-term veto short-term)
4. Backtest each approach

Success: >10% improvement vs single-timeframe

Timeline: 3-4 days
```

---

### Phase 3: Production Integration (Week 2-3)

#### Integration 3.1: Enhanced RegimeDetector
```python
# File: fourier_integration/regime_frequency_features.py

from regime_detector import RegimeDetector
import numpy as np

class FrequencyEnhancedRegimeDetector(RegimeDetector):
    """RegimeDetector with FFT features"""
    
    def _extract_fft_features(self, prices):
        """Extract frequency-domain features"""
        fft = np.fft.fft(prices)
        freqs = np.fft.fftfreq(len(prices))
        power_spectrum = np.abs(fft) ** 2
        
        # Band definitions (based on Experiment 1.3 results)
        n = len(freqs)
        low_band = power_spectrum[:n//4]
        mid_band = power_spectrum[n//4:3*n//4]
        high_band = power_spectrum[3*n//4:]
        
        return {
            'low_freq_power': np.sum(low_band),
            'mid_freq_power': np.sum(mid_band),
            'high_freq_power': np.sum(high_band),
            'dominant_freq': freqs[np.argmax(power_spectrum)],
            'spectral_entropy': self._compute_entropy(power_spectrum),
            'band_ratio': np.sum(low_band) / np.sum(high_band)
        }
    
    def extract_features(self, market_data):
        # Get original time-domain features
        time_features = super().extract_features(market_data)
        
        # Add frequency-domain features
        prices = market_data['close'].values
        freq_features = self._extract_fft_features(prices)
        
        # Combine
        return {**time_features, **freq_features}

# Usage:
detector = FrequencyEnhancedRegimeDetector()
regime = detector.detect_regime(market_data)
```

#### Integration 3.2: Holographic Data Reconstruction
```python
# File: fourier_integration/holographic_reconstruction.py

class HolographicDataReconstructor:
    """Fill gaps in market data using frequency-domain reconstruction"""
    
    def reconstruct_missing_data(self, price_series, missing_indices):
        """
        Reconstruct missing price data points
        
        Args:
            price_series: Array with NaN at missing indices
            missing_indices: List of indices where data is missing
            
        Returns:
            Reconstructed complete series
        """
        # 1. Extract available data
        available = ~np.isnan(price_series)
        available_prices = price_series[available]
        available_indices = np.where(available)[0]
        
        # 2. FFT of available data
        fft_available = np.fft.fft(available_prices)
        
        # 3. Create full-length FFT (interpolate frequency domain)
        # This is the "holographic" property - frequency info is distributed
        fft_full = self._interpolate_fft(
            fft_available, 
            available_indices, 
            len(price_series)
        )
        
        # 4. IFFT to reconstruct
        reconstructed = np.real(np.fft.ifft(fft_full))
        
        return reconstructed
    
    def _interpolate_fft(self, fft_sparse, sparse_indices, full_length):
        """Interpolate sparse FFT to full length"""
        # Use frequency-domain properties to fill gaps
        # Low frequencies are most reliable for long gaps
        # High frequencies for short gaps
        # Implementation based on gap size
        pass

# Usage in production:
reconstructor = HolographicDataReconstructor()

# When exchange API fails
price_data = get_market_data()  # Has gaps (NaN)
if has_gaps(price_data):
    price_data = reconstructor.reconstruct_missing_data(
        price_data, 
        find_gaps(price_data)
    )
```

#### Integration 3.3: Learned Filter Network
```python
# File: fourier_integration/learned_filters.py

import jax
import jax.numpy as jnp
from jax import grad

class LearnedMarketFilter:
    """Neural network that learns optimal frequency-domain filters"""
    
    def __init__(self):
        # Port FTTF's FilterNetwork architecture
        self.network = self._build_network()
        self.trained = False
    
    def train_on_market_data(self, noisy_data, clean_data):
        """
        Train filter to denoise market data
        
        Args:
            noisy_data: Raw price data with noise
            clean_data: Ground truth (aggregated data)
        """
        # Use FTTF's hybrid evolution approach
        # 1. FFT noisy data
        # 2. Neural net generates filter coefficients
        # 3. Apply filter, IFFT
        # 4. Optimize via gradients through IFFT
        pass
    
    def filter_price_data(self, prices):
        """Apply learned filter to new price data"""
        if not self.trained:
            raise ValueError("Filter not trained yet")
        
        # 1. FFT
        price_fft = jnp.fft.fft(prices)
        
        # 2. Generate filter from neural net
        filter_coeffs = self.network.predict(price_fft)
        
        # 3. Apply filter
        filtered_fft = price_fft * filter_coeffs
        
        # 4. IFFT
        filtered_prices = jnp.real(jnp.fft.ifft(filtered_fft))
        
        return filtered_prices

# Usage in specialists:
filter_net = LearnedMarketFilter()
filter_net.train_on_market_data(
    noisy_intraday, 
    smooth_daily
)

# In trading loop
raw_prices = get_latest_prices()
clean_prices = filter_net.filter_price_data(raw_prices)
signals = specialist.generate_signals(clean_prices)
```

---

## 🔬 Experiment Tracking

### Template for Each Experiment

```python
# Header for every experiment file
"""
Experiment: [Name]
Objective: [What we're testing]
Hypothesis: [What we expect to find]
Metrics: [How we measure success]
Timeline: [Expected duration]

Results:
  - [Fill after running]
  - [Key findings]
  - [Plots/data saved to results/]
"""
```

### Results Directory Structure

```
fourier_experiments/results/
├── 01_basic_fft/
│   ├── btc_frequency_spectrum.png
│   ├── reconstruction_quality.json
│   └── experiment_log.md
├── 02_holographic_memory/
│   ├── damage_vs_mse.png
│   ├── visual_reconstruction.gif
│   └── experiment_log.md
├── 03_band_analysis/
│   ├── band_importance_ranking.json
│   ├── frequency_bands.png
│   └── experiment_log.md
└── ...
```

---

## 📊 Success Criteria

### Experiment Success Thresholds

| Experiment | Minimum Success | Target Success | Action If Failed |
|------------|-----------------|----------------|------------------|
| FFT Regime Features | +3% accuracy | +6% accuracy | Try different feature combinations |
| Learned Filters | Beat SMA | Beat Kalman filter | Adjust network architecture |
| Multi-Timeframe | +5% return | +10% return | Refine band definitions |
| Data Reconstruction | MSE < 2x raw | MSE < 1.5x raw | Use more frequency components |

### Integration Success Criteria

- [ ] Enhanced RegimeDetector: 69.2% → 75%+ accuracy
- [ ] Learned filters integrated into at least 1 specialist
- [ ] Data reconstruction handles 20%+ missing data
- [ ] No degradation to existing system performance
- [ ] Production-ready code (error handling, logging, tests)

---

## 🛠️ Development Workflow

### Daily Workflow (During Experiment Phase)

```bash
# Morning: Start new experiment
cd fourier_experiments/01_basic_fft
python crypto_fft_basics.py

# Afternoon: Analyze results
python analyze_results.py > results/experiment_log.md

# Evening: Document findings
git add results/
git commit -m "Experiment 1.1: FFT basics on BTC - Found X dominant frequencies"

# Iterate or move to next experiment
```

### Integration Workflow

```bash
# After successful experiment
cd fourier_integration/

# 1. Create production module
cp ../fourier_experiments/05_regime_features/best_approach.py \
   regime_frequency_features.py

# 2. Add error handling, logging, tests
python -m pytest tests/test_frequency_features.py

# 3. Integrate with existing system
# Modify regime_detector.py to use new features

# 4. Validate
python test_enhanced_regime_detector.py

# 5. Commit
git add fourier_integration/
git commit -m "Integration: Enhanced RegimeDetector with FFT features (+8% accuracy)"
```

---

## 📅 Timeline

### Week 1: Foundation Experiments
- **Day 1-2**: Basic FFT + Holographic memory (Experiments 1.1, 1.2)
- **Day 3-4**: Band analysis (Experiment 1.3)
- **Day 5**: Review findings, document insights

### Week 2: Trading Experiments
- **Day 1-3**: Regime detection features (Experiment 2.1)
- **Day 4-5**: Learned filters (Experiment 2.2)

### Week 3: Integration
- **Day 1-2**: Enhanced RegimeDetector (Integration 3.1)
- **Day 3**: Data reconstruction (Integration 3.2)
- **Day 4-5**: Learned filters in specialists (Integration 3.3)

### Week 4: Validation & Production
- **Day 1-2**: Full system testing
- **Day 3**: Performance benchmarking
- **Day 4-5**: Documentation, merge to main

---

## 🔄 Parallel Tracks

You can work on multiple things simultaneously:

**Track A: Pure Experimentation** (No risk to existing system)
- Run all FTTF experiments
- Learn frequency-domain concepts
- Test hypotheses on historical data
- Build intuition

**Track B: Phase 3D Ensemble Testing** (Continue current work)
- Test ensemble with Phase 3A specialists
- Compare performance vs baseline
- Independent of Fourier work

**Track C: Quick Wins** (Low-hanging fruit)
- Add FFT features to RegimeDetector (1-2 days)
- Test on existing validation set
- If improvement, integrate immediately

---

## 🎯 Decision Gates

### After Week 1 (Foundation Experiments)
**Decision**: Are frequency-domain approaches promising for crypto?

- ✅ **YES**: Proceed to Week 2 (Trading experiments)
- ❌ **NO**: Document findings, return to Phase 3D/4

### After Week 2 (Trading Experiments)
**Decision**: Do FFT features improve trading metrics?

- ✅ **YES**: Proceed to Week 3 (Integration)
- ❌ **NO**: Pivot to subset of promising experiments only

### After Week 3 (Integration)
**Decision**: Does integrated system outperform baseline?

- ✅ **YES**: Merge to main, deploy
- ⚠️ **MIXED**: Keep successful components, iterate on others
- ❌ **NO**: Preserve experiments, return to original system

---

## 📦 Code Templates

### Experiment Template
```python
# fourier_experiments/XX_category/experiment_name.py

"""
Experiment: [Name]
Date: [YYYY-MM-DD]
Objective: [One sentence]
Hypothesis: [What we expect]
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
EXPERIMENT_ID = "XX_experiment_name"
RESULTS_DIR = Path(f"results/{EXPERIMENT_ID}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
def load_crypto_data():
    # Load BTC/ETH price data
    pass

# Run experiment
def run_experiment():
    print(f"Starting: {EXPERIMENT_ID}")
    
    # 1. Setup
    data = load_crypto_data()
    
    # 2. Process
    results = process_data(data)
    
    # 3. Analyze
    metrics = analyze_results(results)
    
    # 4. Visualize
    plot_results(results, save_path=RESULTS_DIR / "plot.png")
    
    # 5. Save
    save_results(metrics, RESULTS_DIR / "metrics.json")
    
    print(f"Complete: {EXPERIMENT_ID}")
    print(f"Results saved to: {RESULTS_DIR}")
    
    return metrics

if __name__ == "__main__":
    metrics = run_experiment()
    
    # Document findings
    print("\nKEY FINDINGS:")
    print(f"  - Finding 1: {metrics['key_metric']}")
    print(f"  - Finding 2: ...")
```

### Integration Template
```python
# fourier_integration/module_name.py

"""
Production module for [functionality]
Based on Experiment: XX_experiment_name
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class ModuleName:
    """
    Production-ready implementation of [functionality]
    
    Usage:
        module = ModuleName()
        result = module.process(data)
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def process(self, data):
        """Main processing method"""
        try:
            result = self._do_processing(data)
            return result
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    def _do_processing(self, data):
        """Internal processing logic"""
        pass
    
    @staticmethod
    def _default_config():
        return {
            'param1': 'value1',
            'param2': 'value2'
        }

# Tests
def test_module():
    module = ModuleName()
    test_data = np.random.randn(100)
    result = module.process(test_data)
    assert result is not None
    print("✓ Tests passed")

if __name__ == "__main__":
    test_module()
```

---

## 🎓 Learning Path

**For Understanding FTTF Concepts**:
1. Read their `README.md` (excellent intro)
2. Run their `fourier_memory_test.py` (interactive demo)
3. Study `jax_hybrid_neuroevolution.py` (core algorithm)
4. Adapt to crypto data (our experiments)

**For Integration**:
1. Start with simplest: FFT features for regime detection
2. Then: Data reconstruction (practical need)
3. Then: Learned filters (more complex)
4. Finally: Advanced (frequency-domain conductor, etc.)

---

## 🚀 Getting Started NOW

### Immediate First Steps (Today!)

```bash
# 1. Create experiment directories
cd PRICE-DETECTION-TEST-1
mkdir -p fourier_experiments/{01_basic_fft,02_holographic_memory,03_band_analysis,04_noise_filtering,05_regime_features,results}
mkdir -p fourier_integration

# 2. Copy FTTF code for reference
cp -r "C:\Users\akbon\OneDrive\Documents\FTTF-fournier (1)\FTTF-fournier" external/FTTF-fournier

# 3. Create first experiment
# I'll create this next!
```

Let me create the first experiment file to get you started!

---

**Status**: Ready to begin experiments! 🚀  
**Next**: Create Experiment 1.1 (Basic FFT on crypto data)
