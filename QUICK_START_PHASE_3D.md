# Quick Start Guide: Phase 3D - Ensemble Testing

**Last Updated**: November 9, 2025  
**Current Status**: Phase 3C Complete, Ready for Phase 3D  
**Branch**: ml-quantum-integration

---

## Current State Summary

### ✅ Completed
- **Phase 3C**: DANN conductor implementation and training
- **DANN Results**: 99.85% param accuracy, 31.67% regime invariance (PERFECT!)
- **Specialist Training**: 6 trainings complete (3 baseline + 3 DANN)
- **Key Finding**: DANN achieved 10-18x cache efficiency improvements

### 📍 Where We Left Off

All specialist training complete. Results show:
- Phase 3A genomes still have best fitness (V: 75.60, T: 47.55, R: 6.99)
- DANN matched baseline fitness but with 10-18x better cache efficiency
- Ready to test ensemble and validate performance

---

## Quick Start Commands

### 1. Check Training Results

```powershell
# View Phase 3A results (best fitness)
cd PRICE-DETECTION-TEST-1
python -c "import json; data = json.load(open('outputs/conductor_enhanced_volatile_20251108_150631.json')); print(f'Phase 3A Volatile: {data[\"best_fitness\"]}')"

# View DANN results (best efficiency)
python -c "import json; data = json.load(open('outputs/conductor_enhanced_volatile_20251108_174554.json')); print(f'DANN Volatile: {data[\"best_fitness\"]}, Cache: {data[\"cache_stats\"][\"hit_rate\"]}%')"
```

### 2. Test Ensemble (Next Step!)

```powershell
cd PRICE-DETECTION-TEST-1

# Test ensemble with Phase 3A specialists (best fitness)
python ensemble_conductor.py

# Expected output:
# - Total return vs +189% baseline
# - Sharpe ratio vs 1.01 baseline
# - 77+ trades expected
# - Regime transition smoothness
```

### 3. Compare Results

```powershell
# After ensemble test, compare with Phase 3A baseline
python -c "
import json

# Phase 3A baseline
print('Phase 3A Baseline:')
print('  Return: +189%')
print('  Sharpe: 1.01')
print('  Trades: 77')
print('  Win Rate: 41.6%')
print('  Max DD: -11%')
print()

# New ensemble results (after running)
# TODO: Extract from latest ensemble_conductor output
"
```

---

## File Locations Reference

### Key Code Files
```
PRICE-DETECTION-TEST-1/
├── domain_adversarial_conductor.py      # DANN implementation (~580 lines)
├── conductor_enhanced_trainer.py        # Trainer with --use-dann flag (~833 lines)
├── extract_dann_training_data.py        # Data extraction (~250 lines)
├── ensemble_conductor.py                # Ensemble testing (ready to run)
├── trading_specialist.py                # Specialist implementation
└── regime_detector.py                   # Regime detection (69.2% accuracy)
```

### Training Data
```
data/
├── dann_train_data.json                 # 720 training samples
├── dann_val_data.json                   # 180 validation samples
├── volatile_regime_data.csv             # Volatile market data (3,248 days)
├── trending_regime_data.csv             # Trending market data (4,136 days)
└── ranging_regime_data.csv              # Ranging market data (2,078 days)
```

### Model Files
```
outputs/
├── dann_conductor_best.pth              # Best DANN model (64KB)
├── dann_conductor_20251108_145051.json  # DANN training history
└── regime_detector_model.pth            # Regime detector (69.2% accuracy)
```

### Training Results

**Phase 3A (Best Fitness)**:
```
outputs/
├── conductor_enhanced_volatile_20251108_150631.json   # 75.60 fitness
├── conductor_enhanced_trending_20251108_151930.json   # 47.55 fitness
└── conductor_enhanced_ranging_20251108_153259.json    # 6.99 fitness
```

**Baseline (Old Conductor)**:
```
outputs/
├── conductor_enhanced_volatile_20251108_161234.json   # 71.92 fitness
├── conductor_enhanced_trending_20251108_163503.json   # 45.67 fitness
└── conductor_enhanced_ranging_20251108_191609.json    # 5.90 fitness (5 extinctions)
```

**DANN (Regime-Invariant)**:
```
outputs/
├── conductor_enhanced_volatile_20251108_174554.json   # 71.92 fitness (14.5% cache!)
├── conductor_enhanced_trending_20251108_180319.json   # 45.67 fitness (10.9% cache!)
└── conductor_enhanced_ranging_20251109_013035.json    # 5.90 fitness (11.4% cache!)
```

---

## Key Metrics to Track

### Ensemble Performance
- [ ] Total Return (target: ≥+189%)
- [ ] Sharpe Ratio (target: ≥1.01)
- [ ] Total Trades (expect: ~77)
- [ ] Win Rate (target: ≥41%)
- [ ] Max Drawdown (target: ≤15%)
- [ ] Regime Transition Smoothness

### Specialist Usage
- [ ] Volatile specialist activation %
- [ ] Trending specialist activation %
- [ ] Ranging specialist activation %
- [ ] Regime detection accuracy (69.2% baseline)

---

## Decision Points

### After Ensemble Test

**If Performance ≥ Phase 3A Baseline:**
✅ **Success!** Proceed to Phase 4 (Advanced Conductor Training)

**Options**:
1. Try DANN specialists ensemble (test efficiency vs performance)
2. Implement hybrid DANN + regime-specific approach
3. Move to production deployment (Phase 5)

**If Performance < Phase 3A Baseline:**
⚠️ **Investigate**:
- Check regime detection accuracy
- Verify specialist genome loading
- Compare individual specialist performance
- Analyze regime transition points

**Recovery Actions**:
1. Debug ensemble coordination logic
2. Retrain specialists with more generations
3. Adjust regime detection thresholds
4. Try alternative ensemble strategies

---

## Phase 4 Options (After Ensemble Success)

### Option A: Hybrid DANN + Regime-Specific 🔥 **RECOMMENDED**

**Concept**: Combine DANN's consistency with regime-specific peak performance

**Timeline**: 2-3 days

**Expected Benefit**: +5-10% fitness improvement, maintained efficiency

**Implementation**:
```python
# Pseudo-code
base_params = dann_conductor.predict(market_state)
regime = detect_regime(market_state)
adjusted_params = regime_conductor[regime].adjust(base_params)
final_params = combine(base_params, adjusted_params)
```

### Option B: Multi-Task DANN

**Concept**: Train DANN to predict GA params + hyperparameters simultaneously

**Timeline**: 3-4 days

**Expected Benefit**: More comprehensive optimization, single unified model

### Option C: Ensemble of DANN Conductors

**Concept**: Multiple DANN models with different architectures, ensemble predictions

**Timeline**: 2-3 days

**Expected Benefit**: Reduced variance, better generalization

---

## Common Commands Reference

### Training Commands

```powershell
# Train specialist with old conductor
python conductor_enhanced_trainer.py [volatile|trending|ranging]

# Train specialist with DANN conductor
python conductor_enhanced_trainer.py [volatile|trending|ranging] --use-dann

# Extract training data for DANN
python extract_dann_training_data.py

# Train new DANN conductor
python domain_adversarial_conductor.py
```

### Testing Commands

```powershell
# Test ensemble
python ensemble_conductor.py

# Test individual specialist
python -c "from trading_specialist import TradingSpecialist; spec = TradingSpecialist('volatile'); spec.load_genome('outputs/conductor_enhanced_volatile_20251108_150631.json'); print(spec.test())"

# Test regime detector
python -c "from regime_detector import RegimeDetector; rd = RegimeDetector(); rd.load('outputs/regime_detector_model.pth'); print(f'Accuracy: {rd.test_accuracy()}')"
```

### Analysis Commands

```powershell
# Compare training results
python -c "
import json
import glob

for file in glob.glob('outputs/conductor_enhanced_*.json'):
    data = json.load(open(file))
    print(f'{file}: Fitness={data[\"best_fitness\"]}, Cache={data.get(\"cache_stats\", {}).get(\"hit_rate\", \"N/A\")}%')
"

# View DANN training progress
python -c "
import json
data = json.load(open('outputs/dann_conductor_20251108_145051.json'))
print(f'Epochs: {data[\"epochs_trained\"]}')
print(f'Best Val Loss: {data[\"best_val_param_loss\"]}')
print(f'Regime Accuracy: {data[\"final_val_regime_accuracy\"]}')
"
```

---

## Troubleshooting

### Issue: Ensemble fails to load specialists

**Solution**:
```powershell
# Check if genome files exist
dir outputs\conductor_enhanced_*.json

# Verify genome structure
python -c "import json; data = json.load(open('outputs/conductor_enhanced_volatile_20251108_150631.json')); print(list(data.keys()))"
```

### Issue: CUDA out of memory

**Solution**:
```powershell
# Reduce batch size in DANN training
# Edit domain_adversarial_conductor.py line ~500
# Change batch_size=32 to batch_size=16

# Or use CPU
# Edit conductor_enhanced_trainer.py
# Change device='cuda' to device='cpu'
```

### Issue: Import errors

**Solution**:
```powershell
# Ensure in correct directory
cd PRICE-DETECTION-TEST-1

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Verify file exists
python -c "import os; print(os.path.exists('domain_adversarial_conductor.py'))"
```

---

## Success Criteria Checklist

### Phase 3D Complete When:
- [ ] Ensemble tested with Phase 3A specialists
- [ ] Performance compared vs Phase 3A baseline
- [ ] Regime transition analysis complete
- [ ] Results documented
- [ ] Decision made on Phase 4 direction

### Ready for Phase 4 When:
- [ ] Ensemble performance validated (≥ baseline)
- [ ] All metrics captured and analyzed
- [ ] Phase 4 option selected
- [ ] Implementation plan created

### Ready for Phase 5 (Production) When:
- [ ] Ensemble consistently outperforms baseline
- [ ] Edge cases tested
- [ ] Error handling robust
- [ ] Monitoring dashboard ready
- [ ] Paper trading plan defined

---

## Quick Context Recovery

**If resuming after break:**

1. Read `PHASE_3C_COMPLETE.md` (comprehensive results)
2. Check this file for commands
3. Run ensemble test (next step)
4. Compare results
5. Decide Phase 4 direction

**Key Numbers to Remember:**
- DANN: 99.85% accuracy, 31.67% regime invariance
- Cache improvement: 10-18x (efficiency breakthrough!)
- Phase 3A best: V=75.60, T=47.55, R=6.99
- Phase 3A ensemble: +189% return, Sharpe 1.01, 77 trades

**Current Goal**: Validate ensemble performance, then proceed to Phase 4 advanced conductor training.

---

## Contact Points / Resources

- **Code Documentation**: See docstrings in each .py file
- **Phase 3C Details**: `PHASE_3C_COMPLETE.md`
- **Architecture Docs**: `PHASE_3C_PATH_A_DANN.md` (implementation plan)
- **Research Paper**: Domain-Adversarial Training (arXiv:1505.07818)
- **Previous Phases**: `PHASE_3A_COMPLETE.md`, `PHASE_2_COMPLETE.md`

---

**Status**: Ready to proceed with Phase 3D Ensemble Testing! 🚀
