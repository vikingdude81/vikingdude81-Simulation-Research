# Phase 3C Path B: Regime-Specific Conductors

**Status**: 🔄 BACKUP PATH - Implement if Path A (DANN) fails  
**Date**: November 8, 2025  
**Time Estimate**: 2-3 hours  
**Risk Level**: Low (proven architecture)

---

## Executive Summary

Train **3 separate conductors** - one for each market regime (volatile, trending, ranging). Each conductor is trained ONLY on its regime's data, ensuring perfect specialization. The ensemble system will load the appropriate conductor based on detected regime.

---

## Motivation

### Current Problem

We have **one conductor trained on volatile data** used across all regimes:
- Trained on: 643 days of volatile data
- Used for: Volatile (643 days), Trending (1,121 days), Ranging (2,078 days)
- **Problem**: Domain mismatch for trending and ranging predictions

### Regime-Specific Solution

Train **3 separate conductors**:
1. **Volatile Conductor**: Trained on 643 days of volatile data
2. **Trending Conductor**: Trained on 1,121 days of trending data
3. **Ranging Conductor**: Trained on 2,078 days of ranging data

**Benefits**:
- Perfect specialization (no domain mismatch)
- Simple, straightforward approach (proven architecture)
- Lower implementation risk (just need more training data)

**Expected Improvements**:
- Volatile: 75.60 → 77-80 (+2-6%)
- Trending: 47.55 → 52-57 (+9-20%) ← BIGGEST GAIN
- Ranging: 6.99 → 8-10 (+14-43%)

---

## Implementation Plan

### Phase 1: Prepare Regime-Specific Training Data (45 min)

**Task**: Extract training samples from ORIGINAL regime data (before conductor training).

**Data Sources**:
```python
# Need to split original BTC data by regime
volatile_data = df[df['regime'] == 'volatile']   # 570 days
trending_data = df[df['regime'] == 'trending']   # 1,137 days
ranging_data = df[df['regime'] == 'ranging']     # 2,179 days
```

**Training Process for Each Conductor**:
1. Load regime-specific data
2. Train Enhanced ML Predictor on regime data (100 epochs)
3. Extract 900 samples from training history
4. Train GA Conductor on samples (50 epochs)

**Files to Create**:
```
models/ga_conductor_volatile.pth
models/ga_conductor_trending.pth
models/ga_conductor_ranging.pth
```

### Phase 2: Train Regime-Specific ML Predictors (1 hour)

**Volatile ML Predictor** (~20 min):
```bash
python train_enhanced_ml_predictor.py --regime volatile --output models/ml_predictor_volatile.pth
```

**Trending ML Predictor** (~20 min):
```bash
python train_enhanced_ml_predictor.py --regime trending --output models/ml_predictor_trending.pth
```

**Ranging ML Predictor** (~20 min):
```bash
python train_enhanced_ml_predictor.py --regime ranging --output models/ml_predictor_ranging.pth
```

**Configuration**:
```python
{
    "input_size": 13,  # Market features
    "hidden_size": 256,
    "output_size": 2,  # Long prob, short prob
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001
}
```

**Expected Results**:
- Volatile val loss: 0.0005-0.0008 (similar to current)
- Trending val loss: 0.0005-0.0008 (should be better than current volatile-trained)
- Ranging val loss: 0.0005-0.0008 (should be better than current)

### Phase 3: Train Regime-Specific GA Conductors (1.5 hours)

**Extract Training Data**:

For each regime, run baseline specialist training (without conductor) to generate training samples:

```bash
# Volatile baseline training (generate samples)
python baseline_trainer.py volatile --generations 300 --save-history

# Trending baseline training (generate samples)  
python baseline_trainer.py trending --generations 300 --save-history

# Ranging baseline training (generate samples)
python baseline_trainer.py ranging --generations 300 --save-history
```

**Train Conductors**:

```bash
# Volatile conductor (~30 min)
python train_ga_conductor.py --regime volatile --samples outputs/baseline_volatile_history.json --output models/ga_conductor_volatile.pth

# Trending conductor (~30 min)
python train_ga_conductor.py --regime trending --samples outputs/baseline_trending_history.json --output models/ga_conductor_trending.pth

# Ranging conductor (~30 min)
python train_ga_conductor.py --regime ranging --samples outputs/baseline_ranging_history.json --output models/ga_conductor_ranging.pth
```

**Configuration**:
```python
{
    "input_size": 25,  # Regime state features
    "hidden_size": 512,
    "output_size": 12,  # 12 GA parameters
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.0001
}
```

**Expected Results**:
- Volatile conductor val loss: 0.045-0.055 (similar to current)
- Trending conductor val loss: 0.040-0.050 (better - trained on trending data!)
- Ranging conductor val loss: 0.040-0.050 (better - trained on ranging data!)

### Phase 4: Modify Ensemble to Use Regime-Specific Conductors (15 min)

**File**: `ensemble_conductor_multimodel.py` (new file, copy from `ensemble_conductor.py`)

**Key Changes**:
```python
class EnsembleConductorMultiModel:
    def __init__(self):
        # Load regime detector
        self.regime_detector = torch.load('models/regime_detector.pth')
        
        # Load ALL 3 specialists (trained with their regime-specific conductors)
        self.specialists = {
            'volatile': self._load_specialist('volatile'),
            'trending': self._load_specialist('trending'),
            'ranging': self._load_specialist('ranging')
        }
        
        # Each specialist has its own conductor baked in from training
        # No need to load conductors separately!
    
    def predict(self, market_data):
        # Detect regime
        regime = self.regime_detector.predict(market_data)
        
        # Use regime-specific specialist
        specialist = self.specialists[regime]
        signal = specialist.generate_signal(market_data)
        
        return signal, regime
```

**Note**: Each specialist was trained with its regime-specific conductor, so the genome already encodes parameters optimized for that regime. No dynamic conductor switching needed!

### Phase 5: Retrain Specialists with Regime-Specific Conductors (1.5 hours)

**Volatile Specialist** (~30 min):
```bash
python conductor_enhanced_trainer_regime_specific.py volatile --conductor models/ga_conductor_volatile.pth
```

**Trending Specialist** (~30 min):
```bash
python conductor_enhanced_trainer_regime_specific.py trending --conductor models/ga_conductor_trending.pth
```

**Ranging Specialist** (~30 min):
```bash
python conductor_enhanced_trainer_regime_specific.py ranging --conductor models/ga_conductor_ranging.pth
```

**Expected Results**:
- Volatile: 75.60 → 77-80 (+2-6% improvement)
- Trending: 47.55 → 52-57 (+9-20% improvement) ← KEY GAIN
- Ranging: 6.99 → 8-10 (+14-43% improvement)

### Phase 6: Ensemble Testing (10 min)

**Test ensemble with regime-specific specialists**:
```bash
python ensemble_conductor_multimodel.py
```

**Expected Results**:
- Total Return: +189% → +210-230%
- Sharpe Ratio: 1.01 → 1.05-1.10
- Max Drawdown: -11% → -9% to -11%
- Number of Trades: 77 → 75-85 (similar activity)
- Win Rate: 41.6% → 43-46% (better predictions)

---

## Architecture Comparison

### Current System (Single Conductor)
```
Market Data → Regime Detector → Regime Label
                                      ↓
Enhanced ML Predictor (volatile-trained) → Signal
                ↓
GA Conductor (volatile-trained) → 12 Parameters
                ↓
Trading Specialist (volatile/trending/ranging) → Trade
```

**Problem**: Conductor trained on volatile data, used for all regimes.

### Path B System (Regime-Specific Conductors)
```
Market Data → Regime Detector → Regime Label
                                      ↓
                   ┌──────────────────┴──────────────────┐
                   ↓                  ↓                   ↓
            [Volatile]          [Trending]          [Ranging]
                   ↓                  ↓                   ↓
     ML Predictor (volatile)  ML Predictor (trending)  ML Predictor (ranging)
                   ↓                  ↓                   ↓
     GA Conductor (volatile)  GA Conductor (trending)  GA Conductor (ranging)
                   ↓                  ↓                   ↓
     Specialist (volatile)    Specialist (trending)    Specialist (ranging)
                   ↓                  ↓                   ↓
                   └──────────────────┬──────────────────┘
                                      ↓
                                   Trade
```

**Benefit**: Perfect specialization - each conductor trained on its regime's data.

---

## Success Criteria

### Minimum Success (Acceptable)
- ✅ All 3 conductors train successfully (val loss < 0.06)
- ✅ All 3 specialists maintain fitness (no regressions)
- ✅ Ensemble Sharpe ratio > 0.95

### Target Success (Expected)
- ✅ Trending conductor val loss < 0.05 (better than current)
- ✅ Trending specialist improves +10%+ (47.55 → 52+)
- ✅ Ranging specialist improves +10%+ (6.99 → 7.7+)
- ✅ Ensemble Sharpe ratio > 1.05
- ✅ Ensemble return > +200%

### Outstanding Success (Stretch Goal)
- ✅ ALL conductors val loss < 0.045
- ✅ ALL specialists improve +15%+
- ✅ Ensemble Sharpe ratio > 1.10
- ✅ Ensemble return > +220%

---

## Risk Assessment

### Technical Risks

1. **Data Splitting Complexity** (LOW)
   - Risk: Need to properly split original data by regime
   - Mitigation: Use existing regime labels from RegimeDetector
   - Fallback: Manual data inspection and validation

2. **Training Data Quality** (LOW)
   - Risk: Some regimes might have insufficient data
   - Mitigation: We have plenty (570 volatile, 1,137 trending, 2,179 ranging)
   - Fallback: Use data augmentation if needed

3. **Model Management Complexity** (MEDIUM)
   - Risk: 6 models to manage (3 ML predictors + 3 conductors)
   - Mitigation: Clear naming conventions, organized file structure
   - Fallback: Scripted model loading/saving

### Performance Risks

1. **No Improvement Over Current** (LOW)
   - Risk: Regime-specific training might not help
   - Impact: Wasted 2-3 hours
   - Mitigation: Very unlikely - specialization almost always helps

2. **Overfitting to Regime** (MEDIUM)
   - Risk: Specialists too specialized, can't handle regime transitions
   - Impact: Poor performance during regime changes
   - Mitigation: Ensemble system already handles transitions via RegimeDetector

3. **Inconsistent Performance Across Regimes** (LOW)
   - Risk: One regime improves, others regress
   - Impact: Mixed results
   - Mitigation: If one regresses, use old model for that regime

---

## Expected Timeline

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| 1 | Prepare regime-specific data | 45 min | 0:45 |
| 2 | Train 3 ML predictors | 1 hour | 1:45 |
| 3 | Train 3 GA conductors | 1.5 hours | 3:15 |
| 4 | Modify ensemble system | 15 min | 3:30 |
| 5 | Retrain all 3 specialists | 1.5 hours | 5:00 |
| 6 | Test ensemble | 10 min | 5:10 |
| | **TOTAL** | **~5 hours** | |

**Note**: Phases 2, 3, and 5 can be partially parallelized (train 3 models at once).

**Wall Time with Parallelization**: ~3-3.5 hours

---

## Comparison to Path A (DANN)

| Aspect | Path A (DANN) | Path B (Regime-Specific) |
|--------|---------------|--------------------------|
| **Implementation Time** | 3-4 hours | 2-3 hours |
| **Complexity** | High (adversarial training) | Medium (standard training) |
| **Risk** | Medium (new approach) | Low (proven approach) |
| **Expected Improvement** | High (10-20%+) | Medium (5-15%) |
| **Elegance** | Very high (single model) | Low (3 separate models) |
| **Regime Transitions** | Seamless (automatic) | Manual (ensemble switches) |
| **Future Extensibility** | High (add regimes easily) | Low (train new conductor each time) |
| **Research Value** | High (novel application) | Low (straightforward) |
| **Model Management** | Simple (1 conductor) | Complex (3 conductors) |

**When to Use Path B**:
- If Path A (DANN) fails or doesn't improve performance
- If you prefer simpler, lower-risk approach
- If time is limited (Path B is faster to implement)
- If you want guaranteed specialization

---

## Implementation Files

### New Files to Create

1. **train_enhanced_ml_predictor.py** (~200 lines)
   - Modified version that accepts regime argument
   - Filters data by regime
   - Trains ML predictor on regime-specific data
   - Saves to regime-specific path

2. **baseline_trainer.py** (~300 lines)
   - Train specialist WITHOUT conductor (baseline)
   - Save generation history for conductor training
   - Used to generate conductor training data

3. **train_ga_conductor.py** (~250 lines)
   - Modified version that accepts regime argument
   - Loads regime-specific training samples
   - Trains conductor on regime-specific data
   - Saves to regime-specific path

4. **conductor_enhanced_trainer_regime_specific.py** (~750 lines)
   - Modified version of conductor_enhanced_trainer.py
   - Accepts --conductor flag to specify conductor path
   - Loads regime-specific conductor
   - Otherwise identical to current trainer

5. **ensemble_conductor_multimodel.py** (~350 lines)
   - Modified version of ensemble_conductor.py
   - Loads all 3 regime-specific specialists
   - Each specialist uses its regime-specific conductor
   - Clearer naming to distinguish from single-conductor version

### Modified Files

None! We'll create new files to keep original system intact.

---

## Validation Checklist

Before declaring Path B successful:

- [ ] All 3 ML predictors trained successfully (val loss < 0.001)
- [ ] All 3 GA conductors trained successfully (val loss < 0.06)
- [ ] Volatile specialist improves or maintains (75.60 → 76+)
- [ ] Trending specialist improves significantly (47.55 → 50+)
- [ ] Ranging specialist improves or maintains (6.99 → 7+)
- [ ] Ensemble return > +200% (better than +189%)
- [ ] Ensemble Sharpe > 1.00 (maintain quality)
- [ ] Ensemble max DD < -12% (maintain risk control)
- [ ] No regressions in any specialist
- [ ] Model files properly organized and labeled

If ≥8 of these criteria met: **Path B SUCCESS** ✅  
If 6-7 met: **Path B PARTIAL SUCCESS** (usable, but room for improvement)  
If ≤5 met: **Path B UNSUCCESSFUL** (investigate issues)

---

## Advantages Over Path A

1. **Lower Risk**: Proven architecture, just needs more training data
2. **Faster**: 2-3 hours vs 3-4 hours (if no parallelization)
3. **Simpler**: No adversarial training, no gradient reversal layer
4. **Guaranteed Specialization**: Each conductor perfectly specialized
5. **Easier Debugging**: If one regime fails, others unaffected

---

## Disadvantages vs Path A

1. **Less Elegant**: 3 models instead of 1
2. **Model Management**: More files to organize and deploy
3. **Regime Transitions**: Manual switching vs seamless adaptation
4. **Future Extensibility**: Adding new regime = train new conductor
5. **Research Value**: Straightforward approach, less novel

---

## Decision Flow

```
Start Phase 3C
      ↓
Try Path A (DANN)
      ↓
Did Path A succeed?
   ↙        ↘
  YES       NO
   ↓         ↓
Use DANN   Try Path B
   ↓         ↓
   ✅     Did Path B succeed?
        ↙            ↘
       YES           NO
        ↓             ↓
   Use Regime    Keep Current
   Specific      (investigate)
        ↓
        ✅
```

**Expected Outcome**: Path A succeeds → use DANN (more elegant)  
**Fallback**: Path A fails → use Path B (safer bet)  
**Worst Case**: Both fail → investigate root causes, keep current system

---

## Code Snippets

### Training Regime-Specific ML Predictor

```python
def train_ml_predictor_regime_specific(regime, data_path, output_path):
    """
    Train Enhanced ML Predictor on regime-specific data.
    
    Args:
        regime: 'volatile', 'trending', or 'ranging'
        data_path: Path to full BTC dataset
        output_path: Where to save trained model
    """
    # Load full data
    df = pd.read_csv(data_path)
    
    # Filter by regime
    regime_data = df[df['regime'] == regime].copy()
    print(f"Training on {len(regime_data)} days of {regime} data")
    
    # Create model
    model = EnhancedMLPredictor(
        input_size=13,
        hidden_size=256,
        output_size=2
    ).to(device)
    
    # Train
    train_model(model, regime_data, epochs=100)
    
    # Save
    torch.save(model.state_dict(), output_path)
    print(f"Saved to {output_path}")
```

### Loading Regime-Specific Conductor

```python
class ConductorEnhancedTrainerRegimeSpecific:
    def __init__(self, regime, conductor_path):
        self.regime = regime
        
        # Load regime-specific data
        self.regime_data = self._load_regime_data(regime)
        
        # Load regime-specific conductor
        self.conductor = GAConductor(input_size=25, hidden_size=512, output_size=12)
        self.conductor.load_state_dict(torch.load(conductor_path))
        self.conductor.eval()
        
        print(f"Loaded {regime} conductor from {conductor_path}")
```

---

## Next Actions

1. ✅ Create this documentation
2. ✅ Commit Phase 3A progress
3. Wait for Path A (DANN) results
4. IF Path A fails:
   - Implement Path B:
     * Create train_enhanced_ml_predictor.py
     * Create baseline_trainer.py
     * Create train_ga_conductor.py
     * Train 3 ML predictors
     * Train 3 GA conductors
     * Modify ensemble system
     * Retrain specialists
     * Test ensemble
5. IF Path B succeeds → Document and commit
6. IF Path B fails → Investigate root causes, analyze data

**Status**: 📋 READY TO IMPLEMENT (if Path A fails)
