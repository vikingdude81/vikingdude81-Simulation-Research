# Phase 3C Path A: Domain-Adversarial Neural Network (DANN) Conductor

**Status**: 🎯 PRIMARY PATH - Implement First  
**Date**: November 8, 2025  
**Time Estimate**: 3-4 hours  
**Risk Level**: Medium (research-based, new architecture)

---

## Executive Summary

Implement a **single universal conductor** using Domain-Adversarial Training of Neural Networks (DANN) that works across all 3 market regimes (volatile, trending, ranging) without manual switching. The conductor will learn regime-invariant features that predict optimal GA parameters regardless of current market conditions.

---

## Motivation

### Current Problem

Our current system has **one conductor trained on volatile data** used across all 3 regimes:
- ✅ Works great for volatile (75.60 fitness)
- ✅ Works okay for trending (47.55 fitness, +3.3% above baseline)
- ✅ Works okay for ranging (6.99 fitness, +529% above baseline)

But there's a **domain mismatch**:
- Volatile conductor sees volatile training data
- But must predict parameters for trending/ranging specialists
- This is a classic **domain adaptation problem**!

### DANN Solution

Train a **single conductor** that:
1. Learns features that work across ALL regimes (domain-invariant)
2. Automatically adapts to regime transitions (no manual switching)
3. Eliminates domain mismatch (trained on all 3 regimes simultaneously)

**Expected Benefits**:
- Trending could improve further: 47.55 → 50-55+ (better parameter predictions)
- Ranging could improve: 6.99 → 8-10+ (more stable predictions)
- Volatile might improve slightly: 75.60 → 78-80+
- Seamless regime transitions (gradual adaptation, no switching lag)

---

## Domain-Adversarial Training (DANN) Explained

### Core Concept

DANN solves the problem: **"Train on one domain, perform well on different domains"**

In our case:
- **Domains**: Volatile, Trending, Ranging market regimes
- **Task**: Predict 12 GA parameters that maximize fitness
- **Challenge**: Features that work in volatile might not work in ranging

DANN forces the model to learn **regime-invariant features** - patterns that indicate good parameter values regardless of which regime we're in.

### Architecture Components

```
Input (Market Features)
        ↓
┌───────────────────┐
│ Feature Extractor │ ← Shared across all regimes
│       (G_f)       │    Learns universal patterns
└───────────────────┘
        ↓
     Features
    ↙       ↘
   ↓         ↓ (Gradient Reversal Layer - GRL)
┌─────────┐ ┌─────────────────┐
│ Label   │ │ Domain          │
│Predictor│ │ Classifier      │
│  (G_y)  │ │     (G_d)       │
└─────────┘ └─────────────────┘
    ↓               ↓
12 GA Params    Regime Class
(mutation,      (0=volatile,
crossover,      1=trending,
etc.)           2=ranging)
```

### How It Works

1. **Feature Extractor (G_f)**:
   - Takes market features as input (13 features: returns, volatility, RSI, etc.)
   - Outputs abstract representation (hidden features)
   - **Goal**: Find patterns that predict good GA parameters

2. **Label Predictor (G_y)**:
   - Takes features from G_f
   - Outputs 12 GA parameters (mutation rate, crossover rate, population diversity, etc.)
   - **Loss**: Mean Squared Error vs. actual parameters that worked
   - **Gradient**: Tells G_f how to improve parameter predictions

3. **Domain Classifier (G_d)**:
   - Takes features from G_f (through GRL)
   - Outputs regime classification (volatile/trending/ranging)
   - **Loss**: Cross-Entropy vs. actual regime label
   - **Gradient (REVERSED)**: Tells G_f to HIDE regime information

4. **Gradient Reversal Layer (GRL)**:
   - **Forward pass**: Identity (just passes features through)
   - **Backward pass**: Multiplies gradient by -1
   - **Effect**: G_f gets conflicting signals:
     * From G_y: "Make features useful for parameter prediction"
     * From G_d (reversed): "Make features USELESS for regime detection"
   - **Result**: G_f learns regime-invariant features that still predict parameters well!

### Training Loss

```python
Total Loss = L_y + (λ * L_d)

where:
  L_y = MSE(predicted_params, true_params)      # Parameter prediction loss
  L_d = CrossEntropy(regime_pred, true_regime)  # Domain classification loss
  λ   = Domain adaptation strength (typically 0.1-1.0)
```

**The Adversarial Game**:
- Domain Classifier (G_d) tries to detect regime from features
- Feature Extractor (G_f) tries to fool Domain Classifier while predicting parameters
- Equilibrium: Features contain parameter-predictive info, but NO regime-specific info

---

## Implementation Plan

### Phase 1: Prepare Training Data (30 min)

**Task**: Extract training samples from all 3 regime training histories.

**Data Structure**:
```python
{
    "regime": "volatile",  # or "trending", "ranging"
    "features": [13 values],  # Market state features
    "parameters": [12 values],  # GA parameters used
    "fitness": 75.60,  # Resulting fitness
    "generation": 150  # Which generation this was from
}
```

**Source Files**:
- `outputs/conductor_enhanced_volatile_20251108_111639.json`
- `outputs/conductor_enhanced_trending_20251108_114301.json`
- `outputs/conductor_enhanced_ranging_20251108_141359.json`

**Expected Samples**:
- Volatile: 300 generations × 200 agents = 60,000 samples
- Trending: 300 generations × 200 agents = 60,000 samples
- Ranging: 300 generations × 200 agents = 60,000 samples
- **Total: ~180,000 training samples**

**Data Split**:
- Training: 80% (144,000 samples)
- Validation: 20% (36,000 samples)
- Balance regimes: Equal representation from each

### Phase 2: Implement DANN Architecture (1 hour)

**File**: `domain_adversarial_conductor.py`

**Components**:

1. **Gradient Reversal Layer**:
```python
class GradientReversalLayer(torch.autograd.Function):
    """
    Passes features forward unchanged.
    Reverses (negates) gradients during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def gradient_reversal_layer(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)
```

2. **Feature Extractor (G_f)**:
```python
class FeatureExtractor(nn.Module):
    def __init__(self, input_size=13, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)
```

3. **Label Predictor (G_y)**:
```python
class ParameterPredictor(nn.Module):
    def __init__(self, feature_size=64, output_size=12):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid()  # Output [0, 1] range for all parameters
        )
    
    def forward(self, features):
        return self.network(features)
```

4. **Domain Classifier (G_d)**:
```python
class RegimeClassifier(nn.Module):
    def __init__(self, feature_size=64, num_regimes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_regimes)  # Softmax applied in loss
        )
    
    def forward(self, features):
        return self.network(features)
```

5. **Complete DANN Conductor**:
```python
class DomainAdversarialConductor(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, output_size=12, num_regimes=3):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_size, hidden_size)
        self.parameter_predictor = ParameterPredictor(64, output_size)
        self.regime_classifier = RegimeClassifier(64, num_regimes)
        self.lambda_ = 0.5  # Domain adaptation strength
    
    def forward(self, x, alpha=1.0):
        # Extract features
        features = self.feature_extractor(x)
        
        # Predict parameters (main task)
        parameters = self.parameter_predictor(features)
        
        # Classify regime (adversarial task, with gradient reversal)
        reversed_features = gradient_reversal_layer(features, alpha * self.lambda_)
        regime_pred = self.regime_classifier(reversed_features)
        
        return parameters, regime_pred
```

### Phase 3: Training Loop (1.5 hours)

**Configuration**:
```python
{
    "batch_size": 256,
    "learning_rate": 0.0003,
    "num_epochs": 100,
    "lambda_schedule": "linear",  # 0 → 1 over training
    "early_stopping_patience": 10,
    "device": "cuda"
}
```

**Training Process**:
```python
def train_dann_conductor(model, train_loader, val_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    param_criterion = nn.MSELoss()
    regime_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config['num_epochs']):
        # Lambda schedule: gradually increase domain adaptation
        p = epoch / config['num_epochs']
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # 0 → 1
        
        model.train()
        train_losses = []
        
        for batch_features, batch_params, batch_regimes in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            pred_params, pred_regimes = model(batch_features, alpha=alpha)
            
            # Compute losses
            L_y = param_criterion(pred_params, batch_params)
            L_d = regime_criterion(pred_regimes, batch_regimes)
            
            # Total loss
            total_loss = L_y + (model.lambda_ * L_d)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_losses.append({
                'total': total_loss.item(),
                'param': L_y.item(),
                'regime': L_d.item()
            })
        
        # Validation
        val_loss = validate(model, val_loader, param_criterion, regime_criterion)
        
        # Early stopping check
        if early_stopping_triggered(val_loss, patience=10):
            break
        
        # Log progress
        print(f"Epoch {epoch}: Train Loss={np.mean([l['total'] for l in train_losses]):.6f}, "
              f"Val Loss={val_loss:.6f}, Alpha={alpha:.3f}")
    
    return model
```

### Phase 4: Validation & Testing (30 min)

**Validation Metrics**:

1. **Parameter Prediction Accuracy**:
   - MSE on validation set
   - Compare to current conductor baseline

2. **Regime Invariance**:
   - Domain classifier accuracy on validation
   - **Goal**: Close to random (33% for 3 regimes)
   - **Interpretation**: Features don't leak regime information

3. **Cross-Regime Generalization**:
   - Test on each regime separately
   - Compare parameter predictions vs. actual optimal parameters

**Expected Results**:
```
Parameter Prediction MSE: 0.02-0.05 (lower is better)
Domain Classifier Accuracy: 35-45% (close to 33% = good!)
Cross-Regime MSE:
  - Volatile:  0.03 ± 0.01
  - Trending:  0.03 ± 0.01
  - Ranging:   0.03 ± 0.01
  (Similar errors across regimes = good invariance!)
```

### Phase 5: Retrain Specialists with DANN Conductor (1.5 hours)

**Modifications to `conductor_enhanced_trainer.py`**:

```python
class ConductorEnhancedTrainer:
    def __init__(self, regime, ...):
        # Load DANN conductor instead of original
        self.conductor = torch.load('models/dann_conductor.pth')
        self.conductor.eval()
        # ... rest of initialization
    
    def _get_adaptive_params(self):
        # Use DANN conductor (no regime label needed!)
        with torch.no_grad():
            features = self._extract_features()
            params, _ = self.conductor(features)  # Ignore regime prediction
        
        # Extract 12 parameters
        mutation_rate = params[0].item()
        crossover_rate = params[1].item()
        # ... etc for all 12 parameters
        
        return {
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            # ... all 12 parameters
        }
```

**Training All 3 Specialists** (~45 min total):
```bash
# Volatile specialist
python conductor_enhanced_trainer_dann.py volatile

# Trending specialist  
python conductor_enhanced_trainer_dann.py trending

# Ranging specialist
python conductor_enhanced_trainer_dann.py ranging
```

**Expected Improvements**:
- Volatile: 75.60 → 78-82 (+4-9% improvement)
- Trending: 47.55 → 52-58 (+9-22% improvement) ← BIGGEST GAIN
- Ranging: 6.99 → 8-11 (+14-57% improvement)

### Phase 6: Ensemble Testing (10 min)

**Test ensemble with DANN-trained specialists**:
```bash
python ensemble_conductor.py
```

**Expected Results**:
- Total Return: +189% → +220-250%
- Sharpe Ratio: 1.01 → 1.10-1.20
- Max Drawdown: -11% → -9% to -11% (similar or better)
- Number of Trades: 77 → 80-100 (slightly more active)
- Win Rate: 41.6% → 43-47% (better parameter predictions)

---

## Success Criteria

### Minimum Success (Acceptable)
- ✅ Parameter prediction MSE < 0.10
- ✅ Domain classifier accuracy < 50% (shows invariance)
- ✅ All 3 specialists maintain or improve fitness
- ✅ Ensemble Sharpe ratio > 0.95

### Target Success (Expected)
- ✅ Parameter prediction MSE < 0.05
- ✅ Domain classifier accuracy 35-45% (near random)
- ✅ Trending specialist improves by +10%+ (47.55 → 52+)
- ✅ Ensemble Sharpe ratio > 1.05
- ✅ Ensemble return > +200%

### Outstanding Success (Stretch Goal)
- ✅ Parameter prediction MSE < 0.03
- ✅ Domain classifier accuracy 30-40% (strong invariance)
- ✅ ALL specialists improve by +15%+
- ✅ Ensemble Sharpe ratio > 1.15
- ✅ Ensemble return > +250%
- ✅ Max drawdown < -10%

---

## Risk Assessment

### Technical Risks

1. **GRL Implementation Complexity** (LOW)
   - Risk: Custom autograd function could have bugs
   - Mitigation: Use well-tested reference implementation
   - Fallback: Manually negate gradients without custom layer

2. **Hyperparameter Sensitivity** (MEDIUM)
   - Risk: Lambda schedule, learning rate, architecture size
   - Mitigation: Start with proven values from paper
   - Fallback: Grid search or manual tuning

3. **Training Instability** (MEDIUM)
   - Risk: Adversarial training can be unstable
   - Mitigation: Careful lambda scheduling (0 → 1 gradually)
   - Fallback: Lower lambda (0.1-0.3) for stability

4. **Overfitting** (LOW)
   - Risk: 180,000 samples should be plenty
   - Mitigation: Dropout (0.2-0.3), early stopping
   - Fallback: Stronger regularization

### Performance Risks

1. **No Improvement Over Current Conductor** (LOW-MEDIUM)
   - Risk: DANN might not outperform current volatile-trained conductor
   - Impact: Wasted 3-4 hours, but learned something
   - Fallback: Use Path B (regime-specific conductors)

2. **Regression in Some Specialists** (LOW)
   - Risk: DANN might hurt one regime while helping others
   - Impact: Mixed results, unclear path forward
   - Fallback: Selective usage (DANN for trending/ranging, old for volatile)

3. **Regime Classifier Too Good** (MEDIUM)
   - Risk: Features still contain regime information
   - Impact: Not truly regime-invariant
   - Mitigation: Increase lambda, stronger GRL
   - Fallback: Add regime labels as explicit input (defeats purpose though)

---

## Expected Timeline

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| 1 | Extract training data from histories | 30 min | 0:30 |
| 2 | Implement DANN architecture | 1 hour | 1:30 |
| 3 | Train DANN conductor | 1.5 hours | 3:00 |
| 4 | Validate & test | 30 min | 3:30 |
| 5 | Retrain all 3 specialists | 1.5 hours | 5:00 |
| 6 | Test ensemble | 10 min | 5:10 |
| | **TOTAL** | **~5 hours** | |

**Note**: Phases 1-4 are sequential (can't parallelize). Phase 5 can parallelize (3 specialists at once = ~45 min wall time).

---

## Comparison to Path B (Regime-Specific)

| Aspect | Path A (DANN) | Path B (Regime-Specific) |
|--------|---------------|--------------------------|
| **Implementation Time** | 3-4 hours | 2-3 hours |
| **Complexity** | High (adversarial training) | Medium (standard training) |
| **Risk** | Medium (new approach) | Low (proven approach) |
| **Expected Improvement** | High (10-20%+) | Medium (5-10%) |
| **Elegance** | Very high (single model) | Low (3 separate models) |
| **Regime Transitions** | Seamless (automatic) | Manual (ensemble switches) |
| **Future Extensibility** | High (add more regimes easily) | Low (need to train new conductor) |
| **Research Value** | High (novel application) | Low (straightforward) |

**Recommendation**: Path A is more sophisticated and potentially more powerful. If it works, it's a better long-term solution. If it doesn't work after 4-5 hours, fall back to Path B.

---

## Implementation Files

### New Files to Create

1. **domain_adversarial_conductor.py** (~400 lines)
   - GradientReversalLayer class
   - FeatureExtractor network
   - ParameterPredictor network
   - RegimeClassifier network
   - DomainAdversarialConductor model
   - Training loop
   - Validation functions

2. **extract_dann_training_data.py** (~200 lines)
   - Load all 3 specialist training histories
   - Extract samples (features, parameters, fitness, regime)
   - Balance regime distribution
   - Save training/validation datasets

3. **conductor_enhanced_trainer_dann.py** (~750 lines)
   - Modified version of conductor_enhanced_trainer.py
   - Loads DANN conductor instead of original
   - Uses DANN for parameter predictions
   - Otherwise identical training loop

4. **validate_dann_conductor.py** (~150 lines)
   - Test parameter prediction accuracy
   - Test regime invariance (domain classifier accuracy)
   - Cross-regime generalization tests
   - Visualization of learned features

### Modified Files

None! We'll create new files to keep original system intact.

---

## Code Snippets

### Full GRL Implementation

```python
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) from Ganin et al.
    
    Forward pass: Identity (x → x)
    Backward pass: Negate gradient (∂L/∂x → -λ·∂L/∂x)
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Return negated gradient, scaled by lambda
        # Second return value is None (no gradient for lambda)
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
```

### Data Extraction Example

```python
def extract_training_samples(specialist_json_path, regime_label):
    """
    Extract training samples from specialist training history.
    
    Args:
        specialist_json_path: Path to outputs/conductor_enhanced_*.json
        regime_label: 0=volatile, 1=trending, 2=ranging
    
    Returns:
        List of (features, parameters, fitness) tuples
    """
    with open(specialist_json_path, 'r') as f:
        data = json.load(f)
    
    samples = []
    for gen_num in range(len(data['generations'])):
        gen_data = data['generations'][gen_num]
        
        # Extract features from this generation
        features = gen_data['features']  # 13 market features
        
        # Extract GA parameters used
        parameters = [
            gen_data['mutation_rate'],
            gen_data['crossover_rate'],
            # ... all 12 parameters
        ]
        
        # Get fitness achieved
        fitness = gen_data['best_fitness']
        
        samples.append({
            'features': features,
            'parameters': parameters,
            'fitness': fitness,
            'regime': regime_label,
            'generation': gen_num
        })
    
    return samples
```

---

## References

**Original Paper**: Ganin et al. (2015), "Domain-Adversarial Training of Neural Networks"  
**arXiv**: https://arxiv.org/abs/1505.07818

**Key Concepts**:
- Domain adaptation without labeled target data
- Gradient reversal for adversarial training
- Learning invariant feature representations
- Applications to cross-domain transfer learning

---

## Success Validation Checklist

Before declaring Path A successful:

- [ ] DANN conductor trained successfully (val loss converges)
- [ ] Domain classifier accuracy 35-45% (shows invariance)
- [ ] Parameter prediction MSE < 0.05 (accurate predictions)
- [ ] Cross-regime generalization uniform (similar MSE across regimes)
- [ ] Volatile specialist improves or maintains (75.60 → 76+)
- [ ] Trending specialist improves significantly (47.55 → 50+)
- [ ] Ranging specialist improves or maintains (6.99 → 7+)
- [ ] Ensemble return > +200% (better than +189%)
- [ ] Ensemble Sharpe > 1.00 (maintain quality)
- [ ] Ensemble max DD < -12% (maintain risk control)

If ≥8 of these criteria met: **Path A SUCCESS** ✅  
If 5-7 met: **Path A PARTIAL SUCCESS** (keep, but consider Path B too)  
If ≤4 met: **Path A UNSUCCESSFUL** (fall back to Path B)

---

## Next Actions

1. ✅ Create this documentation
2. Create Phase 3C Path B documentation (regime-specific backup plan)
3. Commit Phase 3A progress to GitHub
4. Begin Phase 3C Path A implementation:
   - Start with extract_dann_training_data.py
   - Implement domain_adversarial_conductor.py
   - Train DANN conductor
   - Validate regime invariance
   - Retrain specialists
   - Test ensemble
5. If Path A succeeds → Document and commit
6. If Path A fails → Switch to Path B

**Status**: 📋 READY TO IMPLEMENT
