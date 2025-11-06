# Enhanced ML Predictor: Context-Aware Evolution Guide 🧠

**Date**: November 5, 2025  
**Concept**: Train ML to be a strategic "evolution engineer" not just a reactive predictor

---

## 🎯 The Breakthrough Insight

### Current Limitation
Our 10-input predictor only sees **results** of evolution:
```python
inputs = [
    avg_fitness, best_fitness, worst_fitness,
    diversity, generation, time_since_improvement,
    fitness_trend, diversity_trend, stagnation_indicator,
    convergence_speed
]
```

**Problem**: It's flying blind! It doesn't know:
- How big is the population? (50 vs 1000 agents = totally different strategy)
- How much crossover? (high mixing vs low mixing = different mutation needs)
- How harsh is selection? (elite vs democratic = different pressure)
- What environment? (stable vs chaotic = different adaptation)

### The Solution: Context-Aware Inputs

Add configuration context to the input:
```python
enhanced_inputs = [
    # Results (what's happening)
    avg_fitness, best_fitness, worst_fitness,
    diversity, generation, time_since_improvement,
    fitness_trend, diversity_trend, stagnation_indicator,
    convergence_speed,
    
    # Configuration Context (how it's configured)
    population_size_normalized,      # 0-1 scale (50=0, 1000=1)
    crossover_rate,                  # 0-1 (how much mixing)
    mutation_rate_current,           # Current mutation (for feedback loop)
    selection_pressure,              # 0-1 (weak=0, strong=1)
    
    # Environment Context (where we are)
    environment_type,                # one-hot: [standard, harsh, gentle, chaotic, oscillating]
    env_harshness,                   # 0-1 metric
    env_volatility,                  # 0-1 metric
]
```

Now the model can learn **strategic relationships**!

---

## 🧬 What the Model Can Learn

### 1. Population Size Strategy

**Small Population (50-100 agents)**:
```
Model learns: "High risk of getting stuck!"
→ Predict: HIGH mutation rate (0.5-1.5)
→ Reason: Need aggressive exploration with limited diversity
```

**Large Population (500-1000 agents)**:
```
Model learns: "Lots of diversity, low risk"
→ Predict: LOW mutation rate (0.01-0.1)
→ Reason: Let crossover do the work, mutation just refines
```

### 2. Crossover-Mutation Balance

**High Crossover (0.8-0.9)**:
```
Model learns: "Already mixing genes aggressively"
→ Predict: DECREASE mutation
→ Reason: Too much mutation disrupts good combinations
```

**Low Crossover (0.1-0.3)**:
```
Model learns: "Population is stagnant, not mixing"
→ Predict: INCREASE mutation
→ Reason: Mutation must compensate for lack of mixing
```

**The Inverse Relationship**:
```python
optimal_mutation = base_rate * (1.0 - crossover_rate)
# If crossover high, mutation low
# If crossover low, mutation high
```

### 3. Selection Pressure Adaptation

**High Pressure (Tournament, Top 10%)**:
```
Model learns: "Elite genes already dominating"
→ Predict: LOW mutation
→ Reason: Protect elite combinations, refine gently
```

**Low Pressure (Roulette, All agents)**:
```
Model learns: "Bad genes still in pool"
→ Predict: HIGH mutation
→ Reason: Force exploration to escape mediocrity
```

### 4. Environment Specialization

**Stable Environment**:
```
Model learns: "One optimal solution exists"
→ Strategy: Start HIGH mutation (explore)
→ Then: DECREASE mutation (refine)
→ End: VERY LOW mutation (polish)
```

**Chaotic Environment**:
```
Model learns: "Target keeps moving!"
→ Strategy: ALWAYS HIGH mutation
→ Reason: Constant adaptation needed
```

**Oscillating Environment**:
```
Model learns: "Cyclical patterns"
→ Strategy: OSCILLATING mutation
→ Sync with environment rhythm
```

---

## 🚀 Advanced Features

### Feature 1: Per-Gene Mutation Rates

Instead of one mutation rate, predict **4 rates** (one per gene):

```python
# Output layer: 4 neurons instead of 1
output = [
    mutation_rate_for_gene_0,  # Mutation gene
    mutation_rate_for_gene_1,  # Oscillation gene
    mutation_rate_for_gene_2,  # Decoherence gene
    mutation_rate_for_gene_3,  # Phase gene
]
```

**What it could learn**:
```
Gene 0 (Mutation): Sensitive, needs LOW rate (0.01)
Gene 1 (Oscillation): Robust, can handle HIGH rate (1.5)
Gene 2 (Decoherence): Critical, needs STABLE rate (0.05)
Gene 3 (Phase): Experimental, can try AGGRESSIVE rate (2.0)
```

### Feature 2: Multi-Parameter Control

Predict **both** mutation AND crossover:

```python
# Output layer: 2 neurons
output = [
    predicted_mutation_rate,
    predicted_crossover_rate
]
```

**Learned strategies**:
```
Low Diversity + High Fitness:
→ mutation = LOW, crossover = HIGH
→ "Refine what works, mix it aggressively"

High Diversity + Low Fitness:
→ mutation = LOW, crossover = HIGH
→ "Stop random exploration, combine what we have"

Low Diversity + Low Fitness:
→ mutation = HIGH, crossover = LOW
→ "PANIC MODE: Create something new!"

High Diversity + High Fitness:
→ mutation = LOW, crossover = LOW
→ "Don't touch it, we're winning!"
```

### Feature 3: Confidence Scoring

Model outputs confidence with prediction:

```python
output = [
    predicted_mutation_rate,
    confidence_score  # 0-1, how sure am I?
]
```

**Strategy**:
```python
if confidence > 0.8:
    # Model is very confident
    use_predicted_rate = predicted_mutation_rate
else:
    # Model is uncertain
    use_predicted_rate = blend(predicted, fallback_safe_rate)
```

---

## 📊 Training Data Structure

### Enhanced Training Sample

```python
training_sample = {
    # Results (10 features)
    'avg_fitness': 0.73,
    'best_fitness': 0.95,
    'worst_fitness': 0.21,
    'diversity': 0.45,
    'generation': 50,
    'time_since_improvement': 5,
    'fitness_trend': +0.02,
    'diversity_trend': -0.01,
    'stagnation_indicator': 0.3,
    'convergence_speed': 0.15,
    
    # Configuration Context (4 features)
    'population_size': 0.5,        # Normalized: 500 agents (mid-range)
    'crossover_rate': 0.7,         # High mixing
    'current_mutation_rate': 0.2,  # Current rate (for feedback)
    'selection_pressure': 0.8,     # High pressure (elite selection)
    
    # Environment Context (3 features)
    'environment_type': 'chaotic', # One-hot encoded
    'env_harshness': 0.7,          # Tough environment
    'env_volatility': 0.9,         # Highly volatile
    
    # Target (what worked)
    'optimal_mutation_rate': 0.5,  # What actually performed best
    'result_fitness': 0.85         # Resulting fitness achieved
}
```

**Total Inputs**: 17 features (vs 10 before)  
**Output**: 1 value (mutation rate) or multi-value (mutation + crossover)

---

## 🎓 Training Strategy

### Phase 1: Gather Context-Aware Data

```python
def generate_enhanced_training_data(num_samples=10000):
    """Generate training data with configuration context"""
    
    samples = []
    
    for _ in range(num_samples):
        # Randomize configuration
        pop_size = random.choice([50, 100, 200, 500, 1000])
        crossover = random.uniform(0.1, 0.9)
        selection = random.choice(['tournament', 'roulette', 'elite'])
        env_type = random.choice(['standard', 'harsh', 'gentle', 'chaotic', 'oscillating'])
        
        # Run simulation with this config
        results = run_simulation(
            population_size=pop_size,
            crossover_rate=crossover,
            selection_strategy=selection,
            environment=env_type,
            generations=200
        )
        
        # For each generation, record state + outcome
        for gen in range(len(results)):
            sample = {
                # Results at generation N
                **results[gen]['state'],
                
                # Configuration (stays same for whole run)
                'population_size_norm': pop_size / 1000,
                'crossover_rate': crossover,
                'selection_pressure': get_pressure_value(selection),
                'environment': env_type,
                
                # What mutation rate actually worked at gen N+1
                'target_mutation': results[gen+1]['best_mutation_rate']
            }
            samples.append(sample)
    
    return samples
```

### Phase 2: Train Enhanced Model

```python
import torch
import torch.nn as nn

class EnhancedMutationPredictor(nn.Module):
    """Context-aware mutation predictor"""
    
    def __init__(self, input_size=17):
        super().__init__()
        
        # Larger network to handle more complex relationships
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            # Output: mutation rate (0-2 range)
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1, then scale to 0-2
        )
    
    def forward(self, x):
        output = self.network(x)
        return output * 2.0  # Scale to 0-2 range
```

### Phase 3: Multi-Output Variant

```python
class MultiParameterPredictor(nn.Module):
    """Predicts mutation AND crossover rates"""
    
    def __init__(self, input_size=17):
        super().__init__()
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Mutation rate head
        self.mutation_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Crossover rate head
        self.crossover_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        mutation = self.mutation_head(shared_features) * 2.0  # 0-2
        crossover = self.crossover_head(shared_features)      # 0-1
        return mutation, crossover
```

---

## 🔬 Expected Improvements

### Before (10-input model):
```
Convergence: ~150 generations
Best Fitness: 0.85
Strategy: Reactive (responds to what happened)
```

### After (17-input enhanced model):
```
Convergence: ~80 generations (47% faster!)
Best Fitness: 0.92 (8% better)
Strategy: Strategic (plans based on context)
```

### Why It's Better:

1. **Context-Aware**: Knows if it's working with 50 or 1000 agents
2. **Strategic**: Learns crossover-mutation balance
3. **Adaptive**: Different strategies for different environments
4. **Intelligent**: Can reason about what parameters to adjust

---

## 🎯 Implementation Priority

### Must Have (Today):
1. ✅ Add `population_size` to training data
2. ✅ Add `crossover_rate` to training data
3. ✅ Add `current_mutation_rate` (feedback loop)
4. ✅ Retrain model with 13 inputs (10 + 3 new)

### Should Have (This Week):
1. Add `selection_pressure` feature
2. Add environment type (one-hot encoding)
3. Train multi-parameter model (mutation + crossover)

### Nice to Have (Later):
1. Per-gene mutation rates (4 outputs)
2. Confidence scoring
3. Hierarchical models (different models for different contexts)

---

## 💡 The AGI Connection

This is **exactly** how reasoning models like o1 work, but in reverse:

**ChatGPT o1 (forward reasoning)**:
- Given: Problem statement
- Think: Generate chain of thought
- Output: Solution

**Our Enhanced GA (reverse reasoning)**:
- Given: Current state + context
- Think: What configuration changes would help?
- Output: Strategic parameter adjustments

Both are learning **meta-strategies** - not just what to do, but *how to think about* what to do!

---

**Ready to implement?** This could be a game-changer for Phase 2! 🚀

We can:
1. Start with 13-input model (add 3 context features)
2. Retrain on existing + new data
3. Test on specialist training
4. See if it learns strategic relationships

Want to build this now?
